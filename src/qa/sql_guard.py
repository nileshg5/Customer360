"""Validate LLM-written SQL and run it on a read-only SQLite connection.

Layers (each one alone should stop a write; together they also stop data leaks):
  1. tokenizer (string-literal aware): no comments, no parameters, single statement
  2. statement must start with SELECT or WITH; write/DDL/PRAGMA/ATTACH keywords rejected
  3. only allowed tables (plus CTE names); blocked columns (email) rejected; SELECT * rejected
  4. outer LIMIT injected, or capped to max_rows
  5. execution: file opened mode=ro, PRAGMA query_only=ON, authorizer that allows only
     SELECT/READ on allowed tables and columns, progress-handler timeout
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

MAX_SQL_CHARS = 4000

FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "REPLACE", "UPSERT", "MERGE", "DROP", "CREATE", "ALTER",
    "TRUNCATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE", "BEGIN", "COMMIT",
    "ROLLBACK", "SAVEPOINT", "RELEASE", "TRANSACTION", "GRANT", "REVOKE", "LOAD_EXTENSION",
}
BLOCKED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
TABLE_FUNCTIONS = {"json_each", "json_tree"}

_TOKEN_RE = re.compile(r"""
     (?P<ws>\s+)
    |(?P<comment>--[^\n]*|/\*.*?(?:\*/|\Z))
    |(?P<string>'(?:[^']|'')*')
    |(?P<qident>"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\])
    |(?P<number>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+)
    |(?P<word>[A-Za-z_][A-Za-z0-9_$]*)
    |(?P<param>[?:@$][A-Za-z0-9_]*)
    |(?P<op><>|<=|>=|!=|==|\|\||<<|>>|[-+*/%<>=(),.;&|~])
""", re.S | re.X)


class GuardError(ValueError):
    """SQL rejected by policy. `reason` is safe to show to the user."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SqlExecutionError(RuntimeError):
    """SQL passed the guard but SQLite could not run it (e.g. unknown column)."""


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    start: int
    end: int

    @property
    def upper(self) -> str:
        return self.text.upper()

    @property
    def ident(self) -> Optional[str]:
        """Lower-cased identifier name for word / quoted-identifier tokens."""
        if self.kind == "word":
            return self.text.lower()
        if self.kind == "qident":
            return self.text[1:-1].replace('""', '"').lower()
        return None


def tokenize(sql: str) -> List[Token]:
    tokens, pos = [], 0
    while pos < len(sql):
        m = _TOKEN_RE.match(sql, pos)
        if not m:
            raise GuardError(f"could not parse the SQL near: {sql[pos:pos + 20]!r}")
        if m.lastgroup != "ws":
            tokens.append(Token(m.lastgroup, m.group(), m.start(), m.end()))
        pos = m.end()
    return tokens


def _cte_names(tokens: List[Token]) -> Set[str]:
    """Names defined as `name AS (` or `name(cols) AS (` (only valid in a WITH clause)."""
    names = set()
    for i, t in enumerate(tokens):
        if t.upper == "AS" and i + 1 < len(tokens) and tokens[i + 1].text == "(" and i > 0:
            prev = tokens[i - 1]
            if prev.text == ")":                      # name(col, ...) AS (
                depth, j = 0, i - 1
                while j >= 0:
                    depth += {")": 1, "(": -1}.get(tokens[j].text, 0)
                    if depth == 0:
                        break
                    j -= 1
                prev = tokens[j - 1] if j > 0 else prev
            if prev.ident:
                names.add(prev.ident)
    return names


def _referenced_tables(tokens: List[Token]) -> List[str]:
    """Table names after FROM / JOIN, including comma-separated FROM lists."""
    tables = []
    for i, t in enumerate(tokens):
        if t.upper not in ("FROM", "JOIN"):
            continue
        j = i + 1
        while j < len(tokens):
            tok = tokens[j]
            if tok.text == "(":
                break                                    # sub-query: its own FROM is scanned too
            if not tok.ident:
                raise GuardError(f"unexpected token after {t.upper}: {tok.text!r}")
            name = tok.ident
            if j + 2 < len(tokens) and tokens[j + 1].text == "." and tokens[j + 2].ident:
                if name != "main":
                    raise GuardError(f"schema '{name}' is not allowed")
                j += 2
                name = tokens[j].ident
            is_func = j + 1 < len(tokens) and tokens[j + 1].text == "("
            if is_func and name not in TABLE_FUNCTIONS:
                raise GuardError(f"table function '{name}' is not allowed")
            if not is_func:
                tables.append(name)
            # skip optional alias, then continue only on a comma-separated FROM list
            j += 1
            depth = 0
            while j < len(tokens):
                txt = tokens[j].text
                depth += {"(": 1, ")": -1}.get(txt, 0)
                if depth < 0 or (depth == 0 and (txt == "," or tokens[j].upper in _CLAUSE_END)):
                    break
                j += 1
            if j < len(tokens) and tokens[j].text == "," and t.upper == "FROM":
                j += 1
                continue
            break
    return tables


_CLAUSE_END = {"WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "UNION", "EXCEPT", "INTERSECT",
               "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL", "ON", "USING",
               "WINDOW", "SELECT"}


def _check_star(tokens: List[Token]) -> None:
    for i, t in enumerate(tokens):
        if t.text != "*" or i == 0:
            continue
        prev = tokens[i - 1]
        if prev.upper in ("SELECT", "DISTINCT", "ALL") or prev.text in (",", "."):
            raise GuardError("SELECT * is not allowed; name the columns you need")


def _apply_limit(sql: str, tokens: List[Token], max_rows: int) -> str:
    depth, limit_idx = 0, None
    for i, t in enumerate(tokens):
        depth += {"(": 1, ")": -1}.get(t.text, 0)
        if depth == 0 and t.upper == "LIMIT":
            limit_idx = i
    if limit_idx is None:
        return f"{sql} LIMIT {max_rows}"
    rest = tokens[limit_idx + 1:]
    if len(rest) == 1 and rest[0].kind == "number":
        count_tok = rest[0]                                    # LIMIT n
    elif len(rest) == 3 and rest[0].kind == rest[2].kind == "number" and rest[1].upper == "OFFSET":
        count_tok = rest[0]                                    # LIMIT n OFFSET m
    elif len(rest) == 3 and rest[0].kind == rest[2].kind == "number" and rest[1].text == ",":
        count_tok = rest[2]                                    # LIMIT m, n
    else:
        raise GuardError("LIMIT must be a plain number")
    if float(count_tok.text) <= max_rows:
        return sql
    return f"{sql[:count_tok.start]}{max_rows}{sql[count_tok.end:]}"


def guard_sql(sql: str, allowed_tables: Iterable[str], blocked_columns: Iterable[str],
              max_rows: int) -> str:
    """Return a safe, LIMIT-capped version of `sql`, or raise GuardError."""
    if not isinstance(sql, str) or not sql.strip():
        raise GuardError("empty SQL")
    if len(sql) > MAX_SQL_CHARS:
        raise GuardError("SQL is too long")
    allowed = {t.lower() for t in allowed_tables}
    blocked = {c.lower() for c in blocked_columns}

    tokens = tokenize(sql)
    if any(t.kind == "comment" for t in tokens):
        raise GuardError("SQL comments are not allowed")
    if any(t.kind == "param" for t in tokens):
        raise GuardError("SQL parameters are not allowed")
    while tokens and tokens[-1].text == ";":                  # one trailing ';' is harmless
        tokens.pop()
    if any(t.text == ";" for t in tokens):
        raise GuardError("only a single SQL statement is allowed")
    if not tokens or tokens[0].upper not in ("SELECT", "WITH"):
        raise GuardError("only SELECT queries are allowed")
    bad = sorted({t.upper for t in tokens if t.kind == "word" and t.upper in FORBIDDEN_KEYWORDS})
    if bad:
        raise GuardError(f"read-only access: '{bad[0]}' is not allowed")

    cleaned = sql[:tokens[-1].end].strip()
    hidden = sorted({t.ident for t in tokens if t.ident in blocked})
    if hidden:
        raise GuardError(f"column '{hidden[0]}' is not available")
    _check_star(tokens)
    ctes = _cte_names(tokens)
    unknown = sorted({name for name in _referenced_tables(tokens) if name not in allowed | ctes})
    if unknown:
        raise GuardError(f"unknown or blocked table: '{unknown[0]}'")
    return _apply_limit(cleaned, tokens, max_rows)


# ---------- execution ----------

def _authorizer(allowed_tables: Set[str], blocked_columns: Set[str], real_tables: Set[str]):
    """READ is denied on every real table outside the allow-list and on all sqlite_* tables.
    Names that are not real tables (CTEs) are allowed: their rows can only come from real
    tables, and each of those reads is checked here too."""
    recursive = getattr(sqlite3, "SQLITE_RECURSIVE", 33)

    def check(action, arg1, arg2, _db, _source):
        if action in (sqlite3.SQLITE_SELECT, recursive):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table, column = (arg1 or "").lower(), (arg2 or "").lower()
            if table.startswith("sqlite_") or (table in real_tables and table not in allowed_tables):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_DENY if column in blocked_columns else sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (arg2 or "").lower() in BLOCKED_FUNCTIONS else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return check


def open_readonly(db_path: Path, allowed_tables: Iterable[str], blocked_columns: Iterable[str]
                  ) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(f"database not found: {db_path} (run `python main.py` first)")
    con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    real = {r[0].lower() for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    con.set_authorizer(_authorizer({t.lower() for t in allowed_tables},
                                   {c.lower() for c in blocked_columns}, real))
    return con


def run_readonly(con: sqlite3.Connection, sql: str, timeout_seconds: float
                 ) -> Tuple[List[str], List[tuple]]:
    deadline = time.monotonic() + timeout_seconds
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description or []]
        return columns, cur.fetchall()
    except sqlite3.DatabaseError as exc:
        msg = str(exc)
        if "not authorized" in msg or "prohibited" in msg:
            raise GuardError("the query touches data that is not available") from exc
        if "interrupted" in msg:
            raise SqlExecutionError(f"query timed out after {timeout_seconds}s") from exc
        if "readonly" in msg or "read-only" in msg:
            raise GuardError("read-only access") from exc
        raise SqlExecutionError(msg) from exc
    finally:
        con.set_progress_handler(None, 0)
