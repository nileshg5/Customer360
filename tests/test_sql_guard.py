"""SQL guard: static checks, LIMIT handling, and the read-only execution layer."""
from __future__ import annotations

import pytest

from src.publish import write_sqlite
from src.qa.sql_guard import GuardError, SqlExecutionError, guard_sql, open_readonly, run_readonly
from src.enrich import enrich
from src.merge import merge
from src.validate import validate
from tests.conftest import crm_row, order_row, staged, ticket_row

TABLES = ["customer_360", "rejected_records"]
BLOCKED = ["email"]


def g(sql: str, max_rows: int = 200) -> str:
    return guard_sql(sql, TABLES, BLOCKED, max_rows)


@pytest.mark.parametrize("sql,reason", [
    ("DROP TABLE customer_360", "only SELECT"),
    ("SELECT 1; DROP TABLE customer_360", "single SQL statement"),
    ("SELECT customer_id FROM customer_360; SELECT city FROM customer_360", "single SQL statement"),
    ("PRAGMA table_info(customer_360)", "only SELECT"),
    ("SELECT customer_id FROM customer_360 WHERE 1 = (PRAGMA query_only)", "PRAGMA"),
    ("ATTACH DATABASE 'other.db' AS o", "only SELECT"),
    ("WITH x AS (SELECT 1) DELETE FROM customer_360", "DELETE"),
    ("SELECT customer_id FROM customer_360 /* ; DROP TABLE customer_360 */", "comments"),
    ("SELECT customer_id FROM customer_360 -- ; DROP TABLE customer_360", "comments"),
    ("SELECT 1 /**/; DROP TABLE customer_360", "comments"),
    ("SELECT email FROM customer_360", "email"),
    ('SELECT "Email" FROM customer_360', "email"),
    ("SELECT c.email FROM customer_360 c", "email"),
    ("SELECT * FROM customer_360", "SELECT *"),
    ("SELECT c.* FROM customer_360 c", "SELECT *"),
    ("SELECT customer_id FROM (SELECT * FROM customer_360)", "SELECT *"),
    ("SELECT name FROM sqlite_master", "unknown or blocked table"),
    ("SELECT x FROM orders", "unknown or blocked table"),
    ("SELECT customer_id FROM customer_360 JOIN secrets s ON 1=1", "unknown or blocked table"),
    ("SELECT customer_id FROM other.customer_360", "schema"),
    ("SELECT customer_id FROM customer_360 UNION SELECT email FROM customer_360", "email"),
    ("SELECT customer_id FROM customer_360 UNION ALL SELECT name FROM sqlite_master", "unknown or blocked table"),
    ("SELECT customer_id FROM customer_360 WHERE customer_id = ?", "parameters"),
    ("SELECT load_extension('x')", "LOAD_EXTENSION"),
    ("SELECT customer_id FROM customer_360 LIMIT (SELECT 5)", "plain number"),
    ("", "empty"),
])
def test_guard_rejects(sql, reason):
    with pytest.raises(GuardError, match=reason):
        g(sql)


@pytest.mark.parametrize("sql", [
    "SELECT customer_id, lifetime_value FROM customer_360 ORDER BY lifetime_value DESC",
    "select city, count(*) as n from customer_360 group by city",
    "WITH s AS (SELECT segment, COUNT(*) AS n FROM customer_360 GROUP BY segment) SELECT segment, n FROM s",
    "WITH s(seg, n) AS (SELECT segment, COUNT(*) FROM customer_360 GROUP BY segment) SELECT seg, n FROM s",
    "SELECT c.customer_id, r.rule_codes FROM customer_360 c JOIN rejected_records r ON r.customer_id = c.customer_id",
    "SELECT a.customer_id FROM customer_360 a, rejected_records b WHERE a.customer_id = b.customer_id",
    "SELECT name FROM customer_360 WHERE name = 'drop table; -- x'",     # keywords inside a string
    "SELECT customer_id, lifetime_value * 2 AS doubled FROM customer_360",
    "SELECT COUNT(*) FROM customer_360;",
])
def test_guard_accepts(sql):
    out = g(sql)
    assert out.upper().startswith(("SELECT", "WITH")) and out.endswith("LIMIT 200")


def test_limit_injected_capped_and_kept():
    assert g("SELECT customer_id FROM customer_360").endswith("LIMIT 200")
    assert g("SELECT customer_id FROM customer_360 LIMIT 5000").endswith("LIMIT 200")
    assert g("SELECT customer_id FROM customer_360 LIMIT 5").endswith("LIMIT 5")
    assert g("SELECT customer_id FROM customer_360 LIMIT 10, 900").endswith("LIMIT 10, 200")
    sub = g("SELECT customer_id FROM (SELECT customer_id FROM customer_360 LIMIT 999)")
    assert sub.endswith("LIMIT 999) LIMIT 200")                 # only the outer LIMIT matters


# ---------- execution layer ----------

@pytest.fixture()
def db(cfg, tmp_path):
    res = validate(staged([crm_row("C0001"), crm_row("C0002")],
                          [order_row("C0001"), order_row("C0002", lifetime_value="-1")],
                          [ticket_row("C0001")]), cfg, "TEST")
    return write_sqlite(enrich(merge(res.clean, cfg), cfg), res.rejected, tmp_path)


@pytest.mark.parametrize("raw", [
    "SELECT * FROM customer_360",                 # expands to include email -> authorizer denies
    "SELECT email FROM customer_360",
    "SELECT name FROM sqlite_master",
    "DELETE FROM customer_360",
    "UPDATE customer_360 SET city = 'X'",
    "DROP TABLE customer_360",
    "PRAGMA query_only = OFF",
    "WITH x AS (SELECT email AS e FROM customer_360) SELECT e FROM x",        # email hidden in a CTE
    "WITH x AS (SELECT name FROM sqlite_master) SELECT name FROM x",
])
def test_authorizer_blocks_even_without_static_guard(db, raw):
    con = open_readonly(db, TABLES, BLOCKED)
    with pytest.raises(GuardError):
        run_readonly(con, raw, 5)
    con.close()


def test_readonly_connection_runs_valid_query(db):
    con = open_readonly(db, TABLES, BLOCKED)
    cols, rows = run_readonly(con, g("SELECT customer_id, segment FROM customer_360 ORDER BY customer_id"), 5)
    assert cols == ["customer_id", "segment"] and rows == [("C0001", "Silver"), ("C0002", "Unknown")]
    con.close()


def test_query_timeout(db):
    con = open_readonly(db, TABLES, BLOCKED)
    slow = g("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n")
    with pytest.raises(SqlExecutionError, match="timed out"):
        run_readonly(con, slow, 0.05)
    con.close()


def test_unknown_column_is_execution_error(db):
    con = open_readonly(db, TABLES, BLOCKED)
    with pytest.raises(SqlExecutionError, match="no such column"):
        run_readonly(con, g("SELECT nope FROM customer_360"), 5)
    con.close()
