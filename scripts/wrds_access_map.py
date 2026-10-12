"""Access map of everything WRDS shows this account, with a focus on option / volatility / borrow data.

Visible is not accessible: the 90 `cboe` tables were listed but every query failed with 'permission denied for schema cboe_eod',
and the one table that is NOT an option file (`cboe.cboe`, the Cboe index history) was never queried. This script settles both questions:

    WRDS_USERNAME=you python scripts/wrds_access_map.py [extra_regex]

1. One catalogue query (has_schema_privilege / has_table_privilege) for every schema and relation: what the database says you may read.
2. A real `select * ... limit 1` on every option / volatility / index / borrow-like table, and on the first two tables of every other
   library, because a view can carry SELECT privilege while its underlying schema is denied.
3. For tables that really answer, the columns, an exact row count and the date range.
Writes (git-ignored) data/wrds_access_map.txt (paste this back) and data/wrds_access_map.csv (every relation). No data is downloaded
beyond one row per probe; every statement has a 60 s timeout and its own try/except.
"""
import pathlib
import re
import sys

import pandas as pd

from qap.wrds_client import connect

STRONG = re.compile(r"opt|(^|_)iv|vol(a|\b|_)|vix|vxo|vxn|cboe|skew|greek|implied|borrow|lend|vrp|deriv|futur|swap|ovs|hvol|delta", re.I)
CATALOGUE = """
select n.nspname as lib, c.relname as tbl, c.relkind as kind, cast(c.reltuples as bigint) as approx_rows,
       has_schema_privilege(n.oid, 'USAGE') as usage_ok, has_table_privilege(c.oid, 'SELECT') as select_ok
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where c.relkind in ('r', 'v', 'm', 'f', 'p') and n.nspname not in ('pg_catalog', 'information_schema') and n.nspname not like 'pg_%%'
"""
DATE_NAMES = ("date", "datadate", "caldt", "dateff", "trade_date", "tradedate", "quote_date", "exdate")


def try_sql(db, sql):
    try:
        return db.raw_sql(sql)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {' '.join(str(e).split())[:140]}"


def classify(res) -> str:
    if not isinstance(res, str):
        return "OK"
    return "DENIED" if "permission denied" in res.lower() else "ERROR"


def main(extra=None) -> None:
    pat = re.compile(f"{STRONG.pattern}|{extra}", re.I) if extra else STRONG
    db = connect()
    try_sql(db, "set statement_timeout = 60000")
    out = []

    def log(m=""):
        print(m, flush=True)
        out.append(str(m))

    cat = try_sql(db, CATALOGUE)
    if isinstance(cat, str):
        log(f"catalogue query failed ({cat}); falling back to list_libraries/list_tables with real queries only")
        libs = sorted(db.list_libraries())
        cat = pd.concat([pd.DataFrame({"lib": l, "tbl": db.list_tables(library=l)}) for l in libs], ignore_index=True)
        cat["kind"], cat["approx_rows"], cat["usage_ok"], cat["select_ok"] = "?", -1, None, None
    cat["optionlike"] = [bool(pat.search(f"{l} {t}")) for l, t in zip(cat["lib"], cat["tbl"])]
    log(f"# {cat['lib'].nunique()} schemas, {len(cat):,} relations visible; {int(cat['optionlike'].sum())} look option/volatility/index/borrow-like")
    sch = cat.groupby("lib").agg(n=("tbl", "size"), usage=("usage_ok", "first"), sel=("select_ok", "mean"), opt=("optionlike", "sum"))
    log("\n## schemas where the catalogue says USAGE is NOT granted (queries into them will fail):")
    log("  " + (", ".join(f"{i}({r.n})" for i, r in sch[sch["usage"] == False].iterrows()) or "none"))  # noqa: E712

    # real queries: every option-like relation (cap 400), plus the first two relations of every schema
    first_two = cat.sort_values(["lib", "tbl"]).groupby("lib").head(2)
    todo = pd.concat([cat[cat["optionlike"]].head(400), first_two]).drop_duplicates(["lib", "tbl"])
    status, msg = [], []
    for r in todo.itertuples():
        res = try_sql(db, f"select * from {r.lib}.{r.tbl} limit 1")
        status.append(classify(res))
        msg.append("" if not isinstance(res, str) else res)
    todo = todo.assign(status=status, message=msg)
    cat = cat.merge(todo[["lib", "tbl", "status", "message"]], on=["lib", "tbl"], how="left")
    pathlib.Path("data").mkdir(exist_ok=True)
    cat.to_csv("data/wrds_access_map.csv", index=False)

    log("\n## real-query result per schema (tested relations only): OK / DENIED / ERROR")
    t = todo.groupby("lib")["status"].value_counts().unstack(fill_value=0)
    for lib, r in t.iterrows():
        log(f"  {lib:34s} OK {int(r.get('OK', 0)):3d}  DENIED {int(r.get('DENIED', 0)):3d}  ERROR {int(r.get('ERROR', 0)):3d}")

    log("\n## OPTION / VOLATILITY / INDEX / BORROW-LIKE relations that ANSWER a real query (columns, rows, date range)")
    ok = todo[(todo["status"] == "OK") & todo["optionlike"]]
    if ok.empty:
        log("  none")
    for r in ok.head(80).itertuples():
        full = f"{r.lib}.{r.tbl}"
        d = try_sql(db, f"select column_name, data_type from information_schema.columns where table_schema = '{r.lib}' and table_name = '{r.tbl}' order by ordinal_position")
        cols = [] if isinstance(d, str) else list(d["column_name"])
        log(f"\n{full}: {len(cols)} columns: " + ", ".join(cols[:60]))
        n = try_sql(db, f"select count(*) as n from {full}")
        log("  rows: " + (f"{int(n['n'].iloc[0]):,}" if not isinstance(n, str) else n))
        dc = [c for c in cols if c.lower() in DATE_NAMES]
        if dc:
            x = try_sql(db, f"select min({dc[0]}) as lo, max({dc[0]}) as hi from {full}")
            log(f"  {dc[0]} range: " + (f"{x['lo'].iloc[0]} .. {x['hi'].iloc[0]}" if not isinstance(x, str) else x))
        e = try_sql(db, f"select * from {full} limit 2")
        if not isinstance(e, str):
            for row in e.to_dict("records"):
                log("  eg: " + str({k: str(v)[:24] for k, v in list(row.items())[:12]}))
    log("\n## option-like relations that were DENIED or ERRORED (so we know what is behind the wall)")
    bad = todo[(todo["status"] != "OK") & todo["optionlike"]]
    for lib, g in bad.groupby("lib"):
        log(f"  {lib}: {len(g)} relations, e.g. {g['tbl'].iloc[0]} -> {g['message'].iloc[0][:110]}")
    pathlib.Path("data/wrds_access_map.txt").write_text("\n".join(out) + "\n")
    log("\nwrote data/wrds_access_map.txt and data/wrds_access_map.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
