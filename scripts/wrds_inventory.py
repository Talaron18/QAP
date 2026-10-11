"""List EVERY WRDS library this account can see, with table counts and approximate row counts.

The first inspector only looked at libraries whose names matched a few hints (crsp, comp, ibes, ff, ...), so
products such as Thomson 13F/insiders, Cboe, Blockholders, TRACE, Contributed Data, Macro Finance Society,
Historical SPDJI or Intraday Indicators were never listed. Run where your WRDS credentials live:

    WRDS_USERNAME=you python scripts/wrds_inventory.py

Writes (git-ignored): data/wrds_inventory.txt (one block per library: tables, approx rows, first table names;
paste this back) and data/wrds_inventory_tables.csv (every table). Row counts come from the Postgres catalogue
(pg_class.reltuples) and are approximate; -1/0 means 'unknown' (views), not 'empty'. Libraries whose name contains
'samp' are sample data and not usable for research.
"""
import pathlib

import pandas as pd

from qap.wrds_client import connect

SQL = """
select n.nspname as lib, c.relname as tbl, cast(c.reltuples as bigint) as approx_rows, c.relkind as kind
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where c.relkind in ('r', 'v', 'm', 'f', 'p') and n.nspname = any(%(libs)s)
"""


def main() -> None:
    db = connect()
    libs = sorted(db.list_libraries())
    try:
        t = db.raw_sql(SQL, params={"libs": libs})
    except Exception as e:  # catalogue access can be restricted: fall back to table names only
        print(f"catalogue query failed ({e!r}); listing table names only", flush=True)
        t = pd.concat([pd.DataFrame({"lib": l, "tbl": db.list_tables(library=l), "approx_rows": -1, "kind": "?"}) for l in libs],
                      ignore_index=True)
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True)
    t.sort_values(["lib", "tbl"]).to_csv(out / "wrds_inventory_tables.csv", index=False)
    lines = [f"# {len(libs)} libraries visible to this account"]
    for lib, g in t.groupby("lib"):
        rows = g["approx_rows"].clip(lower=0)
        flag = "SAMPLE " if "samp" in lib.lower() else ""
        top = g.sort_values("approx_rows", ascending=False).head(8)
        lines.append(f"\n## {flag}{lib}: {len(g)} tables, ~{int(rows.sum()):,} rows (catalogue estimate)")
        lines.append("  biggest: " + ", ".join(f"{r.tbl}({int(r.approx_rows):,})" for r in top.itertuples()))
        names = sorted(g["tbl"])
        lines.append("  tables: " + ", ".join(names[:40]) + (f" ... (+{len(names) - 40} more)" if len(names) > 40 else ""))
    (out / "wrds_inventory.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print(f"\nwrote {out / 'wrds_inventory.txt'} ({len(libs)} libraries)")


if __name__ == "__main__":
    main()
