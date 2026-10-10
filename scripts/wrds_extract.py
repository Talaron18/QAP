"""Extract raw WRDS tables to parquet under data/raw/ (git-ignored).

Run where your WRDS credentials live:
    WRDS_USERNAME=you python scripts/wrds_extract.py [dataset ...]
Datasets: crsp_monthly crsp_delistings ccm_links compustat_annual compustat_quarterly ff_monthly (default: all)
plus crsp_daily_agg (per-year server-side liquidity aggregation; resumable; run it explicitly).
"""
import pathlib
import sys

from qap.extract import QUERIES
from qap.wrds_client import connect


def extract_daily_agg(db, out: pathlib.Path, first_year: int = 1963, last_year: int = 2025) -> None:
    """Per-year server-side aggregation; each year is a resumable part file (existing parts are kept)."""
    import pandas as pd
    from qap.extract import crsp_daily_agg_sql
    parts = out / "crsp_daily_agg_parts"
    parts.mkdir(exist_ok=True)
    for y in range(first_year, last_year + 1):
        f = parts / f"{y}.parquet"
        if f.exists():
            continue
        df = db.raw_sql(crsp_daily_agg_sql(f"{y}-01-01", f"{y}-12-31"))
        df.to_parquet(f, index=False)
        print(f"[crsp_daily_agg] {y}: {len(df):,} rows", flush=True)
    full = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True)
    full.to_parquet(out / "crsp_daily_agg.parquet", index=False)
    print(f"[crsp_daily_agg] {len(full):,} rows -> {out / 'crsp_daily_agg.parquet'}", flush=True)


def main(names) -> None:
    db = connect()
    out = pathlib.Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    if "crsp_daily_agg" in names:
        extract_daily_agg(db, out)
        names = [n for n in names if n != "crsp_daily_agg"]
        if not names:
            return
    for name in names or QUERIES:
        sql = QUERIES[name]()
        print(f"[{name}] {sql}", flush=True)
        df = db.raw_sql(sql, date_cols=None)
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"[{name}] {len(df):,} rows -> {out / (name + '.parquet')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
