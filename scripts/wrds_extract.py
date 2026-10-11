"""Extract raw WRDS tables to parquet under data/raw/ (git-ignored).

Run where your WRDS credentials live:
    WRDS_USERNAME=you python scripts/wrds_extract.py [dataset ...]
Datasets: crsp_monthly crsp_delistings ccm_links compustat_annual compustat_quarterly ff_monthly (default: all)
plus crsp_daily_agg (per-year server-side liquidity aggregation; resumable; run it explicitly),
rq_data kpss_patents patent_grants (Study B) and jkp_us (Study A: one parquet per year, resumable).
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


def extract_jkp_us(db, out: pathlib.Path, first_year: int = 1963, last_year: int = 2025) -> None:
    """JKP US characteristics, one parquet per year (resumable). Missing theme-member columns are reported, never guessed."""
    import pandas as pd
    from qap import config
    from qap.extract import jkp_us_sql
    from qap.jkp_themes import ALL_CHARACTERISTICS
    lib, table = config.JKP_TABLE.split(".")
    have = set(db.describe_table(library=lib, table=table)["name"])
    required = ["permno", "eom", "size_grp", "ret_exc_lead1m", "me", "excntry", "common", "primary_sec", "exch_main", "obs_main"]
    absent = [c for c in required if c not in have]
    if absent:
        raise SystemExit(f"[jkp_us] required columns absent from {config.JKP_TABLE}: {absent}. Run "
                         "`python scripts/wrds_probe.py contrib_global_factor.global_factor --full` and paste it.")
    cols =[c for c in ALL_CHARACTERISTICS if c in have]
    missing = [c for c in ALL_CHARACTERISTICS if c not in have]
    parts = out / "jkp_us"
    parts.mkdir(exist_ok=True)
    (parts / "_missing_columns.txt").write_text("\n".join(missing) + "\n")
    print(f"[jkp_us] {len(cols)} of {len(ALL_CHARACTERISTICS)} theme members present; missing: {missing}", flush=True)
    for y in range(first_year, last_year + 1):
        f = parts / f"{y}.parquet"
        if f.exists():
            continue
        df = db.raw_sql(jkp_us_sql(cols, y))
        for c in df.columns:
            if str(df[c].dtype) in ("Float64", "float64", "object") and c not in ("size_grp", "eom"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        df.to_parquet(f, index=False)
        print(f"[jkp_us] {y}: {len(df):,} rows", flush=True)


def main(names) -> None:
    db = connect()
    out = pathlib.Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    if "jkp_us" in names:
        extract_jkp_us(db, out)
        names = [n for n in names if n != "jkp_us"]
        if not names:
            return
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
