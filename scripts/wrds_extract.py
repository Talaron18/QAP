"""Extract raw WRDS tables to parquet under data/raw/ (git-ignored).

Run where your WRDS credentials live:
    WRDS_USERNAME=you python scripts/wrds_extract.py [dataset ...]
Datasets: crsp_monthly crsp_delistings ccm_links compustat_annual compustat_quarterly ff_monthly (default: all)
plus crsp_daily_agg (per-year server-side liquidity aggregation; resumable; run it explicitly),
rq_data kpss_patents patent_grants (Study B), arc_data risk_text blockholders class_boards corp_culture fed_litigation (Study C, v9 alt-data), short_interest (Study E, v12; comp.sec_shortint), comp_secm (paper trading), comp_secm_hist (1985- history for the ML check), crsp_names (ETF check) jkp_us (Study A: one parquet per year, resumable) and jkp_intl (v10: developed ex-US, one parquet per year, resumable).
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


def extract_jkp_intl(db, out: pathlib.Path, first_year: int = 1990, last_year: int = 2025) -> None:
    """JKP developed ex-US characteristics (pre-registration v10), one parquet per year (resumable). Required columns are validated."""
    import pandas as pd
    from qap import config
    from qap.extract import jkp_intl_sql
    from qap.jkp_themes import ALL_CHARACTERISTICS
    lib, table = config.JKP_TABLE.split(".")
    have = set(db.describe_table(library=lib, table=table)["name"])
    required = ["id", "excntry", "eom", "size_grp", "ret_exc_lead1m", "me", "bidaskhl_21d", "gp_at",
                "common", "primary_sec", "exch_main", "obs_main"]
    absent = [c for c in required if c not in have]
    if absent:
        raise SystemExit(f"[jkp_intl] required columns absent from {config.JKP_TABLE}: {absent}. Run "
                         "`python scripts/wrds_probe.py contrib_global_factor.global_factor --full` and paste it.")
    cols = [c for c in ALL_CHARACTERISTICS if c in have]
    missing = [c for c in ALL_CHARACTERISTICS if c not in have]
    parts = out / "jkp_intl"
    parts.mkdir(exist_ok=True)
    (parts / "_missing_columns.txt").write_text("\n".join(missing) + "\n")
    print(f"[jkp_intl] {len(cols)} of {len(ALL_CHARACTERISTICS)} theme members present; missing: {missing}", flush=True)
    for y in range(first_year, last_year + 1):
        f = parts / f"{y}.parquet"
        if f.exists():
            continue
        df = db.raw_sql(jkp_intl_sql(cols, y, config.DEV_EX_US))
        df["id"] = df["id"].astype("int64")          # never float32: ids exceed 2**24
        for c in df.columns:
            if c not in ("id", "excntry", "size_grp", "eom") and str(df[c].dtype) in ("Float64", "float64", "object"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        df["me"] = pd.to_numeric(df["me"], errors="coerce").astype("float64")
        df.to_parquet(f, index=False)
        print(f"[jkp_intl] {y}: {len(df):,} rows", flush=True)


def extract_comp_secm(db, out: pathlib.Path, start: str = "2022-01-01", name: str = "comp_secm") -> None:
    """Compustat monthly security file for paper trading. Required columns are validated, never guessed."""
    from qap import config
    from qap.extract import comp_secm_sql
    lib, table = config.COMP_SECM.split(".")
    have = set(db.describe_table(library=lib, table=table)["name"])
    absent = [c for c in config.COMP_SECM_REQUIRED if c not in have]
    if absent:
        raise SystemExit(f"[comp_secm] required columns absent from {config.COMP_SECM}: {absent}. Available: {sorted(have)}")
    cols = config.COMP_SECM_REQUIRED + [c for c in config.COMP_SECM_OPTIONAL if c in have]
    sql = comp_secm_sql(cols, start)
    print(f"[comp_secm] {sql}", flush=True)
    df = db.raw_sql(sql, date_cols=None)
    df.to_parquet(out / f"{name}.parquet", index=False)
    print(f"[{name}] {len(df):,} rows, last datadate {df['datadate'].max()} -> {out / (name + '.parquet')}", flush=True)


def main(names) -> None:
    db = connect()
    out = pathlib.Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    if "comp_secm" in names or "comp_secm_hist" in names:
        if "comp_secm" in names:
            extract_comp_secm(db, out)
        if "comp_secm_hist" in names:
            extract_comp_secm(db, out, start="1985-01-01", name="comp_secm_hist")      # history for the ML check (v8)
        names = [n for n in names if n not in ("comp_secm", "comp_secm_hist")]
        if not names:
            return
    if "jkp_us" in names:
        extract_jkp_us(db, out)
        names = [n for n in names if n != "jkp_us"]
        if not names:
            return
    if "jkp_intl" in names:
        extract_jkp_intl(db, out)
        names = [n for n in names if n != "jkp_intl"]
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
