"""Extraction for Study D (pre-registration v11): daily CRSP returns and the daily FF market factor.

    WRDS_USERNAME=you python scripts/extract_earnings.py [crsp_daily_ret] [ff_daily]     # default: both
Writes data/raw/crsp_daily_ret/{year}.parquet (permno int32, dlycaldt, dlyret float32; years 1989..2025; resumable: existing years are
kept, each year is written to a hidden temp file and renamed so a crash never leaves a partial year) and data/raw/ff_daily.parquet
(date, mktrf, rf as delivered; units are detected when the file is used). Compustat quarterly comes from `wrds_extract.py compustat_quarterly`.
"""
import os
import pathlib
import sys

import pandas as pd

from qap import config
from qap.earnings import FIRST_YEAR, LAST_YEAR, crsp_daily_ret_sql, ff_daily_columns, ff_daily_sql


def extract_crsp_daily_ret(db, out: pathlib.Path, first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> None:
    parts = out / "crsp_daily_ret"
    parts.mkdir(parents=True, exist_ok=True)
    for y in range(first_year, last_year + 1):
        f = parts / f"{y}.parquet"
        if f.exists():
            print(f"[crsp_daily_ret] {y}: kept", flush=True)
            continue
        df = db.raw_sql(crsp_daily_ret_sql(y), date_cols=None)
        df = pd.DataFrame({"permno": pd.to_numeric(df["permno"]).astype("int32"), "dlycaldt": pd.to_datetime(df["dlycaldt"]),
                           "dlyret": pd.to_numeric(df["dlyret"], errors="coerce").astype("float32")})
        tmp = parts / f".{y}.part"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, f)
        print(f"[crsp_daily_ret] {y}: {len(df):,} rows", flush=True)


def extract_ff_daily(db, out: pathlib.Path) -> None:
    lib, table = config.FF_FACTORS_DAILY.split(".")
    have = list(db.describe_table(library=lib, table=table)["name"])
    try:
        cols = ff_daily_columns(have)
    except KeyError as e:
        raise SystemExit(f"[ff_daily] {e.args[0]}")
    df = db.raw_sql(ff_daily_sql(cols), date_cols=None)
    df = pd.DataFrame({"date": pd.to_datetime(df["date"]), "mktrf": pd.to_numeric(df["mktrf"], errors="coerce"),
                       "rf": pd.to_numeric(df["rf"], errors="coerce")})
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "ff_daily.parquet", index=False)
    print(f"[ff_daily] {len(df):,} rows {df['date'].min().date()}..{df['date'].max().date()}; max |mktrf| {df['mktrf'].abs().max():.4g} "
          f"({'percent' if df[['mktrf', 'rf']].abs().max().max() > 1.5 else 'decimal'} units) -> {out / 'ff_daily.parquet'}", flush=True)


def main(names) -> None:
    from qap.wrds_client import connect
    names = list(names) or ["crsp_daily_ret", "ff_daily"]
    bad = [n for n in names if n not in ("crsp_daily_ret", "ff_daily")]
    if bad:
        raise SystemExit(f"unknown dataset(s) {bad}; choose from crsp_daily_ret ff_daily")
    db = connect()
    out = pathlib.Path("data/raw")
    if "ff_daily" in names:
        extract_ff_daily(db, out)
    if "crsp_daily_ret" in names:
        extract_crsp_daily_ret(db, out)


if __name__ == "__main__":
    main(sys.argv[1:])
