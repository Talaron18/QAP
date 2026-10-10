"""Extract raw WRDS tables to parquet under data/raw/ (git-ignored).

Run where your WRDS credentials live:
    WRDS_USERNAME=you python scripts/wrds_extract.py [dataset ...]
Datasets: crsp_monthly ccm_links compustat_annual compustat_quarterly ff_monthly (default: all).
"""
import pathlib
import sys

from qap.extract import QUERIES
from qap.wrds_client import connect


def main(names) -> None:
    db = connect()
    out = pathlib.Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    for name in names or QUERIES:
        sql = QUERIES[name]()
        print(f"[{name}] {sql}", flush=True)
        df = db.raw_sql(sql, date_cols=None)
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"[{name}] {len(df):,} rows -> {out / (name + '.parquet')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
