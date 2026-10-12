"""Extract Compustat North America daily security prices (comp_na_daily_all.secd) for a few tickers (personal journal).

    WRDS_USERNAME=you python scripts/extract_secd.py [TICKER ...] [--start 2023-01-01] [--gvkey-iid GVKEY:IID ...] [--out PATH]

No tickers given -> the project default list (config.SECD_DEFAULT_TICKERS). Writes data/raw/secd_tickers.parquet (git-ignored): one
table, all requested securities, USD rows only (when the curcdd column exists), datadate as datetime64.
Required columns (gvkey iid datadate tic prccd) are validated against the real table; the optional ones that exist are taken.
A ticker that maps to several (gvkey, iid) securities since --start (ticker reuse, share classes) is REFUSED with the candidates
printed; pick one with --gvkey-iid (e.g. --gvkey-iid 012345:01). Run scripts/probe_secd.py first to see the candidates.
Not part of wrds_extract.py's QUERIES because it takes arguments (documented there).
"""
import argparse
import pathlib
import sys

import pandas as pd

from qap import config
from qap.extract import secd_candidates_sql, secd_columns, secd_sql

DEFAULT_OUT = "data/raw/secd_tickers.parquet"


def parse_pair(s: str) -> tuple[str, str]:
    if s.count(":") != 1 or not all(p.strip() for p in s.split(":")):
        raise SystemExit(f"--gvkey-iid expects GVKEY:IID, got {s!r}")
    g, i = s.split(":")
    return g.strip(), i.strip()


def resolve(cand: pd.DataFrame, tickers: list[str], chosen: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """cand: candidates frame (tic, gvkey, iid, ...). Returns (pairs to extract, tickers with no match, refusal messages)."""
    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    refuse: list[str] = []
    chosen_set = set(chosen)
    for t in tickers:
        sub = cand[cand["tic"].astype(str).str.upper() == t] if len(cand) else cand
        secs = sorted({(str(g), str(i)) for g, i in zip(sub["gvkey"], sub["iid"])}) if len(sub) else []
        if not secs:
            missing.append(t)
            continue
        pick = [s for s in secs if s in chosen_set]
        if len(secs) == 1:
            pick = secs
        elif len(pick) != 1:
            why = "several securities" if not pick else "several --gvkey-iid choices"
            lines = [f"{t}: {why}; pass exactly one of these as --gvkey-iid GVKEY:IID:"]
            for r in sub.sort_values(["gvkey", "iid", "last_date"]).itertuples():
                lines.append(f"    {r.gvkey}:{r.iid}  {getattr(r, 'conm', '')}  tpci={getattr(r, 'tpci', '')}  n={r.n}  "
                             f"{str(r.first_date)[:10]} .. {str(r.last_date)[:10]}")
            refuse.append("\n".join(lines))
            continue
        pairs += [p for p in pick if p not in pairs]
    pairs += [p for p in chosen if p not in pairs]
    return pairs, missing, refuse


def summarize(df: pd.DataFrame) -> list[str]:
    out = []
    for (g, i), d in df.groupby(["gvkey", "iid"], sort=True):
        d = d.sort_values("datadate")
        out.append(f"  {d['tic'].iloc[-1]:<8} {g}:{i}  rows {len(d):>6,}  {d['datadate'].iloc[0]:%Y-%m-%d} .. "
                   f"{d['datadate'].iloc[-1]:%Y-%m-%d}  last close {d['prccd'].iloc[-1]:.4f}")
    return out


def run(db, tickers: list[str], chosen: list[tuple[str, str]], start: str, out_path: pathlib.Path) -> pd.DataFrame:
    lib, table = config.COMP_SECD.split(".")
    have = list(db.describe_table(library=lib, table=table)["name"])
    cols, absent = secd_columns(have)
    if absent:
        raise SystemExit(f"[secd] required columns absent from {config.COMP_SECD}: {absent}. Available: {sorted(have)}")
    if "curcdd" not in cols:
        print("[secd] WARNING: no curcdd column, rows are not filtered to USD", flush=True)
    sql = secd_candidates_sql(cols, chosen, tickers, start)
    print(f"[secd] {sql}", flush=True)
    cand = db.raw_sql(sql, date_cols=None)
    pairs, missing, refuse = resolve(cand, tickers, chosen)
    if missing:
        print(f"[secd] no match since {start}: {', '.join(missing)}", flush=True)
    if refuse:
        raise SystemExit("[secd] REFUSED, ambiguous tickers:\n" + "\n".join(refuse))
    if not pairs:
        raise SystemExit("[secd] nothing matched; run scripts/probe_secd.py")
    sql = secd_sql(cols, gvkey_iids=pairs, start=start)
    print(f"[secd] {sql}", flush=True)
    df = db.raw_sql(sql, date_cols=None)
    df["datadate"] = pd.to_datetime(df["datadate"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[secd] {len(df):,} rows -> {out_path}", flush=True)
    for line in summarize(df):
        print(line, flush=True)
    return df


def main(argv=None, db=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--gvkey-iid", action="append", default=[], metavar="G:I")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    tickers = [t.strip().upper() for t in (a.tickers or config.SECD_DEFAULT_TICKERS)]
    if db is None:
        from qap.wrds_client import connect
        db = connect()
    run(db, tickers, [parse_pair(x) for x in a.gvkey_iid], a.start, pathlib.Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
