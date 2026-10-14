"""Probe Compustat North America daily security prices before extracting them (comp_na_daily_all.secd, alias comp.secd).

    WRDS_USERNAME=you python scripts/probe_secd.py [TICKER ...] [--words SANDISK ASTERA HYNIX]
Default tickers: ORCL INTU LITE BE STX AMD CELH ALAB SNDK SKHY AXTI and the leveraged ETFs SNDQ AXTQ DAMD SKDD.
Prints and writes data/secd_probe.txt (git-ignored; paste it back): per table the column names and types, max/min datadate (max restricted
to datadate >= 2025-06-01 to stay fast), the row count of the last 30 days; then a ticker lookup (every match printed, so ticker reuse such
as SNDK old/new with several gvkeys is visible), a case-insensitive company-name search for tickers with no match, and the last 5 rows of
each matched security. Every statement has a 90 s timeout and its own try/except.
"""
import argparse
import datetime as dt
import pathlib
import re
import sys

import pandas as pd

from qap import config

TABLES = [config.COMP_SECD, config.COMP_SECD_ALIAS]
DEFAULT_WORDS = ["SANDISK", "ASTERA", "HYNIX"]
ID_COLS = ["gvkey", "iid", "tic", "conm", "tpci", "exchg", "curcdd"]
ROW_COLS = ["datadate", "prccd", "prcod", "prchd", "prcld", "cshtrd", "ajexdi", "trfd"]
LOOKUP_START = "2024-01-01"
RECENT_START = "2025-06-01"
SAFE = re.compile(r"^[A-Za-z0-9.\-^ ]{1,40}$")


def try_sql(db, sql):
    """DataFrame, or an 'ERROR ...' string (never raises)."""
    try:
        return db.raw_sql(sql, date_cols=None)
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}: {' '.join(str(e).split())[:200]}"


def check_safe(items: list[str], what: str) -> list[str]:
    bad = [x for x in items if not SAFE.match(x)]
    if bad:
        raise SystemExit(f"{what}: unexpected characters in {bad}")
    return items


def future_sql(table: str, n: int = 50) -> str:
    return (f"select gvkey, iid, tic, conm, datadate from {table} where datadate > current_date "
            f"order by datadate desc limit {n}")


def lookup_sql(table: str, tickers: list[str], cols: list[str], start: str = LOOKUP_START) -> str:
    keys = [c for c in ID_COLS if c in cols]
    tl = ", ".join(f"'{t.upper()}'" for t in check_safe(tickers, "tickers"))
    return (f"select {', '.join(keys)}, count(*) as n, min(datadate) as first_date, max(datadate) as last_date from {table} "
            f"where datadate >= '{start}' and datadate <= current_date and tic in ({tl}) group by {', '.join(keys)} order by tic, max(datadate)")


def words_sql(table: str, words: list[str], cols: list[str], start: str = RECENT_START) -> str:
    keys = [c for c in ID_COLS if c in cols]
    cond = " or ".join(f"conm ilike '%%{w}%%'" for w in check_safe(words, "words"))
    return (f"select {', '.join(keys)}, count(*) as n, min(datadate) as first_date, max(datadate) as last_date from {table} "
            f"where datadate >= '{start}' and datadate <= current_date and ({cond}) group by {', '.join(keys)} order by conm, max(datadate)")


def last_rows_sql(table: str, gvkey: str, iid: str, cols: list[str], start: str = RECENT_START, n: int = 5) -> str:
    """Last n rows of one security on or after `start` (callers pass the security's last_date minus ~30 days, so an old security shows too)."""
    check_safe([str(gvkey), str(iid)], "gvkey/iid")
    sel = [c for c in ROW_COLS if c in cols] or ["datadate", "prccd"]
    return (f"select {', '.join(sel)} from {table} where gvkey = '{gvkey}' and iid = '{iid}' and datadate >= '{start}' "
            f"and datadate <= current_date order by datadate desc limit {n}")


def fmt_frame(df: pd.DataFrame, indent: str = "  ") -> list[str]:
    """Plain right-aligned text table (dates cut to YYYY-MM-DD, floats to 6 significant digits)."""
    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)[:10] if hasattr(v, "year") else str(v)
    rows = [[str(c) for c in df.columns]] + [[cell(v) for v in r] for r in df.itertuples(index=False)]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return [indent + "  ".join(c.rjust(w[i]) for i, c in enumerate(r)) for r in rows]


def ticker_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """tic -> sorted distinct 'gvkey:iid' (more than one entry = ambiguous ticker)."""
    out: dict[str, set] = {}
    for r in df.itertuples():
        out.setdefault(str(r.tic).upper(), set()).add(f"{r.gvkey}:{r.iid}")
    return {t: sorted(v) for t, v in out.items()}


def probe_table(db, table: str, log) -> tuple[list[str] | None, str | None]:
    """Columns/types, max/min datadate, last-30-day row count. Returns (column names or None, max datadate string or None)."""
    log(f"\n## {table}")
    lib, tbl = table.split(".")
    cols = None
    try:
        d = db.describe_table(library=lib, table=tbl)
        cols = [str(x) for x in d["name"]]
        log(f"  {len(cols)} columns: " + ", ".join(f"{r.name}:{str(r.type).split('(')[0]}" for r in d.itertuples()))
    except Exception as e:  # noqa: BLE001
        log(f"  describe_table failed: {type(e).__name__}: {' '.join(str(e).split())[:160]}")
        r = try_sql(db, f"select column_name, data_type from information_schema.columns where table_schema = '{lib}' "
                        f"and table_name = '{tbl}' order by ordinal_position")
        if isinstance(r, str):
            log("  information_schema: " + r)
        elif len(r):
            cols = [str(x) for x in r["column_name"]]
            log(f"  {len(cols)} columns (information_schema): " + ", ".join(f"{a}:{b}" for a, b in zip(r["column_name"], r["data_type"])))
    mx = try_sql(db, f"select max(datadate) as hi from {table} where datadate >= '{RECENT_START}' and datadate <= current_date")
    hi = None
    if isinstance(mx, str):
        log("  max(datadate) since 2025-06-01: " + mx)
    else:
        hi = mx["hi"].iloc[0]
        log(f"  max(datadate) (restricted to {RECENT_START} <= datadate <= current_date): {hi}")
    mn = try_sql(db, f"select min(datadate) as lo from {table}")
    log("  min(datadate): " + (mn if isinstance(mn, str) else str(mn["lo"].iloc[0])))
    try:
        base = pd.Timestamp(hi).date() if hi is not None and not pd.isna(hi) else dt.date.today()
    except (ValueError, TypeError):
        base = dt.date.today()
    since = (base - dt.timedelta(days=30)).isoformat()
    cn = try_sql(db, f"select count(*) as n from {table} where datadate >= '{since}' and datadate <= current_date")
    log(f"  rows with datadate >= {since} (last 30 days before the max): " + (cn if isinstance(cn, str) else f"{int(cn['n'].iloc[0]):,}"))
    fu = try_sql(db, future_sql(table))
    if isinstance(fu, str):
        log("  future-dated rows (datadate > current_date): " + fu)
    elif len(fu):
        log(f"  FUTURE-DATED rows (datadate > current_date; first {len(fu)} shown) - never extracted, dropped by prices.py:")
        log("\n".join(fmt_frame(fu, "    ")))
    else:
        log("  future-dated rows (datadate > current_date): none")
    return cols, (None if hi is None else str(hi))


def main(argv=None, db=None, out="data/secd_probe.txt") -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--words", nargs="*", default=DEFAULT_WORDS, help="company-name words for tickers with no match")
    a = ap.parse_args(argv)
    tickers = [t.upper() for t in (a.tickers or config.SECD_DEFAULT_TICKERS)]
    if db is None:
        from qap.wrds_client import connect
        db = connect()
    lines: list[str] = []

    def log(m=""):
        print(m, flush=True)
        lines.append(str(m))

    try_sql(db, "set statement_timeout = 90000")
    log(f"# secd probe {dt.date.today()}   tickers: {' '.join(tickers)}")
    work = None
    for t in TABLES:
        cols, hi = probe_table(db, t, log)
        if work is None and hi is not None:
            work = (t, cols or ID_COLS + ROW_COLS)
    if work is None:
        log("\nno secd table answered; ticker lookup skipped")
    else:
        table, cols = work
        log(f"\n## TICKER LOOKUP in {table} (datadate >= {LOOKUP_START})")
        res = try_sql(db, lookup_sql(table, tickers, cols))
        found = pd.DataFrame()
        if isinstance(res, str):
            log("  " + res)
        elif len(res):
            found = res
            log("\n".join(fmt_frame(res)))
            for t, ids in sorted(ticker_groups(res).items()):
                if len(ids) > 1:
                    log(f"  AMBIGUOUS {t}: {len(ids)} securities ({', '.join(ids)}); extract with --gvkey-iid GVKEY:IID")
        have = {str(x).upper() for x in found["tic"]} if len(found) else set()
        nomatch = [t for t in tickers if t not in have]
        for t in nomatch:
            log(f"  NO MATCH for ticker {t}")
        if nomatch and "conm" in cols and a.words:
            log(f"\n## company-name search (ILIKE, datadate >= {RECENT_START}): {' '.join(a.words)}")
            r = try_sql(db, words_sql(table, a.words, cols))
            log("  " + r if isinstance(r, str) else ("\n".join(fmt_frame(r)) if len(r) else "  no company name matches"))
            if not isinstance(r, str) and len(r):
                found = pd.concat([found, r], ignore_index=True)
        if len(found):
            log("\n## LAST 5 ROWS of each matched security")
            for (g, i), s in found.groupby(["gvkey", "iid"]):
                ld = pd.to_datetime(s["last_date"]).max()
                since = (ld - pd.Timedelta(days=30)).strftime("%Y-%m-%d") if not pd.isna(ld) else RECENT_START
                r = try_sql(db, last_rows_sql(table, g, i, cols, start=since))
                log(f"  {g}:{i} {s['tic'].iloc[0]} {s['conm'].iloc[0] if 'conm' in s else ''}")
                log("  " + r if isinstance(r, str) else "\n".join(fmt_frame(r, "    ")))
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
