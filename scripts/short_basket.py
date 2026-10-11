"""Short-basket sheet for the borrow-cost check (ROADMAP 'break-even borrow cost'). INFORMATION ONLY: no returns, no trials, 2020+ untouched.

    python scripts/short_basket.py [--signal jkp_slow10] [--month 2025-12] [--top 60]
        writes data/short_basket_<signal>_<yyyymm>.csv: bottom-quintile VW basket (ticker, issuer, weight, cumulative weight, ME) with empty
        columns borrow_fee_pct and available. Look the top names up in your broker's shortable-stocks tool and fill them in (available = N if
        it cannot be shorted).
    python scripts/short_basket.py --signal jkp_slow10 --fees data/short_basket_jkp_slow10_202512.csv
        weight-averages the fees you entered and compares with the break-even table. Uses your sheet only (no WRDS data needed).
Note: JKP data ends 2025-12, so the basket is as of that month; borrow fees are today's. Names persist only partly, so treat it as a sample.
"""
import argparse
import pathlib

import pandas as pd

from qap.basket import BREAK_EVEN, attach_names, blended_fee, short_basket, verdict_lines

SIGNALS = list(BREAK_EVEN)


def _month_mi(s: str) -> int:
    y, m = s.split("-")
    return int(y) * 12 + int(m) - 1


def build(signal: str, month: str, top: int) -> None:
    from qap.evaluate import build_research_signals
    from qap.fundamentals import prepare_annual
    from qap.jkp import build_theme_signals
    from qap.study import eligible_universe, load_context, require_inputs
    require_inputs(["data/raw/jkp_us"], "Run: python scripts/wrds_extract.py jkp_us")
    panel, links, funda, _, _ = load_context()
    themes, _ = build_theme_signals(eligible_universe(panel), "data/raw/jkp_us")
    themes = themes.astype({"permno": "int64", "mi": "int64"})
    if signal == "gpa":
        gpa = build_research_signals(panel, prepare_annual(funda), links)[["permno", "mi", "gpa"]].astype({"permno": "int64", "mi": "int64"})
        themes = themes.merge(gpa, on=["permno", "mi"], how="left")
    mi = _month_mi(month) if month else int(themes["mi"].max())
    names_pq = pathlib.Path("data/raw/crsp_names.parquet")
    names = pd.read_parquet(names_pq) if names_pq.exists() else None
    b = attach_names(short_basket(themes, signal, mi), names, mi)
    b["me"], b["me_co"] = b["me"].astype(float).round(0), b["me_co"].astype(float).round(0)
    b["borrow_fee_pct"], b["available"] = "", ""
    out = pathlib.Path("data") / f"short_basket_{signal}_{mi // 12}{mi % 12 + 1:02d}.csv"
    out.parent.mkdir(exist_ok=True)
    b[["ticker", "issuer", "w", "cum_w", "me", "primaryexch", "score", "permno", "borrow_fee_pct", "available"]].to_csv(out, index=False)
    n60 = b.head(top)
    print(f"INFORMATION ONLY. {signal} bottom quintile at {mi // 12}-{mi % 12 + 1:02d}: {len(b)} names; top {top} cover {n60['w'].sum():.1%} of weight; "
          f"largest {b['w'].iloc[0]:.2%}; median ME ${b['me'].median():,.0f}M; NYSE share of weight {b.loc[b['primaryexch'] == 'N', 'w'].sum():.1%}")
    print(f"wrote {out} -- fill borrow_fee_pct (annual %) and available (N if not shortable) for at least the top {top} rows, then run with --fees")


def check(signal: str, path: str) -> None:
    sheet = pd.read_csv(path)
    st = blended_fee(sheet)
    print(f"INFORMATION ONLY. Borrow-fee check for {signal} from {path} ({len(sheet)} names, {int((sheet['borrow_fee_pct'].notna()).sum())} with a fee)")
    for line in verdict_lines(signal, st):
        print(line)
    print("Hurdle caveat: the 1.0% rebate shortfall is an assumption, set by your broker; edit qap.basket.verdict_lines(hurdle_extra=...) if you know yours.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="jkp_slow10", choices=SIGNALS)
    ap.add_argument("--month", default="", help="YYYY-MM (default: last JKP month)")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--fees", default="", help="filled sheet to evaluate")
    a = ap.parse_args()
    check(a.signal, a.fees) if a.fees else build(a.signal, a.month, a.top)
