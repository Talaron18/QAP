"""Study D (pre-registration v11): EAR / SUE / RSUE / EAR+SUE from earnings announcements, judged relative to gpa. Holdout sealed.

    python scripts/wrds_extract.py compustat_quarterly          # if not extracted yet
    python scripts/extract_earnings.py                          # daily CRSP returns (per year, resumable) and daily FF factors
    python scripts/run_earnings.py [--open-holdout] [--jobs N]
Writes data/study_d_report.txt and data/study_d_promoted.json. A signal with fewer than 60 dev months cannot promote.
The information-only block (EW Q5-Q1, rank IC by size tercile and by months since the announcement, top-quintile turnover) uses months < 2020 only.
"""
import argparse
import functools
import json
import pathlib

import pandas as pd

import qap.study
from qap.candidates import promotion_stats
from qap.diagnostics import diagnose_signal, split_summary
from qap.earnings import (BUCKETS, NAMES, break_ties, build_earnings_signals, build_events, earnings_checks, ic_by_bucket, load_daily,
                          prepare_quarters)
from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.study import (SEALED_CUT, eligible_universe, holdout_alpha, load_context, net_lu_series, require_inputs, run_promotion_report)
from qap.trials import trial_count

PREREG = "v11"
MIN_DEV_MONTHS = 60
RAW = pathlib.Path("data/raw")
CONTEXT = ("crsp_monthly", "crsp_delistings", "ccm_links", "compustat_annual", "crsp_daily_agg")


def main(open_holdout: bool, jobs: int = 1) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, open_holdout, jobs)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/study_d_report.txt").write_text("\n".join(report) + "\n")


def _fmt(pair, d=3) -> str:
    m, t = pair
    return "n/a" if m is None else f"{m:.{d}f} ({'n/a' if t is None else f'{t:.2f}'})"


def _info_block(log, signals, panel, inp, aum_t) -> None:
    """Information only (not a decision input, no trial logged): months < 2020 only."""
    log("\n== INFORMATION ONLY (months < 2020; not decision inputs) ==")
    log("EW gross Q5-Q1 %/month (t), top-quintile one-way turnover (mean), rank IC (mean, t) by size tercile (T1 smallest) and by months since the announcement")
    for n in NAMES:
        r = diagnose_signal(signals, panel, inp, aum_t, n, cut=SEALED_CUT)
        ret, cost = r["returns"], r["costs"]
        log(f"  {n}")
        for sp in ("dev", "val"):
            terc = "  ".join(f"T{k} {_fmt(r['tercile_ic'][f'ic_T{k}'][sp])}" for k in (1, 2, 3))
            log(f"    {sp:3s} EW Q5-Q1 {_fmt(ret['Q5-Q1_ew'][sp])}  turnover_L% {_fmt(cost['turnover_L'][sp], 1)}  IC by size: {terc}")
        b = ic_by_bucket(signals, panel, n, SEALED_CUT)
        bs = split_summary({c: b[c] for c in b}, scale=1.0)
        for sp in ("dev", "val"):
            log(f"    {sp:3s} IC by months since announcement (0-1 / 2 / 3+): "
                + "  /  ".join(f"{v} {_fmt(bs[f'ic_{v}'][sp])}" for v in BUCKETS.values()))


def _run(log, open_holdout, jobs) -> None:
    require_inputs([RAW / f"{n}.parquet" for n in CONTEXT] + [RAW / "compustat_quarterly.parquet", RAW / "crsp_daily_ret", RAW / "ff_daily.parquet"],
                   "Run: python scripts/wrds_extract.py compustat_quarterly   and   python scripts/extract_earnings.py")
    panel, links, funda, inp, aum_t = load_context()
    annual = prepare_annual(funda)
    sig_g = build_research_signals(panel, annual, links)
    quarterly = pd.read_parquet(RAW / "compustat_quarterly.parquet")
    daily, daily_rows = load_daily(RAW / "crsp_daily_ret")
    ff = pd.read_parquet(RAW / "ff_daily.parquet")

    log("== CHECKS (printed before any result) ==")
    eligible = eligible_universe(panel)
    events, flags = build_events(quarterly, links, daily, ff)
    del daily
    signals = build_earnings_signals(eligible, events)
    for line in earnings_checks(prepare_quarters(quarterly), flags, signals, daily_rows, len(ff)):
        log(line)
    ok = all(signals[n].notna().any() for n in NAMES)
    log(f"  [{'PASS' if ok else 'FAIL'}] every signal produced values")
    if not ok:
        raise AssertionError("Study D: a signal produced no values: " + ", ".join(n for n in NAMES if not signals[n].notna().any()))

    signals = break_ties(signals)
    log("exact ties are broken by a fixed per-permno hash (as in v9) so every quintile is populated; distinct values keep their order")
    gpa_series = net_lu_series(sig_g, panel, inp, aum_t, "gpa")
    default = qap.study.promotion_stats
    qap.study.promotion_stats = functools.partial(promotion_stats, min_dev_months=MIN_DEV_MONTHS)
    try:
        promo, promoted = run_promotion_report(log, signals, panel, inp, aum_t, gpa_series, NAMES, "D", PREREG, jobs=jobs)
    finally:
        qap.study.promotion_stats = default
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/study_d_promoted.json").write_text(json.dumps(promoted))
    _info_block(log, signals, panel, inp, aum_t)
    if open_holdout:
        log("\n== HOLDOUT OPENED (once) for promoted signals only ==")
        g_full = net_lu_series(sig_g, panel, inp, aum_t, "gpa", cut=None)
        for n in json.loads(pathlib.Path("data/study_d_promoted.json").read_text()):
            log(f"  {n}: " + str(holdout_alpha(net_lu_series(signals, panel, inp, aum_t, n, cut=None), g_full)))
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for the per-signal backtests (fork; memory = one panel copy-on-write)")
    a = ap.parse_args()
    main(a.open_holdout, a.jobs)
