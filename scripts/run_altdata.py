"""Study C (pre-registration v9, commit e239a07): six alternative-data signals, judged relative to gpa. Holdout sealed.

    python scripts/wrds_extract.py arc_data risk_text blockholders class_boards corp_culture fed_litigation   # once (small tables)
    python scripts/run_altdata.py [--open-holdout] [--jobs N]
Writes data/study_c_report.txt and data/study_c_promoted.json. A signal with fewer than 60 dev months cannot promote.
"""
import argparse
import functools
import json
import pathlib

import pandas as pd

import qap.study
from qap.altdata import NAMES, altdata_checks, break_ties, build_altdata_signals
from qap.candidates import promotion_stats
from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.study import (eligible_universe, holdout_alpha, load_context, net_lu_series, require_inputs, run_promotion_report)
from qap.trials import trial_count

PREREG = "v9"
MIN_DEV_MONTHS = 60
RAW_NAMES = ("arc_data", "risk_text", "blockholders", "class_boards", "corp_culture", "fed_litigation")
RAW = pathlib.Path("data/raw")


def main(open_holdout: bool, jobs: int = 1) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, open_holdout, jobs)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/study_c_report.txt").write_text("\n".join(report) + "\n")


def _run(log, open_holdout, jobs) -> None:
    require_inputs([RAW / f"{n}.parquet" for n in RAW_NAMES],
                   "Run: python scripts/wrds_extract.py " + " ".join(RAW_NAMES))
    panel, links, funda, inp, aum_t = load_context()
    annual = prepare_annual(funda)
    sig_g = build_research_signals(panel, annual, links)
    raw = {n: pd.read_parquet(RAW / f"{n}.parquet") for n in RAW_NAMES}

    log("== CHECKS (printed before any result) ==")
    eligible = eligible_universe(panel)
    signals, stats = build_altdata_signals(eligible, funda, links, raw["arc_data"], raw["risk_text"], raw["blockholders"],
                                           raw["class_boards"], raw["corp_culture"], raw["fed_litigation"])
    for line in altdata_checks(signals, raw, funda):
        log(line)
    ok = all(signals[n].notna().any() for n in NAMES)
    log(f"  [{'PASS' if ok else 'FAIL'}] every signal produced values")
    if not ok:
        raise AssertionError("Study C: a signal produced no values: " + ", ".join(n for n in NAMES if not signals[n].notna().any()))

    signals = break_ties(signals)
    log("exact ties are broken by a fixed per-permno hash (v9 amendment) so every quintile is populated; distinct values keep their order")
    gpa_series = net_lu_series(sig_g, panel, inp, aum_t, "gpa")
    # run_promotion_report -> promotion_table -> qap.study.promotion_stats: apply the v9 minimum of 60 dev months for this run only
    default = qap.study.promotion_stats
    qap.study.promotion_stats = functools.partial(promotion_stats, min_dev_months=MIN_DEV_MONTHS)
    try:
        promo, promoted = run_promotion_report(log, signals, panel, inp, aum_t, gpa_series, NAMES, "C", PREREG, jobs=jobs)
    finally:
        qap.study.promotion_stats = default
    pathlib.Path("data/study_c_promoted.json").write_text(json.dumps(promoted))
    if open_holdout:
        log("\n== HOLDOUT OPENED (once) for promoted signals only ==")
        g_full = net_lu_series(sig_g, panel, inp, aum_t, "gpa", cut=None)
        for n in json.loads(pathlib.Path("data/study_c_promoted.json").read_text()):
            log(f"  {n}: " + str(holdout_alpha(net_lu_series(signals, panel, inp, aum_t, n, cut=None), g_full)))
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for the per-signal backtests (fork; memory = one panel copy-on-write)")
    a = ap.parse_args()
    main(a.open_holdout, a.jobs)
