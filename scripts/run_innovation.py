"""Study B (pre-registration v3, commit e10516f): innovation signals, judged relative to gpa. Holdout sealed.

    python scripts/wrds_extract.py rq_data kpss_patents patent_grants     # once (small tables)
    python scripts/run_innovation.py [--open-holdout]
Writes data/study_b_report.txt and data/study_b_promoted.json.
"""
import argparse
import json
import pathlib

import pandas as pd

from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.innovation import build_innovation_signals
from qap.panel import to_mi
from qap.study import (eligible_universe, holdout_alpha, load_context, net_lu_series, require_inputs, run_promotion_report)
from qap.trials import trial_count

PREREG = "e10516f"
JOBS = 1
NAMES = ["rq", "kpss_xi_me", "pat_flow_rd", "pat_quality3"]
RAW = pathlib.Path("data/raw")


def main(open_holdout: bool) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, open_holdout)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/study_b_report.txt").write_text("\n".join(report) + "\n")


def _run(log, open_holdout) -> None:
    require_inputs([RAW / f"{n}.parquet" for n in ("rq_data", "kpss_patents", "patent_grants")],
                   "Run: python scripts/wrds_extract.py rq_data kpss_patents patent_grants")
    panel, links, funda, inp, aum_t = load_context()
    annual = prepare_annual(funda)
    sig_g = build_research_signals(panel, annual, links)
    rq, kpss, grants = (pd.read_parquet(RAW / f"{n}.parquet") for n in ("rq_data", "kpss_patents", "patent_grants"))

    last_issue = pd.to_datetime(kpss["issue_date"]).max()
    end_mi = int(to_mi(pd.Series([last_issue])).iloc[0]) - 3        # last month with complete issue dates: 3 months before the max
    log("== CHECKS (printed before any result) ==")
    log(f"KPSS patents {len(kpss):,}; last issue_date {last_issue.date()} -> kpss_xi_me evaluated through month index {end_mi} "
        f"({end_mi // 12}-{end_mi % 12 + 1:02d}); permnos {kpss['permno'].nunique():,}")
    gy = grants.groupby("grant_year")["n_patents"].sum()
    log(f"patent grants (server-side aggregate) {int(gy.sum()):,} over {len(grants):,} gvkey-years; grant years {int(gy.index.min())}..{int(gy.index.max())}; "
        f"last 6 years: {gy.tail(6).astype(int).to_dict()}")
    log(f"RQ rows {len(rq):,}; rawrq quantiles (1/50/99%): {rq['rawrq'].quantile([0.01, 0.5, 0.99]).round(3).tolist()}; fyears {int(rq['fyear'].min())}..{int(rq['fyear'].max())}")
    eligible = eligible_universe(panel)
    signals = build_innovation_signals(eligible, annual, links, rq, grants, kpss, kpss_end_mi=end_mi)
    dec = (signals["mi"] // 12 // 10) * 10
    log("signal availability (share of eligible rows with a value) by decade:")
    for n in NAMES:
        log(f"  {n:14s} " + "  ".join(f"{int(d)}s={signals.loc[dec == d, n].notna().mean():.0%}" for d in sorted(dec.unique())))
    ok = signals["pat_flow_rd"].notna().any() and signals["rq"].notna().any() and signals["kpss_xi_me"].notna().any()
    log(f"  [{'PASS' if ok else 'FAIL'}] every source produced values")
    if not ok:
        raise AssertionError("Study B: a signal source produced no values")

    gpa_series = net_lu_series(sig_g, panel, inp, aum_t, "gpa")
    promo, promoted = run_promotion_report(log, signals, panel, inp, aum_t, gpa_series, NAMES, "B", PREREG, jobs=JOBS)
    pathlib.Path("data/study_b_promoted.json").write_text(json.dumps(promoted))
    if open_holdout:
        log("\n== HOLDOUT OPENED (once) for promoted signals only ==")
        g_full = net_lu_series(sig_g, panel, inp, aum_t, "gpa", cut=None)
        for n in json.loads(pathlib.Path("data/study_b_promoted.json").read_text()):
            log(f"  {n}: " + str(holdout_alpha(net_lu_series(signals, panel, inp, aum_t, n, cut=None), g_full)))
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for the per-signal backtests (fork; memory = one panel copy-on-write)")
    a = ap.parse_args()
    JOBS = a.jobs
    main(a.open_holdout)
