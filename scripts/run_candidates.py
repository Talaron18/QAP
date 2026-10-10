"""Pre-registered candidate test (see ROADMAP 'Pre-registration v1'). Candidates are judged RELATIVE to gpa.

    python scripts/wrds_extract.py totalq        # once (small)
    python scripts/run_candidates.py
Writes data/candidates_report.txt (checks first, results after; written even if a check fails). 2020+ is sealed and
this script cannot open it. Promotion rule, thresholds and candidate list are fixed in the pre-registration commit.
"""
import pathlib

import numpy as np
import pandas as pd

from qap.candidates import (CANDIDATES, DEV, VAL, add_candidate_features, build_candidate_signals, merge_total_q,
                            promotion_stats, split_adjustment_check, total_q_checks)
from qap.costs import liquidity_inputs
from qap.evaluate import evaluate_signal
from qap.fundamentals import prepare_annual
from qap.links import filter_ccm_links
from qap.panel import build_monthly_panel, calibrate_me_scale
from qap.portfolio import costed_books, market_scaled_aum, quintile_legs
from qap.trials import log_trial, trial_count

RAW = pathlib.Path("data/raw")
AUM, MULT, Y = 1e7, 1.5, 1.0
CUT = 2020 * 12  # sealed
SPLITS = {"pre1990": (1963, 1989), "dev": DEV, "val": VAL}


def _t(x):
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / len(x) ** 0.5)) if len(x) > 2 else float("nan")


def net_lu(sig, panel, inp, aum_t, name):
    r = costed_books(sig, panel, inp, name, aum_t, Y, 0.0)["L_minus_U"]
    idx = r["gross"].index.intersection(r["cost"].index)
    idx = idx[idx < CUT]
    return (r["gross"][idx] - MULT * r["cost"][idx]).rename(name)


def main() -> None:
    report = []

    def log(msg=""):
        print(msg, flush=True)
        report.append(str(msg))

    try:
        _run(log)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/candidates_report.txt").write_text("\n".join(report) + "\n")


def _run(log) -> None:
    monthly = pd.read_parquet(RAW / "crsp_monthly.parquet")
    delists = pd.read_parquet(RAW / "crsp_delistings.parquet")
    links = filter_ccm_links(pd.read_parquet(RAW / "ccm_links.parquet"))
    funda = pd.read_parquet(RAW / "compustat_annual.parquet")
    agg = pd.read_parquet(RAW / "crsp_daily_agg.parquet")
    totalq = pd.read_parquet(RAW / "totalq.parquet")

    panel = build_monthly_panel(monthly, delists, me_scale=1.0)
    scale, _ = calibrate_me_scale(panel, funda, links)
    panel["me"] = panel["mthcap"].astype(float) / scale
    panel["me_co"] = panel.groupby(["permco", "mi"])["me"].transform("sum")
    inp = liquidity_inputs(panel, agg)
    aum_t = market_scaled_aum(panel, AUM)

    annual, match = merge_total_q(add_candidate_features(prepare_annual(funda)), totalq)

    log("== CHECKS (printed before any result) ==")
    lines, checks = total_q_checks(annual, totalq, match)
    for l in lines:
        log(l)
    msg, ok = split_adjustment_check(panel)
    log(msg)
    checks["split adjustment of shares looks right"] = ok
    for n, ok in checks.items():
        log(f"  [{'PASS' if ok else 'FAIL'}] {n}")
    failed = [n for n, ok in checks.items() if not ok]
    if failed:
        raise AssertionError("candidate-data checks failed: " + "; ".join(failed))

    sig = build_candidate_signals(panel, annual, links)
    names = ["gpa", "bm_timely"] + CANDIDATES
    yr = sig["mi"] // 12
    log("\nsignal availability (share of formation rows with a value) | long-leg names per month, by split:")
    for n in names:
        cov = {k: f"{sig.loc[(yr >= a) & (yr <= b), n].notna().mean():.0%}" for k, (a, b) in SPLITS.items()}
        legs = quintile_legs(sig, n)["L"]
        ly = legs["mi"] // 12
        nm = {k: round(float(legs[(ly >= a) & (ly <= b)].groupby("mi").size().mean()), 0) for k, (a, b) in SPLITS.items()}
        log(f"  {n:13s} coverage {cov}  long-leg names {nm}")

    log(f"\n== RESULTS: net long-minus-universe, ${AUM:,.0f} (2019 market-size), cost stress {MULT}x, impact Y={Y}; holdout sealed ==")
    series = {n: net_lu(sig, panel, inp, aum_t, n) for n in names}
    log("standalone (information only): net L-U % per month (t) | gross rank IC (t) on dev / validation")
    for n in names:
        ev = evaluate_signal(sig, panel, n)["by_period"]
        s = series[n]
        dev, val = s[(s.index >= DEV[0] * 12) & (s.index <= DEV[1] * 12 + 11)], s[(s.index >= VAL[0] * 12) & (s.index <= VAL[1] * 12 + 11)]
        log(f"  {n:13s} dev {dev.mean() * 100:+.3f} ({_t(dev):+.2f})  val {val.mean() * 100:+.3f} ({_t(val):+.2f})  | "
            f"IC dev {ev['dev1990_2014']['ic']} ({ev['dev1990_2014']['ic_t']})  val {ev['valid2015_2019']['ic']} ({ev['valid2015_2019']['ic_t']})")

    log(f"\n== PROMOTION RULE (alpha vs gpa: dev alpha t >= 2.0 and validation alpha > 0) ==")
    promoted = []
    for n in CANDIDATES:
        st = promotion_stats(series[n], series["gpa"])
        log(f"  [{'PASS' if st['PROMOTE'] else 'FAIL'}] {n}: {st}")
        log_trial(f"cand:{n}", {"aum": AUM, "mult": MULT, "y": Y, "vs": "gpa", "universe": "ex-NYSE-20pct microcaps", "weights": "VW",
                               "prereg": "e81cb8a"}, st)
        if st["PROMOTE"]:
            promoted.append(n)
    log("\nPROMOTED (provisional; sealed holdout opens once for these): " + (", ".join(promoted) if promoted else "NONE -> next step is the gpa-only long-only tilt"))

    both = sig[sig["bm_int"].notna() & sig["bm_timely"].notna()]
    log(f"\n== bm_int vs bm_timely on the {len(both):,} rows where both exist (same universe) ==")
    for n in ("bm_timely", "bm_int"):
        s = net_lu(both, panel, inp, aum_t, n)
        dev, val = s[(s.index >= DEV[0] * 12) & (s.index <= DEV[1] * 12 + 11)], s[(s.index >= VAL[0] * 12) & (s.index <= VAL[1] * 12 + 11)]
        log(f"  {n:10s} net L-U dev {dev.mean() * 100:+.3f} ({_t(dev):+.2f})  val {val.mean() * 100:+.3f} ({_t(val):+.2f})")
        log_trial(f"cand_common:{n}", {"aum": AUM, "mult": MULT, "rows": "bm_int & bm_timely both present", "prereg": "e81cb8a"},
                  {"dev_pct": float(dev.mean() * 100), "val_pct": float(val.mean() * 100)})
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    main()
