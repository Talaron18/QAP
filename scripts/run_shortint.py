"""Study E (pre-registration v12): short interest - sir_neg, dtc_neg, gpa_veto - judged relative to gpa. Holdout sealed.

    python scripts/wrds_extract.py short_interest      # once (comp.sec_shortint, datadate >= 2006-01-01)
    python scripts/run_shortint.py [--open-holdout] [--jobs N]
Writes data/study_e_report.txt and data/study_e_promoted.json. A signal with fewer than 60 dev months cannot promote.
gpa_veto additionally needs its mean net long-minus-universe improvement over gpa >= 0.05 %/month in development AND > 0 in validation.
"""
import argparse
import functools
import json
import pathlib

import numpy as np
import pandas as pd

import qap.study
from qap.candidates import promotion_stats
from qap.diagnostics import diagnose_signal, quintile_table, split_summary
from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.portfolio import costed_books, leg_gross_return, quintile_legs
from qap.shortint import (NAMES, break_ties, build_shortint_signals, holdout_improvement, monthly_ic, shortint_checks, veto_floor)
from qap.study import (SEALED_CUT, eligible_universe, holdout_alpha, load_context, net_lu_series, require_inputs, run_promotion_report)
from qap.trials import trial_count

PREREG = "v12"
MIN_DEV_MONTHS = 60
RAW_NAMES = ("short_interest",)
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
        pathlib.Path("data/study_e_report.txt").write_text("\n".join(report) + "\n")


def _fmt(pair, d=3) -> str:
    m, t = pair
    return "n/a" if m is None else f"{m:.{d}f} ({'n/a' if t is None else f'{t:.2f}'})"


def _pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.3f}"


def _floor_line(f: dict) -> str:
    return "  ".join(f"{k} {_pct(f[k]['mean'])} %/mo (t {f[k]['t']:.2f}, {f[k]['months']} months)" for k in ("dev", "val"))


def _net_long_series(sig, panel, inp, aum_t, name, mult: float = 1.5) -> pd.Series:
    """Net return of the top-quintile LONG book itself (gross - mult x cost), months before the sealed cut; no universe subtracted."""
    r = costed_books(sig, panel, inp, name, aum_t, 1.0, 0.0)["L_minus_U"]
    lg = leg_gross_return(quintile_legs(sig, name)["L"], panel)
    idx = lg.index.intersection(r["cost"].index)
    idx = idx[idx < SEALED_CUT]
    return (lg[idx] - mult * r["cost"][idx]).rename(name)


def _info_block(log, signals, panel, inp, aum_t) -> None:
    """Information only (no decision weight, no trial logged): months < 2020 only. The top-SIR decile is D1 of sir_neg."""
    log("\n== INFORMATION ONLY (months < 2020; not decision inputs) ==")
    qt = quintile_table(signals, panel, "sir_neg", SEALED_CUT)
    ser = {"top-SIR decile minus universe, gross VW": qt["D1_vw"] - qt["U_vw"], "top-SIR decile minus universe, gross EW": qt["D1_ew"] - qt["U_ew"]}
    ss = split_summary(ser)
    log("short-leg effect, %/month (plain t): lower = highest-SIR names underperform")
    for k, v in ss.items():
        log(f"  {k:42s} dev {_fmt(v['dev'])}  val {_fmt(v['val'])}")
    r = diagnose_signal(signals, panel, inp, aum_t, "sir_neg", cut=SEALED_CUT)
    log("sir_neg rank IC by size tercile (T1 smallest; mean (t)) and top-quintile one-way turnover (mean):")
    for sp in ("dev", "val"):
        terc = "  ".join(f"T{k} {_fmt(r['tercile_ic'][f'ic_T{k}'][sp])}" for k in (1, 2, 3))
        log(f"  {sp:3s} IC by size: {terc}  turnover_L% {_fmt(r['costs']['turnover_L'][sp], 1)}")
    ic = monthly_ic(signals, panel, "sir_neg", SEALED_CUT)
    by = ic.groupby(ic.index // 12).mean()
    log("sir_neg rank IC by year (mean): " + "  ".join(f"{int(y)}={v:.3f}" for y, v in by.dropna().items()))


def _run(log, open_holdout, jobs) -> None:
    require_inputs([RAW / f"{n}.parquet" for n in RAW_NAMES],
                   "Run: python scripts/wrds_extract.py " + " ".join(RAW_NAMES))
    panel, links, funda, inp, aum_t = load_context()
    annual = prepare_annual(funda)
    sig_g = build_research_signals(panel, annual, links)
    si = pd.read_parquet(RAW / "short_interest.parquet")

    log("== CHECKS (printed before any result) ==")
    eligible = eligible_universe(panel)
    signals, stats = build_shortint_signals(eligible, links, si, sig_g, panel)
    for line in shortint_checks(si, stats, signals):
        log(line)
    ok = all(signals[n].notna().any() for n in NAMES)
    log(f"  [{'PASS' if ok else 'FAIL'}] every signal produced values")
    if not ok:
        raise AssertionError("Study E: a signal produced no values: " + ", ".join(n for n in NAMES if not signals[n].notna().any()))
    log("signals are attached to the rows of the incumbent gpa frame (same universe as gpa); the veto decile is formed among eligible_universe names with a SIR")

    signals = break_ties(signals)
    log("exact ties are broken by a fixed per-permno hash (as in v9) so every quintile is populated; distinct values keep their order")
    gpa_series = net_lu_series(sig_g, panel, inp, aum_t, "gpa")
    default = qap.study.promotion_stats
    qap.study.promotion_stats = functools.partial(promotion_stats, min_dev_months=MIN_DEV_MONTHS)
    try:
        promo, promoted = run_promotion_report(log, signals, panel, inp, aum_t, gpa_series, NAMES, "E", PREREG, jobs=jobs)
    finally:
        qap.study.promotion_stats = default

    log("\n== SIZE FLOOR for gpa_veto (net long-minus-universe, gpa_veto minus gpa, aligned months; need dev >= 0.050 %/mo AND val > 0) ==")
    floor = veto_floor(net_lu_series(signals, panel, inp, aum_t, "gpa_veto"), gpa_series)
    log(f"  improvement over gpa: {_floor_line(floor)}")
    log(f"  [{'PASS' if floor['PASS'] else 'FAIL'}] size floor")
    # information only: the registered measure subtracts each book's own universe, and gpa_veto's universe excludes the vetoed names,
    # which understates the veto if high-SIR names underperform; the direct long-book difference (same benchmark by construction) is shown beside it
    direct = veto_floor(_net_long_series(signals, panel, inp, aum_t, "gpa_veto"), _net_long_series(sig_g, panel, inp, aum_t, "gpa"))
    log(f"  information only - direct long-book net improvement (gpa_veto long book minus gpa long book, no universe subtracted): {_floor_line(direct)}")
    if "gpa_veto" in promoted and not floor["PASS"]:
        promoted = [n for n in promoted if n != "gpa_veto"]
        log("  gpa_veto passed the standard promotion rule but FAILS the size floor: REMOVED from the promoted list")
    log("\nPROMOTED after the size floor (sealed holdout opens once for these): " + (", ".join(promoted) if promoted else "NONE"))
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/study_e_promoted.json").write_text(json.dumps(promoted))
    _info_block(log, signals, panel, inp, aum_t)
    if open_holdout:
        log("\n== HOLDOUT OPENED (once) for promoted signals only ==")
        g_full = net_lu_series(sig_g, panel, inp, aum_t, "gpa", cut=None)
        for n in json.loads(pathlib.Path("data/study_e_promoted.json").read_text()):
            full = net_lu_series(signals, panel, inp, aum_t, n, cut=None)
            log(f"  {n}: " + str(holdout_alpha(full, g_full)))
            if n == "gpa_veto":
                h = holdout_improvement(full, g_full, SEALED_CUT)
                log(f"  gpa_veto holdout improvement over gpa: {_pct(h['mean'])} %/mo (t {h['t']:.2f}, {h['months']} months); "
                    f"the veto is licensed only if this is > 0: {'YES' if h['mean'] > 0 else 'NO'}")
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for the per-signal backtests (fork; memory = one panel copy-on-write)")
    a = ap.parse_args()
    main(a.open_holdout, a.jobs)
