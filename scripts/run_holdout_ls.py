"""One-shot sealed-window test of a long-short jkp_slow10 (pre-registration v6).

    python scripts/run_holdout_ls.py --rehearsal    # pre-2020 only: reproduces the diagnostic numbers (gate); never reads 2020+ results
    python scripts/run_holdout_ls.py                # ONE SHOT: gate, then marker, then the 2020-01..2025-12 result and the decision
Needs the Study A inputs (data/raw incl. jkp_us); uses data/raw/ff_monthly.parquet for the information-only FF block if present.
Writes data/holdout_v6_report.txt (rehearsal: data/holdout_v6_rehearsal.txt). Logs one trial (v6:slow10_LS), only on the real run.
"""
import argparse
import datetime
import pathlib

import pandas as pd

from qap.holdout_ls import (EXPECTED, POWER_NOTE, assert_unopened, borrow_series, decide, ls_series, mark_opened, rehearsal_gate,
                            split_mean, summarize)
from qap.candidates import DEV, VAL
from qap.jkp import build_theme_signals
from qap.replicate import ff_to_mi
from qap.study import SEALED_CUT, eligible_universe, load_context, require_inputs
from qap.trials import log_trial

NAME, JKP_DIR, FF_PQ = "jkp_slow10", "data/raw/jkp_us", "data/raw/ff_monthly.parquet"
FIRST, LAST = 2020 * 12, 2025 * 12 + 11
BANNER = "ONE SHOT, pre-registration v6: long-short jkp_slow10, base-scenario borrow, 1.5x costs, $10M (2019 market-size)"


def main(rehearsal: bool) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, rehearsal)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/holdout_v6_rehearsal.txt" if rehearsal else "data/holdout_v6_report.txt").write_text("\n".join(report) + "\n")


def _fmt(s: dict) -> str:
    return f"mean {s['mean_pct']:.3f} %/mo, t {s['t']:.2f} (NW6 {s['t_nw6']:.2f}), n {s['n']}, sd {s['sd_pct']:.2f}"


def _run(log, rehearsal: bool) -> None:
    require_inputs([JKP_DIR], "Run: python scripts/wrds_extract.py jkp_us")
    log(BANNER + (" [REHEARSAL: months < 2020 only]" if rehearsal else ""))
    if not rehearsal:
        assert_unopened()                                  # refuse before any heavy work if the shot was already used
    panel, _, _, inp, aum_t = load_context()
    themes, _ = build_theme_signals(eligible_universe(panel), JKP_DIR)
    sig = themes.astype({"permno": "int64", "mi": "int64"})
    full = ls_series(sig, panel[["permno", "mi", "mthret_adj"]], inp, aum_t, NAME)      # ALL months, base scenario
    pre = full[full.index < SEALED_CUT]                                                  # everything below sees only months < 2020
    ok, msgs = rehearsal_gate(pre, EXPECTED)
    log("rehearsal gate vs diagnostic (b) jkp_slow10 (gross L-S, net L-S before borrow; dev 1990-2014, validation 2015-2019):")
    for m in msgs:
        log(m)
    for sp, w in (("dev", DEV), ("val", VAL)):
        log(f"  {sp}: net L-S before borrow {split_mean(pre['net_LS_preborrow'], w):.3f} %/mo, after base borrow {split_mean(pre['net_LS'], w):.3f} %/mo")
    if not ok:
        raise SystemExit("GATE FAILED: the pre-2020 series does not reproduce the diagnostic numbers; the 2020+ window was NOT opened "
                         "and no marker was written. Fix the pipeline (not the thresholds) and re-run.")
    if rehearsal:
        log("gate passed (rehearsal only: no marker, 2020+ not summarised)")
        return
    assert_unopened()
    ff = ff_to_mi(pd.read_parquet(FF_PQ)) if pathlib.Path(FF_PQ).exists() else None
    w = full[(full.index >= FIRST) & (full.index <= LAST)]
    base = summarize(w["net_LS"], ff)                                  # compute EVERYTHING before the marker: a crash here wastes nothing
    alts = {sc: summarize(w["net_LS_preborrow"] - borrow_series(sig, NAME, sc).reindex(w.index)) for sc in ("optimistic", "pessimistic")}
    label = decide(base["mean_pct"], base["t"])
    mark_opened(f"v6 opened {datetime.datetime.now().isoformat(timespec='seconds')}")        # marker, then (and only then) anything is shown
    log(f"\n== SEALED WINDOW 2020-01..2025-12: {len(w)} realised months (pre-registered 72) ==")
    log(f"PRIMARY (base borrow): net L-S {_fmt(base)}")
    log(f"  gross L-S {w['gross_LS'].mean() * 100:.3f}, L cost {w['cost_L'].mean() * 100:.3f}, S cost {w['cost_S'].mean() * 100:.3f}, "
        f"borrow {w['borrow'].mean() * 100:.3f} %/mo")
    for sc, alt in alts.items():
        log(f"sensitivity ({sc} borrow, NOT a decision input): {_fmt(alt)}")
    log("by calendar year (compounded net L-S, %): " + ", ".join(f"{y} {100 * r:.1f}" for y, r in base["by_year"].items()))
    log(f"halves: first {base['half1_pct']:.3f}, second {base['half2_pct']:.3f} %/mo; cumulative {100 * base['cum_return']:.1f}%, max drawdown {100 * base['max_drawdown']:.1f}%")
    for k in ("capm", "ff4"):
        if k in base:
            log(f"FF information ({k}): " + ", ".join(f"{a} {b:.3f}" for a, b in base[k].items()))
    log(f"legs (info): long L-U net of cost {w['long_LU'].mean() * 100:.3f}, short U-S net of cost and borrow {w['short_UmS'].mean() * 100:.3f} %/mo; "
        f"turnover one-way L {w['turnover_L'].mean():.1f}%, S {w['turnover_S'].mean():.1f}%")
    log(f"\nDECISION (base net mean {base['mean_pct']:.3f} %/mo, t {base['t']:.2f}): {label}")
    log(POWER_NOTE)
    log("No result of this test promotes the strategy to deployment; a failure retires long-short slow10.")
    spec = {"signal": NAME, "weights": "VW", "legs": "top/bottom quintile", "mult": 1.5, "aum": 1e7, "y": 1.0, "borrow": "base + 1.0 rebate",
            "window": "2020-01..2025-12", "prereg": "v6"}
    log_trial("v6:slow10_LS", spec, {"mean_pct": base["mean_pct"], "t": base["t"], "t_nw6": base["t_nw6"], "n": base["n"], "decision": label})
    log("one shot used: result must be recorded in ROADMAP")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    main(ap.parse_args().rehearsal)
