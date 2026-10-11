"""Study A (pre-registration v3, commit e10516f): JKP themes through 2025, judged relative to gpa. Holdout sealed.

    python scripts/wrds_extract.py jkp_us totalq   # once (jkp_us: per-year, resumable)
    python scripts/run_jkp_themes.py [--open-holdout]
Writes data/study_a_report.txt and data/study_a_promoted.json. --open-holdout (once) looks at 2020+ ONLY for promoted themes.
"""
import argparse
import json
import pathlib

import pandas as pd

from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.jkp import build_theme_signals, rank_corr_by_month
from qap.jkp_themes import ALL_CHARACTERISTICS, THEMES
from qap.study import (eligible_universe, holdout_alpha, load_context, net_lu_series, require_inputs, run_promotion_report)
from qap.trials import trial_count

PREREG = "e10516f"
NAMES = list(THEMES) + ["jkp_all13", "jkp_slow10"]
JKP_DIR = "data/raw/jkp_us"


def main(open_holdout: bool) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, open_holdout)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/study_a_report.txt").write_text("\n".join(report) + "\n")


def _run(log, open_holdout) -> None:
    require_inputs([JKP_DIR], "Run: python scripts/wrds_extract.py jkp_us")
    panel, links, funda, inp, aum_t = load_context()
    sig_g = build_research_signals(panel, prepare_annual(funda), links)          # incumbent gpa, as before
    eligible = eligible_universe(panel)
    signals, chk = build_theme_signals(eligible, JKP_DIR)

    log("== CHECKS (printed before any result) ==")
    missing = pathlib.Path(JKP_DIR, "_missing_columns.txt").read_text().split() if pathlib.Path(JKP_DIR, "_missing_columns.txt").exists() else []
    log(f"theme-member columns missing from the table: {len(missing)} of {len(ALL_CHARACTERISTICS)} {missing}")
    yrs = chk["mi"] // 12
    log(f"JKP rows merged onto our eligible universe: {len(chk):,}; eligible rows {len(eligible):,}; "
        f"coverage {len(chk) / len(eligible):.1%}; years {int(yrs.min())}..{int(yrs.max())}")
    log("size_grp values: " + str(chk["size_grp"].value_counts().to_dict()))
    ours = sig_g[["permno", "mi", "gpa", "bm_timely"]].astype({"permno": "int64"})
    fwd = panel[["permno", "mi", "mthret_adj"]].copy()
    fwd["mi"] = fwd["mi"] - 1
    fwd = fwd.rename(columns={"mthret_adj": "fwd"}).astype({"permno": "int64"})
    m = chk.merge(ours, on=["permno", "mi"], how="left").merge(fwd, on=["permno", "mi"], how="left")
    c1, c2, c3 = (rank_corr_by_month(m, "gp_at", "gpa"), rank_corr_by_month(m, "be_me", "bm_timely"),
                  rank_corr_by_month(m, "ret_exc_lead1m", "fwd"))
    log(f"monthly rank corr, JKP gp_at vs our gpa: mean {c1.mean():.3f} (min {c1.min():.3f}); JKP be_me vs our be/ME: mean {c2.mean():.3f}; "
        f"JKP ret_exc_lead1m vs our next-month return: mean {c3.mean():.3f} (min {c3.min():.3f})")
    dec = (signals["mi"] // 12 // 10) * 10
    log("theme availability (share of rows with a score) by decade:")
    for n in NAMES:
        log(f"  {n:20s} " + "  ".join(f"{int(d)}s={signals.loc[dec == d, n].notna().mean():.0%}" for d in sorted(dec.unique())))
    checks = {"JKP next-month return matches ours (mean rank corr >= 0.95)": bool(c3.mean() >= 0.95),
              "<= 15% of theme-member columns missing": len(missing) <= 0.15 * len(ALL_CHARACTERISTICS)}
    for k, ok in checks.items():
        log(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise AssertionError("Study A data checks failed: " + "; ".join(k for k, ok in checks.items() if not ok))

    gpa_series = net_lu_series(sig_g, panel, inp, aum_t, "gpa")
    promo, promoted = run_promotion_report(log, signals, panel, inp, aum_t, gpa_series, NAMES, "A", PREREG)
    pathlib.Path("data/study_a_promoted.json").write_text(json.dumps(promoted))

    if open_holdout:
        log("\n== HOLDOUT OPENED (once) for promoted themes only: 2020+ alpha vs gpa with the DEV beta ==")
        g_full = net_lu_series(sig_g, panel, inp, aum_t, "gpa", cut=None)
        for n in json.loads(pathlib.Path("data/study_a_promoted.json").read_text()):
            log(f"  {n}: " + str(holdout_alpha(net_lu_series(signals, panel, inp, aum_t, n, cut=None), g_full)))
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true", help="2020+ look, once, for promoted themes only")
    main(ap.parse_args().open_holdout)
