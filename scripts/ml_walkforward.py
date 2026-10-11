"""INFORMATION-ONLY walk-forward run of the frozen v8 ML protocol over formation months 1995-01..2025-11 (ROADMAP 'Pre-registration v8').

    python scripts/ml_walkforward.py [--first-year 1995] [--jobs 4]
Needs data/raw/comp_secm_hist.parquet and data/raw/compustat_annual.parquet. Retrains every January on formation months 1990-01..previous
November (labels through December) and scores that year's months; compares the ML score with the equal-weight composite and with gpa
(rank IC, VW top-quintile minus VW universe, gross and net of 10 bp per one-way turnover). Writes data/ml_walkforward_report.txt.
Formation months from 2025-12 on (incl. the pseudo-forward 2025-12..2026-08) are deliberately NEVER built, trained on, scored or evaluated here: they
are reserved for the forward-only evaluation (ml_form / ml_settle).
NOT EVIDENCE: the inputs, features and model class were chosen knowing this period; it exists to catch gross implementation errors.
"""
import argparse
import pathlib
import time

import numpy as np
import pandas as pd

import qap.mlcheck as ml
from qap.parallel import context, pmap, set_context
from qap.study import require_inputs

RAW = pathlib.Path("data/raw")
CAVEAT = ("*** CAVEAT (information only, NOT evidence): the inputs, the 20 features, the signs and the model class were chosen with knowledge of "
          "1995-2025. This run only checks the implementation for gross errors (leakage, unit or alignment bugs). No claim may rest on it. ***")


def _fit_year(year: int) -> pd.DataFrame:
    c = context()
    if c["jobs"] > 1:                      # fork + OpenMP: pin each worker's thread pool explicitly
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return ml.fit_predict_year(c["R"], year, c["last_mi"], c["train_start"])
    return ml.fit_predict_year(c["R"], year, c["last_mi"], c["train_start"])


def _fmt(mean, t, scale=100.0, nd=3):
    return f"{mean * scale:+.{nd}f} ({t:+.1f})" if np.isfinite(mean) else "   n/a"


def block_table(m: pd.DataFrame, years: int = 5) -> list:
    y = (m.index // 12).to_numpy()
    start = int(y.min())
    lines = [f"{'block':<11}{'n':>4} | " + " | ".join(f"{n:^26}" for n in ("IC ml / comp / gpa", "net L-U %/mo ml / comp / gpa")) + " | ml-comp, ml-gpa (net %/mo, t)"]
    blocks = [(a, min(a + years - 1, int(y.max()))) for a in range(start, int(y.max()) + 1, years)] + [(start, int(y.max()))]
    for a, b in blocks:
        d = m[(y >= a) & (y <= b)]
        s = ml.summarize(d)
        ic = " / ".join(f"{s[k]['ic']:+.3f}" for k in ml.SIGNALS)
        net = " / ".join(f"{s[k]['net'] * 100:+.2f}" for k in ml.SIGNALS)
        tag = f"{a}-{b}" + ("  ALL" if (a, b) == blocks[-1] and len(blocks) > 1 else "")
        lines.append(f"{tag:<11}{s['n']:>4} | {ic:^26} | {net:^26} | {_fmt(s['ml_minus_comp']['mean'], s['ml_minus_comp']['t'])}, {_fmt(s['ml_minus_gpa']['mean'], s['ml_minus_gpa']['t'])}")
    return lines


def main(first_year: int, jobs: int, train_start: str) -> None:
    report = []

    def log(x=""):
        print(x, flush=True)
        report.append(str(x))

    try:
        _run(log, first_year, jobs, train_start)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/ml_walkforward_report.txt").write_text("\n".join(report) + "\n")


def _run(log, first_year, jobs, train_start) -> None:
    require_inputs([RAW / "comp_secm_hist.parquet", RAW / "compustat_annual.parquet"], "Run: python scripts/wrds_extract.py comp_secm_hist compustat_annual")
    t0 = time.time()
    log(CAVEAT)
    secm, funda = pd.read_parquet(RAW / "comp_secm_hist.parquet"), pd.read_parquet(RAW / "compustat_annual.parquet")
    b = ml.build_from_raw(secm, funda, max_mi=ml.WALKFORWARD_LAST_MI + 1)       # nothing after formation 2025-11 / return 2025-12 is ever built
    w = ml.walkforward_inputs(b, first_year)
    F, last_mi = w["features"], w["cap"]
    log(f"\nsecm rows {len(secm):,}; ME scale {b['scale']:g} (ratio {b['ratio']:.3f}); months {ml.ym(int(F['mi'].min()))}..{ml.ym(last_mi)}; "
        f"feature rows {len(F):,} (eligible universe); build {time.time() - t0:.0f}s")
    cov, share = ml.coverage(F, last_mi)
    log("feature coverage, latest month: " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items()) + f"; >= {ml.MIN_FEATURES_COVERAGE} of 20: {share:.1%}")
    R = ml.rank_features(F)
    years = w["years"]
    set_context(R=R, last_mi=last_mi, train_start=ml.parse_ym(train_start), jobs=jobs)
    t1 = time.time()
    parts = pmap(_fit_year, years, jobs)
    wf = pd.concat(parts, ignore_index=True)
    log(f"retrained {len(years)} January models ({years[0]}..{years[-1]}), jobs={jobs}, {time.time() - t1:.0f}s; scored rows {len(wf):,}")
    m = ml.evaluate_scores(wf, w["ret"])
    log("\n== WALK-FORWARD (formation-month blocks; IC = mean rank IC; net = VW top quintile minus VW eligible universe, 10 bp per one-way turnover) ==")
    for ln in block_table(m):
        log(ln)
    s = ml.summarize(m)
    log("\nfull sample, per signal: " + "; ".join(f"{k} IC {s[k]['ic']:+.4f} (t {s[k]['ic_t']:+.1f}), gross {s[k]['gross'] * 100:+.3f}, net {s[k]['net'] * 100:+.3f} "
                                              f"(t {s[k]['net_t']:+.1f}), turnover {s[k]['turn']:.2f}" for k in ml.SIGNALS))
    log("\n== LEAKAGE SANITY ==")
    ic_ml, ic_c = s["score_ml"]["ic"], s["score_comp"]["ic"]
    ratio = ic_ml / ic_c if ic_c > 0 else float("nan")
    log(f"ML IC {ic_ml:+.4f}, composite IC {ic_c:+.4f}, gpa IC {s['gpa']['ic']:+.4f}; ML/composite IC ratio {ratio:.2f}")
    why = [t for t, hit in (("ML IC > 2x composite IC", np.isfinite(ratio) and ratio > 2.0), ("ML IC > 0.10", ic_ml > 0.10),
                            ("composite IC <= 0 while ML IC > 0.05 (extra rule, not in the pre-registration)", not ic_c > 0 and ic_ml > 0.05)) if hit]
    log(f"WARNING ({'; '.join(why)}): suspect leakage or an alignment bug before reading anything else. (A single dominant feature that the 20-feature composite "
        f"dilutes can also trigger the 2x rule without any leakage.)" if why else "ok: ML IC is within a plausible multiple of the composite IC and below 0.10.")
    log(f"\n{CAVEAT}")
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-year", type=int, default=1995)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--train-start", default="1990-01")
    a = ap.parse_args()
    main(a.first_year, a.jobs, a.train_start)
