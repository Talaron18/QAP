"""Forward formation of the v8 ML check (ROADMAP 'Pre-registration v8'): freeze one score file per formation month.

    python scripts/wrds_extract.py comp_secm_hist compustat_annual      # refresh before every formation
    python scripts/ml_form.py [--from 2025-12]
Builds the Compustat-only features through the latest COMPLETE month, prints a CHECKS block first, then for every formation month from
`--from` that is not yet frozen trains the model the protocol allows (the January-Y model, trained on formation months 1990-01..(Y-1)-11,
labels through (Y-1)-12, scores formation months Y-01..Y-12) and writes data/ml/forms/{yyyymm}.parquet (git-ignored; gvkey, iid, mi, me,
score_ml, score_comp, gpa = within-month gpa rank) for ALL eligible names, appending its sha256 to paper_ledger/ml_manifest.jsonl (tracked:
commit and push it before the month starts). An existing frozen month is never overwritten. Writes data/ml_form_report.txt.
"""
import argparse
import datetime as dt
import pathlib
import subprocess

import pandas as pd

import qap.mlcheck as ml
from qap.study import require_inputs

RAW = pathlib.Path("data/raw")


def main(first: str) -> None:
    report = []

    def log(x=""):
        print(x, flush=True)
        report.append(str(x))

    try:
        _run(log, first)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/ml_form_report.txt").write_text("\n".join(report) + "\n")


def _run(log, first) -> None:
    require_inputs([RAW / "comp_secm_hist.parquet", RAW / "compustat_annual.parquet"], "Run: python scripts/wrds_extract.py comp_secm_hist compustat_annual")
    secm, funda = pd.read_parquet(RAW / "comp_secm_hist.parquet"), pd.read_parquet(RAW / "compustat_annual.parquet")
    log("== COVERAGE DIAGNOSTIC (printed first) ==")
    for ln in ml.raw_coverage_lines(secm):
        log(ln)
    b = ml.build_from_raw(secm, funda)
    F, last_mi = b["features"], b["last_mi"]
    log("")
    for ln in ml.panel_coverage_lines(F, b["pseudo"]):
        log(ln)
    log("")
    if b["partial_mi"] is not None:
        log(f"NOTE: {ml.ym(b['partial_mi'])} is a partial month in comp.secm and is ignored; last complete month {ml.ym(last_mi)}")
    log("== CHECKS (printed before any score is written) ==")
    cov, share = ml.coverage(F, last_mi)
    n_last = int((F["mi"] == last_mi).sum())
    log(f"comp.secm rows {len(secm):,}; ME scale {b['scale']:g} (median ratio {b['ratio']:.3f}); NYSE share of rows {(b['pseudo']['primaryexch'] == 'N').mean():.1%}")
    log(f"latest complete month {ml.ym(last_mi)}: {n_last:,} eligible names; share with >= {ml.MIN_FEATURES_COVERAGE} of 20 features: {share:.1%}")
    log("feature coverage in the latest month: " + ", ".join(f"{k} {v:.0%}" for k, v in cov.items()))
    if share < 0.8:
        log(f"WARNING: only {share:.1%} of eligible names have >= {ml.MIN_FEATURES_COVERAGE} features; inspect the coverage line above before relying on these scores")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=pathlib.Path(__file__).resolve().parent).stdout.strip() or "unknown"
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    frozen = {m["mi"] for m in ml.read_manifest()}
    months = [mi for mi in range(ml.parse_ym(first), last_mi + 1) if mi not in frozen and not ml.form_path(mi).exists()]
    log(f"\n== LEDGER (spec commit {commit[:10]}, formed_on {today}) ==")
    for mi in range(ml.parse_ym(first), last_mi + 1):
        if mi not in months:
            log(f"  {ml.ym(mi)}: frozen file kept")
    if not months:
        log("nothing new to freeze")
        return
    try:
        ts = ml.choose_windows(F, last_mi)["train_start"]               # same data-driven start as the walk-forward
        counts = ml.usable_counts(F)
        for year in sorted({ml.model_year(mi) for mi in months}):
            rows, nmon = ml.train_stats(counts, ts, year)
            if rows < ml.MIN_TRAIN_ROWS or nmon < ml.MIN_TRAIN_MONTHS:
                raise ml.InfeasibleWindow(f"the {year} model would train on {rows} usable rows over {nmon} months (needs >= {ml.MIN_TRAIN_ROWS} and "
                                          f">= {ml.MIN_TRAIN_MONTHS}) from train_start {ml.ym(ts)}")
    except ml.InfeasibleWindow as e:
        raise SystemExit(f"INFEASIBLE WINDOW, nothing written: {e}")
    log(f"train_start {ml.ym(ts)} (data-driven)")
    R = ml.rank_features(F)
    for year in sorted({ml.model_year(mi) for mi in months}):
        need = [mi for mi in months if ml.model_year(mi) == year]
        sc = ml.fit_predict_year(R, year, last_mi, train_start_mi=ts, pred_months=need)
        mid = ml.model_id(ml.train_window(year, ts)[1], ts)
        for mi in need:
            form = ml.make_form(sc, mi)
            if form.empty:
                log(f"  {ml.ym(mi)}: no eligible names (skipped)")
                continue
            try:
                sha = ml.freeze_month(form, mi, commit, today, mid)
            except FileExistsError as e:
                log(f"  {ml.ym(mi)}: {e}; kept")
                continue
            log(f"  {ml.ym(mi)}: written, {len(form):,} names, {ml.form_group(mi, today)}, model train {ml.ym(ts)}..{ml.ym(ml.train_window(year, ts)[1])}, sha {sha[:10]}")
    log("\nNext: git add paper_ledger/ml_manifest.jsonl && git commit && git push; settle with `python scripts/ml_settle.py` after the next month's refresh.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", default="2025-12", help="first formation month (YYYY-MM)")
    main(ap.parse_args().first)
