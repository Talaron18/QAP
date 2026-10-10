"""Replicate FF factors, then (only if the gates pass) build and evaluate Tier-1 research signals.

    python scripts/run_signals.py [--force-signals]
Reads data/raw/*.parquet; writes data/signals.parquet and data/signals_report.txt (summary numbers).
Every check is written to the report BEFORE any assertion fires, so a failure still leaves the numbers.
"""
import argparse
import pathlib

import pandas as pd

from qap.evaluate import SIGNALS, build_research_signals, evaluate_signal
from qap.fundamentals import prepare_annual
from qap.links import filter_ccm_links
from qap.panel import assert_month_grid, build_monthly_panel, calibrate_me_scale, to_mi
from qap.replicate import compare, ff_to_mi, form_portfolios, portfolio_returns, smb_hml, umd, vw_market_return
from qap.trials import log_trial

RAW = pathlib.Path("data/raw")
GATE_MKT, GATE_FACTOR = 0.99, 0.95
GATE_UMD = 0.95


def main(force: bool) -> None:
    report = []
    try:
        _run(force, report)
    finally:  # a failed precondition must still leave the numbers gathered so far
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/signals_report.txt").write_text("\n".join(report) + "\n")


def _run(force: bool, report: list) -> None:

    def log(msg=""):
        print(msg, flush=True)
        report.append(str(msg))

    failures = []
    monthly = pd.read_parquet(RAW / "crsp_monthly.parquet")
    delists = pd.read_parquet(RAW / "crsp_delistings.parquet")
    links = filter_ccm_links(pd.read_parquet(RAW / "ccm_links.parquet"))
    funda = pd.read_parquet(RAW / "compustat_annual.parquet")
    ff = ff_to_mi(pd.read_parquet(RAW / "ff_monthly.parquet"))

    panel = build_monthly_panel(monthly, delists, me_scale=1.0)
    try:
        assert_month_grid(panel)
        log(f"month grid OK: {panel['mi'].nunique()} unique months, no duplicate (permno, month)")
    except AssertionError as e:
        failures.append(f"month grid: {e}")
        log(f"FAIL month grid: {e}")

    scale, ratio = calibrate_me_scale(panel, funda, links)
    log(f"unit calibration: median mthcap/(csho*prcc_f) = {ratio:.3f} -> dividing mthcap by {scale:g} to get $ millions")
    panel["me"] = panel["mthcap"].astype(float) / scale
    panel["me_co"] = panel.groupby(["permco", "mi"])["me"].transform("sum")

    annual = prepare_annual(funda)

    mkt = vw_market_return(panel)
    c1 = compare(mkt - ff["rf"], ff["mktrf"])
    log(f"GATE 1 market excess: {c1}  (need corr >= {GATE_MKT})")
    if c1["corr"] < GATE_MKT:
        failures.append(f"gate 1 market corr {c1['corr']:.4f} < {GATE_MKT}")

    assign = form_portfolios(panel, annual, links)
    log(f"FF 2x3 formation: {assign['t'].nunique()} years, {len(assign):,} (permno, year) assignments; "
        + str(assign["port"].value_counts().to_dict()))
    fac = smb_hml(portfolio_returns(panel, assign))
    for name in ("smb", "hml"):
        c = compare(fac[name], ff[name])
        log(f"GATE 2 {name.upper()}: {c}  (need corr >= {GATE_FACTOR})")
        if c["corr"] < GATE_FACTOR:
            failures.append(f"gate 2 {name} corr {c['corr']:.4f} < {GATE_FACTOR}")
    c3 = compare(umd(panel), ff["umd"])
    log(f"GATE 3 UMD: {c3}  (need corr >= {GATE_FACTOR})")
    if c3["corr"] < GATE_UMD:
        failures.append(f"gate 3 umd corr {c3['corr']:.4f} < {GATE_UMD}")

    if failures and not force:
        log("\nGATES FAILED -> research signals NOT built. " + "; ".join(failures))
    else:
        if failures:
            log("\nWARNING: gates failed, signals below are UNVALIDATED (--force-signals)")
        sig = build_research_signals(panel, annual, links)
        log(f"\nresearch signals: {len(sig):,} formation rows, CCM+PIT hit rate on formation universe {sig.attrs['link_hit_rate']:.1%}")
        log("  link/PIT diagnostics: " + str(sig.attrs["link_diag"]))
        for name in SIGNALS:
            res = evaluate_signal(sig, panel, name)
            by = res.pop("by_period")
            log("  " + str(res))
            for per, v in by.items():
                log(f"    {per}: {v}")
            log_trial(name, {"universe": "eq, ex-NYSE-20pct microcaps", "weights": "VW", "pit_lag_months": 4}, {"by_period": by})
        pathlib.Path("data").mkdir(exist_ok=True)
        sig.to_parquet("data/signals.parquet", index=False)

    assert not failures or force, "gates failed: " + "; ".join(failures)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-signals", action="store_true")
    main(ap.parse_args().force_signals)
