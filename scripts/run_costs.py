"""Transaction-cost model and costed backtests for the Tier-1 signals.

    python scripts/wrds_extract.py crsp_daily_agg     # once; per-year, resumable
    python scripts/run_costs.py [--open-holdout] [--y 1.0] [--commission-bps 0]
Reads data/raw/*.parquet; writes data/costs_report.txt. The report starts with sanity checks on the
liquidity inputs and is written BEFORE any assertion fires. 2020+ stays sealed unless --open-holdout.
Books: 'LS' = long top quintile / short bottom (borrow NOT modelled); 'L-U' = top quintile minus the VW
universe, net of the long leg's trading costs (benchmark treated as costless).
"""
import argparse
import pathlib

import pandas as pd

from qap.costs import liquidity_inputs, sanity_report
from qap.evaluate import add_composite, add_momentum, build_research_signals
from qap.fundamentals import prepare_annual
from qap.links import filter_ccm_links
from qap.panel import build_monthly_panel, calibrate_me_scale
from qap.portfolio import costed_books, period_stats
from qap.trials import log_trial, trial_count

RAW = pathlib.Path("data/raw")
SIGNALS = ["composite_vpi", "gpa", "neg_ag", "mom", "bm_timely"]
AUMS = [1e6, 1e7, 1e8]
MULTS = [1.0, 1.5, 2.0]


def _t(x):
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / len(x) ** 0.5)) if len(x) > 2 else float("nan")


def main(open_holdout: bool, y: float, commission: float, enforce_sanity: bool) -> None:
    report = []
    try:
        _run(open_holdout, y, commission, enforce_sanity, report)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/costs_report.txt").write_text("\n".join(report) + "\n")


def _run(open_holdout, y, commission, enforce_sanity, report) -> None:
    def log(msg=""):
        print(msg, flush=True)
        report.append(str(msg))

    monthly = pd.read_parquet(RAW / "crsp_monthly.parquet")
    delists = pd.read_parquet(RAW / "crsp_delistings.parquet")
    links = filter_ccm_links(pd.read_parquet(RAW / "ccm_links.parquet"))
    funda = pd.read_parquet(RAW / "compustat_annual.parquet")
    agg = pd.read_parquet(RAW / "crsp_daily_agg.parquet")

    panel = build_monthly_panel(monthly, delists, me_scale=1.0)
    scale, ratio = calibrate_me_scale(panel, funda, links)
    panel["me"] = panel["mthcap"].astype(float) / scale
    panel["me_co"] = panel.groupby(["permco", "mi"])["me"].transform("sum")
    log(f"units: mthcap / {scale:g} -> $ millions (calibration ratio {ratio:.3f}); cost inputs from {len(agg):,} permno-months")

    inp = liquidity_inputs(panel, agg)
    lines, checks = sanity_report(inp)
    log("== SANITY CHECKS ON LIQUIDITY INPUTS ==")
    for l in lines:
        log(l)
    for name, ok in checks.items():
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok in checks.items() if not ok]
    if failed and enforce_sanity:
        log("SANITY FAILED -> costed backtests NOT run: " + "; ".join(failed))
        raise AssertionError("cost-input sanity failed: " + "; ".join(failed))

    annual = prepare_annual(funda)
    sig = add_momentum(add_composite(build_research_signals(panel, annual, links)), panel)
    log(f"\n== COSTED BACKTESTS (impact Y={y}, commission {commission * 1e4:g} bp, stress multiples {MULTS}); "
        f"holdout {'OPENED' if open_holdout else 'sealed'} ==")
    cut = None if open_holdout else 2020 * 12
    for name in SIGNALS:
        for aum in AUMS:
            res = costed_books(sig, panel, inp, name, aum, y, commission)
            tl = res["turnover_L"]
            log(f"\n{name} @ AUM ${aum:,.0f}: monthly one-way turnover L={tl.mean():.1%} S={res['turnover_S'].mean():.1%}; "
                f"long-leg trade $ above 5% ADV: {res['adv5'].mean():.2%}, above 10% ADV: {res['adv10'].mean():.2%}")
            metrics = {}
            for book, label in (("LS", "LS (borrow not modelled)"), ("L_minus_U", "L-U (benchmark costless)")):
                g, c = res[book]["gross"], res[book]["cost"]
                idx = g.index.intersection(c.index)
                if cut is not None:
                    idx = idx[idx < cut]
                g, c = g[idx], c[idx]
                row = {"gross_pct": round(float(g.mean()) * 100, 3), "cost1x_pct": round(float(c.mean()) * 100, 3),
                       "breakeven_mult": round(float(g.mean() / c.mean()), 2) if c.mean() > 0 else None}
                for m in MULTS:
                    n = g - m * c
                    row[f"net{m:g}x_pct"] = round(float(n.mean()) * 100, 3)
                    row[f"net{m:g}x_t"] = round(_t(n), 2)
                log(f"  {label}: {row}")
                metrics[book] = row
                if aum == 1e7:
                    for m in (1.5,):
                        log(f"    by period, net at {m}x: " + str(period_stats(res[book]["gross"] - m * res[book]["cost"].reindex(res[book]["gross"].index), open_holdout)))
            log_trial(f"cost:{name}", {"aum": aum, "y": y, "commission": commission, "universe": "ex-NYSE-20pct microcaps", "weights": "VW"},
                      metrics)
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--y", type=float, default=1.0, help="impact coefficient (report: order 1)")
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--no-sanity-assert", action="store_true", help="run backtests even if sanity checks fail (unvalidated)")
    a = ap.parse_args()
    main(a.open_holdout, a.y, a.commission_bps / 1e4, not a.no_sanity_assert)
