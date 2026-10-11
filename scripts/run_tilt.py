"""Long-only gpa tilt (see ROADMAP 'Pre-registration v2', commit 0fef4ed). Final spec = (c); (a) and (b) are ablations.

    python scripts/run_tilt.py [--open-holdout]
Writes data/tilt_report.txt. 2020+ is sealed; --open-holdout opens it ONLY for spec (c) (once, for the final spec).
Net active return = gross tilt return - gross benchmark return - 1.5 x tilt trading costs (benchmark costless; borrow n/a).
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

from qap.costs import liquidity_inputs
from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.links import filter_ccm_links
from qap.panel import build_monthly_panel, calibrate_me_scale
from qap.portfolio import leg_costs, leg_gross_return, market_scaled_aum
from qap.replicate import ff_to_mi
from qap.tilt import (active_stats, benchmark_weights, factor_regression, holding_stats, tilt_weights)
from qap.trials import log_trial, trial_count

RAW = pathlib.Path("data/raw")
AUM, MULT, Y = 1e7, 1.5, 1.0
SPECS = {"a_baseline": dict(entry=0.8, exit_=0.8, cap=None),
         "b_buffer": dict(entry=0.8, exit_=0.7, cap=None),
         "c_buffer_cap": dict(entry=0.8, exit_=0.7, cap=0.03)}
FINAL = "c_buffer_cap"
SPLITS = {"pre1990": (1963, 1989), "dev": (1990, 2014), "val": (2015, 2019)}


def main(open_holdout: bool) -> None:
    report = []

    def log(msg=""):
        print(msg, flush=True)
        report.append(str(msg))

    try:
        _run(log, open_holdout)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/tilt_report.txt").write_text("\n".join(report) + "\n")


def _run(log, open_holdout) -> None:
    monthly = pd.read_parquet(RAW / "crsp_monthly.parquet")
    delists = pd.read_parquet(RAW / "crsp_delistings.parquet")
    links = filter_ccm_links(pd.read_parquet(RAW / "ccm_links.parquet"))
    funda = pd.read_parquet(RAW / "compustat_annual.parquet")
    agg = pd.read_parquet(RAW / "crsp_daily_agg.parquet")
    ff = ff_to_mi(pd.read_parquet(RAW / "ff_monthly.parquet"))

    panel = build_monthly_panel(monthly, delists, me_scale=1.0)
    scale, _ = calibrate_me_scale(panel, funda, links)
    panel["me"] = panel["mthcap"].astype(float) / scale
    panel["me_co"] = panel.groupby(["permco", "mi"])["me"].transform("sum")
    inp = liquidity_inputs(panel, agg)
    aum_t = market_scaled_aum(panel, AUM)
    sig = build_research_signals(panel, prepare_annual(funda), links)
    siccd = panel[["permno", "mi", "siccd"]] if "siccd" in panel else None
    B = benchmark_weights(sig, "gpa")
    gross_b = leg_gross_return(B, panel)

    log(f"== LONG-ONLY gpa TILT: ${AUM:,.0f} (2019 market-size), cost stress {MULT}x, impact Y={Y}; benchmark = VW eligible universe (costless) ==")
    log("holdout " + ("OPENED for the final spec only" if open_holdout else "sealed"))
    results = {}
    for name, kw in SPECS.items():
        W = tilt_weights(sig, "gpa", **kw)
        gross = leg_gross_return(W, panel)
        cost = leg_costs(W, panel, inp, aum_t, Y, 0.0)
        c1 = cost["cost"].set_axis(cost.index + 1)
        idx = gross.index.intersection(gross_b.index).intersection(c1.index)
        net = (gross[idx] - gross_b[idx] - MULT * c1[idx]).rename("net")
        gross_act = (gross[idx] - gross_b[idx]).rename("gross")
        hs = holding_stats(W, B, siccd)
        hs["turnover"] = cost["turnover"]
        hs["cost1x_pct"] = cost["cost"] * 100
        results[name] = (W, net, gross_act, hs, c1)
        incl = open_holdout and name == FINAL
        log(f"\n[{name}] {kw}" + ("   <-- FINAL SPEC" if name == FINAL else "   (ablation, information only)"))
        log("  net active return / TE / IR by split: ")
        for k, v in active_stats(net, incl).items():
            log(f"    {k}: {v}")
        log("  gross active return (before costs) by split: " + str({k: v.get("active_pct_mo") for k, v in active_stats(gross_act, incl).items()}))
        log("  holdings, max weight, top-10 weight, active share, one-way turnover, cost@1x (%/mo), max |active SIC2| (monthly means by split):")
        for sp, (y0, y1) in SPLITS.items():
            h = hs[(hs.index >= y0 * 12) & (hs.index <= y1 * 12 + 11)].mean(numeric_only=True).round(3)
            log(f"    {sp}: " + ", ".join(f"{c}={h[c]}" for c in hs.columns if c in h.index))
        log_trial(f"tilt:{name}", dict(kw, aum=AUM, mult=MULT, y=Y, benchmark="VW eligible", prereg="0fef4ed"),
                  {k: v for k, v in active_stats(net, False).items()})

    W, net, _, hs, _ = results[FINAL]
    log("\n== INFORMATION ONLY (final spec): net active return regressed on mktrf/smb/hml/umd ==")
    for sp, (y0, y1) in SPLITS.items():
        if not open_holdout and y1 >= 2020:
            continue
        n = net[(net.index >= y0 * 12) & (net.index <= y1 * 12 + 11)]
        log(f"  {sp}: " + str(factor_regression(n, ff)))
    if siccd is not None:
        j = W.merge(B, on=["permno", "mi"], how="outer", suffixes=("", "_b")).fillna({"w": 0.0, "w_b": 0.0}).merge(siccd, on=["permno", "mi"], how="left")
        j["sic2"] = (j["siccd"] // 100).fillna(-1).astype(int)
        dev = j[(j["mi"] >= 1990 * 12) & (j["mi"] <= 2014 * 12 + 11)]
        act = (dev.groupby(["mi", "sic2"])[["w", "w_b"]].sum().assign(a=lambda d: d["w"] - d["w_b"])["a"]).groupby("sic2").mean().sort_values()
        log("  industry (SIC2) average active weight, dev: most underweight " + str(act.head(3).round(3).to_dict())
            + "; most overweight " + str(act.tail(3).round(3).to_dict()))
    log(f"\nunique trials logged so far: {trial_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true", help="open 2020+ for the FINAL spec only (once)")
    main(ap.parse_args().open_holdout)
