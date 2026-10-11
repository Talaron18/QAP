"""Pre-registration v7 (ROADMAP.md): tax-aware direct indexing / loss-harvesting simulation. INFORMATION ONLY, NOT an alpha test.

    python scripts/run_taxsim.py [--jobs N] [--start-year 1963 --end-year 2015] [--contrib 500 --outside-gains 3000 --cost-bp 5
                                  --harvest-pct 10 --n-names 150] [--sensitivities]
Needs data/raw: crsp_monthly (with mthretx), crsp_delistings (+ ccm_links, compustat_annual for the ME unit calibration).
Writes data/taxsim_report.txt. No signals, no trial logging, deterministic (no seeds).
"""
import argparse
import pathlib
import time

import pandas as pd

from qap.study import require_inputs
from qap.taxsim import TaxParams, build_inputs, format_summary, run_windows, summarize_windows

RAW = pathlib.Path("data/raw")
BANNER = "INFORMATION ONLY: mechanical tax simulation, no signal used, no trials logged, sealed 2020+ window not relevant"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--start-year", type=int, default=1963)
    ap.add_argument("--end-year", type=int, default=2015)
    ap.add_argument("--contrib", type=float, default=500.0)
    ap.add_argument("--outside-gains", type=float, default=3000.0)
    ap.add_argument("--cost-bp", type=float, default=5.0)
    ap.add_argument("--harvest-pct", type=float, default=10.0)
    ap.add_argument("--n-names", type=int, default=150)
    ap.add_argument("--wash-rule", choices=["proportional", "literal"], default="proportional")
    ap.add_argument("--sensitivities", action="store_true")
    args = ap.parse_args()
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(args, log)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/taxsim_report.txt").write_text("\n".join(report) + "\n")


def load_panel(log):
    from qap.links import filter_ccm_links
    from qap.panel import build_monthly_panel, calibrate_me_scale
    monthly = pd.read_parquet(RAW / "crsp_monthly.parquet")
    delists = pd.read_parquet(RAW / "crsp_delistings.parquet")
    if "mthretx" not in monthly.columns:
        log("WARNING: crsp_monthly has no mthretx: dividend yield set to 0 (all return treated as price return)")
    panel = build_monthly_panel(monthly, delists, me_scale=1.0)
    try:
        links = filter_ccm_links(pd.read_parquet(RAW / "ccm_links.parquet"))
        funda = pd.read_parquet(RAW / "compustat_annual.parquet")
        scale, ratio = calibrate_me_scale(panel, funda, links)
        panel["me"] = panel["mthcap"].astype(float) / scale
        panel["me_co"] = panel.groupby(["permco", "mi"])["me"].transform("sum")
        log(f"ME unit calibration (as qap.study.load_context): mthcap/(csho*prcc_f) median {ratio:.3f} -> scale {scale:g} ($ millions); "
            f"ranks and VW weights are scale-invariant")
    except FileNotFoundError as e:
        log(f"ME calibration skipped ({e}); ranks and VW weights are scale-invariant")
    return panel


def _run(args, log) -> None:
    require_inputs([RAW / "crsp_monthly.parquet", RAW / "crsp_delistings.parquet"], "Run: python scripts/wrds_extract.py crsp_monthly crsp_delistings")
    t0 = time.time()
    log(BANNER)
    panel = load_panel(log)
    inp = build_inputs(panel)
    log(f"panel: {len(panel):,} rows; wide arrays {inp['pr'].shape[0]} months x {inp['pr'].shape[1]:,} ever-equity permnos "
        f"(mi {inp['months'][0]}..{inp['months'][-1]})")
    base = TaxParams(contrib=args.contrib, outside_lt_gains=args.outside_gains, cost_bp=args.cost_bp, harvest_pct=args.harvest_pct / 100.0,
                     n_names=args.n_names, wash_rule=args.wash_rule)
    log("\n== ASSUMPTIONS (pre-registration v7; constants, to be confirmed with a tax adviser) ==")
    log(f"  tax rates: ST {base.st_rate:.3f}  LT {base.lt_rate:.3f}  dividends {base.div_rate:.3f} (federal + 3.8% NIIT + 5% state folded in)")
    log(f"  net loss cap vs ordinary income USD {base.loss_cap:,.0f}/yr at ST rate; remainder carried forward indefinitely (ST/LT character kept)")
    log(f"  outside realised LT gains USD {base.outside_lt_gains:,.0f}/yr in ALL arms (absorbs harvested losses at the LT rate)")
    log(f"  start USD {base.start_capital:,.0f}; contributions USD {base.contrib:,.0f}/month at month end; cost {base.cost_bp:g} bp per side (DIY arms)")
    log(f"  universe: {base.n_names} largest company-ME common stocks at window start (one permno per permco), VW at purchase")
    log(f"  harvest: lot loss >= {base.harvest_pct * 100:g}% of basis and >= USD {base.harvest_min_usd:g}; pro-rata reinvestment; sold name blocked "
        f"from purchases that and the next month; wash rule '{base.wash_rule}': (proportional) a name's loss lots may be sold up to their "
        f"value minus dollars of that name bought last month, largest loss % first, partial lots allowed; (literal) names bought last month skipped")
    log("  windows: 10 years starting January of each year (overlapping); long-term = held MORE than 12 months")
    log("  tax cash runs through the account: each year's tax is settled at the December close against the next month's investable cash "
        "(refunds add, tax owed reduces; a negative balance, e.g. with zero contributions, is carried at zero interest); the final year's tax "
        "is deducted from terminal value. Annualisation is geometric: (W_a/W_b)^(1/10)-1")
    log("  month-0 market weights use their own month (no lag available at the panel's first month); universe ranked on the month BEFORE the window when available")

    res = run_windows(inp, args.start_year, args.end_year, base, args.jobs, say=log)
    s = summarize_windows(res)
    log(f"\n== PRE-REGISTERED STATISTIC ({s['n_windows']} windows, start years {args.start_year}..{args.end_year}) ==")
    for line in format_summary(s):
        log(line)
    log("\n== PRIMARY STATISTIC BY DECADE OF START YEAR (harvest - no-harvest, hold-indefinitely, %/yr) ==")
    for d, v in s["by_decade"].items():
        log(f"  {d}s: median {v['median']:+.3f}  IQR [{v['q25']:+.3f}, {v['q75']:+.3f}]  p10 {v['p10']:+.3f}  share>0 {v['share_pos']:.2f}  (n={v['n']})")

    if args.sensitivities:
        log("\n== SENSITIVITIES (information only; the decision rests on the primary assumptions above) ==")
        variants = [("contributions 0", dict(contrib=0.0)), ("contributions 1500", dict(contrib=1500.0)),
                    ("outside gains 0", dict(outside_lt_gains=0.0)), ("outside gains 10000", dict(outside_lt_gains=10000.0)),
                    ("cost 0 bp", dict(cost_bp=0.0)), ("cost 15 bp", dict(cost_bp=15.0)),
                    ("threshold 5%", dict(harvest_pct=0.05)), ("threshold 15%", dict(harvest_pct=0.15)),
                    ("100 names", dict(n_names=100)), ("250 names", dict(n_names=250)),
                    ("wash rule literal" if base.wash_rule == "proportional" else "wash rule proportional",
                     dict(wash_rule="literal" if base.wash_rule == "proportional" else "proportional"))]
        for name, kw in variants:
            ss = summarize_windows(run_windows(inp, args.start_year, args.end_year, base.with_(**kw), args.jobs, say=log))
            p, q = ss["primary"], ss["liq"]
            log(f"  {name:20s} hold: median {p['median']:+.3f} IQR [{p['q25']:+.3f}, {p['q75']:+.3f}] p10 {p['p10']:+.3f} share>0 {p['share_pos']:.2f} | "
                f"liquidate median {q['median']:+.3f} | TE hv-vs-nohv {ss['te_hv_vs_nohv']['median']:.2f} | withheld ${ss['withheld_median']:,.0f} | {ss['decision']}")
    log(f"\nDECISION (primary assumptions): {s['decision']}")
    log(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
