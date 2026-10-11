"""Single-stock leveraged/inverse ETF check (pre-registration v5). INFORMATION ONLY.

    python scripts/wrds_extract.py crsp_names      # once (select * from crsp.stocknames_v2)
    python scripts/run_etf_check.py
Needs data/raw: crsp_names, crsp_monthly (+ the Study A inputs), crsp_daily_agg, jkp_us. Writes data/etf_check_report.txt.
Uses signal RANKS (bottom / top quintile membership) and CRSP returns of ETFs and underlyings only; no signal return, no trial log.
"""
import pathlib

import pandas as pd

from qap.etfcheck import build_etf_table, coverage, excluded_etfs, format_report, shortfall_panel, shortfall_summary
from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.jkp import build_theme_signals
from qap.replicate import ff_to_mi
from qap.study import eligible_universe, load_context, require_inputs

NAMES_PQ, AGG_PQ, JKP_DIR, FF_PQ = "data/raw/crsp_names.parquet", "data/raw/crsp_daily_agg.parquet", "data/raw/jkp_us", "data/raw/ff_monthly.parquet"
SIGNALS = ["gpa", "Quality", "Profitability", "jkp_slow10", "jkp_all13"]
BANNER = "INFORMATION ONLY: no signal returns used, 2020+ signal performance not opened, no trials logged"


def main() -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/etf_check_report.txt").write_text("\n".join(report) + "\n")


def _run(log) -> None:
    require_inputs([NAMES_PQ, JKP_DIR, AGG_PQ, FF_PQ], "Run: python scripts/wrds_extract.py crsp_names jkp_us crsp_daily_agg ff_monthly")
    log(BANNER)
    names = pd.read_parquet(NAMES_PQ)
    agg = pd.read_parquet(AGG_PQ)
    panel, links, funda, _, _ = load_context()      # panel holds ALL CRSP monthly rows incl. ETFs (eq False): returns come from here
    log(f"crsp_names: {len(names):,} rows; columns: {list(names.columns)}")
    etf = build_etf_table(names, panel)
    excl = excluded_etfs(names)
    gpa = build_research_signals(panel, prepare_annual(funda), links)[["permno", "mi", "gpa"]].astype({"permno": "int64", "mi": "int64"})
    themes, _ = build_theme_signals(eligible_universe(panel), JKP_DIR)
    sig = themes.astype({"permno": "int64", "mi": "int64"}).merge(gpa, on=["permno", "mi"], how="left")
    cov = {n: coverage(sig, etf, n, panel) for n in SIGNALS}
    rf = ff_to_mi(pd.read_parquet(FF_PQ))["rf"]
    summ = shortfall_summary(shortfall_panel(etf, panel, agg, rf))
    log(f"signals (ranks only): {', '.join(SIGNALS)}; eligible universe as the diagnostic, gpa merged on (permno, mi)\n")
    for line in format_report(etf, excl, cov, summ):
        log(line)
    log("\n" + BANNER)


if __name__ == "__main__":
    main()
