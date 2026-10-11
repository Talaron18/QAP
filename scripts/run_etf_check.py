"""Single-stock leveraged/inverse ETF check (pre-registration v5). INFORMATION ONLY.

The inventory is RETURN-BASED (crsp.stocknames_v2 has no fund names): a never-eq permno whose monthly returns since 2019 track one
eq stock with |rho| >= 0.90 and slope near an allowed leverage is a leveraged / inverse single-stock product. The name parser is run
only as a cross-check. With fewer than 10 matched ETFs the verdict is INVALID, never NOT VIABLE.

    python scripts/wrds_extract.py crsp_names      # once (select * from crsp.stocknames_v2)
    python scripts/run_etf_check.py
Needs data/raw: crsp_names, crsp_monthly (+ the Study A inputs), crsp_daily_agg, jkp_us. Writes data/etf_check_report.txt.
Uses signal RANKS (bottom / top quintile membership) and CRSP returns of ETFs and underlyings only; no signal return, no trial log.
"""
import pathlib

import pandas as pd

from qap.etfcheck import (build_etf_table, coverage, detect_by_returns, excluded_etfs, format_report, mi_str, shortfall_panel,
                          shortfall_summary)
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
    tc = [c for c in ("securitytype", "securitysubtype") if c in names.columns]
    if tc:
        log("names: securitytype x securitysubtype (top 15 of rows)")
        for k, v in names.groupby(tc, dropna=False).size().sort_values(ascending=False).head(15).items():
            log(f"  {k}: {v:,}")
    etf = detect_by_returns(panel, names)
    parsed = build_etf_table(names, panel)
    found, byname = set(etf["etf_permno"]), set(parsed["etf_permno"])
    log(f"return-detected ETFs: {len(etf)} ({int(etf['underlying_permno'].notna().sum())} matched); name-parsed: {len(parsed)}; "
        f"name-parsed also return-detected: {len(found & byname)}")
    extra = parsed[~parsed["etf_permno"].isin(found)]
    for r in extra.itertuples():
        log(f"  name-parsed only: {r.etf_ticker} {r.name} L={r.L:+.1f} underlying {r.underlying_ticker} ({r.method})")
    log("eyeball for false positives: ticker, issuernm, L, underlying, rho, beta, months, first-last")
    for r in etf.itertuples():
        log(f"  {r.etf_ticker or '-':8s} {r.name[:40]:40s} L={r.L:+.1f} {r.underlying_ticker or r.underlying_permno:>8} rho {r.rho:+.3f} "
            f"beta {r.beta:+.2f} n {int(r.n_months)} {mi_str(r.first_mi)}..{mi_str(r.last_mi)}")
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
