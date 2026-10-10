"""Sanity-check data/raw/*.parquet and write data/raw_report.txt (summary statistics only).

    python scripts/check_raw.py
Share the report: it has counts and distributions, no row-level data.
"""
import pathlib

import pandas as pd

RAW = pathlib.Path("data/raw")


def main() -> None:
    out = []

    def log(msg=""):
        print(msg)
        out.append(str(msg))

    m = pd.read_parquet(RAW / "crsp_monthly.parquet")
    log(f"crsp_monthly: {len(m):,} rows, {m.permno.nunique():,} permnos, {m.mthcaldt.min()} .. {m.mthcaldt.max()}")
    log(f"  duplicate (permno, mthcaldt): {m.duplicated(['permno', 'mthcaldt']).sum()}")
    log(f"  null mthret: {m.mthret.isna().mean():.2%}   mthret min/max: {m.mthret.min():.3f} / {m.mthret.max():.3f}")
    for c in ["mthdelflg", "securitytype", "securitysubtype", "sharetype", "usincflg", "issuertype",
              "primaryexch", "conditionaltype", "tradingstatusflg"]:
        log(f"  {c}: " + m[c].value_counts(dropna=False).head(12).to_dict().__repr__())
    # delisting behaviour: last row per permno, by delisting flag
    last = m.sort_values("mthcaldt").groupby("permno").tail(1)
    log("  last row per permno, mthdelflg vs mthret null share:")
    g = last.groupby("mthdelflg", dropna=False).mthret.agg(n="size", null_share=lambda s: s.isna().mean(), median="median", mean="mean")
    log(g.to_string())

    for name in ["ccm_links", "compustat_annual", "compustat_quarterly", "ff_monthly"]:
        p = RAW / f"{name}.parquet"
        if not p.exists():
            log(f"{name}: MISSING")
            continue
        d = pd.read_parquet(p)
        log(f"{name}: {len(d):,} rows, {d.shape[1]} cols")
        if name == "ccm_links":
            log("  linktype: " + d.linktype.value_counts().to_dict().__repr__() + "  linkprim: " + d.linkprim.value_counts().to_dict().__repr__())
            log(f"  null linkenddt: {d.linkenddt.isna().mean():.2%}")
        if name == "compustat_annual":
            log(f"  datadate {d.datadate.min()} .. {d.datadate.max()}; dup (gvkey,datadate): {d.duplicated(['gvkey', 'datadate']).sum()}")
        if name == "compustat_quarterly":
            log(f"  null rdq: {d.rdq.isna().mean():.2%}")

    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/raw_report.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
