"""CRSP CIZ (v2) monthly helpers.

Delisting flag semantics are confirmed from crsp.metaflaginfo (flagtype DE): mthret already
contains the delisting return when mthdelflg is A, P or V; it EXCLUDES it when mthdelflg is G
(delisting price more than 10 days from the delisting date) or M (missing); N means active.
"""
import pandas as pd

RET_INCLUDES_DELISTING = ("A", "P", "V")
RET_EXCLUDES_DELISTING = ("G", "M")

# CIZ analogue of legacy "common stock on NYSE/AMEX/Nasdaq" (shrcd 10/11, exchcd 1/2/3).
# This mapping follows CRSP's conventions as I understand them; every value used exists in the
# user's data (see data/raw_report.txt) but the equivalence itself is NOT verified against
# crsp.metasiztociz. Keep it explicit and adjustable.
DEFAULT_ISSUER_TYPES = ("ACOR", "CORP")
DEFAULT_EXCHANGES = ("N", "A", "Q")


def common_equity_mask(monthly: pd.DataFrame, issuer_types=DEFAULT_ISSUER_TYPES,
                       exchanges=DEFAULT_EXCHANGES) -> pd.Series:
    """US-incorporated ordinary common stock on a primary exchange that is regular-way and active."""
    return (
        (monthly["securitytype"] == "EQTY")
        & (monthly["securitysubtype"] == "COM")
        & (monthly["sharetype"] == "NS")
        & (monthly["usincflg"] == "Y")
        & monthly["issuertype"].isin(issuer_types)
        & monthly["primaryexch"].isin(exchanges)
        & (monthly["conditionaltype"] == "RW")
        & (monthly["tradingstatusflg"] == "A")
    )


def flag_delisting_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns saying whether mthret contains the delisting return.

    delist_ret_included: delisting month with the return folded into mthret (A/P/V)
    delist_ret_missing:  delisting month where it was left out (G/M); needs explicit treatment
    """
    out = monthly.copy()
    out["delist_ret_included"] = out["mthdelflg"].isin(RET_INCLUDES_DELISTING)
    out["delist_ret_missing"] = out["mthdelflg"].isin(RET_EXCLUDES_DELISTING)
    return out


def monthly_return_panel(monthly: pd.DataFrame, value: str = "mthret") -> pd.DataFrame:
    """Pivot to a month-end x permno panel (sorted)."""
    return monthly.pivot(index="mthcaldt", columns="permno", values=value).sort_index()


# Delisting reasons I treat as performance-related, following the legacy Shumway (1997) code groups
# (500, 520-584: unavailable/OTC/capital/float/market makers/shareholders/bankruptcy/delinquent/SEC).
# This mapping onto CIZ reason types is a judgement, not a CRSP-published equivalence. The user's
# delisting file supports it: FING, INSC, LP, DELQ, BKPY, EQRQ, SERQ have clearly negative mean
# delisting returns. UNAV (65% of all delistings, mostly mergers) is NOT included.
PERFORMANCE_REASONS = (
    "BKPY", "DELQ", "DEEX", "EQRQ", "FARG", "FING", "INSC", "INSF", "LP", "MTMK",
    "MVOT", "SERQ", "SHLD", "UNL", "VIO",
)
SHUMWAY_IMPUTED_RETURN = -0.30


def adjust_delisting_returns(monthly: pd.DataFrame, delists: pd.DataFrame,
                             performance_reasons=PERFORMANCE_REASONS,
                             imputed_return: float = SHUMWAY_IMPUTED_RETURN) -> pd.DataFrame:
    """Add `mthret_adj` and `delist_adj_method`, fixing only the months where mthret EXCLUDES the delisting return.

    Rows with mthdelflg A/P/V already include it, N are active: mthret_adj == mthret, method 'none'.
    For G/M rows, matched on (permno, calendar month of delistingdt):
      - 'delret':  stkdelists has a usable delret (delretmisstype == 'NA'): compound it onto mthret
      - 'imputed': no usable delret and a performance-related reason: compound `imputed_return`
      - 'zero':    no usable delret, other reason: leave mthret (delisting return taken as 0)
      - 'unmatched': G/M row with no stkdelists row: leave mthret and flag it
    A missing mthret in a G/M month is treated as 0 before compounding.
    """
    d = delists.copy()
    d["delistingdt"] = pd.to_datetime(d["delistingdt"])
    d["_ym"] = d["delistingdt"].dt.to_period("M")
    d = d.sort_values("delistingdt").drop_duplicates(["permno", "_ym"], keep="last")
    out = monthly.copy()
    out["_ym"] = pd.to_datetime(out["mthcaldt"]).dt.to_period("M")
    cols = ["permno", "_ym", "delret", "delretmisstype", "delreasontype"]
    out = out.merge(d[cols], on=["permno", "_ym"], how="left")
    out["delist_adj_method"] = "none"
    out["mthret_adj"] = out["mthret"]

    gm = out["mthdelflg"].isin(RET_EXCLUDES_DELISTING)
    matched = out["delreasontype"].notna()
    base = out["mthret"].fillna(0.0)
    usable = out["delret"].notna() & (out["delretmisstype"] == "NA")
    perf = out["delreasontype"].isin(performance_reasons)

    def apply(mask, delisting_ret, label):
        out.loc[mask, "mthret_adj"] = (1.0 + base[mask]) * (1.0 + delisting_ret) - 1.0
        out.loc[mask, "delist_adj_method"] = label

    apply(gm & matched & usable, out["delret"].astype(float).fillna(0.0)[gm & matched & usable], "delret")
    apply(gm & matched & ~usable & perf, imputed_return, "imputed")
    z = gm & matched & ~usable & ~perf
    out.loc[z, "delist_adj_method"] = "zero"
    out.loc[z, "mthret_adj"] = base[z]
    out.loc[gm & ~matched, "delist_adj_method"] = "unmatched"
    return out.drop(columns=["_ym", "delret", "delretmisstype", "delreasontype"])
