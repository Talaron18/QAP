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
