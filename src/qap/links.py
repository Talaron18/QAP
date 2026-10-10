"""CRSP-Compustat (CCM) link filtering."""
import pandas as pd

VALID_LINKTYPES = ("LC", "LU")
VALID_LINKPRIMS = ("P", "C")


def filter_ccm_links(links: pd.DataFrame) -> pd.DataFrame:
    """Keep usable links: linktype LC/LU and linkprim P/C.

    Expects columns gvkey, permno (or CCM's lpermno), linktype, linkprim, linkdt, linkenddt.
    A null linkenddt means the link is still current and is filled with
    pd.Timestamp.max so date-bounded joins work.
    """
    if "permno" not in links.columns and "lpermno" in links.columns:
        links = links.rename(columns={"lpermno": "permno"})
    out = links[links["linktype"].isin(VALID_LINKTYPES) & links["linkprim"].isin(VALID_LINKPRIMS)].copy()
    out["permno"] = out["permno"].astype("int64")
    out["linkdt"] = pd.to_datetime(out["linkdt"])
    out["linkenddt"] = pd.to_datetime(out["linkenddt"]).fillna(pd.Timestamp.max)
    return out.reset_index(drop=True)


def links_valid_on(links: pd.DataFrame, date) -> pd.DataFrame:
    """Links from `filter_ccm_links` that are in force on `date`."""
    d = pd.Timestamp(date)
    return links[(links["linkdt"] <= d) & (d <= links["linkenddt"])]
