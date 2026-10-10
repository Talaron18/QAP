"""Compustat annual features and point-in-time attachment."""
import numpy as np
import pandas as pd

from qap.panel import attach_gvkey
from qap.pit import annual_available_date


def book_equity(f: pd.DataFrame) -> pd.Series:
    """Fama-French book equity: SEQ (else CEQ+PSTK, else AT-LT) + TXDITC (0 if missing) - PS.

    PS = PSTKRV, else PSTKL, else PSTK, else 0. Non-positive book equity becomes NaN.
    """
    seq = f["seq"].copy()
    seq = seq.fillna(f["ceq"] + f["pstk"])
    seq = seq.fillna(f["at"] - f["lt"])
    ps = f["pstkrv"].fillna(f["pstkl"]).fillna(f["pstk"]).fillna(0.0)
    be = seq + f["txditc"].fillna(0.0) - ps
    return be.where(be > 0)


def prepare_annual(funda: pd.DataFrame, lag_months: int = 4) -> pd.DataFrame:
    """Annual features and availability dates. gvkey is zero-padded to 6 characters."""
    f = funda.copy()
    f["gvkey"] = f["gvkey"].astype(str).str.strip().str.zfill(6)
    f["datadate"] = pd.to_datetime(f["datadate"])
    f = f.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    f["be"] = book_equity(f)
    f["gp"] = f["revt"] - f["cogs"]
    prev_at = f.groupby("gvkey")["at"].shift(1)
    prev_dd = f.groupby("gvkey")["datadate"].shift(1)
    gap = (f["datadate"] - prev_dd).dt.days
    f["at_lag"] = prev_at.where(gap.between(300, 430))
    f["avail"] = annual_available_date(f["datadate"], months=lag_months)
    return f


PIT_COLS = ["be", "gp", "at", "at_lag"]


def attach_pit(keys: pd.DataFrame, annual: pd.DataFrame, links: pd.DataFrame, max_stale_days: int = 548,
               cols=None) -> pd.DataFrame:
    """Attach the latest annual record available on each (permno, month-end), via CCM links.

    keys needs permno, mi, mthcaldt. Records older than `max_stale_days` past availability are ignored.
    Returns keys (that have a link) plus gvkey, datadate, be, gp, at, at_lag.
    """
    k = attach_gvkey(keys, links, "mthcaldt")
    k["mthcaldt"] = pd.to_datetime(k["mthcaldt"])
    k = k.sort_values("mthcaldt")
    a = annual[["gvkey", "avail", "datadate", *(cols or PIT_COLS)]].sort_values("avail")
    out = pd.merge_asof(k, a, left_on="mthcaldt", right_on="avail", by="gvkey", direction="backward",
                        tolerance=pd.Timedelta(days=max_stale_days))
    return out.drop(columns=["avail"])
