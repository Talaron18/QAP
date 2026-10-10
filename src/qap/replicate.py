"""Replicate Fama-French market, SMB and HML from my own CRSP + Compustat + CCM pipeline."""
import numpy as np
import pandas as pd

from qap.fundamentals import prepare_annual
from qap.panel import attach_gvkey, lag_one_month, to_mi

PORTS = ["SL", "SM", "SH", "BL", "BM", "BH"]


def vw_market_return(panel: pd.DataFrame) -> pd.Series:
    """Value-weighted return by month. Universe and weights are as of the previous month-end."""
    lag = lag_one_month(panel, ["me", "eq"])
    m = lag[lag["eq_lag"].fillna(False).astype(bool) & (lag["me_lag"] > 0) & lag["mthret_adj"].notna()]
    num = (m["mthret_adj"] * m["me_lag"]).groupby(m["mi"]).sum()
    den = m["me_lag"].groupby(m["mi"]).sum()
    return (num / den).rename("mkt")


def _be_by_fy(annual: pd.DataFrame) -> pd.DataFrame:
    a = annual.dropna(subset=["be"]).copy()
    a["fy_year"] = a["datadate"].dt.year
    return a.sort_values("datadate").drop_duplicates(["gvkey", "fy_year"], keep="last")[["gvkey", "fy_year", "be"]]


def form_portfolios(panel: pd.DataFrame, annual: pd.DataFrame, links: pd.DataFrame,
                    start_year: int = 1964, end_year: int = 2025) -> pd.DataFrame:
    """Canonical FF 2x3 assignments, one row per (permno, formation year).

    End of June t: size = company ME in June t; BM = BE(fiscal year ending in t-1) / company ME at
    December t-1; NYSE breakpoints (size median, BM 30th/70th) from one row per company.
    """
    be = _be_by_fy(annual)
    rows = []
    for t in range(start_year, end_year + 1):
        mj, md = t * 12 + 5, (t - 1) * 12 + 11
        j = panel[(panel["mi"] == mj) & panel["eq"] & (panel["me_co"] > 0)][["permno", "permco", "mi", "mthcaldt", "me_co", "primaryexch"]]
        d = panel[(panel["mi"] == md) & (panel["me_co"] > 0)][["permno", "me_co"]].rename(columns={"me_co": "me_dec"})
        x = j.merge(d, on="permno")
        if x.empty:
            continue
        x = attach_gvkey(x, links, "mthcaldt")
        x = x.merge(be.assign(fy_year=be["fy_year"] + 1).rename(columns={"fy_year": "t"}).query("t == @t"), on="gvkey")
        x["bm"] = x["be"] / x["me_dec"]
        firm = x.sort_values("me_co", ascending=False).drop_duplicates("permco")
        nyse = firm[firm["primaryexch"] == "N"]
        if len(nyse) < 100:
            continue
        s_med = nyse["me_co"].median()
        b30, b70 = nyse["bm"].quantile([0.3, 0.7])
        x["size_grp"] = np.where(x["me_co"] <= s_med, "S", "B")
        x["bm_grp"] = np.where(x["bm"] <= b30, "L", np.where(x["bm"] <= b70, "M", "H"))
        x["port"] = x["size_grp"] + x["bm_grp"]
        x["t"] = t
        rows.append(x[["permno", "t", "port"]])
    return pd.concat(rows, ignore_index=True)


def expand_holding(assign: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Each June-t assignment is held July t .. June t+1 (offsets 1..12 from the June month index)."""
    parts = []
    for k in range(1, months + 1):
        a = assign.copy()
        a["mi"] = a["t"] * 12 + 5 + k
        parts.append(a)
    return pd.concat(parts, ignore_index=True)


def portfolio_returns(panel: pd.DataFrame, assign: pd.DataFrame) -> pd.DataFrame:
    """Value-weighted monthly returns of the six portfolios (weights: permno ME at previous month-end)."""
    held = expand_holding(assign)
    lag = lag_one_month(panel, ["me"])[["permno", "mi", "mthret_adj", "me_lag"]]
    m = held.merge(lag, on=["permno", "mi"])
    m = m[m["mthret_adj"].notna() & (m["me_lag"] > 0)]
    m["w"] = m["me_lag"]
    m["wr"] = m["w"] * m["mthret_adj"]
    g = m.groupby(["mi", "port"])[["wr", "w"]].sum()
    r = (g["wr"] / g["w"]).unstack("port")
    return r.reindex(columns=PORTS)


def smb_hml(r: pd.DataFrame) -> pd.DataFrame:
    smb = r[["SL", "SM", "SH"]].mean(axis=1) - r[["BL", "BM", "BH"]].mean(axis=1)
    hml = r[["SH", "BH"]].mean(axis=1) - r[["SL", "BL"]].mean(axis=1)
    return pd.DataFrame({"smb": smb, "hml": hml})


def ff_to_mi(ff: pd.DataFrame) -> pd.DataFrame:
    f = ff.copy()
    f["mi"] = to_mi(f["date"]).values
    f = f.set_index("mi")
    cols = [c for c in ["mktrf", "smb", "hml", "umd", "rf"] if c in f]
    f[cols] = f[cols].astype(float)
    if f[cols].abs().max().max() > 1.5:  # percent units
        f[cols] = f[cols] / 100.0
    return f


def compare(mine: pd.Series, theirs: pd.Series) -> dict:
    j = pd.concat([mine.rename("mine"), theirs.rename("ff")], axis=1).dropna()
    return {
        "corr": float(j["mine"].corr(j["ff"])),
        "mean_diff_pct_per_month": float((j["mine"] - j["ff"]).mean() * 100),
        "n_months": int(len(j)),
        "window": f"{int(j.index.min() // 12)}-{int(j.index.min() % 12) + 1:02d} .. {int(j.index.max() // 12)}-{int(j.index.max() % 12) + 1:02d}",
    }
