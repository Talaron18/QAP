"""Monthly panel keyed on calendar month index `mi` (= year*12 + month - 1), not on mthcaldt."""
import numpy as np
import pandas as pd

from qap.crsp import adjust_delisting_returns, common_equity_mask


def to_mi(dates) -> pd.Series:
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year * 12 + d.dt.month - 1).astype("int64")


def mi_to_period(mi):
    return pd.PeriodIndex([pd.Period(year=int(m // 12), month=int(m % 12) + 1, freq="M") for m in np.atleast_1d(mi)])


def assert_month_grid(df: pd.DataFrame, start: str = "1963-01", end: str = "2025-12") -> None:
    first, last = to_mi(pd.Series([start + "-01"])).iloc[0], to_mi(pd.Series([end + "-01"])).iloc[0]
    expected = int(last - first + 1)
    got = df["mi"].nunique()
    assert got == expected, f"expected {expected} unique months {start}..{end}, found {got}"
    assert df["mi"].min() == first and df["mi"].max() == last, "month range differs from expected"
    assert not df.duplicated(["permno", "mi"]).any(), "duplicate (permno, month) rows"


def calibrate_me_scale(panel: pd.DataFrame, funda: pd.DataFrame, links: pd.DataFrame) -> tuple:
    """Median of mthcap / (csho*prcc_f) in each firm's fiscal-year-end month; returns (scale, ratio).

    Compustat csho*prcc_f is in $ millions, so scale = 10**round(log10(ratio)) converts mthcap to
    $ millions (1 -> already millions, 1000 -> thousands). Asserts the corrected ratio is near 1.
    """
    f = funda.dropna(subset=["csho", "prcc_f"]).copy()
    f["gvkey"] = f["gvkey"].astype(str).str.strip().str.zfill(6)
    f = f[(f.csho > 0) & (f.prcc_f > 0)]
    f["mi"] = to_mi(f["datadate"])
    f["cmp_me"] = f["csho"] * f["prcc_f"]
    f["date"] = pd.to_datetime(f["datadate"])
    k = panel[["permno", "mi", "mthcap", "mthcaldt"]].dropna(subset=["mthcap"])
    k = k[k.mthcap > 0]
    k = attach_gvkey(k, links, date_col="mthcaldt")
    j = k.merge(f[["gvkey", "mi", "cmp_me"]], on=["gvkey", "mi"])
    assert len(j) > 1000, f"too few linked fiscal-year-end months for unit calibration: {len(j)}"
    ratio = float((j["mthcap"] / j["cmp_me"]).median())
    scale = float(10 ** round(np.log10(ratio)))
    assert 0.7 < ratio / scale < 1.4, f"mthcap/(csho*prcc_f) median {ratio:.3f} is not near a power of ten"
    return scale, ratio


def attach_gvkey(df: pd.DataFrame, links: pd.DataFrame, date_col: str = "mthcaldt") -> pd.DataFrame:
    """Add gvkey using date-bounded CCM links (filtered). Ties: linkprim P over C, then latest linkdt.

    Asserts at most one gvkey per (permno, month). Rows without a link are dropped.
    """
    l = links[["gvkey", "permno", "linkprim", "linkdt", "linkenddt"]].copy()
    l["gvkey"] = l["gvkey"].astype(str).str.strip().str.zfill(6)
    l["prim_rank"] = (l["linkprim"] != "P").astype(int)
    m = df.merge(l, on="permno")
    d = pd.to_datetime(m[date_col])
    m = m[(m["linkdt"] <= d) & (d <= m["linkenddt"])]
    m = m.sort_values(["permno", "mi", "prim_rank", "linkdt"], ascending=[True, True, True, False])
    m = m.drop_duplicates(["permno", "mi"], keep="first")
    assert not m.duplicated(["permno", "mi"]).any(), "more than one gvkey for a (permno, month)"
    return m.drop(columns=["linkprim", "linkdt", "linkenddt", "prim_rank"])


def build_monthly_panel(monthly: pd.DataFrame, delists: pd.DataFrame, me_scale: float = 1.0) -> pd.DataFrame:
    """Month-keyed panel with delisting-adjusted returns, company-level ME and the equity mask.

    `eq` marks the formation universe. Returns are NEVER filtered by it: labels use mthret_adj of
    every row so names leaving the universe keep their realised return (no survivorship bias).
    """
    p = adjust_delisting_returns(monthly, delists)
    p["mi"] = to_mi(p["mthcaldt"]).values
    p["me"] = p["mthcap"].astype(float) / me_scale
    me_co = p.groupby(["permco", "mi"])["me"].sum(min_count=1).rename("me_co").reset_index()
    p = p.merge(me_co, on=["permco", "mi"], how="left")
    p["eq"] = common_equity_mask(p).values
    keep = ["permno", "permco", "mthcaldt", "mi", "mthret", "mthretx", "mthret_adj", "delist_adj_method", "mthdelflg",
            "mthcap", "me", "me_co", "eq", "primaryexch", "shrout", "mthcumfacshr", "siccd", "mthvol"]
    return p[[c for c in keep if c in p.columns]]


def lag_one_month(df: pd.DataFrame, cols, suffix: str = "_lag") -> pd.DataFrame:
    """Merge each (permno, month) with its values from the previous calendar month."""
    prev = df[["permno", "mi", *cols]].copy()
    prev["mi"] = prev["mi"] + 1
    return df.merge(prev.rename(columns={c: c + suffix for c in cols}), on=["permno", "mi"], how="left")
