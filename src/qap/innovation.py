"""Study B: innovation signals (pre-registration v3). All inputs are point in time by construction (see each function)."""
import numpy as np
import pandas as pd

from qap.fundamentals import attach_pit
from qap.panel import to_mi

RD_WEIGHTS = (1.0, 0.8, 0.6, 0.4, 0.2)   # sum_{j=0..4} (1 - 0.2 j) xrd_{y-j}


def june30_after(year: pd.Series) -> pd.Series:
    """Availability date: June 30 of year+1."""
    return pd.to_datetime((year.astype("int64") + 1).astype(str) + "-06-30")


def rd_capital(annual: pd.DataFrame) -> pd.DataFrame:
    """R&D capital by (gvkey, fyear): sum of weighted xrd over the last 5 fiscal years; needs >= 3 observed years, stock > 0."""
    a = annual[["gvkey", "fyear", "xrd"]].dropna(subset=["fyear"]).copy()
    a["fyear"] = a["fyear"].astype("int64")
    a = a.drop_duplicates(["gvkey", "fyear"], keep="last")
    base = a[["gvkey", "fyear"]].copy()
    stock = pd.Series(0.0, index=base.index)
    seen = pd.Series(0, index=base.index)
    for j, w in enumerate(RD_WEIGHTS):
        lag = a.assign(fyear=a["fyear"] + j)[["gvkey", "fyear", "xrd"]].rename(columns={"xrd": "x"})
        x = base.merge(lag, on=["gvkey", "fyear"], how="left")["x"]
        stock = stock + w * x.fillna(0.0).to_numpy()
        seen = seen + x.notna().astype(int).to_numpy()
    base["rd_capital"] = stock.where((seen >= 3) & (stock > 0))
    return base


def rq_table(rq: pd.DataFrame) -> pd.DataFrame:
    t = rq.dropna(subset=["fyear", "rawrq"]).copy()
    t["gvkey"] = t["gvkey"].astype(str).str.strip().str.zfill(6)
    t["avail"] = june30_after(t["fyear"])
    t["datadate"] = t["avail"]
    return t[["gvkey", "avail", "datadate", "rawrq"]].rename(columns={"rawrq": "rq"})


def patent_tables(grants: pd.DataFrame, rdcap: pd.DataFrame) -> tuple:
    """(flow table, quality table) keyed for point-in-time attachment.

    flow:    patents granted in calendar year y / R&D capital at fiscal year y, available June 30 of y+1; firms with R&D capital
             but no grant rows count as 0 patents, only for years the grant data fully cover (y < last grant year)
    quality: cohort y-3's 3-year citations per patent (complete at the end of year y), available June 30 of y+1
    """
    g = grants.copy()
    g["gvkey"] = g["gvkey"].astype(str).str.strip().str.zfill(6)
    last_year = int(g["grant_year"].max())
    r = rdcap.dropna(subset=["rd_capital"]).copy()
    r = r[r["fyear"] < last_year]
    f = r.merge(g.rename(columns={"grant_year": "fyear"})[["gvkey", "fyear", "n_patents"]], on=["gvkey", "fyear"], how="left")
    f["pat_flow_rd"] = f["n_patents"].fillna(0.0) / f["rd_capital"]
    f["avail"] = june30_after(f["fyear"])
    f["datadate"] = f["avail"]
    q = g[g["n_cite3y"] >= 1].copy()
    q["pat_quality3"] = q["cites3y_sum"] / q["n_cite3y"]
    q["avail"] = june30_after(q["grant_year"] + 3)
    q["datadate"] = q["avail"]
    return f[["gvkey", "avail", "datadate", "pat_flow_rd"]], q[["gvkey", "avail", "datadate", "pat_quality3"]]


def kpss_value_36m(keys: pd.DataFrame, kpss: pd.DataFrame, end_mi: int = None) -> pd.Series:
    """Sum of patent values (xi_nominal) issued in the 36 months to each (permno, month), 0 if none in the window.

    Missing before the permno's first patent and after `end_mi` (the last month with complete issue dates).
    Returns a Series aligned to keys.index.
    """
    k = kpss.dropna(subset=["permno", "issue_date", "xi_nominal"]).copy()
    k["permno"] = k["permno"].astype("int64")
    k["imi"] = to_mi(k["issue_date"]).values
    m = k.groupby(["permno", "imi"], as_index=False)["xi_nominal"].sum().sort_values(["imi", "permno"])
    m["cum"] = m.groupby("permno")["xi_nominal"].cumsum()
    kk = pd.DataFrame({"permno": keys["permno"].astype("int64").to_numpy(), "mi": keys["mi"].to_numpy()}, index=keys.index)

    def cum_at(shift: int) -> pd.Series:
        q = kk.assign(q=kk["mi"] - shift).sort_values("q")
        out = pd.merge_asof(q, m[["permno", "imi", "cum"]], left_on="q", right_on="imi", by="permno", direction="backward")
        return pd.Series(out["cum"].to_numpy(), index=q.index).reindex(kk.index)

    now, before = cum_at(0), cum_at(36)
    val = (now - before.fillna(0.0)).where(now.notna())
    if end_mi is not None:
        val = val.where(kk["mi"] <= end_mi)
    return val


def build_innovation_signals(keys: pd.DataFrame, annual: pd.DataFrame, links: pd.DataFrame, rq: pd.DataFrame,
                             grants: pd.DataFrame, kpss: pd.DataFrame, kpss_end_mi: int = None) -> pd.DataFrame:
    """Eligible (permno, month) rows with rq, kpss_xi_me, pat_flow_rd, pat_quality3 (keys needs permno, mi, mthcaldt, me_co)."""
    k = keys.copy()
    k["permno"] = k["permno"].astype("int64")
    rdcap = rd_capital(annual)
    flow, qual = patent_tables(grants, rdcap)
    for table, col in ((rq_table(rq), "rq"), (flow, "pat_flow_rd"), (qual, "pat_quality3")):
        a = attach_pit(k[["permno", "mi", "mthcaldt"]], table, links, cols=[col])
        k = k.merge(a[["permno", "mi", col]], on=["permno", "mi"], how="left")
    k["kpss_xi_me"] = (kpss_value_36m(k, kpss, kpss_end_mi) / k["me_co"].where(k["me_co"] > 0)).to_numpy()
    return k
