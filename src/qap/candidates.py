"""Pre-registered profitability / quality / intangible-value candidates (see ROADMAP pre-registration v1)."""
import numpy as np
import pandas as pd

from qap.evaluate import add_composite
from qap.fundamentals import attach_pit

CANDIDATES = ["op_ff", "neg_acc", "fscore", "neg_issuance", "bm_int", "quality3"]
ANNUAL_CAND_COLS = ["be", "gp", "at", "at_lag", "op_ff", "neg_acc", "fscore", "k_int"]
DEV, VAL = (1990, 2014), (2015, 2019)


def _prior_year(a: pd.DataFrame, cols) -> pd.DataFrame:
    """Previous fiscal year's value of each column (same gvkey, 300-430 days earlier), else NaN."""
    g = a.groupby("gvkey")
    gap = (a["datadate"] - g["datadate"].shift(1)).dt.days.between(300, 430)
    return pd.DataFrame({c + "_l": g[c].shift(1).where(gap) for c in cols}, index=a.index)


def add_candidate_features(annual: pd.DataFrame) -> pd.DataFrame:
    """op_ff, neg_acc and the Piotroski fscore on annual rows (a sorted by gvkey, datadate; from `prepare_annual`)."""
    a = annual.copy()
    a = pd.concat([a, _prior_year(a, ["ni", "dltt", "act", "lct", "revt", "cogs"])], axis=1)
    if "at_lag" in a:
        a["at_l"] = a["at_lag"]
    # 1. operating profitability (Fama-French 2015)
    parts = a[["cogs", "xsga", "xint"]]
    op = a["revt"] - a["cogs"].fillna(0) - a["xsga"].fillna(0) - a["xint"].fillna(0)
    a["op_ff"] = (op / a["be"]).where(parts.notna().any(axis=1) & a["be"].notna() & a["revt"].notna())
    # 2. cash-flow accruals, negated (low accruals are good)
    a["neg_acc"] = (-(a["ib"] - a["oancf"]) / a["at"].where(a["at"] > 0)).where(a["oancf"].notna())
    # 3. Piotroski F-score, nine binary components; components with missing inputs are missing
    at, at_l = a["at"].where(a["at"] > 0), a["at_l"].where(a["at_l"] > 0)
    roa, roa_l = a["ni"] / at, a["ni_l"] / at_l
    gm, gm_l = (a["revt"] - a["cogs"]) / a["revt"].where(a["revt"] > 0), (a["revt_l"] - a["cogs_l"]) / a["revt_l"].where(a["revt_l"] > 0)
    flag = lambda cond, *inputs: cond.astype(float).where(pd.concat(inputs, axis=1).notna().all(axis=1))
    comp = pd.DataFrame({
        "roa_pos": flag(roa > 0, roa),
        "cfo_pos": flag(a["oancf"] > 0, a["oancf"]),
        "d_roa": flag(roa > roa_l, roa, roa_l),
        "accrual": flag(a["oancf"] / at > roa, a["oancf"], at, roa),
        "d_lev": flag(a["dltt"] / at < a["dltt_l"] / at_l, a["dltt"], at, a["dltt_l"], at_l),
        "d_liq": flag(a["act"] / a["lct"].where(a["lct"] > 0) > a["act_l"] / a["lct_l"].where(a["lct_l"] > 0),
                      a["act"], a["lct"], a["act_l"], a["lct_l"]),
        "no_issue": (~(a["sstk"].fillna(0) > 0)).astype(float),            # missing or 0 -> no issuance
        "d_margin": flag(gm > gm_l, gm, gm_l),
        "d_turn": flag(a["revt"] / at > a["revt_l"] / at_l, a["revt"], at, a["revt_l"], at_l),
    }, index=a.index)
    a["fscore"] = comp.sum(axis=1).where(comp.notna().sum(axis=1) >= 8)
    return a


def merge_total_q(annual: pd.DataFrame, totalq: pd.DataFrame) -> tuple:
    """Attach k_int by (gvkey, datadate), gvkey zero-padded to 6 on both sides. Returns (annual, match_rate)."""
    t = totalq[["gvkey", "datadate", "k_int"]].copy()
    t["gvkey"] = t["gvkey"].astype(str).str.strip().str.zfill(6)
    t["datadate"] = pd.to_datetime(t["datadate"])
    t = t.drop_duplicates(["gvkey", "datadate"])
    a = annual.drop(columns=[c for c in ["k_int"] if c in annual]).merge(t, on=["gvkey", "datadate"], how="left")
    lo, hi = t["datadate"].min(), t["datadate"].max()
    in_cov = a[(a["datadate"] >= lo) & (a["datadate"] <= hi) & (a["at"] > 0)]
    return a, float(in_cov["k_int"].notna().mean())


def total_q_checks(annual: pd.DataFrame, totalq: pd.DataFrame, match_rate: float) -> tuple:
    """Printable lines and {check: bool} for the Total Q join: coverage by year, match rate, k_int/at scale."""
    t = totalq.copy()
    t["datadate"] = pd.to_datetime(t["datadate"])
    by_year = t.groupby(t["datadate"].dt.year).size()
    lines = [f"Total Q rows by fiscal-year-end year (first/last 3): {by_year.head(3).to_dict()} ... {by_year.tail(3).to_dict()}",
             f"Total Q last datadate {t['datadate'].max().date()}; (gvkey, datadate) match rate onto funda within coverage: {match_rate:.1%}"]
    ratio = (annual["k_int"] / annual["at"].where(annual["at"] > 0)).dropna()
    med = float(ratio.median())
    lines.append(f"median k_int / at = {med:.3f} over {len(ratio):,} matched rows")
    return lines, {"Total Q covers validation years (>= 2019)": bool(t["datadate"].max() >= pd.Timestamp("2019-12-31")),
                   "median k_int/at in [0.05, 2] (units)": bool(0.05 <= med <= 2.0)}


def net_issuance(panel: pd.DataFrame) -> pd.DataFrame:
    """-ln(adj shares_t / adj shares_{t-12}) with adj shares = shrout * mthcumfacshr; needs both months present."""
    s = panel[["permno", "mi"]].copy()
    s["lsh"] = np.log((panel["shrout"] * panel["mthcumfacshr"]).where(panel["shrout"] * panel["mthcumfacshr"] > 0))
    prev = s.rename(columns={"lsh": "lsh_l"}).assign(mi=lambda d: d["mi"] + 12)
    j = s.merge(prev, on=["permno", "mi"], how="left")
    j["neg_issuance"] = -(j["lsh"] - j["lsh_l"])
    return j[["permno", "mi", "neg_issuance"]]


def split_adjustment_check(panel: pd.DataFrame, jump: float = 0.4, max_share: float = 0.005) -> tuple:
    """Share of 1-month |d ln adjusted shares| above `jump`; a broken split adjustment shows jumps near 0.69."""
    s = panel[["permno", "mi"]].copy()
    s["lsh"] = np.log((panel["shrout"] * panel["mthcumfacshr"]).where(panel["shrout"] * panel["mthcumfacshr"] > 0))
    prev = s.rename(columns={"lsh": "lsh_l"}).assign(mi=lambda d: d["mi"] + 1)
    d = s.merge(prev, on=["permno", "mi"])
    dl = (d["lsh"] - d["lsh_l"]).abs().dropna()
    share = float((dl > jump).mean())
    return f"split-adjustment check: share of |monthly d ln adj shares| > {jump} = {share:.3%} over {len(dl):,} pairs", share < max_share


def build_candidate_signals(panel: pd.DataFrame, annual: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Formation-universe rows with gpa, bm_timely and the six candidates (annual data point-in-time, +4 months)."""
    keys = panel.loc[panel["eq"] & (panel["me_co"] > 0), ["permno", "permco", "mi", "mthcaldt", "me", "me_co", "primaryexch"]]
    k = attach_pit(keys, annual, links, cols=ANNUAL_CAND_COLS)
    k["gpa"] = k["gp"] / k["at"].where(k["at"] > 0)
    k["bm_timely"] = k["be"] / k["me_co"]
    k["bm_int"] = (k["be"] + k["k_int"]) / k["me_co"]
    k = k.merge(net_issuance(panel), on=["permno", "mi"], how="left")
    return add_composite(k, names=("gpa", "op_ff", "neg_acc"), min_signals=2, out_col="quality3")


def _ols_alpha(y: np.ndarray, x: np.ndarray, nw_lags: int = 0) -> tuple:
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    u = y - X @ beta
    if nw_lags == 0:
        V = XtXi * (u @ u) / (len(y) - 2)
    else:
        Xu = X * u[:, None]
        S = Xu.T @ Xu
        for l in range(1, nw_lags + 1):
            G = Xu[l:].T @ Xu[:-l]
            S += (1 - l / (nw_lags + 1)) * (G + G.T)
        V = XtXi @ S @ XtXi
    return float(beta[0]), float(beta[1]), float(beta[0] / np.sqrt(V[0, 0]))


def _window(s: pd.Series, y0: int, y1: int) -> pd.Series:
    return s[(s.index >= y0 * 12) & (s.index <= y1 * 12 + 11)]


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def promotion_stats(rc: pd.Series, rg: pd.Series, alpha_t_min: float = 2.0) -> dict:
    """Pre-registered rule. rc, rg: monthly NET long-minus-universe returns of the candidate and of gpa.

    Promote iff the dev alpha t >= alpha_t_min (plain OLS of rc on rg) AND the validation alpha, computed
    with the DEV beta (mean(rc - b_dev * rg)), is > 0. Everything else is information only.
    """
    j = pd.concat([rc.rename("c"), rg.rename("g")], axis=1).dropna()
    dev, val = _window(j, *DEV), _window(j, *VAL)
    if len(dev) < 24 or len(val) < 12:
        return {"dev_months": int(len(dev)), "val_months": int(len(val)), "PROMOTE": False,
                "note": "too few overlapping months (empty quintile or sparse signal)"}
    a, b, t = _ols_alpha(dev["c"].to_numpy(), dev["g"].to_numpy())
    _, _, t_nw = _ols_alpha(dev["c"].to_numpy(), dev["g"].to_numpy(), nw_lags=6)
    a_val = float((val["c"] - b * val["g"]).mean())
    ew_dev, ew_val = (dev["c"] + dev["g"]) / 2, (val["c"] + val["g"]) / 2
    return {
        "dev_months": int(len(dev)), "val_months": int(len(val)),
        "dev_alpha_pct": round(a * 100, 3), "dev_alpha_t": round(t, 2), "dev_alpha_t_nw6": round(t_nw, 2), "beta_on_gpa": round(b, 2),
        "val_alpha_pct": round(a_val * 100, 3),
        "corr_with_gpa_dev": round(float(dev["c"].corr(dev["g"])), 2),
        "standalone_dev_pct": round(float(dev["c"].mean()) * 100, 3), "standalone_dev_t": round(_tstat(dev["c"]), 2),
        "standalone_val_pct": round(float(val["c"].mean()) * 100, 3),
        "ew_gpa_plus_c_dev_pct": round(float(ew_dev.mean()) * 100, 3), "ew_dev_t": round(_tstat(ew_dev), 2),
        "gpa_alone_dev_pct": round(float(dev["g"].mean()) * 100, 3), "gpa_alone_dev_t": round(_tstat(dev["g"]), 2),
        "ew_gpa_plus_c_val_pct": round(float(ew_val.mean()) * 100, 3), "gpa_alone_val_pct": round(float(val["g"].mean()) * 100, 3),
        "PROMOTE": bool(t >= alpha_t_min and a_val > 0),
    }
