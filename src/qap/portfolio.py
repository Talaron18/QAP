"""Costed quintile portfolios: drifted weights, turnover, trading costs, net returns by split."""
import numpy as np
import pandas as pd

from qap.costs import trade_cost
from qap.evaluate import PERIODS, SEALED, _tstat, exclude_microcaps


def quintile_legs(sig: pd.DataFrame, name: str, quantiles: int = 5) -> dict:
    """VW weights (permno, mi, w) for the top quintile 'L', bottom quintile 'S' and the whole universe 'U'.

    Same universe and ranking as `evaluate_signal` (ex-microcaps, signal and ME present).
    """
    s = exclude_microcaps(sig).dropna(subset=[name, "me"])
    s = s[s["me"] > 0].copy()
    pct = s.groupby("mi")[name].rank(pct=True)
    s["q"] = np.minimum(np.ceil(pct * quantiles), quantiles).astype(int)
    out = {}
    for leg, mask in (("L", s["q"] == quantiles), ("S", s["q"] == 1), ("U", s["q"] > 0)):
        t = s.loc[mask, ["permno", "mi", "me"]].copy()
        t["w"] = t["me"] / t.groupby("mi")["me"].transform("sum")
        out[leg] = t[["permno", "mi", "w"]]
    return out


def leg_gross_return(W: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Return earned in month f+1 by weights formed at month f (missing returns count as 0). Indexed by f+1."""
    r = panel[["permno", "mi", "mthret_adj"]].copy()
    r["mi"] = r["mi"] - 1
    m = W.merge(r, on=["permno", "mi"], how="left")
    m["wr"] = m["w"] * m["mthret_adj"].fillna(0.0)
    s = m.groupby("mi")["wr"].sum()
    s.index = s.index + 1
    return s.rename("gross")


def leg_trades(W: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per (permno, month) trade |w - w_pre| where w_pre is last month's weight drifted by this month's return.

    The first formation month is dropped (initial build is not a rebalance) and nothing is traded after the last.
    """
    r = panel[["permno", "mi", "mthret_adj"]]
    prev = W.rename(columns={"w": "w_prev"}).assign(mi=lambda d: d["mi"] + 1)
    prev = prev.merge(r, on=["permno", "mi"], how="left")
    prev["g"] = prev["w_prev"] * (1.0 + prev["mthret_adj"].fillna(0.0))
    prev["w_pre"] = prev["g"] / prev.groupby("mi")["g"].transform("sum")
    m = W.merge(prev[["permno", "mi", "w_pre"]], on=["permno", "mi"], how="outer")
    m[["w", "w_pre"]] = m[["w", "w_pre"]].fillna(0.0)
    m = m[(m["mi"] > W["mi"].min()) & (m["mi"] <= W["mi"].max())]   # no phantom trades after the last formation month
    m["dw"] = (m["w"] - m["w_pre"]).abs()
    return m


def market_scaled_aum(panel: pd.DataFrame, aum: float, ref_year: int = 2019) -> pd.Series:
    """Book size by month that is a CONSTANT SHARE of the equity market.

    `aum` is the book size in `ref_year` market-size dollars: aum_t = aum * total_ME_t / total_ME_ref. A fixed
    nominal book would be enormous relative to 1960s-70s volumes and overstate early-era impact.
    """
    tot = panel.loc[panel["eq"] & (panel["me"] > 0)].groupby("mi")["me"].sum()
    ref = tot[(tot.index >= ref_year * 12) & (tot.index <= ref_year * 12 + 11)].mean()
    return (aum * tot / ref).rename("aum")


def leg_costs(W: pd.DataFrame, panel: pd.DataFrame, inputs: pd.DataFrame, aum,
              y: float = 1.0, commission: float = 0.0) -> pd.DataFrame:
    """Per rebalance month: cost (fraction of leg capital, mult=1), one-way turnover, share of trade dollars above 5%/10% ADV.

    `aum` is a float (nominal, constant) or a Series indexed by month (see `market_scaled_aum`).
    """
    t = leg_trades(W, panel).merge(inputs[["permno", "mi", "spread", "sigma_d", "adv"]], on=["permno", "mi"], how="left")
    t["aum"] = t["mi"].map(aum) if isinstance(aum, pd.Series) else float(aum)
    t["aum"] = t["aum"].fillna(t["aum"].median())
    for col in ("spread", "sigma_d", "adv"):  # exited names have no row: use that month's median
        t[col] = t[col].fillna(t.groupby("mi")[col].transform("median")).fillna(inputs[col].median())
    t["cost"] = trade_cost(t["dw"], t["spread"], t["sigma_d"], t["adv"], t["aum"].to_numpy(), y, commission)
    t["pct_adv"] = t["dw"] * t["aum"] / t["adv"]
    t["d5"] = t["dw"] * (t["pct_adv"] > 0.05)
    t["d10"] = t["dw"] * (t["pct_adv"] > 0.10)
    g = t.groupby("mi")
    out = pd.DataFrame({"cost": g["cost"].sum(), "turnover": g["dw"].sum() / 2.0})
    tot = g["dw"].sum().replace(0, np.nan)
    out["share_gt5pct_adv"] = g["d5"].sum() / tot
    out["share_gt10pct_adv"] = g["d10"].sum() / tot
    return out


def period_stats(x: pd.Series, include_holdout: bool = False) -> dict:
    """Mean (% per month) and t-stat by split; x indexed by month index."""
    res = {}
    for name, (y0, y1) in PERIODS.items():
        if name in SEALED and not include_holdout:
            continue
        v = x[(x.index >= y0 * 12) & (x.index <= y1 * 12 + 11)].dropna()
        res[name] = {"months": int(len(v)), "mean_pct": round(float(v.mean() * 100), 3) if len(v) else None,
                     "t": round(_tstat(v), 2)}
    return res


def costed_books(sig: pd.DataFrame, panel: pd.DataFrame, inputs: pd.DataFrame, name: str, aum,
                 y: float = 1.0, commission: float = 0.0) -> dict:
    """Gross and cost series for the two books.

    'LS':     long top quintile, short bottom quintile (borrow NOT modelled), capital 1 on each leg
    'L_minus_U': long top quintile minus the VW universe; the benchmark is treated as costless
    Cost at the end of month f is charged against the month f+1 return. Costs are returned at mult=1
    (cost is linear in the stress multiplier, so any multiple is cost * mult).
    """
    legs = quintile_legs(sig, name)
    gross = {k: leg_gross_return(legs[k], panel) for k in ("L", "S", "U")}
    cost = {k: leg_costs(legs[k], panel, inputs, aum, y, commission) for k in ("L", "S")}
    shift = lambda s: s.set_axis(s.index + 1)
    cL, cS = shift(cost["L"]["cost"]), shift(cost["S"]["cost"])
    ls_gross = (gross["L"] - gross["S"]).dropna()
    lu_gross = (gross["L"] - gross["U"]).dropna()
    return {
        "LS": {"gross": ls_gross, "cost": (cL + cS).reindex(ls_gross.index).dropna()},
        "L_minus_U": {"gross": lu_gross, "cost": cL.reindex(lu_gross.index).dropna()},
        "turnover_L": cost["L"]["turnover"], "turnover_S": cost["S"]["turnover"],
        "adv5": cost["L"]["share_gt5pct_adv"], "adv10": cost["L"]["share_gt10pct_adv"],
    }
