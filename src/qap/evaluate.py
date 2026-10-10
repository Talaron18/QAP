"""Research signals on point-in-time fundamentals and their evaluation (rank IC, quintile spreads)."""
import numpy as np
import pandas as pd

from qap.fundamentals import attach_pit

SIGNALS = ["bm_timely", "gpa", "neg_ag"]


def build_research_signals(panel: pd.DataFrame, annual: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Signals at each month-end on the formation universe (eq, positive company ME).

    bm_timely: BE (latest PIT annual) / current company ME  (Asness-Frazzini timely price)
    gpa:       gross profits / total assets (Novy-Marx)
    neg_ag:    minus asset growth (low investment is good)
    """
    keys = panel.loc[panel["eq"] & (panel["me_co"] > 0), ["permno", "permco", "mi", "mthcaldt", "me", "me_co", "primaryexch"]]
    k = attach_pit(keys, annual, links)
    k["bm_timely"] = k["be"] / k["me_co"]
    k["gpa"] = k["gp"] / k["at"].where(k["at"] > 0)
    k["neg_ag"] = -(k["at"] / k["at_lag"].where(k["at_lag"] > 0) - 1.0)
    k.attrs["link_hit_rate"] = len(k) / len(keys)
    return k


def _with_next_return(sig: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Label = NEXT month's delisting-adjusted return from the UNFILTERED panel."""
    lab = panel[["permno", "mi", "mthret_adj"]].copy()
    lab["mi"] = lab["mi"] - 1
    return sig.merge(lab.rename(columns={"mthret_adj": "fwd"}), on=["permno", "mi"], how="left")


def exclude_microcaps(sig: pd.DataFrame, pct: float = 0.2) -> pd.DataFrame:
    """Drop stocks below the NYSE `pct` quantile of company ME in each month."""
    cut = sig[sig["primaryexch"] == "N"].groupby("mi")["me_co"].quantile(pct).rename("cut")
    s = sig.merge(cut, on="mi", how="left")
    return s[s["me_co"] >= s["cut"]].drop(columns="cut")


def evaluate_signal(sig: pd.DataFrame, panel: pd.DataFrame, name: str, quantiles: int = 5) -> dict:
    s = exclude_microcaps(_with_next_return(sig, panel))
    s = s.dropna(subset=[name, "fwd", "me"])
    s = s[s["me"] > 0]
    g = s.groupby("mi")
    s["r_sig"] = g[name].rank()
    s["r_ret"] = g["fwd"].rank()
    ic = s.groupby("mi").apply(lambda d: d["r_sig"].corr(d["r_ret"]), include_groups=False)
    pct = g[name].rank(pct=True)
    s["q"] = np.minimum(np.ceil(pct * quantiles), quantiles).astype(int)
    s["wr"] = s["fwd"] * s["me"]
    qq = s.groupby(["mi", "q"])[["wr", "me"]].sum()
    qr = (qq["wr"] / qq["me"]).unstack("q")
    spread = (qr[quantiles] - qr[1]).dropna()
    t = lambda x: float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
    return {
        "signal": name, "months": int(len(ic.dropna())),
        "rank_ic_mean": float(ic.mean()), "rank_ic_t": t(ic.dropna()), "icir": float(ic.mean() / ic.std(ddof=1)),
        "ls_vw_mean_pct_month": float(spread.mean() * 100), "ls_vw_t": t(spread),
        "avg_names_per_month": float(s.groupby("mi").size().mean()),
    }
