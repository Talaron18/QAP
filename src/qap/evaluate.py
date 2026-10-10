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
    k.attrs["link_diag"] = {
        "formation_rows": len(keys),
        "with_gvkey": len(k),
        "with_annual_record": int(k["datadate"].notna().sum()),
        "with_positive_be": int(k["be"].notna().sum()),
        "with_gpa_inputs": int((k["gp"].notna() & (k["at"] > 0)).sum()),
    }
    k["bm_timely"] = k["be"] / k["me_co"]
    k["gpa"] = k["gp"] / k["at"].where(k["at"] > 0)
    k["neg_ag"] = -(k["at"] / k["at_lag"].where(k["at_lag"] > 0) - 1.0)
    k.attrs["link_hit_rate"] = len(k) / len(keys)
    k.attrs["link_diag"]["with_gvkey_pct"] = round(100 * len(k) / len(keys), 1)
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


PERIODS = {  # report section 9.3 split, plus the pre-development years
    "pre1990": (1963, 1989),
    "dev1990_2014": (1990, 2014),
    "valid2015_2019": (2015, 2019),
    "pre2020": (1963, 2019),
    "holdout2020+": (2020, 2025),
    "full": (1963, 2025),
}
SEALED = ("holdout2020+", "full")  # hidden unless the holdout is explicitly opened


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else float("nan")


def _period_summary(ic: pd.Series, spread: pd.Series, include_holdout: bool = False) -> dict:
    out = {}
    for name, (y0, y1) in PERIODS.items():
        if name in SEALED and not include_holdout:
            continue
        lo, hi = y0 * 12, y1 * 12 + 11
        i = ic[(ic.index >= lo) & (ic.index <= hi)].dropna()
        sp = spread[(spread.index >= lo) & (spread.index <= hi)].dropna()
        out[name] = {"months": int(len(i)), "ic": round(float(i.mean()), 4) if len(i) else None, "ic_t": round(_tstat(i), 2),
                     "ls_pct": round(float(sp.mean() * 100), 3) if len(sp) else None, "ls_t": round(_tstat(sp), 2)}
    return out


def evaluate_signal(sig: pd.DataFrame, panel: pd.DataFrame, name: str, quantiles: int = 5,
                    include_holdout: bool = False) -> dict:
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
    if not include_holdout:  # headline numbers use pre-2020 months only
        ic, spread = ic[ic.index < 2020 * 12], spread[spread.index < 2020 * 12]
    return {
        "signal": name, "months": int(len(ic.dropna())),
        "rank_ic_mean": float(ic.mean()), "rank_ic_t": t(ic.dropna()), "icir": float(ic.mean() / ic.std(ddof=1)),
        "ls_vw_mean_pct_month": float(spread.mean() * 100), "ls_vw_t": t(spread),
        "avg_names_per_month": float(s.groupby("mi").size().mean()),
        "by_period": _period_summary(ic, spread, include_holdout),
    }


def add_composite(sig: pd.DataFrame, names=("bm_timely", "gpa", "neg_ag"), min_signals: int = 2) -> pd.DataFrame:
    """Equal-weight composite of within-month rank scores (each in [-0.5, 0.5]); needs >= min_signals present.

    Fixed, untuned weights: one trial, no fitting. Rank scores are robust to outliers.
    """
    out = sig.copy()
    scores = pd.concat([out.groupby("mi")[n].rank(pct=True) - 0.5 for n in names], axis=1)
    out["composite_vpi"] = scores.mean(axis=1).where(scores.notna().sum(axis=1) >= min_signals)
    return out


def add_momentum(sig: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach 12-1 momentum (formation month f uses returns f-11 .. f-1)."""
    from qap.replicate import momentum_signal
    return sig.merge(momentum_signal(panel), on=["permno", "mi"], how="left")
