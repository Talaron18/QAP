"""Conversion diagnostic (pre-registration v4 b): why does rank IC not turn into net long-minus-universe return?

INFORMATION ONLY: every statistic uses months < SEALED_CUT only and nothing here logs a trial.
"""
import numpy as np
import pandas as pd

from qap.candidates import DEV, VAL
from qap.evaluate import _with_next_return, exclude_microcaps
from qap.portfolio import costed_books
from qap.study import SEALED_CUT

SPLITS = {"dev": DEV, "val": VAL}
CLEAR_PP, CLEAR_RATIO = 0.10, 1.5   # reading rule: "clearly larger" = at least 0.10 %/month AND 1.5x
CONCENTRATED, SHARE_HALF = 0.5, 0.5  # smallest-tercile IC share and short-leg / cost shares that count as "most"


def _prep(sig: pd.DataFrame, panel: pd.DataFrame, name: str, cut: int):
    """Slim copies restricted to months < cut with explicit numpy dtypes (parquet nullable dtypes are converted)."""
    cols = [c for c in ("permno", "mi", "me", "me_co", "primaryexch", name) if c in sig.columns]
    s = sig.loc[sig["mi"] < cut, cols].copy()
    s["permno"], s["mi"] = s["permno"].astype("int64"), s["mi"].astype("int64")
    for c in ("me", "me_co", name):
        s[c] = s[c].to_numpy(dtype="float64", na_value=np.nan)
    p = panel.loc[panel["mi"] < cut, ["permno", "mi", "mthret_adj"]].copy()
    p["permno"], p["mi"] = p["permno"].astype("int64"), p["mi"].astype("int64")
    p["mthret_adj"] = p["mthret_adj"].to_numpy(dtype="float64", na_value=np.nan)
    return s, p


def _eligible(s: pd.DataFrame, p: pd.DataFrame, name: str) -> pd.DataFrame:
    """Same eligibility as `portfolio.quintile_legs`, plus next-month return `fwd` and `r0` (missing counts as 0)."""
    e = exclude_microcaps(s).dropna(subset=[name, "me"])
    e = e[e["me"] > 0].copy()
    e = _with_next_return(e, p)
    e["r0"] = e["fwd"].fillna(0.0)
    return e


def _bucket(e: pd.DataFrame, key: str) -> tuple:
    e = e.assign(wr=e["me"] * e["r0"])
    g = e.groupby(["mi", key])[["wr", "me"]].sum()
    vw = (g["wr"] / g["me"]).unstack(key)
    ew = e.groupby(["mi", key])["r0"].mean().unstack(key)
    return vw, ew


def quintile_table(sig: pd.DataFrame, panel: pd.DataFrame, name: str, cut: int = SEALED_CUT) -> pd.DataFrame:
    """Per-month gross returns (index = month f+1) of Q1..Q5, universe U and deciles D1/D10, `_vw` and `_ew` columns."""
    s, p = _prep(sig, panel, name, cut)
    e = _eligible(s, p, name)
    pct = e.groupby("mi")[name].rank(pct=True)
    e["q"] = np.minimum(np.ceil(pct * 5), 5).astype(int)
    e["d"] = np.minimum(np.ceil(pct * 10), 10).astype(int)
    out = {}
    for key, ks, tag in (("q", range(1, 6), "Q"), ("d", (1, 10), "D")):
        vw, ew = _bucket(e, key)
        for k in ks:
            out[f"{tag}{k}_vw"] = vw[k] if k in vw else pd.Series(dtype=float)
            out[f"{tag}{k}_ew"] = ew[k] if k in ew else pd.Series(dtype=float)
    g = e.assign(wr=e["me"] * e["r0"]).groupby("mi")
    out["U_vw"], out["U_ew"] = g["wr"].sum() / g["me"].sum(), g["r0"].mean()
    t = pd.DataFrame(out)
    t = t.set_axis(t.index + 1)
    return t[t.index < cut]


def size_tercile_stats(sig: pd.DataFrame, panel: pd.DataFrame, name: str, cut: int = SEALED_CUT) -> pd.DataFrame:
    """Per month (index f+1) and size tercile T1 (smallest) .. T3 of company ME among eligible names: `ic_T*` and `ew_T*`.

    ic: Spearman of the signal vs next-month return (names with a return); ew: EW Q5-Q1 gross, quintiles formed within the tercile.
    """
    s, p = _prep(sig, panel, name, cut)
    e = _eligible(s, p, name)
    e["t"] = np.minimum(np.ceil(e.groupby("mi")["me_co"].rank(pct=True) * 3), 3).astype(int)
    pct = e.groupby(["mi", "t"])[name].rank(pct=True)
    e["q"] = np.minimum(np.ceil(pct * 5), 5).astype(int)
    qr = e.groupby(["mi", "t", "q"])["r0"].mean().unstack("q").reindex(columns=range(1, 6))
    spread = (qr[5] - qr[1]).unstack("t")
    d = e.dropna(subset=["fwd"])
    keys = [d["mi"], d["t"]]
    rx, rz = d.groupby(["mi", "t"])[name].rank(), d.groupby(["mi", "t"])["fwd"].rank()
    x, z = rx - rx.groupby(keys).transform("mean"), rz - rz.groupby(keys).transform("mean")
    a = pd.DataFrame({"xy": x * z, "xx": x * x, "zz": z * z, "mi": d["mi"], "t": d["t"]})
    sm = a.groupby(["mi", "t"]).agg(xy=("xy", "sum"), xx=("xx", "sum"), zz=("zz", "sum"), n=("xy", "size"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = (sm["xy"] / np.sqrt(sm["xx"] * sm["zz"])).where(sm["n"] >= 3).unstack("t")
    out = pd.concat([ic.rename(columns=lambda k: f"ic_T{k}"), spread.rename(columns=lambda k: f"ew_T{k}")], axis=1)
    for k in (1, 2, 3):
        for c in (f"ic_T{k}", f"ew_T{k}"):
            if c not in out:
                out[c] = np.nan
    out = out[[f"{m}_T{k}" for m in ("ic", "ew") for k in (1, 2, 3)]]
    out = out.set_axis(out.index + 1)
    return out[out.index < cut]


def split_summary(series_dict: dict, splits: dict = None, scale: float = 100.0) -> dict:
    """{series: {split: (mean * scale, t)}} on the dev / validation windows (candidates.DEV / VAL, as `study.split_means`)."""
    out = {}
    for n, s in series_dict.items():
        out[n] = {}
        for label, (y0, y1) in (splits or SPLITS).items():
            v = s[(s.index >= y0 * 12) & (s.index <= y1 * 12 + 11)].dropna()
            with np.errstate(divide="ignore", invalid="ignore"):
                t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 else None
            out[n][label] = (float(v.mean()) * scale if len(v) else None, t)
    return out


def diagnose_signal(sig: pd.DataFrame, panel: pd.DataFrame, inp: pd.DataFrame, aum_t, name: str,
                    mult: float = 1.5, y: float = 1.0, cut: int = SEALED_CUT) -> dict:
    """Quintile/decile returns, spreads, costs and size-tercile stats for one signal, summarised by split (months < cut)."""
    s, p = _prep(sig, panel, name, cut)
    inp = inp.loc[inp["mi"] < cut]
    aum_t = aum_t[aum_t.index < cut] if isinstance(aum_t, pd.Series) else aum_t
    qt = quintile_table(s, p, name, cut)
    ser = {}
    for w in ("vw", "ew"):
        for c in ("Q1", "Q2", "Q3", "Q4", "Q5", "U", "D1", "D10"):
            ser[f"{c}_{w}"] = qt[f"{c}_{w}"]
        ser[f"Q5-Q1_{w}"] = qt[f"Q5_{w}"] - qt[f"Q1_{w}"]
        ser[f"Q5-U_{w}"] = qt[f"Q5_{w}"] - qt[f"U_{w}"]
        ser[f"Q1-U_{w}"] = qt[f"Q1_{w}"] - qt[f"U_{w}"]
        ser[f"D10-D1_{w}"] = qt[f"D10_{w}"] - qt[f"D1_{w}"]
    b = costed_books(s, p, inp, name, aum_t, y, 0.0)
    lu, ls = b["L_minus_U"], b["LS"]
    iu = lu["gross"].index.intersection(lu["cost"].index)
    iu = iu[iu < cut]
    il = ls["gross"].index.intersection(ls["cost"].index)
    il = il[il < cut]
    cl = mult * lu["cost"][iu]
    both = iu.intersection(il)
    cs = mult * (ls["cost"][both] - lu["cost"][both])
    shift = lambda x: x.set_axis(x.index + 1)
    costs = {"gross_LU": lu["gross"][iu], "cost_L": cl, "net_LU": lu["gross"][iu] - cl,
             "gross_LS": ls["gross"][il], "cost_S": cs, "net_LS": ls["gross"][il] - mult * ls["cost"][il],
             "turnover_L": shift(b["turnover_L"]), "turnover_S": shift(b["turnover_S"])}
    costs = {k: v[v.index < cut] for k, v in costs.items()}
    tr = size_tercile_stats(s, p, name, cut)
    return {"name": name, "mult": mult, "returns": split_summary(ser), "costs": split_summary(costs),
            "tercile_ic": split_summary({c: tr[c] for c in tr if c.startswith("ic")}, scale=1.0),
            "tercile_ew": split_summary({c: tr[c] for c in tr if c.startswith("ew")}),
            "months": {k: int(((qt.index >= y0 * 12) & (qt.index <= y1 * 12 + 11)).sum()) for k, (y0, y1) in SPLITS.items()}}


def _m(x) -> float:
    return float("nan") if x is None else float(x)


def _f(x, d=3) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _mt(pair, d=3) -> str:
    return f"{_f(pair[0], d)}({_f(pair[1], 2)})"


def _clear(a: float, b: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and a - b >= CLEAR_PP and a >= CLEAR_RATIO * b)


def reading(r: dict, split: str) -> str:
    """Data-driven labels from the fixed pre-registered reading rule for one split."""
    ret, co, ic = r["returns"], r["costs"], r["tercile_ic"]
    vw, ew = _m(ret["Q5-Q1_vw"][split][0]), _m(ret["Q5-Q1_ew"][split][0])
    ics = [_m(ic[f"ic_T{k}"][split][0]) for k in (1, 2, 3)]
    tot = sum(ics)
    share_ic = ics[0] / tot if np.isfinite(tot) and tot > 0 else float("nan")
    small = bool(np.isfinite(share_ic) and share_ic >= CONCENTRATED and ics[0] > max(ics[1:]))
    q1u, q5u = _m(ret["Q1-U_vw"][split][0]), _m(ret["Q5-U_vw"][split][0])
    short_share = -q1u / vw if np.isfinite(vw) and vw > 0 and np.isfinite(q1u) else float("nan")
    long_share = q5u / vw if np.isfinite(vw) and vw > 0 and np.isfinite(q5u) else float("nan")
    g, c = _m(co["gross_LU"][split][0]), _m(co["cost_L"][split][0])
    cost_share = c / g if np.isfinite(g) and g > 0 and np.isfinite(c) else float("nan")
    flags = []
    if np.isfinite(vw) and vw <= 0:
        flags.append("VW_Q5Q1<=0")
    if np.isfinite(g) and g <= 0:
        flags.append("GROSS_LU<=0")
    if _clear(ew, vw) and small:
        flags.append("SMALLCAP_EW_EFFECT")
    elif _clear(ew, vw):
        flags.append("EW>>VW")
    elif small:
        flags.append("IC_IN_SMALLEST_TERCILE")
    if np.isfinite(short_share) and short_share >= SHARE_HALF:
        flags.append("SHORT_LEG_CARRIES")
    if np.isfinite(cost_share) and cost_share >= SHARE_HALF:
        flags.append("COSTS_EXPLAIN_GAP")
    return (f"reading [{split}]: VW Q5-Q1 {_f(vw)}, EW Q5-Q1 {_f(ew)}; IC share smallest tercile {_f(share_ic)} "
            f"(T1 {_f(ics[0], 4)} T2 {_f(ics[1], 4)} T3 {_f(ics[2], 4)}); VW short-leg share {_f(short_share)} long-leg share "
            f"{_f(long_share)}; cost/gross L-U {_f(cost_share)}; flags: {', '.join(flags) if flags else 'NONE'}")


def format_report(results: list) -> list:
    """Compact text block per signal (means in % per month with t-stats in parentheses; IC in correlation units)."""
    L = []
    for r in results:
        ret, co = r["returns"], r["costs"]
        L.append(f"== {r['name']}  (months dev {r['months']['dev']}, val {r['months']['val']}; costs {r['mult']}x; "
                 f"borrow NOT modelled) ==")
        for sp in SPLITS:
            for w in ("vw", "ew"):
                q = " ".join(f"Q{k} {_mt(ret[f'Q{k}_{w}'][sp])}" for k in range(1, 6))
                sd = " ".join(f"{n} {_mt(ret[f'{n}_{w}'][sp])}" for n in ("Q5-Q1", "Q5-U", "Q1-U", "D10-D1"))
                L.append(f"  {sp} {w.upper()}: {q} | U {_mt(ret[f'U_{w}'][sp])} | {sd}")
            L.append(f"  {sp} costs: gross L-U {_mt(co['gross_LU'][sp])} L-cost {_f(co['cost_L'][sp][0])} S-cost "
                     f"{_f(co['cost_S'][sp][0])} turnover1way L {_f(co['turnover_L'][sp][0])}% S {_f(co['turnover_S'][sp][0])}% | "
                     f"net L-U {_mt(co['net_LU'][sp])} | gross LS {_mt(co['gross_LS'][sp])} net LS {_mt(co['net_LS'][sp])}")
            L.append(f"  {sp} size terciles (T1 smallest): IC " + " ".join(f"T{k} {_mt(r['tercile_ic'][f'ic_T{k}'][sp], 4)}" for k in (1, 2, 3))
                     + " | EW Q5-Q1 " + " ".join(f"T{k} {_mt(r['tercile_ew'][f'ew_T{k}'][sp])}" for k in (1, 2, 3)))
        for sp in SPLITS:
            L.append("  " + reading(r, sp))
        L.append("")
    return L
