"""Long-only gpa tilt (pre-registration v2): rank-buffered, capped, value-weighted, benchmarked to the eligible universe."""
import numpy as np
import pandas as pd

from qap.evaluate import PERIODS, SEALED, exclude_microcaps

ENTRY, EXIT, CAP = 0.80, 0.70, 0.03


def cap_weights(me: np.ndarray, cap: float) -> np.ndarray:
    """Value weights with no weight above `cap`; excess is redistributed pro rata (water-filling).

    If the cap is infeasible (fewer than 1/cap names) it falls back to equal weights.
    """
    w = me / me.sum()
    if cap is None:
        return w
    n = len(w)
    if n * cap < 1.0 - 1e-12:
        return np.full(n, 1.0 / n)
    capped = np.zeros(n, dtype=bool)
    for _ in range(n):
        over = (w > cap + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        w = np.where(capped, cap, 0.0) + np.where(capped, 0.0, me / me[~capped].sum() * (1.0 - cap * capped.sum()))
    return w


def eligible(sig: pd.DataFrame, name: str) -> pd.DataFrame:
    """Same eligibility and ranking as the evaluation: ex-microcaps, signal and ME present, within-month percentile."""
    s = exclude_microcaps(sig).dropna(subset=[name, "me"])
    s = s[s["me"] > 0].copy()
    s["pct"] = s.groupby("mi")[name].rank(pct=True)
    return s


def tilt_weights(sig: pd.DataFrame, name: str = "gpa", entry: float = ENTRY, exit_: float = EXIT,
                 cap: float = CAP) -> pd.DataFrame:
    """Long-only weights (permno, mi, w). Enter at pct > entry; an existing holding stays while pct > exit_.

    entry == exit_ disables the buffer; cap=None disables the position cap. Weights are value weights within the held set.
    """
    s = eligible(sig, name)
    out, held = [], np.array([], dtype="int64")
    for mi, g in s.groupby("mi", sort=True):
        # explicit dtypes: WRDS parquet gives pandas nullable Int64/Float64, which np.array(dtype=...) cannot interpret
        pct = g["pct"].to_numpy(dtype="float64")
        perm = g["permno"].to_numpy(dtype="int64")
        me = g["me"].to_numpy(dtype="float64")
        keep = pct > entry
        if exit_ < entry:
            keep |= np.isin(perm, held) & (pct > exit_)
        if not keep.any():
            held = perm[:0]
            continue
        w = cap_weights(me[keep], cap)
        out.append(pd.DataFrame({"permno": perm[keep], "mi": mi, "w": w}))
        held = perm[keep]
    return pd.concat(out, ignore_index=True)


def benchmark_weights(sig: pd.DataFrame, name: str = "gpa") -> pd.DataFrame:
    """Value-weighted eligible universe (permno, mi, w)."""
    s = eligible(sig, name)[["permno", "mi", "me"]]
    s["w"] = s["me"] / s.groupby("mi")["me"].transform("sum")
    return s[["permno", "mi", "w"]]


def holding_stats(W: pd.DataFrame, B: pd.DataFrame, siccd: pd.DataFrame = None) -> pd.DataFrame:
    """Per month: holdings, max weight, top-10 weight, active share vs B, (optional) max |active SIC2 weight|."""
    g = W.groupby("mi")["w"]
    top10 = W.sort_values(["mi", "w"], ascending=[True, False]).groupby("mi").head(10).groupby("mi")["w"].sum()
    j = W.merge(B, on=["permno", "mi"], how="outer", suffixes=("", "_b")).fillna({"w": 0.0, "w_b": 0.0})
    active_share = 0.5 * (j["w"] - j["w_b"]).abs().groupby(j["mi"]).sum()
    out = pd.DataFrame({"holdings": g.size(), "max_w": g.max(), "top10": top10, "active_share": active_share})
    if siccd is not None:
        j = j.merge(siccd, on=["permno", "mi"], how="left")
        j["sic2"] = (j["siccd"] // 100).fillna(-1)
        ind = j.groupby(["mi", "sic2"])[["w", "w_b"]].sum()
        ind["act"] = ind["w"] - ind["w_b"]
        out["max_abs_active_sic2"] = ind["act"].abs().groupby("mi").max()
    return out


def active_stats(active: pd.Series, include_holdout: bool = False) -> dict:
    """Mean (% per month), t, tracking error and information ratio (annualised) by split."""
    res = {}
    for name, (y0, y1) in PERIODS.items():
        if name in SEALED and not include_holdout:
            continue
        v = active[(active.index >= y0 * 12) & (active.index <= y1 * 12 + 11)].dropna()
        if len(v) < 3:
            res[name] = {"months": int(len(v))}
            continue
        te = float(v.std(ddof=1) * np.sqrt(12))
        res[name] = {"months": int(len(v)), "active_pct_mo": round(float(v.mean()) * 100, 3),
                     "t": round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2),
                     "TE_ann_pct": round(te * 100, 2), "IR_ann": round(float(v.mean() * 12 / te), 2)}
    return res


def factor_regression(active: pd.Series, ff: pd.DataFrame, cols=("mktrf", "smb", "hml", "umd")) -> dict:
    """OLS of monthly net active return on factor returns: alpha (%/mo), t, loadings and their t-stats."""
    j = pd.concat([active.rename("y"), ff[list(cols)]], axis=1).dropna()
    if len(j) < len(cols) + 10:
        return {"months": int(len(j))}
    X = np.column_stack([np.ones(len(j)), j[list(cols)].to_numpy()])
    y = j["y"].to_numpy()
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    se = np.sqrt(np.diag(XtXi) * (u @ u) / (len(y) - X.shape[1]))
    t = b / se
    out = {"months": int(len(j)), "alpha_pct_mo": round(float(b[0]) * 100, 3), "alpha_t": round(float(t[0]), 2)}
    for i, c in enumerate(cols, start=1):
        out[c] = f"{b[i]:+.2f} (t {t[i]:+.1f})"
    return out
