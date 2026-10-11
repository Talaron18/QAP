"""One-shot sealed-window test of a long-short jkp_slow10 (pre-registration v6): series, statistics, decision rule, gate, marker.

The functions are pure; the caller slices by date (the rehearsal slices to months < SEALED_CUT BEFORE summarising). The 2020+ window is
read only by `scripts/run_holdout_ls.py` without --rehearsal, after the marker file is written.
"""
import pathlib

import numpy as np
import pandas as pd

from qap.basket import assumed_fee
from qap.candidates import DEV, VAL
from qap.portfolio import costed_books, quintile_legs

MARKER = "data/holdout_v6.done"
MIN_MEAN_PCT, MIN_T = 0.20, 1.0            # SUPPORTS FURTHER WORK only if net mean >= 0.20 %/mo AND OLS t >= 1.0
EXPECTED = {"dev": (0.899, 0.545), "val": (0.450, 0.386)}   # jkp_slow10 diagnostic (b), %/mo: (gross L-S, net L-S before borrow)
POWER_NOTE = ("Power (pre-registered): monthly sd of net L-S ~3.9% => SE ~0.46%/mo over 72 months. If the true edge equals the pooled "
              "dev+validation estimate (0.52%/mo) the expected t is ~1.1 and P(t >= 2) is ~20%; the test can refute a large failure, "
              "it cannot confirm a modest edge.")


def _prep(sig, panel, inp, name):
    """Slim copies with explicit numpy dtypes (parquet nullable dtypes are converted)."""
    cols = [c for c in ("permno", "mi", "me", "me_co", "primaryexch", name) if c in sig.columns]
    s = sig[cols].copy()
    s["permno"], s["mi"] = s["permno"].astype("int64"), s["mi"].astype("int64")
    for c in ("me", "me_co", name):
        s[c] = s[c].to_numpy(dtype="float64", na_value=np.nan)
    s["primaryexch"] = s["primaryexch"].astype("string").fillna("").to_numpy(dtype=object)
    p = panel[["permno", "mi", "mthret_adj"]].copy()
    p["permno"], p["mi"] = p["permno"].astype("int64"), p["mi"].astype("int64")
    p["mthret_adj"] = p["mthret_adj"].to_numpy(dtype="float64", na_value=np.nan)
    i = inp[["permno", "mi", "spread", "sigma_d", "adv"]].copy()
    i["permno"], i["mi"] = i["permno"].astype("int64"), i["mi"].astype("int64")
    for c in ("spread", "sigma_d", "adv"):
        i[c] = i[c].to_numpy(dtype="float64", na_value=np.nan)
    return s, p, i


def borrow_series(sig: pd.DataFrame, name: str, scenario: str = "base", rebate_shortfall: float = 1.0) -> pd.Series:
    """Monthly borrow charge (fraction, >= 0) indexed by realised month f+1: sum_i w_i fee_i/100/12 over the short-leg weights formed at f
    (fee_i = basket.assumed_fee(company ME_i, scenario), %/yr) plus rebate_shortfall/100/12 on the short notional (sum of weights)."""
    s = sig[[c for c in ("permno", "mi", "me", "me_co", "primaryexch", name) if c in sig.columns]].copy()
    s["permno"], s["mi"] = s["permno"].astype("int64"), s["mi"].astype("int64")
    for c in ("me", "me_co", name):
        s[c] = s[c].to_numpy(dtype="float64", na_value=np.nan)
    s["primaryexch"] = s["primaryexch"].astype("string").fillna("").to_numpy(dtype=object)
    W = quintile_legs(s, name)["S"].merge(s[["permno", "mi", "me"]], on=["permno", "mi"], how="left")
    W["fee"] = assumed_fee(W["me"].to_numpy(dtype="float64"), scenario)
    W["wf"] = W["w"] * W["fee"]
    g = W.groupby("mi")
    b = (g["wf"].sum() / 100.0 + rebate_shortfall / 100.0 * g["w"].sum()) / 12.0
    return b.set_axis(b.index + 1).rename("borrow")


def ls_series(sig, panel, inp, aum_t, name, mult: float = 1.5, scenario: str = "base", rebate_shortfall: float = 1.0,
              y: float = 1.0) -> pd.DataFrame:
    """Long top / short bottom quintile (VW, capital 1 per leg) by realised month (f+1), ALL months (the caller slices).

    Columns: gross_LS, cost_L, cost_S (x mult), borrow, net_LS_preborrow = gross - cost_L - cost_S, net_LS = preborrow - borrow, plus
    info: long_LU (gross L-U - cost_L), short_UmS (gross U-S - cost_S - borrow), turnover_L / turnover_S (one-way %, by realised month).
    """
    s, p, i = _prep(sig, panel, inp, name)
    b = costed_books(s, p, i, name, aum_t, y, 0.0)
    ls, lu = b["LS"], b["L_minus_U"]
    idx = ls["gross"].index.intersection(ls["cost"].index)
    cL = (mult * lu["cost"]).reindex(idx).fillna(0.0)
    tot = mult * ls["cost"][idx]
    out = pd.DataFrame({"gross_LS": ls["gross"][idx], "cost_L": cL, "cost_S": tot - cL})
    out["borrow"] = borrow_series(s, name, scenario, rebate_shortfall).reindex(idx)
    out["net_LS_preborrow"] = out["gross_LS"] - out["cost_L"] - out["cost_S"]
    out["net_LS"] = out["net_LS_preborrow"] - out["borrow"]
    gLU = lu["gross"].reindex(idx)
    out["long_LU"] = gLU - out["cost_L"]
    out["short_UmS"] = (out["gross_LS"] - gLU) - out["cost_S"] - out["borrow"]
    shift = lambda x: x.set_axis(x.index + 1).reindex(idx)
    out["turnover_L"], out["turnover_S"] = shift(b["turnover_L"]), shift(b["turnover_S"])
    return out


def nw_t(x, lags: int = 6) -> float:
    """t-statistic of the mean with Newey-West (Bartlett) standard error."""
    x = np.asarray(pd.Series(x).dropna(), dtype="float64")
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    S = float(e @ e) / n
    for l in range(1, min(lags, n - 1) + 1):
        S += 2.0 * (1.0 - l / (lags + 1.0)) * float(e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(S / n)) if S > 0 else float("nan")


def _ols(y, X) -> dict:
    X = np.column_stack([np.ones(len(y)), X])
    n, k = X.shape
    if n <= k + 1 or np.linalg.matrix_rank(X) < k:
        return {}
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    e = y - X @ beta
    se = np.sqrt(np.diag(inv) * float(e @ e) / (n - k))
    return {"beta": beta, "t": beta / se}


def summarize(net: pd.Series, ff: pd.DataFrame = None) -> dict:
    """Statistics of a monthly net return series indexed by month index (mean in %/mo; t = OLS and NW6). FF block is information only."""
    x = net.dropna().astype("float64")
    n = len(x)
    out = {"n": int(n), "mean_pct": float("nan"), "sd_pct": float("nan"), "t": float("nan"), "t_nw6": float("nan"), "cum_return": float("nan"),
           "max_drawdown": float("nan"), "by_year": {}, "half1_pct": float("nan"), "half2_pct": float("nan")}
    if n == 0:
        return out
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    out.update(mean_pct=float(x.mean()) * 100, sd_pct=sd * 100, t=float(x.mean() / (sd / np.sqrt(n))) if n > 2 and sd > 0 else float("nan"),
               t_nw6=nw_t(x))
    w = (1.0 + x).cumprod()
    out["cum_return"] = float(w.iloc[-1] - 1.0)
    out["max_drawdown"] = float((w / np.maximum.accumulate(np.r_[1.0, w.to_numpy()])[1:] - 1.0).min())
    out["by_year"] = {int(y): float((1.0 + g).prod() - 1.0) for y, g in x.groupby(x.index // 12)}
    h = n // 2
    out["half1_pct"], out["half2_pct"] = float(x.iloc[:h].mean()) * 100, float(x.iloc[h:].mean()) * 100
    if ff is not None:
        j = pd.concat([x.rename("y"), ff], axis=1, join="inner").dropna()
        for tag, fs in (("capm", ["mktrf"]), ("ff4", ["mktrf", "smb", "hml", "umd"])):
            if all(f in j for f in fs) and len(j) > len(fs) + 2:
                r = _ols(j["y"].to_numpy(dtype="float64"), j[fs].to_numpy(dtype="float64"))
                if r:
                    out[tag] = {"alpha_pct": float(r["beta"][0]) * 100, "alpha_t": float(r["t"][0]), "n": int(len(j)),
                                **{f"b_{f}": float(r["beta"][k + 1]) for k, f in enumerate(fs)},
                                **{f"t_{f}": float(r["t"][k + 1]) for k, f in enumerate(fs)}}
    return out


def decide(mean_pct: float, t: float) -> str:
    """The pre-registered rule: REFUTED if mean <= 0; SUPPORTS FURTHER WORK if mean >= 0.20 %/mo and t >= 1.0; else INCONCLUSIVE."""
    if not np.isfinite(mean_pct) or not np.isfinite(t):
        return "INCONCLUSIVE"
    if mean_pct <= 0:
        return "REFUTED"
    return "SUPPORTS FURTHER WORK" if mean_pct >= MIN_MEAN_PCT and t >= MIN_T else "INCONCLUSIVE"


def split_mean(x: pd.Series, window: tuple) -> float:
    v = x[(x.index >= window[0] * 12) & (x.index <= window[1] * 12 + 11)].dropna()
    return float(v.mean()) * 100 if len(v) else float("nan")


def rehearsal_gate(series: pd.DataFrame, expected: dict = None, tol: float = 0.01) -> tuple:
    """Dev (1990-2014) and validation (2015-2019) means (%/mo) of gross_LS and net_LS_preborrow vs `expected` {'dev': (g, n), 'val': (g, n)}."""
    expected = EXPECTED if expected is None else expected
    ok, msgs = True, []
    for sp, win in (("dev", DEV), ("val", VAL)):
        for k, col in enumerate(("gross_LS", "net_LS_preborrow")):
            got = split_mean(series[col], win) if col in series else float("nan")
            good = bool(np.isfinite(got) and abs(got - expected[sp][k]) <= tol)
            ok &= good
            msgs.append(f"  [{'ok' if good else 'MISMATCH'}] {sp} {col}: {got:.3f} %/mo vs expected {expected[sp][k]:.3f} (tol {tol})")
    return ok, msgs


def assert_unopened(marker: str = None) -> None:
    m = pathlib.Path(MARKER if marker is None else marker)
    if m.exists():
        raise SystemExit(f"The v6 sealed-window test was ALREADY RUN ({m} exists: {m.read_text().strip()!r}). One shot only; "
                         "record the result in ROADMAP.md. Not running again.")


def mark_opened(text: str, marker: str = None) -> None:
    m = pathlib.Path(MARKER if marker is None else marker)
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(text.strip() + "\n")
