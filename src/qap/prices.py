"""Compustat NA daily security prices (data/raw/secd_tickers.parquet) for the personal journal. Pure functions, no I/O except load.

Conventions (pinned):
  * one security per `tic`: a security is (gvkey, iid), labelled by its latest `tic`; two securities with the same label raise
    (disambiguate at extraction with scripts/extract_secd.py --gvkey-iid);
  * adjusted close = prccd / ajexdi (ajexdi missing, non-positive or absent -> 1); total-return factor trfd (missing -> 1);
    TRI_t = (prccd / ajexdi) * trfd ; return_t = TRI_t / TRI_{t-1} - 1 on consecutive trading ROWS of one security;
  * a gap of more than GAP_DAYS calendar days between two rows breaks the chain (that return is NaN) and is counted;
  * duplicates on (security, datadate) are dropped (last kept), rows with a missing or non-positive prccd are dropped; both counted;
  * volatility is annualised with 252 days.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

GAP_DAYS = 5
ANN = 252
LAMBDA = 0.94
SEED_N = 20
MIN_EWMA = 60
MIN_SAMPLE = 20


def prepare_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Clean a secd extract. Returns (frame, diag): frame columns tic, gvkey, iid, datadate, prccd, ajexdi, trfd, adj_close, tri,
    gap, ret (NaN on a security's first row and after a gap); diag[tic] = rows, dups, nonpositive, gaps, first, last."""
    need = ["gvkey", "iid", "datadate", "tic", "prccd"]
    absent = [c for c in need if c not in raw.columns]
    if absent:
        raise ValueError(f"price file lacks required columns {absent}")
    d = raw.copy()
    d["datadate"] = pd.to_datetime(d["datadate"]).dt.normalize()
    d["tic"] = d["tic"].astype(str).str.strip().str.upper()
    d["gvkey"], d["iid"] = d["gvkey"].astype(str), d["iid"].astype(str)
    for c in ("prccd", "ajexdi", "trfd"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    d = d.sort_values(["gvkey", "iid", "datadate"], kind="stable")
    n_dup = d.duplicated(["gvkey", "iid", "datadate"], keep="last")
    dups = d[n_dup].groupby(["gvkey", "iid"]).size()
    d = d[~n_dup]
    bad = ~(d["prccd"] > 0)
    nonpos = d[bad].groupby(["gvkey", "iid"]).size()
    d = d[~bad].copy()
    # label each security by its latest tic; one security per label
    lab = d.groupby(["gvkey", "iid"])["tic"].last()
    d["tic"] = [lab[(g, i)] for g, i in zip(d["gvkey"], d["iid"])]
    owners = lab.reset_index().groupby("tic").size()
    amb = owners[owners > 1].index.tolist()
    if amb:
        cand = {t: [f"{g}:{i}" for (g, i), v in lab.items() if v == t] for t in amb}
        raise ValueError(f"ticker maps to several securities in the price file (re-extract with --gvkey-iid): {cand}")
    # a missing / non-positive adjustment factor is filled from the same security's neighbouring rows (forward, then backward);
    # only a factor that is missing for the WHOLE security becomes 1. Filling with 1 row by row would fake a return jump wherever the
    # cumulative factor is not 1.
    for c in ("ajexdi", "trfd"):
        d[c] = d[c].where(d[c] > 0)
        d[c] = d.groupby(["gvkey", "iid"], sort=False)[c].transform(lambda x: x.ffill().bfill()).fillna(1.0)
    adj = d["ajexdi"]
    d["adj_close"] = d["prccd"] / adj
    d["tri"] = d["adj_close"] * d["trfd"]
    g = d.groupby(["gvkey", "iid"], sort=False)
    d["gap"] = g["datadate"].diff().dt.days > GAP_DAYS
    d["ret"] = g["tri"].pct_change(fill_method=None).where(~d["gap"])
    d = d.reset_index(drop=True)
    diag = {}
    for t, s in d.groupby("tic"):
        key = (s["gvkey"].iloc[0], s["iid"].iloc[0])
        diag[t] = dict(rows=len(s), dups=int(dups.get(key, 0)), nonpositive=int(nonpos.get(key, 0)),
                       gaps=int(s["gap"].sum()), first=s["datadate"].min(), last=s["datadate"].max())
    return d, diag


def load_prices(path: str | Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Read the extract parquet and clean it (see prepare_prices)."""
    return prepare_prices(pd.read_parquet(path))


def last_price(df: pd.DataFrame, ticker: str) -> tuple[float, pd.Timestamp | None]:
    """(latest unadjusted close prccd, its date) of a ticker in a prepared frame; (nan, None) when absent."""
    s = df[df["tic"] == str(ticker).upper()]
    if s.empty:
        return float("nan"), None
    r = s.loc[s["datadate"].idxmax()]
    return float(r["prccd"]), pd.Timestamp(r["datadate"])


def close_on_or_before(df: pd.DataFrame, ticker: str, day, max_back: int = GAP_DAYS) -> tuple[float, pd.Timestamp | None]:
    """Unadjusted close on `day`, else the last row at most `max_back` calendar days earlier; (nan, None) when none."""
    s = df[df["tic"] == str(ticker).upper()]
    day = pd.Timestamp(day).normalize()
    s = s[(s["datadate"] <= day) & (s["datadate"] >= day - pd.Timedelta(days=max_back))]
    if s.empty:
        return float("nan"), None
    r = s.loc[s["datadate"].idxmax()]
    return float(r["prccd"]), pd.Timestamp(r["datadate"])


def return_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily total-return series of a ticker indexed by datadate (chain breaks and first row dropped)."""
    s = df[df["tic"] == str(ticker).upper()]
    return pd.Series(s["ret"].to_numpy(), index=pd.DatetimeIndex(s["datadate"]), name=str(ticker).upper()).dropna()


# ----------------------------------------------------------------------------------------------- volatility
def _clean(returns) -> np.ndarray:
    x = np.asarray(returns, float)
    return x[np.isfinite(x)]


def ewma_vol(returns, lam: float = LAMBDA, seed_n: int = SEED_N, ann: int = ANN) -> float:
    """sqrt(ann * v_T), v_t = lam v_{t-1} + (1 - lam) r_t^2 over ALL returns, v_0 = mean of squares of the first `seed_n` returns."""
    x = _clean(returns)
    if len(x) == 0:
        return float("nan")
    v = float(np.mean(x[:seed_n] ** 2))
    for r in x:
        v = lam * v + (1.0 - lam) * r * r
    return math.sqrt(ann * v)


def sample_vol(returns, n: int | None = None, ann: int = ANN) -> float:
    """Sample (ddof=1) standard deviation x sqrt(ann) of the last `n` returns (all when n is None); NaN with fewer than 2."""
    x = _clean(returns)
    if n is not None:
        x = x[-n:]
    return float(np.std(x, ddof=1) * math.sqrt(ann)) if len(x) >= 2 else float("nan")


def best_vol(returns) -> tuple[float, int, str]:
    """(annual vol, number of observations, label): EWMA(0.94) when >= 60 returns, sample vol when 20..59, else NaN 'insufficient'."""
    x = _clean(returns)
    n = len(x)
    if n >= MIN_EWMA:
        return ewma_vol(x), n, "ewma94"
    if n >= MIN_SAMPLE:
        return sample_vol(x), n, "sample"
    return float("nan"), n, "insufficient"


# ----------------------------------------------------------------------------------------------- leveraged products
def leveraged_check(etf_returns: pd.Series, underlying_returns: pd.Series, L: float) -> dict:
    """Compare a daily-reset product with its underlying on the common dates.

    slope/r2: OLS of etf on underlying (nominal slope = L); te_ann: annualised std of etf - L x underlying;
    drag_realised = L log(S_T/S_0) - log(ETF_T/ETF_0) over the common returns (S_0, ETF_0 = the close before the first common return);
    drag_expected = (L^2 - L) sigma2 T / 2 with sigma2 = annualised sample variance of the underlying's returns on those dates,
    T = n / 252 years. All NaN with fewer than 3 common observations.
    """
    j = pd.concat([etf_returns.rename("e"), underlying_returns.rename("u")], axis=1, join="inner").dropna()
    n = len(j)
    out = dict(n=n, L=float(L), start=None, end=None, slope=float("nan"), intercept=float("nan"), r2=float("nan"),
               te_ann=float("nan"), drag_realised=float("nan"), drag_expected=float("nan"), sigma_ann=float("nan"),
               t_years=n / ANN)
    if n < 3:
        return out
    e, u = j["e"].to_numpy(), j["u"].to_numpy()
    out["start"], out["end"] = j.index[0], j.index[-1]
    vu = float(np.var(u, ddof=1))
    if vu > 0:
        slope = float(np.cov(e, u, ddof=1)[0, 1] / vu)
        out["slope"], out["intercept"] = slope, float(e.mean() - slope * u.mean())
        ve = float(np.var(e, ddof=1))
        out["r2"] = float(np.corrcoef(e, u)[0, 1] ** 2) if ve > 0 else float("nan")
    out["te_ann"] = float(np.std(e - L * u, ddof=1) * math.sqrt(ANN))
    out["drag_realised"] = float(L * np.log1p(u).sum() - np.log1p(e).sum())
    out["sigma_ann"] = math.sqrt(ANN * vu)
    out["drag_expected"] = (L * L - L) * out["sigma_ann"] ** 2 * out["t_years"] / 2.0
    return out
