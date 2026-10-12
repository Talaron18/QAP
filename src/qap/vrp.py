"""Study G (pre-registration v14, Stage B at the INDEX level): does the model forecast carry information beyond VIX, and is there a
variance-risk-premium signal? Pure functions (numpy / pandas); the engine, calendar, windows and forecasts are Study F's (`qap.volfc`).

Units (one convention for everything; window variance over 21 trading days)
---------------------------------------------------------------------------
IV2_win(t) = (VIX_t / 100)^2 * 30 / 365        RV_win(t) = sum r^2 over t+1..t+21 = 21 * y21        F_win(t, m) = 21 * f21_m(t)
Primary sample = forecast dates whose CALENDAR INDEX is a multiple of 21 (the refit dates), complete RV window, non-null VIX, finite positive
forecasts of all six models (one common mask, `volfc.valid_masks`). Regressions: OLS with Bartlett Newey-West covariance, 3 lags
(30 lags for the information-only daily-overlap versions), no small-sample correction (the same estimator as `volfc.newey_west_t`).
"""
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd

from qap import volfc as V

# ------------------------------------------------------------------------------------------------ pre-registered constants
WIN = 21                          # trading days in the RV window (= volfc.REFIT)
VIX_DAYS, CAL_DAYS = 30, 365      # VIX's own convention
NW_PRIMARY, NW_DAILY = 3, 30
C_MIN, T_MIN = 0.10, 2.0          # T1: dev c >= 0.10 AND t(c) >= 2.0 AND validation c > 0
T3_T_MIN = 2.0                    # T3: dev slope t >= 2.0 AND validation slope > 0
HOLD = ("2020-01-01", "2026-12-31")          # "holdout 2020-01..2026" as written (volfc.HOLD ends 2025-12-31; the data end earlier)
SEAL = "2020-01-01"
PRE_END = "2019-12-31"
EX_CRASH = ("2008-09-01", "2008-12-31")      # T3 information-only rerun excludes windows touching these dates (development)
MARKER_V13 = "data/holdout_v13.done"
MARKER_V14 = "data/holdout_v14.done"
TRIAL_NAME = "studyG:vrp_payoff"
PREREG = "v14"
PRIMARY_MODEL, SECOND_MODEL = "gjr_t", "garch"

# v13 market-table mean QLIKE, (h, model) -> (dev, validation). The v14 pipeline must reproduce them within REPRO_TOL or the run aborts.
REPRO_EXPECTED = {(1, "gjr_t"): (-8.2590, -8.8061), (1, "garch"): (-8.2144, -8.7425),
                  (21, "gjr_t"): (-8.1053, -8.5679), (21, "garch"): (-8.0991, -8.5666)}
REPRO_TOL = 5e-4


# --------------------------------------------------------------------------------------------------------- unit conversions
def iv2_win(vix):
    """Implied window variance (VIX_t / 100)^2 * 30 / 365."""
    return (np.asarray(vix, dtype="float64") / 100.0) ** 2 * VIX_DAYS / CAL_DAYS


def rv_win(y21):
    """Realised window variance = 21 x the v13 proxy y21 (= sum of r^2 over t+1..t+21)."""
    return WIN * np.asarray(y21, dtype="float64")


def f_win(f21):
    """Window variance forecast = 21 x the model's h = 21 average-variance forecast."""
    return WIN * np.asarray(f21, dtype="float64")


def vol_points(window_var):
    """Annualised volatility points of a window variance under VIX's own convention: 100 sqrt(var x 365 / 30) (VIX itself maps back to VIX)."""
    return 100.0 * np.sqrt(np.asarray(window_var, dtype="float64") * CAL_DAYS / VIX_DAYS)


# ------------------------------------------------------------------------------------------------------------------ loaders
def load_calendar(directory) -> tuple:
    """(sorted calendar of ALL daily dates as datetime64[D], {year: rows}) reading ONLY the dlycaldt column of each yearly file. Identical to the
    calendar of `volfc.load_daily_subset`."""
    import glob
    days, rows = [], {}
    for f in sorted(glob.glob(str(pathlib.Path(directory) / "[0-9]*.parquet"))):
        d = pd.to_datetime(pd.read_parquet(f, columns=["dlycaldt"])["dlycaldt"])
        rows[int(pathlib.Path(f).stem)] = len(d)
        days.append(d.drop_duplicates().to_numpy().astype("datetime64[D]"))
    cal = np.unique(np.concatenate(days)) if days else np.empty(0, "datetime64[D]")
    return cal, rows


def prepare_vix(df, cal, before=None) -> dict:
    """VIX on the calendar. `df` has date, vix (extra columns ignored). Rows dated on/after `before` (ISO date) are dropped FIRST (the seal), so
    nothing later is counted or reported. Null / non-finite / non-positive VIX and dates absent from the calendar are dropped and counted.
    Returns {'vix': float array on cal (NaN = none), 'stats': dict}."""
    d = pd.DataFrame({"date": pd.to_datetime(df["date"], errors="coerce"), "vix": pd.to_numeric(df["vix"], errors="coerce")})
    st = {"rows_read": int(len(df))}
    st["bad_date"] = int(d["date"].isna().sum())
    d = d.dropna(subset=["date"])
    if before is not None:
        d = d[d["date"] < pd.Timestamp(before)]
    st["rows_in_scope"] = int(len(d))
    st["duplicates"] = int(d["date"].duplicated(keep="last").sum())
    d = d.drop_duplicates("date", keep="last").sort_values("date")
    bad = ~np.isfinite(d["vix"].to_numpy()) | (d["vix"].to_numpy() <= 0)
    st["null_dropped"] = int(bad.sum())
    d = d[~bad]
    dd = d["date"].to_numpy().astype("datetime64[D]")
    pos = np.searchsorted(cal, dd)
    pc = np.minimum(pos, max(len(cal) - 1, 0))
    hit = (pos < len(cal)) & (cal[pc] == dd) if len(cal) else np.zeros(len(dd), bool)
    st["not_in_calendar"] = int((~hit).sum())
    out = np.full(len(cal), np.nan)
    out[pos[hit]] = d["vix"].to_numpy("float64")[hit]
    st["matched"] = int(hit.sum())
    st["first"] = str(dd[0]) if len(dd) else "n/a"
    st["last"] = str(dd[-1]) if len(dd) else "n/a"
    st["median_vix"] = float(np.median(d["vix"])) if len(d) else float("nan")
    return {"vix": out, "stats": st}


def rows_frame(rows: dict, cal) -> pd.DataFrame:
    """DataFrame (with `date`) from the arrays returned by `volfc.run_asset(...)['rows']`."""
    df = pd.DataFrame({c: np.asarray(v) for c, v in rows.items()})
    df["date"] = cal[df["tidx"].to_numpy("int64")].astype("datetime64[ns]") if len(df) else pd.to_datetime(pd.Series([], dtype="float64"))
    return df


def market_rows(ret, cal, last_year: int, mc: bool = False) -> pd.DataFrame:
    """The v13 market-series forecasts (`volfc.run_asset(..., mc=False)`, FIRST_FC_YEAR..last_year) as a DataFrame."""
    ranges = V.year_ranges(cal, range(V.FIRST_FC_YEAR, last_year + 1))
    return rows_frame(V.run_asset(ret, ranges, V.MKT_ID, mc=mc)["rows"], cal)


# ------------------------------------------------------------------------------------------------------ reproduction check
def repro_values(dm, cal) -> dict:
    """{(h, model): (dev mean QLIKE, validation mean QLIKE)} computed exactly as run_volfc builds its market table
    (valid_masks over all six models -> date_losses -> window_slice mean)."""
    masks = V.valid_masks(dm)
    out = {}
    for h in V.HORIZONS:
        dl = V.date_losses(dm, masks[h], h, "qlike")
        for m in V.MODELS:
            out[(h, m)] = (float(V.window_slice(dl, cal, V.DEV)[m].mean()), float(V.window_slice(dl, cal, V.VAL)[m].mean()))
    return out


def repro_check(dm, cal, expected=None, tol=None) -> tuple:
    """(ok, lines): the v14 market pipeline versus the v13 market-table constants (dev and validation, h = 1 and 21, gjr_t and garch)."""
    expected = REPRO_EXPECTED if expected is None else expected
    tol = REPRO_TOL if tol is None else tol
    got = repro_values(dm, cal)
    ok_all, lines = True, []
    for (h, m), want in expected.items():
        have = got[(h, m)]
        diffs = [have[i] - want[i] for i in (0, 1)]
        ok = all(np.isfinite(x) and abs(x) <= tol for x in diffs)
        ok_all &= ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] reproduction h={h:<2d} {m:6s}: dev {have[0]:.4f} (v13 {want[0]:.4f}, diff {diffs[0]:+.5f})  "
                     f"val {have[1]:.4f} (v13 {want[1]:.4f}, diff {diffs[1]:+.5f})  tolerance {tol:g}")
    return ok_all, lines


# ---------------------------------------------------------------------------------------------------------------- samples
def select_samples(df, cal, vix_cal) -> dict:
    """Primary and daily samples from the market-series forecast rows `df` (needs tidx, y21 and every f21_<model>, f1_<model>, y1).

    common mask = volfc.valid_masks(df)[21] (complete 21-day window, finite positive forecasts for ALL six models) & a VIX on exactly that
    date & VIX > 0 & RV_win > 0 (the logs need it). Primary = the common-mask rows whose calendar index is a multiple of REFIT.
    Returns {'daily', 'primary': DataFrames [tidx, date, vix, iv2, rv, F_<model>...], 'counts': dict}."""
    tidx = df["tidx"].to_numpy("int64")
    vix = vix_cal[tidx] if len(tidx) else np.empty(0)
    masks = V.valid_masks(df)
    m21 = masks[21]
    has_vix = np.isfinite(vix) & (vix > 0)
    rv = rv_win(df["y21"].to_numpy())
    pos_rv = np.isfinite(rv) & (rv > 0)
    base = m21 & has_vix & pos_rv
    prim = base & (tidx % V.REFIT == 0)
    counts = {"rows": int(len(df)), "complete_window": int(np.isfinite(df["y21"].to_numpy()).sum()), "valid_six_models": int(m21.sum()),
              "with_vix": int((m21 & has_vix).sum()), "daily": int(base.sum()), "primary": int(prim.sum()),
              "no_vix": int((np.isfinite(df["y21"].to_numpy()) & ~has_vix).sum())}

    def frame(mask):
        d = pd.DataFrame({"tidx": tidx[mask], "vix": vix[mask]})
        d["date"] = cal[d["tidx"].to_numpy("int64")].astype("datetime64[ns]")
        d["iv2"] = iv2_win(d["vix"].to_numpy())
        d["rv"] = rv[mask]
        for m in V.MODELS:
            d[f"F_{m}"] = f_win(df[f"f21_{m}"].to_numpy()[mask])
        return d.reset_index(drop=True)
    return {"daily": frame(base), "primary": frame(prim), "counts": counts}


def in_window(s, window) -> pd.DataFrame:
    """Rows of a sample whose forecast date lies in [window[0], window[1]] (inclusive ISO dates)."""
    d = s["date"].to_numpy()
    return s[(d >= np.datetime64(window[0])) & (d <= np.datetime64(window[1]))].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------- HAC regression
def ols_hac(y, X, lags: int) -> dict:
    """OLS with a Bartlett-kernel Newey-West (HAC) covariance: V = (X'X)^-1 S (X'X)^-1, S = G0 + sum_{l=1..L} (1 - l/(L+1)) (G_l + G_l'),
    G_l = sum_t x_t e_t e_{t-l} x_{t-l}'; no degrees-of-freedom correction (an intercept-only fit equals `volfc.newey_west_t`). lags = 0 gives the
    White HC0 covariance. Rows with a non-finite value are dropped. Returns coef, se, t, n, r2 (NaN arrays when n <= number of columns)."""
    y = np.asarray(y, dtype="float64")
    X = np.asarray(X, dtype="float64")
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    nan = np.full(k, np.nan)
    if n <= k:
        return {"coef": nan, "se": nan.copy(), "t": nan.copy(), "n": int(n), "r2": float("nan")}
    xtx = X.T @ X
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return {"coef": nan, "se": nan.copy(), "t": nan.copy(), "n": int(n), "r2": float("nan")}
    beta = inv @ (X.T @ y)
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, min(int(lags), n - 1) + 1):
        G = xe[l:].T @ xe[:-l]
        S = S + (1.0 - l / (lags + 1.0)) * (G + G.T)
    cov = inv @ S @ inv
    var = np.diag(cov)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.where(var > 0, np.sqrt(np.where(var > 0, var, 1.0)), np.nan)
        t = beta / se
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(e @ e) / tss if tss > 0 else float("nan")
    return {"coef": beta, "se": se, "t": t, "n": int(n), "r2": r2}


def _fit1(y, x, lags):
    """Regression of y on [1, x]: (intercept, slope, t(slope), n, r2)."""
    r = ols_hac(y, np.column_stack([np.ones(len(np.asarray(y))), x]), lags)
    return {"a": float(r["coef"][0]), "slope": float(r["coef"][1]), "t": float(r["t"][1]), "n": r["n"], "r2": r["r2"]}


# --------------------------------------------------------------------------------------------------------------------- T1
def t1_fit(s, model: str, lags: int = NW_PRIMARY) -> dict:
    """log RV_win = a + b log IV2_win + c log F_win(model) + e on the rows of `s`. Also the single-regressor fits (R2s) and the c / b of each."""
    lrv, liv, lf = np.log(s["rv"].to_numpy()), np.log(s["iv2"].to_numpy()), np.log(s[f"F_{model}"].to_numpy())
    r = ols_hac(lrv, np.column_stack([np.ones(len(lrv)), liv, lf]), lags)
    iv_only, f_only = _fit1(lrv, liv, lags), _fit1(lrv, lf, lags)
    return {"a": float(r["coef"][0]), "b": float(r["coef"][1]), "c": float(r["coef"][2]), "t_b": float(r["t"][1]), "t_c": float(r["t"][2]),
            "n": r["n"], "r2": r["r2"], "r2_iv": iv_only["r2"], "r2_f": f_only["r2"], "slope_iv": iv_only["slope"], "slope_f": f_only["slope"]}


def t1_rule(dev: dict, val: dict) -> dict:
    """'THE FORECAST ADDS INFORMATION BEYOND VIX': dev c >= 0.10 AND t(c) >= 2.0 AND validation c > 0 (NaN fails)."""
    dev_ok = bool(dev["c"] >= C_MIN and dev["t_c"] >= T_MIN)
    val_ok = bool(val["c"] > 0)
    return {"dev_ok": dev_ok, "val_ok": val_ok, "passed": dev_ok and val_ok}


# --------------------------------------------------------------------------------------------------------------------- T3
def payoff(s):
    """Y = (RV_win - IV2_win) / IV2_win: return per unit strike of a long 30-day variance swap struck at VIX^2 (before any cost)."""
    return (s["rv"].to_numpy() - s["iv2"].to_numpy()) / s["iv2"].to_numpy()


def signal_d(s, model: str = PRIMARY_MODEL, level: bool = False):
    """D = log F_win(model) - log IV2_win (primary) or the level version F_win - IV2_win."""
    f, iv = s[f"F_{model}"].to_numpy(), s["iv2"].to_numpy()
    return (f - iv) if level else (np.log(f) - np.log(iv))


def t3_fit(s, lags: int = NW_PRIMARY, model: str = PRIMARY_MODEL, level: bool = False) -> dict:
    """Y on [1, D] (Newey-West): slope, t, n, intercept, r2."""
    return _fit1(payoff(s), signal_d(s, model, level), lags)


def t3_rule(dev: dict, val: dict) -> dict:
    """'A VARIANCE-RISK-PREMIUM SIGNAL EXISTS': development slope t >= 2.0 AND validation slope > 0 (NaN fails)."""
    dev_ok = bool(dev["t"] >= T3_T_MIN)
    val_ok = bool(val["slope"] > 0)
    return {"dev_ok": dev_ok, "val_ok": val_ok, "passed": dev_ok and val_ok}


def tercile_breaks(d_dev) -> np.ndarray:
    """Breakpoints at the 1/3 and 2/3 quantiles (linear interpolation) of D on DEVELOPMENT; frozen for every other window."""
    d = np.asarray(d_dev, dtype="float64")
    d = d[np.isfinite(d)]
    return np.quantile(d, [1.0 / 3.0, 2.0 / 3.0]) if len(d) else np.full(2, np.nan)


def tercile_table(s, breaks, lags: int = NW_PRIMARY, model: str = PRIMARY_MODEL) -> list:
    """Mean Y (and Newey-West t of the mean, n) by tercile of D with FROZEN breakpoints: tercile 0 = D <= b1, 1 = b1 < D <= b2, 2 = D > b2."""
    d, y = signal_d(s, model), payoff(s)
    out = []
    if len(d) == 0 or not np.all(np.isfinite(breaks)):
        return [{"tercile": k, "n": 0, "mean": float("nan"), "t": float("nan")} for k in range(3)]
    which = np.searchsorted(np.asarray(breaks), d, side="left")
    for k in range(3):
        v = y[which == k]
        m, t = V.newey_west_t(v, lags)
        out.append({"tercile": k, "n": int(len(v)), "mean": m, "t": t})
    return out


def touches(s, cal, lo: str, hi: str) -> np.ndarray:
    """True where the 21-day RV window (calendar days t+1..t+21) overlaps [lo, hi]."""
    t = s["tidx"].to_numpy("int64")
    first, last = cal[np.minimum(t + 1, len(cal) - 1)], cal[np.minimum(t + WIN, len(cal) - 1)]
    return (last >= np.datetime64(lo)) & (first <= np.datetime64(hi))


# ------------------------------------------------------------------------------------------------- T2 / T4 / scaled-IV QLIKE
def t2_stats(s, lags: int = NW_PRIMARY) -> dict:
    """Unconditional premium: mean IV2_win - RV_win (and in volatility points VIX - realised, same convention), share positive, NW t."""
    d = s["iv2"].to_numpy() - s["rv"].to_numpy()
    m, t = V.newey_west_t(d, lags)
    dv = vol_points(s["iv2"].to_numpy()) - vol_points(s["rv"].to_numpy())
    mv, tv = V.newey_west_t(dv, lags)
    return {"n": int(len(d)), "mean_var": m, "t_var": t, "mean_vol_pts": mv, "t_vol_pts": tv, "share_pos": float(np.mean(d > 0)) if len(d) else float("nan"),
            "mean_iv2": float(np.mean(s["iv2"])) if len(d) else float("nan"), "mean_rv": float(np.mean(s["rv"])) if len(d) else float("nan"),
            "mean_vix": float(np.mean(s["vix"])) if len(d) else float("nan"), "mean_rv_vol": float(np.mean(vol_points(s["rv"].to_numpy()))) if len(d) else float("nan")}


def t4_fit(s, r21, regressor: str, lags: int = NW_PRIMARY, model: str = PRIMARY_MODEL) -> dict:
    """Next-21-day FF excess return (`r21`, aligned with the rows of `s`) on D (regressor 'D') or on IV2_win - F_win (regressor 'iv_minus_f')."""
    x = signal_d(s, model) if regressor == "D" else s["iv2"].to_numpy() - s[f"F_{model}"].to_numpy()
    return _fit1(np.asarray(r21, dtype="float64"), x, lags)


def scaled_iv_k(s_dev) -> float:
    """k = mean RV_win / mean IV2_win on DEVELOPMENT only (frozen)."""
    return float(np.mean(s_dev["rv"]) / np.mean(s_dev["iv2"])) if len(s_dev) else float("nan")


def qlike_vs_scaled_iv(s, k: float, models=(PRIMARY_MODEL, SECOND_MODEL), lags: int = NW_PRIMARY) -> dict:
    """{model: (mean QLIKE difference model - k x IV2_win, DM t with NW `lags`, n)} in window-variance units (QLIKE = ln F + RV / F; the common
    ln 21 and the unit scale cancel in the difference). Negative = the model beats scaled implied variance."""
    rv = s["rv"].to_numpy()
    l_iv = V.qlike(k * s["iv2"].to_numpy(), rv)
    out = {}
    for m in models:
        d = V.qlike(s[f"F_{m}"].to_numpy(), rv) - l_iv
        mean, t = V.newey_west_t(d, lags)
        out[m] = (mean, t, int(np.isfinite(d).sum()))
    out["scaled_iv_mean"] = float(np.mean(l_iv)) if len(l_iv) else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------------- decision
def decision(t1: dict, t3: dict, open_holdout: bool = False, hold: dict | None = None) -> dict:
    """Final decision. t1 / t3 = the `t1_rule` / `t3_rule` outputs (M3). The case for buying stock-level option data is made ONLY if BOTH pass in
    development AND validation. The holdout (dict {'c': T1 holdout c, 'slope': T3 holdout slope}) is consulted ONLY when `open_holdout` is True
    and both passed; the case STANDS only if c > 0 and slope > 0 there. Returns {'dev_val', 'opened', 'stands', 'status'}."""
    dv = bool(t1["passed"] and t3["passed"])
    if not dv:
        return dict(dev_val=False, opened=False, stands=None,
                    status="NO CASE FOR BUYING STOCK-LEVEL DATA ON THIS EVIDENCE: the index-level implied variance encompasses the model forecast "
                           "(the stock-level question stays open but unfunded)")
    if not open_holdout:
        return dict(dev_val=True, opened=False, stands=None,
                    status="PASSED IN DEV/VALIDATION (T1 and T3 both); holdout sealed (run with --open-holdout, after the v13 holdout, to open it once)")
    if hold is None:
        raise ValueError("open_holdout without holdout statistics")
    stands = bool(hold["c"] > 0 and hold["slope"] > 0)
    return dict(dev_val=True, opened=True, stands=stands,
                status="CASE FOR BUYING STOCK-LEVEL DATA STANDS (dev/validation and holdout)" if stands
                else "passed dev/validation, failed in the holdout")


# -------------------------------------------------------------------------------------------------------------- trial / markers
def _spec_key(name, spec) -> tuple:
    return name, json.dumps(spec, sort_keys=True, default=str)


def log_trial_once(name: str, spec: dict, metrics: dict, path: str = "data/trials.jsonl") -> bool:
    """Append the trial unless a record with the same (name, spec) is already in the file (the registry counts unique pairs; this also keeps the
    file free of duplicates). Returns True when a line was written."""
    from qap.trials import log_trial
    p = pathlib.Path(path)
    if p.exists():
        want = _spec_key(name, spec)
        for line in p.open():
            if line.strip():
                r = json.loads(line)
                if _spec_key(r["name"], r["spec"]) == want:
                    return False
    log_trial(name, spec, metrics, path)
    return True


def trial_spec() -> dict:
    """Deterministic T3 spec (no timestamps, no results)."""
    return {"prereg": PREREG, "test": "T3", "Y": "(RV_win - IV2_win) / IV2_win", "D": "log F_win(gjr_t) - log IV2_win",
            "IV2_win": "(VIX/100)^2 * 30/365", "RV_win": "sum r^2 over t+1..t+21", "F_win": "21 x f21 (h=21 average-variance forecast)",
            "series": "FF market mktrf+rf, Study F engine", "sample": "forecast dates with calendar index % 21 == 0", "regression": "OLS Newey-West 3",
            "rule": "dev slope t >= 2.0 and validation slope > 0", "windows": {"dev": V.DEV, "val": V.VAL}}


def require_v13_marker(marker: str = MARKER_V13) -> None:
    """Gate for `run_vrp.py --open-holdout`: the v13 holdout must already have been opened (its one-shot marker exists)."""
    if not pathlib.Path(marker).exists():
        raise V.HoldoutRefused(f"--open-holdout REFUSED: the v13 (Study F) holdout has not been opened ({marker} is absent). The 2020+ window scores "
                               "the same market gjr_t forecasts, so run `python scripts/run_volfc.py --open-holdout` first.")


def crash_note() -> str:
    return f"windows touching {EX_CRASH[0][:7]}..{EX_CRASH[1][:7]}"


def fmt_num(x, d=3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


# ----------------------------------------------------------------------------------------------------------------- checks
def vrp_checks(vstats: dict, ncal_pre: int, cal_overlap: tuple, counts: dict, sample_counts: dict, sanity: dict, ff_cov: tuple, rows_per_year: dict) -> list:
    """CHECKS lines printed before any result. vstats = prepare_vix()['stats']; cal_overlap = (calendar days in the evaluation span that have a
    VIX, calendar days in the span); counts = select_samples()['counts']; sample_counts = {'dev': (primary, daily), 'val': (primary, daily)};
    sanity = {'median_vix', 'mean_iv2', 'mean_rv', 'corr_log'}; ff_cov = (finite market days, calendar days) through the seal."""
    L = []
    yrs = [y for y in sorted(rows_per_year) if y <= int(PRE_END[:4])]                # nothing after the seal is reported
    L.append("daily returns files: " + (f"{len(yrs)} yearly files through {yrs[-1]} (first {yrs[0]}); calendar read from the dlycaldt column only" if yrs else "NO FILES"))
    L.append(f"calendar (sorted union of daily dates) before the seal: {ncal_pre:,} trading days; ff coverage: {ff_cov[0]:,} of {ff_cov[1]:,} "
             f"({ff_cov[0] / max(ff_cov[1], 1):.2%})")
    s = vstats
    L.append(f"cboe_vix: {s['rows_in_scope']:,} rows dated before the seal (later rows are not read), date range {s['first']}..{s['last']}")
    L.append(f"VIX days dropped: null / non-finite / non-positive {s['null_dropped']:,}; date not in the calendar {s['not_in_calendar']:,}; "
             f"duplicate dates {s['duplicates']:,}; unparseable dates {s['bad_date']:,}; matched to the calendar {s['matched']:,}")
    L.append(f"calendar overlap: {cal_overlap[0]:,} of {cal_overlap[1]:,} calendar days in the forecast span {V.FIRST_FC_YEAR}..2019 have a VIX "
             f"({cal_overlap[0] / max(cal_overlap[1], 1):.1%})")
    med = sanity["median_vix"]
    L.append(f"  [{'PASS' if 10.0 <= med < 30.0 else 'FAIL'}] median VIX {med:.2f} (expected in the teens or twenties)")
    L.append(f"  [{'PASS' if sanity['mean_iv2'] > sanity['mean_rv'] else 'FAIL'}] mean IV2_win {sanity['mean_iv2']:.5f} versus mean RV_win "
             f"{sanity['mean_rv']:.5f} on the primary sample (implied above realised on average: {sanity['mean_iv2'] / sanity['mean_rv']:.2f}x)")
    L.append(f"  [{'PASS' if np.isfinite(sanity['corr_log']) and sanity['corr_log'] > 0 else 'FAIL'}] correlation of log IV2_win with log F_win(gjr_t) "
             f"on the primary sample: {sanity['corr_log']:.3f}")
    L.append(f"primary sample (calendar index multiple of {V.REFIT}, complete window, VIX present, common six-model mask): dev {sample_counts['dev'][0]} "
             f"observations, validation {sample_counts['val'][0]}; daily-overlap sample: dev {sample_counts['dev'][1]:,}, validation {sample_counts['val'][1]:,}")
    n_inc = counts["rows"] - counts["complete_window"]
    L.append(f"incomplete RV windows (market forecast rows without all 21 following returns): {n_inc:,} of {counts['rows']:,} "
             f"({n_inc / max(counts['rows'], 1):.2%}); forecast rows with a complete window but no VIX: {counts['no_vix']:,}; "
             f"rows on the common six-model mask: {counts['valid_six_models']:,}")
    L.append("with only about 60 validation observations the validation sign checks have little power")
    return L
