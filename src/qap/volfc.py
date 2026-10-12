"""Study F (pre-registration v13, Stage A): out-of-sample volatility forecasting, GJR-GARCH-t versus simple benchmarks.

Pure functions (numpy / scipy only; no `arch`). Forecast date t = information through the close of t; targets use returns on t+1... only.

Definitions (exact; see ROADMAP.md, pre-registration v13)
-----------------------------------------------------------
recursion   sigma2[0] = s2 (window second moment), sigma2[k+1] = omega + (alpha + gamma 1[r_k<0]) r_k^2 + beta sigma2[k]; omega = (1-p) s2,
            p = alpha + beta (GARCH) or alpha + gamma/2 + beta (GJR). sigma2[k] is the variance OF r_k, sigma2[n] the forecast for day n.
            The forcing term depends on observed returns only, so it is a linear filter in sigma2 (scipy.signal.lfilter, zi = beta * s2).
window      the asset's non-missing returns among the 1000 calendar trading days t-999..t (>= 950 required); refit when the calendar index
            is a multiple of 21; parameters held for t_r..t_r+20; the recursion is re-run from the window start through each forecast date.
fallback    M3 / M4 not ok (not converged, non-finite, or p >= 0.999) -> the M2 forecast; M2 not ok -> M1 (EWMA). Scored as the heavy model.
"""
import contextlib
import math

import numpy as np
from scipy import optimize, special, stats
from scipy.signal import lfilter

MODELS = ("hist", "ewma", "garch", "gjr_t", "gjr_t_wins", "har")
SIZING = ("hist", "ewma", "garch", "gjr_t", "har")          # M4 is an ablation, not a sizing candidate
HORIZONS = (1, 21)

# ------------------------------------------------------------------------------------------- pre-registered constants
WINDOW = 1000            # calendar trading days in the estimation window
MIN_OBS = 950            # minimum non-missing returns in the window
REFIT = 21               # refit every 21st calendar index
HIST_DAYS = 252
EWMA_LAMBDA = 0.94
WINS_Q = (0.5, 99.5)
P_BREACH = 0.999
NU_BOUNDS = (2.1, 100.0)
START = dict(alpha=0.05, gamma=0.05, beta=0.88, nu=8.0)
HAR_FLOOR = 0.1
MC_PATHS = 20000
NW_LAGS = {1: 5, 21: 30}
MATERIALITY = -0.01
DM_T = -2.0
DEV = ("1995-01-01", "2014-12-31")
VAL = ("2015-01-01", "2019-12-31")
HOLD = ("2020-01-01", "2025-12-31")
FIRST_FC_YEAR = 1995
TOP_N = 100
P_CAP = 0.9999           # numerical ceiling for p inside the optimiser (breach is judged at P_BREACH)
BETA_MAX = 0.9999
VERIFY_TOL = 1e-10
MKT_ID = -1              # asset id of the FF market series (cannot collide with a CRSP permno)


# ------------------------------------------------------------------------------------------------- BLAS threads
@contextlib.contextmanager
def single_thread_blas():
    """Run with one BLAS thread. The tiny BLAS calls inside L-BFGS-B are 10x slower with OpenBLAS threads under any load (spin waits), and
    worker processes would oversubscribe the cores. Uses threadpoolctl if installed, else the OpenBLAS symbols via ctypes; otherwise a no-op."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is not None:
        with threadpool_limits(limits=1, user_api="blas"):
            yield
        return
    restore = []
    try:
        import ctypes
        with open("/proc/self/maps") as fh:
            libs = sorted({ln.split()[-1] for ln in fh if "openblas" in ln.lower() and ln.split()[-1].startswith("/")})
        for lib in libs:
            dll = ctypes.CDLL(lib)
            for pre in ("scipy_openblas", "openblas"):
                for suf in ("", "64_"):
                    setter, getter = getattr(dll, f"{pre}_set_num_threads{suf}", None), getattr(dll, f"{pre}_get_num_threads{suf}", None)
                    if setter is not None and getter is not None:
                        restore.append((setter, int(getter())))
                        setter(1)
                        break
                else:
                    continue
                break
    except Exception:
        pass
    try:
        yield
    finally:
        for setter, n in restore:
            setter(n)


# ---------------------------------------------------------------------------------------------------- recursion
def persistence(alpha: float, gamma: float, beta: float) -> float:
    return alpha + gamma / 2.0 + beta


def sigma2_path(r, alpha: float, gamma: float, beta: float, s2: float, omega: float | None = None) -> np.ndarray:
    """Length n+1 variance path (see module docstring) by a linear filter. omega defaults to the variance-targeting value."""
    r = np.asarray(r, dtype="float64")
    if omega is None:
        omega = (1.0 - persistence(alpha, gamma, beta)) * s2
    x = omega + (alpha + gamma * (r < 0)) * r * r
    y = lfilter([1.0], [1.0, -beta], x, zi=[beta * s2])[0]
    return np.concatenate(([s2], y))


def recursion_slow(r, alpha: float, gamma: float, beta: float, s2: float, omega: float | None = None) -> np.ndarray:
    """Reference loop for `sigma2_path`."""
    r = np.asarray(r, dtype="float64")
    if omega is None:
        omega = (1.0 - persistence(alpha, gamma, beta)) * s2
    out = np.empty(len(r) + 1)
    out[0] = s2
    for k in range(len(r)):
        out[k + 1] = omega + (alpha + (gamma if r[k] < 0 else 0.0)) * r[k] ** 2 + beta * out[k]
    return out


def verify_recursion(r, tol: float = VERIFY_TOL) -> float:
    """Vectorised vs slow recursion on a sample (relative max difference); raises AssertionError above `tol`."""
    r = np.asarray(r, dtype="float64")
    r = r[np.isfinite(r)]
    s2 = float(np.mean(r * r))
    worst = 0.0
    for a, g, b in ((0.05, 0.0, 0.93), (0.03, 0.08, 0.90), (0.0, 0.0, 0.5), (0.1, 0.1, 0.8), (0.2, 0.0, 0.0)):
        v, w = sigma2_path(r, a, g, b, s2), recursion_slow(r, a, g, b, s2)
        worst = max(worst, float(np.max(np.abs(v - w) / w)))
    if not worst <= tol:
        raise AssertionError(f"vectorised variance recursion differs from the reference loop by {worst:.3e} (> {tol:g})")
    return worst


# --------------------------------------------------------------------------------------------------- densities
def std_t_logpdf(x, sigma2, nu):
    """Log density of x = sigma * z, z standardised Student-t (unit variance; scale sqrt((nu-2)/nu)); nu > 2."""
    x, sigma2, nu = np.asarray(x, dtype="float64"), np.asarray(sigma2, dtype="float64"), np.asarray(nu, dtype="float64")
    c = special.gammaln((nu + 1.0) / 2.0) - special.gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    return c - 0.5 * np.log(sigma2) - (nu + 1.0) / 2.0 * np.log1p(x * x / (sigma2 * (nu - 2.0)))


def normal_logpdf(x, sigma2):
    x, sigma2 = np.asarray(x, dtype="float64"), np.asarray(sigma2, dtype="float64")
    return -0.5 * (np.log(2.0 * np.pi * sigma2) + x * x / sigma2)


def t_density_check(nus=(2.1, 3.0, 5.0, 8.0, 30.0, 100.0)) -> dict:
    """Numerical check that the standardised-t density integrates to 1 and has unit variance: {nu: (integral, variance)}."""
    from scipy import integrate
    out = {}
    for nu in nus:
        f = lambda x: math.exp(float(std_t_logpdf(x, 1.0, nu)))
        # x = tan(theta) substitution turns the infinite range into a finite one and tames the heavy tails
        g0 = lambda th: f(math.tan(th)) / math.cos(th) ** 2
        g2 = lambda th: math.tan(th) ** 2 * f(math.tan(th)) / math.cos(th) ** 2
        half = math.pi / 2.0 - 1e-12
        one = integrate.quad(g0, -half, half, limit=400, epsabs=1e-12, epsrel=1e-12)[0]
        var = integrate.quad(g2, -half, half, limit=800, epsabs=1e-12, epsrel=1e-12)[0] if nu > 2.5 else float("nan")
        out[nu] = (one, var)
    return out


# ------------------------------------------------------------------------------------------------- estimators
def winsorise(window, q=WINS_Q):
    """Clip at the window's OWN 0.5th / 99.5th percentiles (np.percentile, linear interpolation); returns (clipped, lo, hi)."""
    w = np.asarray(window, dtype="float64")
    lo, hi = np.percentile(w, q[0]), np.percentile(w, q[1])
    return np.clip(w, lo, hi), float(lo), float(hi)


def _nll(x, z, tdist: bool, gjr: bool) -> float:
    """Mean negative log-likelihood on unit-second-moment returns z; x = (alpha, gamma, beta[, log nu])."""
    a, g, b = x[0], (x[1] if gjr else 0.0), x[2 if gjr else 1]
    p = a + g / 2.0 + b
    if p > P_CAP:                                   # keep omega > 0: evaluate at the projection onto p = P_CAP (a flat ridge, never an error)
        k = P_CAP / p
        a, g, b = a * k, g * k, b * k
        p = P_CAP
    sig = sigma2_path(z, a, g, b, 1.0, 1.0 - p)[:-1]
    if tdist:
        ll = std_t_logpdf(z, sig, math.exp(x[-1]))
    else:
        ll = normal_logpdf(z, sig)
    v = -float(np.mean(ll))
    return v if np.isfinite(v) else 1e10


def _fit(window, tdist: bool, gjr: bool) -> dict:
    w = np.asarray(window, dtype="float64")
    s2 = float(np.mean(w * w)) if len(w) else float("nan")
    bad = dict(status="failed", alpha=np.nan, gamma=np.nan, beta=np.nan, nu=np.nan, p=np.nan, s2=s2, nll=np.nan, nit=0, nfev=0)
    if len(w) < 10 or not np.isfinite(s2) or s2 <= 0 or not np.all(np.isfinite(w)):
        return bad
    z = w / math.sqrt(s2)
    if gjr:
        x0, lb, ub = [START["alpha"], START["gamma"], START["beta"]], [0.0, 0.0, 0.0], [1.0, 1.0, BETA_MAX]
    else:
        x0, lb, ub = [START["alpha"], START["beta"]], [0.0, 0.0], [1.0, BETA_MAX]
    if tdist:
        x0, lb, ub = x0 + [math.log(START["nu"])], lb + [math.log(NU_BOUNDS[0])], ub + [math.log(NU_BOUNDS[1])]
    try:
        res = optimize.minimize(_nll, np.array(x0), args=(z, tdist, gjr), method="L-BFGS-B", bounds=list(zip(lb, ub)),
                                options=dict(maxiter=500, maxfun=5000))
    except Exception:
        return bad
    x = res.x
    a, g, b = x[0], (x[1] if gjr else 0.0), x[2 if gjr else 1]
    p = a + g / 2.0 + b
    if p > P_CAP:
        k = P_CAP / p
        a, g, b, p = a * k, g * k, b * k, P_CAP
    nu = math.exp(x[-1]) if tdist else float("nan")
    out = dict(bad, alpha=float(a), gamma=float(g), beta=float(b), nu=nu, p=float(p), nll=float(res.fun), nit=int(res.nit), nfev=int(res.nfev))
    if not (res.success and np.isfinite(res.fun) and np.all(np.isfinite([a, g, b, p])) and (not tdist or np.isfinite(nu))):
        out["status"] = "failed"
    elif p >= P_BREACH:
        out["status"] = "breach"
    else:
        out["status"] = "ok"
    return out


def fit_garch_normal(window) -> dict:
    """GARCH(1,1) Gaussian QMLE with variance targeting. Keys: status ('ok' / 'failed' / 'breach'), alpha, gamma(=0), beta, nu(nan), p, s2."""
    return _fit(window, tdist=False, gjr=False)


def fit_gjr_t(window) -> dict:
    """GJR-GARCH(1,1), standardised Student-t, exact ML with variance targeting (nu in [2.1, 100])."""
    return _fit(window, tdist=True, gjr=True)


# -------------------------------------------------------------------------------------------------- forecasters
def avg_var_forecast(next_var, s2: float, p: float, h: int):
    """Mean over k = 1..h of E sigma2_{t+k} = s2 + p^(k-1) (sigma2_{t+1} - s2). `next_var` may be an array."""
    k = np.arange(h, dtype="float64")
    w = float(np.mean(p ** k))
    return s2 + w * (np.asarray(next_var, dtype="float64") - s2)


def har_fit(window, h: int) -> np.ndarray | None:
    """OLS in levels: y = mean r^2 over t+1..t+h on [1, r_t^2, mean r^2 (t-4..t), mean r^2 (t-21..t)]; rows whose target is inside the window."""
    w = np.asarray(window, dtype="float64")
    r2 = w * w
    s2 = float(np.mean(r2))
    if not s2 > 0:
        return None
    u = r2 / s2
    n = len(u)
    cs = np.concatenate(([0.0], np.cumsum(u)))
    t = np.arange(21, n - h)
    if len(t) < 20:
        return None
    X = np.column_stack([np.ones(len(t)), u[t], (cs[t + 1] - cs[t - 4]) / 5.0, (cs[t + 1] - cs[t - 21]) / 22.0])
    y = (cs[t + h + 1] - cs[t + 1]) / h
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta                                      # coefficients in units of s2 for the level of r^2


def har_forecast(x, pos, beta, s2: float, floor: float = HAR_FLOOR):
    """HAR forecasts (and floor-hit flags) at positions `pos` of the series x (needs pos >= 21)."""
    x = np.asarray(x, dtype="float64")
    u = x * x / s2
    cs = np.concatenate(([0.0], np.cumsum(u)))
    pos = np.asarray(pos)
    f = (beta[0] + beta[1] * u[pos] + beta[2] * (cs[pos + 1] - cs[pos - 4]) / 5.0 + beta[3] * (cs[pos + 1] - cs[pos - 21]) / 22.0) * s2
    lo = floor * s2
    return np.maximum(f, lo), f < lo


# ----------------------------------------------------------------------------------------------------- losses
def qlike(sigma2_hat, proxy):
    sigma2_hat, proxy = np.asarray(sigma2_hat, dtype="float64"), np.asarray(proxy, dtype="float64")
    return np.log(sigma2_hat) + proxy / sigma2_hat


def mse_loss(sigma2_hat, proxy):
    return (np.asarray(sigma2_hat, dtype="float64") - np.asarray(proxy, dtype="float64")) ** 2


def newey_west_t(d, lags: int) -> tuple:
    """(mean, t) of a series with a Bartlett-kernel Newey-West long-run variance: var = g0 + 2 sum_{l=1..L} (1 - l/(L+1)) g_l; t = mean / sqrt(var/T)."""
    d = np.asarray(d, dtype="float64")
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 3:
        return float("nan"), float("nan")
    m = float(d.mean())
    e = d - m
    lr = float(e @ e) / T
    for l in range(1, min(lags, T - 1) + 1):
        lr += 2.0 * (1.0 - l / (lags + 1.0)) * float(e[l:] @ e[:-l]) / T
    if not lr > 0:
        return m, float("nan")
    return m, m / math.sqrt(lr / T)


def kupiec_pvalue(x: int, n: int, p: float) -> float:
    """Kupiec proportion-of-failures likelihood-ratio p-value (chi-square 1) for x violations in n at nominal rate p."""
    if n <= 0:
        return float("nan")
    ph = x / n
    def ll(q):
        a = (n - x) * math.log(1 - q) if n - x > 0 else 0.0
        b = x * math.log(q) if x > 0 else 0.0
        return a + b
    lr = -2.0 * (ll(p) - ll(ph)) if 0 < ph < 1 else -2.0 * ll(p) + (0.0)
    return float(stats.chi2.sf(max(lr, 0.0), 1))


# ------------------------------------------------------------------------------------------------- Monte Carlo
def mc_cumret(next_var: float, alpha: float, gamma: float, beta: float, omega: float, nu, rng, n_paths: int, horizon: int = 21) -> tuple:
    """Simulate `n_paths` paths of r_{t+1..t+horizon} = sigma_k z_k with GJR dynamics; z standardised-t (nu given) or Gaussian (nu None).

    Returns (cumulative return per path [sum of daily returns], mean r^2 per step, mean sigma2 per step).
    """
    var = np.full(n_paths, float(next_var))
    cum = np.zeros(n_paths)
    m_r2, m_var = np.empty(horizon), np.empty(horizon)
    scale = math.sqrt((nu - 2.0) / nu) if nu is not None else 1.0
    for k in range(horizon):
        z = rng.standard_normal(n_paths) if nu is None else rng.standard_t(nu, n_paths) * scale
        r = np.sqrt(var) * z
        cum += r
        m_r2[k], m_var[k] = float(np.mean(r * r)), float(np.mean(var))
        var = omega + (alpha + gamma * (r < 0)) * r * r + beta * var
    return cum, m_r2, m_var


# ------------------------------------------------------------------------------------------------- rolling engine
def targets(ret) -> tuple:
    """Per calendar index j: y1 = r_{j+1}^2, r1 = r_{j+1}, y21 = mean r^2 over j+1..j+21 and r21s = sum r over j+1..j+21 (NaN unless ALL 21
    calendar days are non-missing; NaN at the end of the data). Only returns AFTER j enter."""
    ret = np.asarray(ret, dtype="float64")
    n = len(ret)
    fin = np.isfinite(ret)
    r0 = np.where(fin, ret, 0.0)
    cq = np.concatenate(([0.0], np.cumsum(r0 * r0)))
    cr = np.concatenate(([0.0], np.cumsum(r0)))
    cf = np.concatenate(([0], np.cumsum(fin.astype("int64"))))
    j = np.arange(n)
    y1, r1 = np.full(n, np.nan), np.full(n, np.nan)
    y1[:-1], r1[:-1] = np.where(fin[1:], ret[1:] ** 2, np.nan), np.where(fin[1:], ret[1:], np.nan)
    y21, r21 = np.full(n, np.nan), np.full(n, np.nan)
    ok = (j + 21 <= n - 1)
    jj = j[ok]
    full = (cf[jj + 22] - cf[jj + 1]) == 21
    y21[jj] = np.where(full, (cq[jj + 22] - cq[jj + 1]) / 21.0, np.nan)
    r21[jj] = np.where(full, cr[jj + 22] - cr[jj + 1], np.nan)
    return y1, r1, y21, r21


_FB = ("garch", "gjr_t", "gjr_t_wins")
ROW_FLOATS = ([f"f{h}_{m}" for m in MODELS for h in HORIZONS] + ["nu_gjr_t", "nu_gjr_t_wins", "y1", "r1", "y21", "r21s"]
              + [f"mc{q}_{m}" for m in _FB for q in ("01", "05")])
ROW_BOOLS = [f"fb_{m}" for m in _FB] + ["har_floor1", "har_floor21"]
FIT_COLS = ["aid", "refit", "model", "status", "alpha", "gamma", "beta", "nu", "p", "nit", "nfev"]


def _empty_rows() -> dict:
    d = {"aid": np.empty(0, "int64"), "tidx": np.empty(0, "int64"), "refit": np.empty(0, "int64")}
    d.update({c: np.empty(0) for c in ROW_FLOATS})
    d.update({c: np.empty(0, bool) for c in ROW_BOOLS})
    return d


def _blocks_for(ranges, ncal: int) -> list:
    out = set()
    for lo, hi in ranges:
        lo, hi = max(int(lo), 0), min(int(hi), ncal - 1)
        if hi < lo:
            continue
        m0 = max(0, -(-(lo - (REFIT - 1)) // REFIT) * REFIT)
        out.update(range(m0, (hi // REFIT) * REFIT + 1, REFIT))
    return sorted(out)


def run_asset(ret, ranges, asset_id: int, mc: bool = True) -> dict:
    with single_thread_blas():
        return _run_asset(ret, ranges, asset_id, mc)


def _run_asset(ret, ranges, asset_id: int, mc: bool = True) -> dict:
    """Rolling forecasts for ONE asset (see the module docstring).

    ret: daily returns on the market calendar (NaN = no return). ranges: [(lo, hi)] inclusive calendar-index ranges in which forecasts are
    wanted. Returns {'rows': dict of equal-length arrays, 'fits': list of tuples (FIT_COLS), 'blocks': int array (refit index, 1 if the history rule
    is met) for every candidate block}.
    """
    ret = np.asarray(ret, dtype="float64")
    ncal = len(ret)
    fin = np.isfinite(ret)
    y1_all, r1_all, y21_all, r21_all = targets(ret)
    allowed = np.zeros(ncal, bool)
    for lo, hi in ranges:
        allowed[max(int(lo), 0):min(int(hi), ncal - 1) + 1] = True
    allowed &= fin
    parts, fits, blk = [], [], []
    for t_r in _blocks_for(ranges, ncal):
        dates = np.arange(t_r, min(t_r + REFIT, ncal))
        fd = dates[allowed[dates]]
        if len(fd) == 0:
            continue
        seg = ret[max(0, t_r - WINDOW + 1):t_r + 1]
        w = seg[np.isfinite(seg)]
        blk.append((t_r, int(len(w) >= MIN_OBS)))
        if len(w) < MIN_OBS:
            continue
        ext = dates[(dates > t_r) & fin[dates]]
        x = np.concatenate([w, ret[ext]])
        n = len(w)
        pos = np.where(fd == t_r, n - 1, n + np.searchsorted(ext, fd))
        k = len(fd)
        row = {"aid": np.full(k, asset_id, "int64"), "tidx": fd.astype("int64"), "refit": np.full(k, t_r, "int64"),
               "y1": y1_all[fd], "r1": r1_all[fd], "y21": y21_all[fd], "r21s": r21_all[fd]}
        s2 = float(np.mean(w * w))
        # M0 hist
        cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
        hist = (cs2[pos + 1] - cs2[pos + 1 - HIST_DAYS]) / HIST_DAYS
        row["f1_hist"] = row["f21_hist"] = hist
        # M1 ewma
        ew = lfilter([1.0], [1.0, -EWMA_LAMBDA], (1.0 - EWMA_LAMBDA) * x * x, zi=[EWMA_LAMBDA * s2])[0]
        row["f1_ewma"] = row["f21_ewma"] = ew[pos]
        # M2 garch (fallback M1)
        fg = fit_garch_normal(w)
        fj = fit_gjr_t(w)
        wc, lo_c, hi_c = winsorise(w)
        fw = fit_gjr_t(wc)
        for nm, f in (("garch", fg), ("gjr_t", fj), ("gjr_t_wins", fw)):
            fits.append((asset_id, t_r, nm, f["status"], f["alpha"], f["gamma"], f["beta"], f["nu"], f["p"], f["nit"], f["nfev"]))
        sims = {}
        if fg["status"] == "ok":
            nxt = sigma2_path(x, fg["alpha"], 0.0, fg["beta"], s2)[pos + 1]
            g1, g21 = nxt, avg_var_forecast(nxt, s2, fg["p"], 21)
            g_par = (fg["alpha"], 0.0, fg["beta"], (1.0 - fg["p"]) * s2, None)
        else:
            g1, g21 = ew[pos], ew[pos]
            g_par = (1.0 - EWMA_LAMBDA, 0.0, EWMA_LAMBDA, 0.0, None)         # EWMA = IGARCH(0.06, 0.94) with omega 0
        row["f1_garch"], row["f21_garch"] = g1, g21
        row["fb_garch"] = np.full(k, fg["status"] != "ok")
        sims["garch"] = (g_par, float(g1[pos == n - 1][0]) if (pos == n - 1).any() else None)
        for nm, f, xs, s2x in (("gjr_t", fj, x, s2), ("gjr_t_wins", fw, np.clip(x, lo_c, hi_c), float(np.mean(wc * wc)))):
            if f["status"] == "ok":
                nxt = sigma2_path(xs, f["alpha"], f["gamma"], f["beta"], s2x)[pos + 1]
                a1, a21 = nxt, avg_var_forecast(nxt, s2x, f["p"], 21)
                par = (f["alpha"], f["gamma"], f["beta"], (1.0 - f["p"]) * s2x, f["nu"])
                row["nu_" + nm] = np.full(k, f["nu"])
            else:
                a1, a21, par = g1, g21, g_par
                row["nu_" + nm] = np.full(k, np.nan)
            row["f1_" + nm], row["f21_" + nm] = a1, a21
            row["fb_" + nm] = np.full(k, f["status"] != "ok")
            sims[nm] = (par, float(a1[pos == n - 1][0]) if (pos == n - 1).any() else None)
        # M5 har
        for h in HORIZONS:
            b = har_fit(w, h)
            if b is None:
                row[f"f{h}_har"], row[f"har_floor{h}"] = np.full(k, np.nan), np.zeros(k, bool)
            else:
                row[f"f{h}_har"], row[f"har_floor{h}"] = har_forecast(x, pos, b, s2)
        # Monte Carlo 21-day return VaR coverage at the refit date only (non-overlapping 21-day returns)
        for nm in _FB:
            for q in ("01", "05"):
                row[f"mc{q}_{nm}"] = np.full(k, np.nan)
        if mc and (pos == n - 1).any():
            i0 = int(np.flatnonzero(pos == n - 1)[0])
            rng = np.random.default_rng([asset_id + 1, t_r])
            for nm in _FB:                                   # fixed draw order: the seed depends on (asset, refit index) only
                par, nv = sims[nm]
                cum, _, _ = mc_cumret(nv, par[0], par[1], par[2], par[3], par[4], rng, MC_PATHS)
                real = row["r21s"][i0]
                if np.isfinite(real):
                    for q, a in (("01", 0.01), ("05", 0.05)):
                        row[f"mc{q}_{nm}"][i0] = float(real < np.quantile(cum, a))
        parts.append(row)
    if not parts:
        return {"rows": _empty_rows(), "fits": fits, "blocks": np.array(blk, dtype="int64").reshape(-1, 2)}
    rows = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    return {"rows": rows, "fits": fits, "blocks": np.array(blk, dtype="int64").reshape(-1, 2)}


# ----------------------------------------------------------------------------------------------------- universe
def top_universe(panel, years, n: int = TOP_N) -> dict:
    """{year: sorted permnos}: for each calendar year Y the top `n` companies (permco) by prior-December company ME (`me_co`) among common
    equity (`eq`); the representative security of a company is its permno with the largest `me` (ties: smaller permno). No survivor filter."""
    import pandas as pd
    out = {}
    for y in years:
        d = panel.loc[(panel["mi"] == (int(y) - 1) * 12 + 11) & panel["eq"] & (panel["me_co"] > 0), ["permno", "permco", "me", "me_co"]]
        d = d.sort_values(["permco", "me", "permno"], ascending=[True, False, True], kind="mergesort").drop_duplicates("permco", keep="first")
        d = d.sort_values(["me_co", "permco"], ascending=[False, True], kind="mergesort").head(n)
        out[int(y)] = np.sort(d["permno"].to_numpy("int64"))
    return out


def stock_years(universe: dict) -> dict:
    """{permno: sorted years in the universe}."""
    out = {}
    for y, ps in universe.items():
        for p in ps:
            out.setdefault(int(p), []).append(int(y))
    return {p: sorted(v) for p, v in sorted(out.items())}


def year_ranges(cal, years) -> list:
    """Inclusive calendar-index ranges [(lo, hi)] of the given calendar years."""
    out = []
    for y in years:
        lo = int(np.searchsorted(cal, np.datetime64(f"{int(y)}-01-01"), side="left"))
        hi = int(np.searchsorted(cal, np.datetime64(f"{int(y) + 1}-01-01"), side="left")) - 1
        if hi >= lo:
            out.append((lo, hi))
    return out


# ------------------------------------------------------------------------------------------------------ loaders
def load_daily_subset(directory, permnos) -> tuple:
    """Read data/raw/crsp_daily_ret/{year}.parquet one year at a time, keep only `permnos`: (DataFrame[permno, dlycaldt, dlyret], {year: rows},
    sorted calendar of ALL dates as datetime64[D])."""
    import glob
    import pathlib

    import pandas as pd
    keep = np.asarray(sorted(set(int(p) for p in permnos)), dtype="int64")
    parts, rows, days = [], {}, []
    for f in sorted(glob.glob(str(pathlib.Path(directory) / "[0-9]*.parquet"))):
        d = pd.read_parquet(f, columns=["permno", "dlycaldt", "dlyret"])
        rows[int(pathlib.Path(f).stem)] = len(d)
        d["dlycaldt"] = pd.to_datetime(d["dlycaldt"])
        days.append(d["dlycaldt"].drop_duplicates().to_numpy().astype("datetime64[D]"))
        parts.append(d[np.isin(d["permno"].to_numpy("int64"), keep)].astype({"permno": "int64", "dlyret": "float64"}))
    cal = np.unique(np.concatenate(days)) if days else np.empty(0, "datetime64[D]")
    sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"permno": [], "dlycaldt": [], "dlyret": []})
    return sub, rows, cal


def return_matrix(sub, cal, permnos) -> np.ndarray:
    """float64 (len(permnos), len(cal)) of daily returns, NaN where the stock has no return."""
    permnos = np.asarray(list(permnos), dtype="int64")
    R = np.full((len(permnos), len(cal)), np.nan)
    if len(sub) == 0:
        return R
    ri = np.searchsorted(permnos, sub["permno"].to_numpy("int64"))        # permnos must be sorted
    ci = np.searchsorted(cal, sub["dlycaldt"].to_numpy().astype("datetime64[D]"))
    ok = (ri < len(permnos)) & (permnos[np.minimum(ri, len(permnos) - 1)] == sub["permno"].to_numpy("int64"))
    R[ri[ok], ci[ok]] = sub["dlyret"].to_numpy("float64")[ok]
    return R


def market_series(ff, cal) -> np.ndarray:
    """mktrf + rf (decimals via earnings.ff_decimals) on the calendar; NaN where FF is missing."""
    from qap.earnings import ff_decimals
    f = ff_decimals(ff)
    fd = f["date"].to_numpy().astype("datetime64[D]")
    pos = np.searchsorted(cal, fd)
    pc = np.minimum(pos, max(len(cal) - 1, 0))
    hit = (pos < len(cal)) & (cal[pc] == fd) if len(cal) else np.zeros(len(fd), bool)
    out = np.full(len(cal), np.nan)
    out[pos[hit]] = (f["mktrf"] + f["rf"]).to_numpy()[hit]
    return out


# ------------------------------------------------------------------------------------------------ ticker mapping
TICKERS = ("ORCL", "INTU", "LITE", "BE", "STX", "AMD", "CELH", "ALAB", "SNDK")
NOT_CRSP = ("SKHY",)
MIN_TICKER_DAYS = 1000


def map_tickers(names, tickers, on_date) -> dict:
    """{ticker: ('ok', permno) | ('missing', None) | ('ambiguous', [permnos])}: the permno whose CRSP name record is in force on `on_date`
    (namedt <= on_date <= nameenddt). Columns are located with etfcheck.pick_columns (ticker, namedt, nameenddt)."""
    from qap.etfcheck import prep_names
    nm = prep_names(names)
    d = np.datetime64(on_date, "ns")
    inforce = nm[(nm["start"].to_numpy() <= d) & (nm["end"].to_numpy() >= d)]
    out = {}
    for t in tickers:
        hit = sorted(set(int(p) for p in inforce.loc[inforce["ticker"] == t.upper(), "permno"]))
        out[t] = ("missing", None) if not hit else (("ok", hit[0]) if len(hit) == 1 else ("ambiguous", hit))
    return out


# --------------------------------------------------------------------------------------------- merged analysis
def valid_masks(df) -> dict:
    """{h: boolean mask}: proxy available AND every model's forecast finite and > 0 (one common row set for all models)."""
    out = {}
    for h in HORIZONS:
        ok = np.isfinite(df["y1" if h == 1 else "y21"].to_numpy())
        for m in MODELS:
            f = df[f"f{h}_{m}"].to_numpy()
            ok &= np.isfinite(f) & (f > 0)
        out[h] = ok
    return out


def date_losses(df, mask, h: int, kind: str = "qlike", models=MODELS):
    """DataFrame (index = forecast date index tidx, columns = models): cross-asset mean loss per forecast date on the rows of `mask`."""
    import pandas as pd
    proxy = df["y1" if h == 1 else "y21"].to_numpy()[mask]
    fn = qlike if kind == "qlike" else mse_loss
    cols = {m: fn(df[f"f{h}_{m}"].to_numpy()[mask], proxy) for m in models}
    g = pd.DataFrame(cols)
    g["tidx"] = df["tidx"].to_numpy()[mask]
    return g.groupby("tidx", sort=True).mean()


def window_slice(dl, cal, window):
    """Rows of a per-date frame whose forecast date lies in [window[0], window[1]] (inclusive ISO dates)."""
    d = cal[dl.index.to_numpy()]
    return dl[(d >= np.datetime64(window[0])) & (d <= np.datetime64(window[1]))]


def compare(dl, cal, a: str, b: str, h: int, window) -> tuple:
    """(mean loss difference a - b, Newey-West DM t, number of forecast dates) on `window`."""
    s = window_slice(dl, cal, window)
    d = (s[a] - s[b]).to_numpy()
    m, t = newey_west_t(d, NW_LAGS[h])
    return m, t, int(np.isfinite(d).sum())


COMPARISONS = (("gjr_t", "garch"), ("gjr_t", "har"))


def licence_verdict(comps: dict, holdout: dict | None = None, open_holdout: bool = False) -> dict:
    """Pre-registered licence rule. comps: {(model, benchmark, h): {'dev': (mean, t), 'val': (mean, t)}} for ALL FOUR comparisons.
    LICENSED in dev/validation only if every one has dev mean <= -0.01 AND dev t <= -2.0 AND validation mean < 0. The holdout (a dict
    {(model, benchmark, h): mean difference}) is consulted ONLY when `open_holdout` is True and dev/validation passed; it then has to be
    < 0 for all four. Returns {'dev_val': bool, 'opened': bool, 'holdout_all_negative': bool|None, 'status': str}."""
    need = {(m, b, h) for m, b in COMPARISONS for h in HORIZONS}
    if set(comps) != need:
        raise KeyError(f"licence rule needs exactly the four comparisons {sorted(need)}; got {sorted(comps)}")
    dv = all(c["dev"][0] <= MATERIALITY and c["dev"][1] <= DM_T and c["val"][0] < 0 for c in comps.values())
    if not dv:
        return dict(dev_val=False, opened=False, holdout_all_negative=None, status="NOT LICENSED (fails the dev / validation rule)")
    if not open_holdout:
        return dict(dev_val=True, opened=False, holdout_all_negative=None,
                    status="LICENSED IN DEV/VALIDATION, holdout sealed (run with --open-holdout to open it once)")
    if holdout is None:
        raise ValueError("open_holdout without holdout differences")
    allneg = all(np.isfinite(holdout[k]) and holdout[k] < 0 for k in need)
    return dict(dev_val=True, opened=True, holdout_all_negative=bool(allneg),
                status="LICENSED (dev/validation and holdout)" if allneg else "licensed in dev/validation, failed in the holdout")


def default_sizing_model(dev_means: dict) -> str:
    """Lowest dev mean QLIKE at h = 21 among M0, M1, M2, M3, M5 (M4 excluded). Ties: the first in SIZING order."""
    best = min(SIZING, key=lambda m: (dev_means[m] if np.isfinite(dev_means[m]) else np.inf))
    return best


def row_logscore(df, mask) -> dict:
    """Per-model predictive log score of r_{t+1} (rows of `mask`): Normal, except standardised-t with the fitted nu for gjr_t / gjr_t_wins
    when not a fallback."""
    r = df["r1"].to_numpy()[mask]
    out = {}
    for m in MODELS:
        f = df[f"f1_{m}"].to_numpy()[mask]
        ls = normal_logpdf(r, f)
        if m in ("gjr_t", "gjr_t_wins"):
            fb = df[f"fb_{m}"].to_numpy()[mask]
            nu = df[f"nu_{m}"].to_numpy()[mask]
            t_ok = ~fb & np.isfinite(nu)
            if t_ok.any():
                ls = np.where(t_ok, std_t_logpdf(r, f, np.where(t_ok, nu, 8.0)), ls)
        out[m] = ls
    return out


def var_violations(df, mask, alpha: float) -> dict:
    """Per-model 1-day VaR violation indicator r_{t+1} < q_alpha (Normal quantile, standardised-t for gjr_t / gjr_t_wins unless fallback)."""
    r = df["r1"].to_numpy()[mask]
    out = {}
    for m in MODELS:
        f = df[f"f1_{m}"].to_numpy()[mask]
        q = np.sqrt(f) * stats.norm.ppf(alpha)
        if m in ("gjr_t", "gjr_t_wins"):
            fb = df[f"fb_{m}"].to_numpy()[mask]
            nu = df[f"nu_{m}"].to_numpy()[mask]
            t_ok = ~fb & np.isfinite(nu)
            if t_ok.any():
                nn = np.where(t_ok, nu, 8.0)
                q = np.where(t_ok, np.sqrt(f) * stats.t.ppf(alpha, nn) * np.sqrt((nn - 2.0) / nn), q)
        out[m] = r < q
    return out


def fit_summary(fits, model: str) -> dict:
    """Parameter summary over the converged ('ok') fits of a model plus status shares. `fits` = DataFrame with FIT_COLS."""
    f = fits[fits["model"] == model]
    ok = f[f["status"] == "ok"]
    n = len(f)
    out = {"fits": n, "ok": len(ok) / n if n else float("nan"), "failed": float((f["status"] == "failed").mean()) if n else float("nan"),
           "breach": float((f["status"] == "breach").mean()) if n else float("nan"),
           "alpha": float(ok["alpha"].median()) if len(ok) else float("nan"), "beta": float(ok["beta"].median()) if len(ok) else float("nan"),
           "p": float(ok["p"].median()) if len(ok) else float("nan"), "nit": float(f["nit"].median()) if n else float("nan"),
           "nfev": float(f["nfev"].median()) if n else float("nan")}
    out["gamma"] = float(ok["gamma"].median()) if len(ok) and model != "garch" else float("nan")
    if model != "garch" and len(ok):
        nu = ok["nu"].to_numpy()
        out["nu"] = float(np.median(nu))
        out["nu_at_bound"] = float(np.mean((nu <= NU_BOUNDS[0] * 1.001) | (nu >= NU_BOUNDS[1] * 0.999)))
    else:
        out["nu"] = out["nu_at_bound"] = float("nan")
    return out


# ----------------------------------------------------------------------------------------------------- checks
def volfc_checks(daily_rows: dict, ncal: int, ff_cov: tuple, universe: dict, df, fits, blocks: tuple, verify_diff: float, tcheck: dict,
                 masks: dict, notes=()) -> list:
    """CHECKS lines (printed before any result). ff_cov = (finite market days, calendar days); blocks = (candidate blocks, meeting the history
    rule). `df` = merged primary forecast rows with a `year` column."""
    L = []
    yrs = sorted(daily_rows)
    L.append("daily returns file: " + (f"{sum(daily_rows.values()):,} rows in {len(yrs)} yearly files ({yrs[0]}..{yrs[-1]}); rows per year: "
             + " ".join(f"{y}={daily_rows[y]:,}" for y in yrs) if yrs else "NO ROWS"))
    L.append(f"calendar (sorted union of daily dates): {ncal:,} trading days")
    L.append(f"ff coverage: {ff_cov[0]:,} of {ff_cov[1]:,} calendar days have a market return ({ff_cov[0] / max(ff_cov[1], 1):.2%})")
    L.append("universe size per year (top-100 permco by prior-December ME, common equity): " + " ".join(f"{y}={len(p)}" for y, p in sorted(universe.items())))
    sy = df.groupby(["aid", "year"]).size() if len(df) else []
    L.append(f"stock-years forecast: {len(sy):,} of {sum(len(p) for p in universe.values()):,} universe stock-years "
             f"({len(df):,} forecast rows; per year " + (" ".join(f"{y}={n}" for y, n in sorted(df.groupby('year')['aid'].nunique().items())) + ")" if len(df) else "none)"))
    L.append(f"history rule (>= {MIN_OBS} non-missing of the last {WINDOW} calendar days): {blocks[1]:,} of {blocks[0]:,} refit blocks pass "
             f"({blocks[1] / max(blocks[0], 1):.1%})")
    for m in _FB:
        s = fit_summary(fits, m)
        rows = df[f"fb_{m}"].mean() if len(df) else float("nan")
        L.append(f"fit {m:10s}: ok {s['ok']:.1%} failed {s['failed']:.1%} persistence>=0.999 {s['breach']:.1%} of {s['fits']:,} fits; "
                 f"fallback share of forecast rows {rows:.1%}")
    L.append("optimiser (L-BFGS-B, deterministic start alpha .05 gamma .05 beta .88 nu 8): median iterations / function evaluations "
             + "; ".join(f"{m} {fit_summary(fits, m)['nit']:.0f}/{fit_summary(fits, m)['nfev']:.0f}" for m in _FB))
    L.append(f"HAR floor hits (forecast < 0.1 x window second moment): h=1 {df['har_floor1'].mean():.2%}, h=21 {df['har_floor21'].mean():.2%} of rows"
             if len(df) else "HAR floor hits: n/a")
    inc = df.loc[~np.isfinite(df["y21"]), "year"] if len(df) else []
    per = {} if not len(df) else inc.value_counts().sort_index().to_dict()
    L.append(f"incomplete h=21 windows dropped: {int(sum(per.values())):,} of {len(df):,} forecast rows; by year " + (" ".join(f"{y}={n}" for y, n in per.items()) or "none"))
    for h in HORIZONS:
        bad = int((np.isfinite(df["y1" if h == 1 else "y21"].to_numpy()) & ~masks[h]).sum()) if len(df) else 0
        L.append(f"rows with a proxy but a non-finite / non-positive forecast excluded from every model's evaluation (h={h}): {bad:,}")
    L.append(f"[PASS] vectorised GARCH/GJR variance recursion equals the reference loop on a sample (max relative difference {verify_diff:.2e} <= {VERIFY_TOL:g})")
    L.append("[PASS] standardised-t density: integral and variance " + "; ".join(
        f"nu={nu:g}: {v[0]:.8f}" + (f" / {v[1]:.6f}" if np.isfinite(v[1]) else "") for nu, v in tcheck.items()))
    L.extend(notes)
    return L


def t_density_ok(tcheck: dict, tol: float = 1e-6) -> bool:
    return all(abs(one - 1.0) < tol and (not np.isfinite(var) or abs(var - 1.0) < tol) for one, var in tcheck.values())


def last_daily_date(directory):
    """Latest date in the last yearly file of the daily returns directory (datetime64[D])."""
    import glob
    import pathlib

    import pandas as pd
    files = sorted(glob.glob(str(pathlib.Path(directory) / "[0-9]*.parquet")))
    if not files:
        raise FileNotFoundError(f"no yearly parquet files in {directory}")
    return pd.to_datetime(pd.read_parquet(files[-1], columns=["dlycaldt"])["dlycaldt"]).max().to_datetime64().astype("datetime64[D]")


# ----------------------------------------------------------------------------------------------------- tickers
def ticker_summary(rows: dict, cal, years: int = 3, levs: float = 2.0) -> dict:
    """Information-only block for ONE ticker from its `run_asset` rows: mean QLIKE by model (h = 21), mean ratio forecast / realised 21-day
    variance by model over the last `years` years of data, and the implied annualised daily-reset drag (L^2 - L) sigma^2 / 2 of an L = `levs`
    product from the garch and gjr_t forecasts versus realised (sigma^2 = 252 x daily variance)."""
    import pandas as pd
    df = {k: np.asarray(v) for k, v in rows.items()}
    if len(df["tidx"]) == 0:
        return {"rows": 0}
    d = pd.DataFrame({k: df[k] for k in df if k in ("tidx", "y21") or k.startswith("f21_")})
    ok = np.isfinite(d["y21"].to_numpy())
    for m in MODELS:
        f = d[f"f21_{m}"].to_numpy()
        ok &= np.isfinite(f) & (f > 0)
    d = d[ok]
    out = {"rows": int(len(d)), "qlike21": {m: float(np.mean(qlike(d[f"f21_{m}"], d["y21"]))) if len(d) else float("nan") for m in MODELS}}
    cut = (pd.Timestamp(cal[-1]) - pd.DateOffset(years=years)).to_datetime64()
    rec = d[cal[d["tidx"].to_numpy()].astype("datetime64[ns]") > cut]
    rec = rec[rec["y21"] > 0]
    out["recent_rows"] = int(len(rec))
    out["ratio_mean"] = {m: float(np.mean(rec[f"f21_{m}"] / rec["y21"])) if len(rec) else float("nan") for m in MODELS}
    out["ratio_of_means"] = {m: float(rec[f"f21_{m}"].mean() / rec["y21"].mean()) if len(rec) else float("nan") for m in MODELS}
    k = (levs * levs - levs) / 2.0
    out["drag"] = {m: float(k * 252.0 * rec[f"f21_{m}"].mean()) if len(rec) else float("nan") for m in ("garch", "gjr_t")}
    out["drag"]["realised"] = float(k * 252.0 * rec["y21"].mean()) if len(rec) else float("nan")
    return out


def asset_worker(job):
    """pmap worker: job = (asset id, row of the return matrix or -1 for the market series, forecast ranges, run Monte Carlo). Big arrays come from
    the parallel context (`R`, `mkt`), shared copy-on-write by fork."""
    from qap.parallel import context
    aid, row, ranges, mc = job
    c = context()
    return run_asset(c["mkt"] if row < 0 else c["R"][row], ranges, aid, mc=mc)
