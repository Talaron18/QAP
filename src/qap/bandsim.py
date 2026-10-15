"""Study H (pre-registration v15): the mechanical skeleton of a band-switch and a momentum rule on simulated 2x daily-reset products.

Pure functions (numpy / pandas only). Everything here is deterministic; the placebo seeds depend on (permno, year, draw) only.

TIMING CONVENTION (exact; the tests pin it with hand-built paths)
-----------------------------------------------------------------
Index k = 0..n-1 are the closes of one sleeve-year. ret[k] is the instrument return of day k (close k-1 -> close k).
X[k] in {-1, 0, +1} is the position HELD AFTER the trades of close k, so it is the position held during day k+1 and earns ret[k+1].
A trade at close u therefore changes the position that earns from day u+1.  X[n-1] = 0 always (forced close at the last close).

  signal at close t  ->  trade at close t + lag  ->  the position first earns the instrument return of day t + lag + 1
  (primary lag = 1: trade at t+1, first return day t+2; lag 0 = same-close execution, information only)

Net sleeve return of day k, with H = X[k-1] (the position held during day k; the sleeve starts flat, so H = 0 on the first day) and
sides[k] = |X[k] - X[k-1]| (an entry or an exit is one side, a flip is two sides):
        net[k] = (1 + g[k]) (1 - cost)^sides[k] - 1,   g[k] = ret_long[k] if H = +1, ret_short[k] if H = -1, rf[k] if H = 0
so an entry-day return is rf (the sleeve was flat) less one side of cost, an exit-day return is the instrument return less one side,
a flip-day return is the instrument return of the OLD leg less two sides.  Excess return = net - rf.

Per close u the engine does, in this order:
  1. mark: a held position multiplies q (instrument value since the entry close, q = 1 at the entry close) by 1 + ret[u];
     STOP if q <= 1 - 0.20 (checked on closes, filled AT the breaching close, overshoot included): the sleeve goes flat at u, any pending
     trade is cancelled and no entry may execute at the closes u or u+1 (one trading-day pause); the earliest new entry executes at u+2 (i.e.
     from a signal at u+1 with lag 1). Signals keep being computed; re-entry is allowed.
  2. the last close of the sleeve-year: any open position is closed here (an exit with costs; 'yearend', or 'stop' when it is a stop) and
     NOTHING is entered (an entry at the final close would be closed at the same price for two sides of cost).
  3. execution of what was decided at close u - lag (R1: the desired position Z[u-lag]; R2 / R3: a pending entry or exit).
  4. signals at close u (R2 / R3 only): entry signals are evaluated only when the sleeve is flat after step 3 (signals are ignored while in
     the trade; an entry signal at the stop close or on the pause close is lost); take-profit signals only when in the trade and are
     LAGGED like every other signal (only the stop is a same-close fill).

R1 (band switch): M_t = mean of the N closes P_{t-N+1..t} of the total-return price index (INCLUDES the current close); Z_t = +1 (LONG2) if
P_t < M_t (1-d), -1 (SHORT2) if P_t >= M_t (1+d), else Z_{t-1}; Z starts neutral (0) on the first trading day of the year (only M may use
prior-year closes). The desired position at close u is Z[u-lag]; a state change flips the whole position (two sides).
R2 (momentum): signal at close t if r_t > 0 and r_{t-1} > 0 (r = the underlying's total return) -> LONG2; exit signal when the instrument has
gained +15% since the entry close (q >= 1.15); no time exit.
R3 (information only): L = 1 on the stock itself (no fee); entry signal P_t < 0.967 M_t, exit signal P_t >= 1.133 M_t, same -20% stop.

PLACEBO. The FINAL position series X (after stops, pauses, forced close) of a sleeve-year is rotated circularly by k trading days and
evaluated against the unshifted instrument returns with the formula above, the circular previous position of the first day being X[n-1]
(= 0 for the real series, so k = 0 reproduces the real net returns exactly). Fractions of time long / short / flat, the run lengths and
the number of position changes are invariants of a rotation, and sides[] are counted circularly, so the number of transitions and the
trading costs are EQUAL BY CONSTRUCTION; only the timing against the returns is broken. Stops are NOT re-applied to the shifted series.
k is drawn uniformly from [63, n - 63] (circular distance at least 63 days both ways; a sleeve-year shorter than 126 days uses
[n//2, n - n//2], counted in CHECKS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qap import volfc as V

PREREG = "v15"
MARKER = "data/holdout_v15.done"
STUDY = "v15 (Study H)"
DEV, VAL, HOLD = V.DEV, V.VAL, V.HOLD
FIRST_YEAR = 1995
TOP_N = 300
TOP_N_INFO = 100
MIN_VOL = 0.45
MIN_OBS = 200
VOL_DAYS = 252
ANN = 252
FEE = 0.015
COST = 0.0010
STOP = 0.20
TP_R2 = 0.15
BAND_D = 0.059
BAND_N = 21
R3_BUY = 0.967
R3_TP = 1.133
NW_LAGS = 10
DRAWS = 200
MIN_SHIFT = 63
SEED_BASE = 15
T_MIN = 2.0
EPS = 1e-12
MIN_PAIR_DAYS = 60
SLOPE_TOL = 0.15
GAP_TOL = 0.15

PAIRS = (("SNXX", "SNDK", 2.0), ("LABX", "ALAB", 2.0), ("LITX", "LITE", 2.0), ("SKHX", "SKHY", 2.0),
         ("SNDQ", "SNDK", -2.0), ("DAMD", "AMD", -2.0), ("SKDD", "SKHY", -2.0), ("AXTQ", "AXTI", -2.0))
USER_NAMES = ("SNDK", "ALAB", "LITE", "SKHY", "AMD", "INTU", "STX", "ORCL", "CELH", "BE", "AXTI")
REASONS = ("stop", "tp", "flip", "yearend", "signal")


@dataclass(frozen=True)
class Params:
    """One simulation configuration (a sensitivity is just another Params)."""
    rule: str = "R1"
    fee: float = FEE
    cost: float = COST
    lag: int = 1
    d: float = BAND_D
    n: int = BAND_N
    tp: float = TP_R2
    stop: float = STOP
    lev: float = 2.0
    buy: float = R3_BUY
    sell: float = R3_TP


def primary(rule: str) -> Params:
    """The pre-registered ONE parameter set of a rule."""
    return r3_params() if rule == "R3" else Params(rule=rule)


def r3_params(**kw) -> Params:
    return Params(rule="R3", fee=0.0, lev=1.0, **kw)


# ----------------------------------------------------------------------------------------------------------- instruments
def instrument_returns(r, rf, L: float, fee: float, dead_stays: bool = True) -> np.ndarray:
    """Daily return of a daily-reset product: L r + (1 - L) rf - fee / 252, floored at -1 (value 0). NaN r or rf count as 0 (callers count them).
    dead_stays: once the value is 0 (return <= -1) every later return is 0."""
    r = np.nan_to_num(np.asarray(r, dtype="float64"))
    rf = np.nan_to_num(np.asarray(rf, dtype="float64"))
    x = np.maximum(L * r + (1.0 - L) * rf - fee / ANN, -1.0)
    if dead_stays and len(x):
        dead = np.flatnonzero(x <= -1.0)
        if len(dead):
            x[dead[0] + 1:] = 0.0
    return x


def instrument_value(ret) -> np.ndarray:
    """Cumulative value path (start 1) of a return series; stays 0 once 0."""
    return np.cumprod(1.0 + np.asarray(ret, dtype="float64"))


def price_index(r) -> np.ndarray:
    """Cumulative product of (1 + r) (total-return price index). NaN before the first finite return; an interior NaN return is a flat day."""
    r = np.asarray(r, dtype="float64")
    P = np.cumprod(np.maximum(1.0 + np.nan_to_num(r), 0.0))
    fin = np.flatnonzero(np.isfinite(r))
    if len(fin) == 0:
        return np.full(len(r), np.nan)
    P[:fin[0]] = np.nan
    return P


def rolling_mean(P, a: int, b: int, n: int) -> np.ndarray:
    """M_t for t = a..b: mean of the n closes P[t-n+1..t] (current close included); NaN when fewer than n finite closes."""
    out = np.full(b - a + 1, np.nan)
    lo = max(a - (n - 1), 0)
    seg = np.asarray(P[lo:b + 1], dtype="float64")
    if len(seg) >= n:
        w = np.lib.stride_tricks.sliding_window_view(seg, n).mean(axis=1)
        end = lo + np.arange(len(w)) + n - 1
        sel = end >= a
        out[end[sel] - a] = w[sel]
    return out


def ffill_state(raw) -> np.ndarray:
    """Hysteresis: the last non-zero entry of `raw` carried forward (0 until the first one)."""
    raw = np.asarray(raw)
    idx = np.where(raw != 0, np.arange(len(raw)), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, raw[np.maximum(idx, 0)], 0).astype(np.int8)


def sleeve_signals(P, r_full, a: int, b: int, p: Params) -> dict:
    """Signal arrays for the closes a..b of a sleeve (index 0 = close a): R1 -> {'Z'}, R2 -> {'entry'}, R3 -> {'entry', 'exit_u'}."""
    if p.rule == "R2":
        r = np.asarray(r_full, dtype="float64")
        cur = r[a:b + 1]
        prev = np.concatenate([[r[a - 1] if a >= 1 else np.nan], r[a:b]])
        with np.errstate(invalid="ignore"):
            return {"entry": (cur > 0) & (prev > 0)}
    Ps = np.asarray(P[a:b + 1], dtype="float64")
    M = rolling_mean(P, a, b, p.n)
    with np.errstate(invalid="ignore"):
        if p.rule == "R1":
            raw = np.where(Ps < M * (1.0 - p.d), 1, np.where(Ps >= M * (1.0 + p.d), -1, 0))
            return {"Z": ffill_state(raw)}
        if p.rule == "R3":
            return {"entry": Ps < p.buy * M, "exit_u": Ps >= p.sell * M}
    raise ValueError(f"unknown rule {p.rule!r}")


# --------------------------------------------------------------------------------------------------------- state machine
def simulate(p: Params, rL, rS, Z=None, entry=None, exit_u=None) -> tuple:
    """Exact sequential state machine of one sleeve-year (see the module docstring for the timing convention).
    Returns (X int8 array, trades) with trades = [(side, i0, i1, reason, q)]: entry close, exit close, reason in REASONS, q = instrument
    value at exit relative to the entry close (gross of costs)."""
    n = len(rL)
    lag = int(p.lag)
    X = np.zeros(n, dtype=np.int8)
    trades = []
    lo_q = 1.0 - p.stop + EPS
    hi_q = 1.0 + p.tp - EPS
    pos, q, i0 = 0, 1.0, -1
    blocked = -1                    # entries may not execute at closes <= blocked
    pend_entry, pend_exit = 0, False
    for u in range(n):
        if pos != 0:                                                             # 1. mark and stop
            q *= 1.0 + (rL[u] if pos > 0 else rS[u])
            if q <= lo_q:
                trades.append((pos, i0, u, "stop", q))
                pos = 0
                blocked = u + 1
                pend_entry, pend_exit = 0, False
        if u == n - 1:                                                           # 2. forced close, no entry
            if pos != 0:
                trades.append((pos, i0, u, "yearend", q))
            X[u] = 0
            break
        if Z is not None:                                                        # 3. R1 execution of the desired position
            want = int(Z[u - lag]) if u - lag >= 0 else 0
            if want != pos and u > blocked:
                if pos != 0:
                    trades.append((pos, i0, u, "flip" if want != 0 else "signal", q))
                    pos = 0
                if want != 0:
                    pos, q, i0 = want, 1.0, u
        else:                                                                    # 3./4. R2 / R3
            if lag >= 1:
                if pend_exit and pos != 0:
                    trades.append((pos, i0, u, "tp", q))
                    pos = 0
                if pend_entry != 0 and pos == 0 and u > blocked:
                    pos, q, i0 = pend_entry, 1.0, u
                pend_entry, pend_exit = 0, False
            if pos == 0:
                if entry[u]:
                    if lag >= 1:
                        pend_entry = 1
                    elif u > blocked:
                        pos, q, i0 = 1, 1.0, u
            else:
                hit = (q >= hi_q) if p.rule == "R2" else bool(exit_u[u])
                if hit:
                    if lag >= 1:
                        pend_exit = True
                    else:
                        trades.append((pos, i0, u, "tp", q))
                        pos = 0
        X[u] = pos
    return X, trades


# ------------------------------------------------------------------------------------------------------------ evaluation
def evaluate_positions(X, rL, rS, rf, cost: float, ks=None) -> tuple:
    """(excess, net), both (D, n): the formula of the module docstring for the position series X rotated by each k in `ks` (default [0] = the
    real series). rL / rS / rf are the unshifted per-day instrument returns and risk-free rate of the sleeve. A held position whose
    value reached 0 stays at 0 until the position changes."""
    X = np.asarray(X, dtype=np.int8)
    n = len(X)
    ks = np.zeros(1, dtype=np.int64) if ks is None else np.asarray(ks, dtype=np.int64)
    idx = (np.arange(n)[None, :] + ks[:, None]) % n
    Y = X[idx]
    Yp = np.roll(Y, 1, axis=1)
    sides = np.abs(Y.astype(np.int16) - Yp.astype(np.int16))
    rL, rS, rf = (np.asarray(a, dtype="float64") for a in (rL, rS, rf))
    g = np.where(Yp > 0, rL[None, :], np.where(Yp < 0, rS[None, :], rf[None, :]))
    bad = (Yp != 0) & (g <= -1.0)
    if bad.any():
        for d in np.flatnonzero(bad.any(axis=1)):
            dead = False
            for j in range(n):
                if j > 0 and Yp[d, j] != Yp[d, j - 1]:
                    dead = False
                if Yp[d, j] != 0:
                    if dead:
                        g[d, j] = 0.0
                    elif g[d, j] <= -1.0:
                        dead = True
    net = (1.0 + g) * (1.0 - cost) ** sides - 1.0
    return net - rf[None, :], net


def placebo_shifts(permno: int, year: int, n: int, draws: int, min_shift: int = MIN_SHIFT) -> np.ndarray:
    """Circular shifts k, one per draw, from default_rng([SEED_BASE, permno, year, draw]); k in [m, n - m], m = min(63, n // 2)."""
    m = min(int(min_shift), n // 2)
    ks = np.zeros(int(draws), dtype=np.int64)
    if n < 2:
        return ks
    for d in range(int(draws)):
        ks[d] = np.random.default_rng([SEED_BASE, int(permno), int(year), d]).integers(m, n - m + 1)
    return ks


def max_drawdown(net) -> float:
    eq = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(net, dtype="float64"))])
    return float(np.min(eq / np.maximum.accumulate(eq) - 1.0))


def trade_returns(trades, cost: float) -> np.ndarray:
    """Net return of each trade: instrument value ratio less one entry and one exit side."""
    return np.array([t[4] * (1.0 - cost) ** 2 - 1.0 for t in trades], dtype="float64")


# --------------------------------------------------------------------------------------------------------------- universe
def ranked_candidates(panel, year: int, n: int) -> np.ndarray:
    """Permnos of the n largest companies (permco) by prior-December company ME among common equity, in rank order (same selection as
    volfc.top_universe; the representative permno of a company is the one with the largest `me`)."""
    d = panel.loc[(panel["mi"] == (int(year) - 1) * 12 + 11) & panel["eq"] & (panel["me_co"] > 0), ["permno", "permco", "me", "me_co"]]
    d = d.sort_values(["permco", "me", "permno"], ascending=[True, False, True], kind="mergesort").drop_duplicates("permco", keep="first")
    d = d.sort_values(["me_co", "permco"], ascending=[False, True], kind="mergesort").head(n)
    return d["permno"].to_numpy("int64")


def dec_index(cal, year: int) -> int:
    """Calendar index of the last trading day of December of year - 1 (-1 when there is none)."""
    return int(np.searchsorted(cal, np.datetime64(f"{int(year)}-01-01"), side="left")) - 1


def trailing_vol(R, rows, e: int, days: int | None = None) -> tuple:
    """(annualised volatility, number of non-missing returns) over the `days` calendar indices ending at e (inclusive) for each row of R:
    sample standard deviation (ddof = 1) x sqrt(252) of the non-missing returns; NaN with fewer than 2."""
    days = VOL_DAYS if days is None else days
    rows = np.asarray(rows, dtype=np.int64)
    if e < 0 or len(rows) == 0:
        return np.full(len(rows), np.nan), np.zeros(len(rows), dtype=np.int64)
    w = R[rows, max(e - days + 1, 0):e + 1]
    fin = np.isfinite(w)
    cnt = fin.sum(axis=1)
    mean = np.where(cnt > 0, np.where(fin, w, 0.0).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    ss = (np.where(fin, w - mean[:, None], 0.0) ** 2).sum(axis=1)
    vol = np.where(cnt >= 2, np.sqrt(ss / np.maximum(cnt - 1, 1)) * math.sqrt(ANN), np.nan)
    return vol, cnt


def select_universe(panel, years, cal, R, row_of: dict, top_n: int | None = None, min_vol: float | None = None, min_obs: int | None = None,
                    days: int | None = None, assets: int | None = None) -> dict:
    """{year: dict(permno=array in ME rank order, vol=array, cands=int, obs_ok=int, top100=array)}: per calendar year Y the `top_n` companies
    by prior-December company ME, kept when the trailing-252-day annualised volatility at the last trading day of December has >= `min_obs`
    non-missing returns and is >= `min_vol`. No survivor filter. `assets` (quick runs) keeps only the first N of each list."""
    top_n = TOP_N if top_n is None else top_n
    min_vol = MIN_VOL if min_vol is None else min_vol
    min_obs = MIN_OBS if min_obs is None else min_obs
    out = {}
    for y in years:
        cand = ranked_candidates(panel, y, top_n)
        e = dec_index(cal, y)
        rows = np.array([row_of.get(int(p), -1) for p in cand], dtype=np.int64)
        known = rows >= 0
        vol = np.full(len(cand), np.nan)
        cnt = np.zeros(len(cand), dtype=np.int64)
        if known.any():
            vol[known], cnt[known] = trailing_vol(R, rows[known], e, days)
        ok = (cnt >= min_obs) & (vol >= min_vol)
        keep = np.flatnonzero(ok)
        top100 = ranked_candidates(panel, y, TOP_N_INFO)
        if assets is not None:
            keep, top100 = keep[:int(assets)], top100[:int(assets)]
        out[int(y)] = dict(permno=cand[keep], vol=vol[keep], cands=int(len(cand)), obs_ok=int((cnt >= min_obs).sum()), top100=top100)
    return out


def terciles(universe: dict) -> tuple:
    """Pooled 1/3 and 2/3 quantiles of the trailing volatility over all sleeve-years of `universe`."""
    v = np.concatenate([u["vol"] for u in universe.values()]) if universe else np.empty(0)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))


# ------------------------------------------------------------------------------------------------------ loaders / series
def load_daily(directory, permnos, max_year: int | None = None) -> tuple:
    """Like volfc.load_daily_subset, but yearly files after `max_year` are never read (sealed runs). Returns (R matrix with one row per sorted
    permno, sorted permno array, calendar of ALL dates read as datetime64[D], {year: rows})."""
    import glob
    import pathlib
    keep = np.asarray(sorted(set(int(p) for p in permnos)), dtype="int64")
    parts, rows, days = [], {}, []
    for f in sorted(glob.glob(str(pathlib.Path(directory) / "[0-9]*.parquet"))):
        y = int(pathlib.Path(f).stem)
        if max_year is not None and y > max_year:
            continue
        d = pd.read_parquet(f, columns=["permno", "dlycaldt", "dlyret"])
        rows[y] = len(d)
        d["dlycaldt"] = pd.to_datetime(d["dlycaldt"])
        days.append(d["dlycaldt"].drop_duplicates().to_numpy().astype("datetime64[D]"))
        parts.append(d[np.isin(d["permno"].to_numpy("int64"), keep)].astype({"permno": "int64", "dlyret": "float64"}))
    cal = np.unique(np.concatenate(days)) if days else np.empty(0, "datetime64[D]")
    sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"permno": [], "dlycaldt": [], "dlyret": []})
    return V.return_matrix(sub, cal, keep), keep, cal, rows


def rf_series(ff, dates) -> np.ndarray:
    """FF daily rf (decimals via earnings.ff_decimals) on `dates`; NaN where FF has no row for the exact date, except days after the last FF row, which carry the last rf forward (so 2x inverse products
    are not under-financed on the 2026 Compustat days of the replication check)."""
    from qap.earnings import ff_decimals
    f = ff_decimals(ff)
    fd = f["date"].to_numpy().astype("datetime64[D]")
    dates = np.asarray(dates).astype("datetime64[D]")
    out = np.full(len(dates), np.nan)
    if len(fd) == 0 or len(dates) == 0:
        return out
    pos = np.searchsorted(fd, dates)
    pc = np.minimum(pos, len(fd) - 1)
    hit = (pos < len(fd)) & (fd[pc] == dates)
    out[hit] = f["rf"].to_numpy()[pc[hit]]
    beyond = dates > fd[-1]                       # days AFTER the last FF row (Compustat prices run past the FF file): carry the last rf forward
    last_rf = f["rf"].to_numpy()[-1]
    if beyond.any() and np.isfinite(last_rf):
        out[beyond] = last_rf
    return out


# ---------------------------------------------------------------------------------------------------- aggregation worker
@dataclass(frozen=True)
class Spec:
    """A configuration evaluated by the year worker. kind: 'rule' | 'bh_u' | 'bh_l' | 'bh_s' (buy-and-hold underlying / 2x long / 2x inverse)."""
    name: str
    kind: str = "rule"
    params: Params = Params()
    sets: tuple = ("main",)
    placebo: bool = False
    groups: bool = False


def _new_acc(nd: int, draws: int, s: Spec) -> dict:
    return dict(S=np.zeros(nd), C=np.zeros(nd, dtype=np.int64), SP=np.zeros((draws, nd)) if s.placebo else None,
                G={g: (np.zeros(nd), np.zeros(nd, dtype=np.int64)) for g in range(3)} if s.groups else None,
                trades=[], dd=[], sleeves=0, short=0, days=np.zeros(3, dtype=np.int64), trans=0, miss_r=0, miss_rf=0, sleeve_days=0)


def run_spec(s: Spec, P, r_full, a: int, b: int, r, rfs):
    """(X, rL, rS, trades) of one sleeve for one spec (r, rfs: NaN-free slices a..b)."""
    p = s.params
    n = b - a + 1
    if s.kind == "rule":
        rL = instrument_returns(r, rfs, p.lev, p.fee, False)
        rS = instrument_returns(r, rfs, -p.lev, p.fee, False) if p.rule == "R1" else np.zeros(n)
        X, trades = simulate(p, rL, rS, **sleeve_signals(P, r_full, a, b, p))
        return X, rL, rS, trades
    X = np.ones(n, dtype=np.int8) * (-1 if s.kind == "bh_s" else 1)
    X[-1] = 0
    if s.kind == "bh_u":
        return X, instrument_returns(r, rfs, 1.0, 0.0, False), np.zeros(n), []
    return X, instrument_returns(r, rfs, p.lev, p.fee, False), instrument_returns(r, rfs, -p.lev, p.fee, False), []


def year_worker(year: int) -> dict:
    """pmap worker: every spec over every sleeve of one calendar year. Big inputs come from the parallel context (R, rf, ranges, sleeves, specs,
    draws). Sleeves are processed in their stored order, so the per-year arrays are the same whatever --jobs is."""
    from qap.parallel import context
    c = context()
    R, rf, draws = c["R"], c["rf"], c["draws"]
    lo, hi = c["ranges"][year]
    nd = hi - lo + 1
    out = {}
    for set_name, sleeves in c["sleeves"][year].items():
        specs = [s for s in c["specs"] if set_name in s.sets]
        acc = {s.name: _new_acc(nd, draws, s) for s in specs}
        for permno, row, terc in sleeves:
            r_full = R[row]
            fin = np.flatnonzero(np.isfinite(r_full[lo:hi + 1]))
            if len(fin) == 0:
                continue
            a, b = lo + int(fin[0]), lo + int(fin[-1])
            n = b - a + 1
            P = price_index(r_full[:b + 1])
            r = np.nan_to_num(r_full[a:b + 1])
            rfv = rf[a:b + 1]
            rfs = np.nan_to_num(rfv)
            ks = None
            sl = slice(a - lo, b - lo + 1)
            for s in specs:
                A = acc[s.name]
                X, rL, rS, trades = run_spec(s, P, r_full, a, b, r, rfs)
                ex, net = evaluate_positions(X, rL, rS, rfs, s.params.cost)
                A["S"][sl] += ex[0]
                A["C"][sl] += 1
                if A["SP"] is not None:
                    if ks is None:
                        ks = placebo_shifts(permno, year, n, draws)
                    A["SP"][:, sl] += evaluate_positions(X, rL, rS, rfs, s.params.cost, ks)[0]
                if A["G"] is not None:
                    A["G"][terc][0][sl] += ex[0]
                    A["G"][terc][1][sl] += 1
                A["sleeves"] += 1
                A["short"] += int(n < 2 * MIN_SHIFT)
                A["sleeve_days"] += n
                A["miss_r"] += int(n - np.isfinite(r_full[a:b + 1]).sum())
                A["miss_rf"] += int(np.isnan(rfv).sum())
                held = np.roll(X, 1)
                A["days"] += np.array([int((held == 0).sum()), int((held > 0).sum()), int((held < 0).sum())])
                A["trans"] += int((X != held).sum())
                A["trades"] += [(t[0], t[2] - t[1], REASONS.index(t[3]), t[4] * (1.0 - s.params.cost) ** 2 - 1.0) for t in trades]
                if s.kind == "rule":
                    A["dd"].append((terc, max_drawdown(net[0])))
        out[set_name] = acc
    return {"year": int(year), "lo": lo, "nd": nd, "acc": out}


def combine(results: list, T: int, draws: int) -> dict:
    """{(set, name): dict(S, C, SP, G, trades, dd, counters, by_year)} summed over the per-year results (years are disjoint date ranges, the
    additions happen in the order of `results`, so the outcome does not depend on how many processes produced them)."""
    agg = {}
    for res in results:
        lo, nd = res["lo"], res["nd"]
        for set_name, accs in res["acc"].items():
            for name, A in accs.items():
                g = agg.setdefault((set_name, name), dict(
                    S=np.zeros(T), C=np.zeros(T, dtype=np.int64), SP=None if A["SP"] is None else np.zeros((draws, T)),
                    G=None if A["G"] is None else {k: (np.zeros(T), np.zeros(T, dtype=np.int64)) for k in range(3)},
                    trades=[], dd=[], sleeves=0, short=0, days=np.zeros(3, dtype=np.int64), trans=0, miss_r=0, miss_rf=0, sleeve_days=0,
                    by_year={}))
                sl = slice(lo, lo + nd)
                g["S"][sl] += A["S"]
                g["C"][sl] += A["C"]
                if A["SP"] is not None:
                    g["SP"][:, sl] += A["SP"]
                if A["G"] is not None:
                    for k in range(3):
                        g["G"][k][0][sl] += A["G"][k][0]
                        g["G"][k][1][sl] += A["G"][k][1]
                g["trades"] += A["trades"]
                g["dd"] += A["dd"]
                for key in ("sleeves", "short", "trans", "miss_r", "miss_rf", "sleeve_days"):
                    g[key] += A[key]
                g["days"] = g["days"] + A["days"]
                g["by_year"][res["year"]] = (A["trans"], A["sleeves"])
    return agg


# -------------------------------------------------------------------------------------------------------------- statistics
def window_mask(cal, window) -> np.ndarray:
    return (cal >= np.datetime64(window[0])) & (cal <= np.datetime64(window[1]))


def daily_excess(S, C) -> np.ndarray:
    """Per-date cross-sectional mean excess return over the active sleeves; NaN where no sleeve is active."""
    return np.where(C > 0, S / np.maximum(C, 1), np.nan)


def window_stat(S, C, cal, window, lags: int = NW_LAGS) -> dict:
    """Mean daily cross-sectional excess return and its Newey-West t (qap.volfc.newey_west_t, `lags` lags) over the dates of `window`."""
    m = window_mask(cal, window) & (C > 0)
    d = daily_excess(S, C)[m]
    mean, t = V.newey_west_t(d, lags) if len(d) else (float("nan"), float("nan"))
    return dict(mean=mean, t=t, n=int(m.sum()))


def placebo_stats(SP, C, cal, window) -> np.ndarray:
    """The same window statistic (mean daily cross-sectional excess) for every placebo draw."""
    m = window_mask(cal, window) & (C > 0)
    if SP is None or not m.any():
        return np.empty(0)
    return (SP[:, m] / C[m][None, :]).mean(axis=1)


def placebo_summary(draws_stat) -> dict:
    x = np.asarray(draws_stat, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(p95=float("nan"), median=float("nan"), n=0)
    return dict(p95=float(np.percentile(x, 95)), median=float(np.percentile(x, 50)), n=int(len(x)))


def licence_verdict(dev: dict, val: dict, plc: dict, hold: dict | None = None, plc_hold: dict | None = None, open_holdout: bool = False) -> dict:
    """Pre-registered licence of ONE rule. dev / val / hold: {'mean', 't'}; plc: placebo_summary of the dev statistic, plc_hold of the holdout
    statistic. Dev: mean > 0 AND t >= 2 AND mean > the placebo 95th percentile; validation: mean > 0. Only then may the holdout be opened
    (and only with open_holdout): the licence stands only if the holdout mean > 0 AND above the placebo median. NaN never passes."""
    c = {"dev mean > 0": bool(dev["mean"] > 0), "dev t >= 2.0": bool(dev["t"] >= T_MIN), "dev > placebo 95th pct": bool(dev["mean"] > plc["p95"]),
         "validation mean > 0": bool(val["mean"] > 0)}
    dev_val = all(c.values())
    opened = bool(open_holdout and dev_val and hold is not None)
    stands = None
    if opened:
        stands = bool(hold["mean"] > 0 and plc_hold is not None and hold["mean"] > plc_hold["median"])
    if not dev_val:
        status = "NOT LICENSED (failed: " + ", ".join(k for k, v in c.items() if not v) + ")"
    elif not opened:
        status = "PASSED IN DEV/VALIDATION: holdout sealed (opens once, only with --open-holdout)"
    else:
        status = "LICENSED (holdout stands)" if stands else "FAILED IN THE HOLDOUT (recorded as passed in dev/validation, failed in the holdout)"
    return dict(conditions=c, dev_val=dev_val, opened=opened, stands=stands, status=status)


# ------------------------------------------------------------------------------------------------------------ trial spec
def trial_spec(rule: str) -> dict:
    p = primary(rule)
    spec = {"prereg": PREREG, "study": "H", "rule": rule, "fee": p.fee, "cost_per_side": p.cost, "lag": p.lag, "stop": p.stop,
            "universe": f"top{TOP_N} prior-December company ME, trailing-252d vol >= {MIN_VOL}", "placebo": f"circular shift >= {MIN_SHIFT}d, {DRAWS} draws",
            "licence": "dev mean>0, NW10 t>=2, > placebo p95, validation>0; holdout once: >0 and > placebo median"}
    if rule == "R1":
        spec.update({"N": p.n, "d": p.d, "L": [2, -2]})
    else:
        spec.update({"entry": "two consecutive up closes", "take_profit": p.tp, "L": [2]})
    return spec


# ------------------------------------------------------------------------------------------------------ replication check
def replication_check(frame, ff, pairs=PAIRS, fee: float = FEE, min_days: int = MIN_PAIR_DAYS, slope_tol: float = SLOPE_TOL,
                      gap_tol: float = GAP_TOL) -> dict:
    """Simulated product versus the actual ETF on the common days of each pair (frame = qap.prices.prepare_prices output). Per pair with >= min_days
    common returns: slope of the ETF's daily return on the underlying's (must be within slope_tol x |L| of L), tracking error, and the cumulative
    log gap actual - simulated per calendar year of common data (the simulation uses the SAME formula incl. rf and fee). ok = every used pair
    has a slope in range AND the median over all (pair, year) cells of |gap| <= gap_tol AND at least one pair was usable."""
    from qap import prices as PR
    rows, cells, reasons = [], [], []
    for etf, und, L in pairs:
        e, u = PR.return_series(frame, etf), PR.return_series(frame, und)
        j = pd.concat([e.rename("e"), u.rename("u")], axis=1, join="inner").dropna()
        if len(j) < min_days:
            rows.append(dict(etf=etf, und=und, L=L, n=int(len(j)), used=False))
            continue
        chk = PR.leveraged_check(e, u, L)
        rfv = rf_series(ff, j.index.to_numpy())
        sim = instrument_returns(j["u"].to_numpy(), np.nan_to_num(rfv), L, fee)
        gap = np.log1p(np.maximum(j["e"].to_numpy(), -1 + 1e-12)) - np.log1p(np.maximum(sim, -1 + 1e-12))
        yrs = j.index.year.to_numpy()
        per_year = []
        for k, yv in enumerate(sorted(set(yrs))):
            m = yrs == yv
            per_year.append(dict(rank=k + 1, year=int(yv), days=int(m.sum()), gap=float(gap[m].sum())))
            cells.append(abs(float(gap[m].sum())))
        slope_ok = bool(abs(chk["slope"] - L) <= slope_tol * abs(L))
        if not slope_ok:
            reasons.append(f"{etf}/{und}: slope {chk['slope']:.3f} is not within {slope_tol:.0%} of L = {L:+.0f}")
        rows.append(dict(etf=etf, und=und, L=L, n=int(len(j)), used=True, slope=float(chk["slope"]), te_ann=float(chk["te_ann"]), slope_ok=slope_ok,
                         rf_missing=int(np.isnan(rfv).sum()), years=per_year))
    used = [r for r in rows if r["used"]]
    med = float(np.median(cells)) if cells else float("nan")
    if not used:
        reasons.append(f"no pair has >= {min_days} common days (nothing to check)")
    elif not med <= gap_tol:
        reasons.append(f"median absolute cumulative log gap per year {med:.3f} exceeds {gap_tol}")
    return dict(ok=not reasons, reasons=reasons, rows=rows, median_abs_gap=med, cells=len(cells))


# --------------------------------------------------------------------------------------------------- the user's own names
def name_sleeve(r, rfv, p: Params) -> dict | None:
    """One sleeve over the whole available history of a name (r: its total-return series, rfv: rf on the same dates). None when the history is
    too short for the anchor. Returns trades, win rate, sleeve total return (net, rf while flat) and buy-and-hold of the underlying."""
    r = np.nan_to_num(np.asarray(r, dtype="float64"))
    n = len(r)
    if n < max(p.n, 3) + 2:
        return None
    rfs = np.nan_to_num(np.asarray(rfv, dtype="float64"))
    P = price_index(r)
    rL = instrument_returns(r, rfs, p.lev, p.fee, False)
    rS = instrument_returns(r, rfs, -p.lev, p.fee, False) if p.rule == "R1" else np.zeros(n)
    X, trades = simulate(p, rL, rS, **sleeve_signals(P, r, 0, n - 1, p))
    _, net = evaluate_positions(X, rL, rS, rfs, p.cost)
    tr = trade_returns(trades, p.cost)
    return dict(trades=len(trades), win=float((tr > 0).mean()) if len(tr) else float("nan"), total=float(np.prod(1.0 + net[0]) - 1.0),
                bh=float(np.prod(1.0 + r) - 1.0), stops=int(sum(t[3] == "stop" for t in trades)))
