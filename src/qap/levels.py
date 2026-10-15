"""Price-level decision support: odds behind buy / take-profit / stop / account-stop rules. NOT a predictor.

A Monte Carlo of the underlying's daily returns (GJR-GARCH(1,1)-t from qap.volfc when the ticker has >= 750 returns, else a flat-volatility
Student-t(5) fallback), zero mean unless a drift is given. A leveraged / inverse daily-reset product is the exact recursion
V_t = V_{t-1} (1 + L r_t), floored at 0, driven by the underlying's returns.

Level arguments: a value ending in '%' is a return on the INSTRUMENT (+15%, -20%); a plain number is a PRICE of the UNDERLYING stock.
Levels are treated as limit / stop orders that fill AT the level:
 * default (bridge): a level is hit when a daily close crosses it or, when both closes of a day lie on the same side of it, when the Brownian
   bridge of the underlying's log price touches it, with probability exp(-2 ln(b/S_t) ln(b/S_t+1) / sigma_day^2) (sigma_day = that step's
   conditional vol), decided by an independent uniform draw. An instrument-return barrier of a leveraged product is mapped into the underlying
   within the day (S* = S_{t-1} (1 + (b/V_{t-1} - 1)/L)). This keeps a driftless stock a martingale: expected trade return about 0 before costs.
 * --close-only: levels are checked on closes only and the fill is the instrument's value at the trigger close (no intraday touches).
 * a price level on a leveraged product triggers on the UNDERLYING and fills at the instrument's value when the underlying is at the level.
If target and stop are hit on the same day the stop is assumed first. Not modelled: overshoot past a level (fills are exactly at the level),
gaps through a stop (see the worst-day statistic in the report), costs and slippage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qap import prices as px
from qap import volfc

MIN_GJR = 750
FIT_WINDOW = 1000
FALLBACK_NU = 5.0
QUANTILES = (5, 25, 50, 75, 95)
CHUNK_CELLS = 2_000_000       # paths x horizon per chunk (memory bound)
EPS = 1e-12


# ----------------------------------------------------------------------------------------------- arguments
@dataclass(frozen=True)
class Level:
    kind: str      # 'ret' (instrument return, fraction) or 'price' (underlying price)
    value: float


def parse_level(text) -> Level | None:
    """'15%' -> Level('ret', 0.15) ; '-20%' -> ('ret', -0.2) ; '48.5' -> ('price', 48.5) ; None -> None."""
    if text is None:
        return None
    t = str(text).strip().replace("$", "").replace(",", "")
    if t.endswith("%"):
        return Level("ret", float(t[:-1]) / 100.0)
    v = float(t)
    if v <= 0:
        raise ValueError(f"a price level must be positive (use a trailing % for returns): {text!r}")
    return Level("price", v)


# ----------------------------------------------------------------------------------------------- model
@dataclass
class Model:
    kind: str                  # 'gjr' or 'fallback'
    n_returns: int
    sigma_annual: float        # unconditional (gjr) or EWMA / sample (fallback)
    note: str
    params: dict


def build_model(returns, min_gjr: int = MIN_GJR) -> Model:
    """GJR-GARCH-t on the last min(n, 1000) returns when n >= 750 and the fit is 'ok'; else flat Student-t(5) at best_vol."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    why = f"only {n} returns (< {min_gjr})"
    if n >= min_gjr:
        w = r[-min(n, FIT_WINDOW):]
        f = volfc.fit_gjr_t(w)
        if f["status"] == "ok":
            s2 = f["s2"]
            nxt = float(volfc.sigma2_path(w, f["alpha"], f["gamma"], f["beta"], s2)[-1])
            p = dict(alpha=f["alpha"], gamma=f["gamma"], beta=f["beta"], nu=f["nu"], persistence=f["p"], s2=s2,
                     omega=(1.0 - f["p"]) * s2, next_var=nxt, window=len(w))
            return Model("gjr", n, math.sqrt(252 * s2),
                         f"GJR-GARCH(1,1)-t fitted on the last {len(w)} returns", p)
        why = f"GJR-t fit status '{f['status']}'"
    v, k, lab = px.best_vol(r)
    if lab == "insufficient":
        raise ValueError(f"only {k} usable returns: need at least {px.MIN_SAMPLE} for any volatility estimate")
    return Model("fallback", n, v, f"model: fallback, no leverage effect ({why}); flat volatility {v:.1%}/yr ({lab}, {k} returns), "
                 f"Student-t({FALLBACK_NU:g}) daily returns", dict(nu=FALLBACK_NU, label=lab, window=k))


def draw_paths(model: Model, n: int, horizon: int, rng, drift_annual: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """((n, horizon) simple daily returns floored at -1, (n, horizon) conditional daily vol of each step).

    Zero mean (before the optional drift); shocks are standardised so the step variance is exactly sigma_t^2."""
    out = np.empty((n, horizon))
    sig = np.empty((n, horizon))
    mu = drift_annual / 252.0
    if model.kind == "fallback":
        nu = model.params["nu"]
        sd = model.sigma_annual / math.sqrt(252)
        out[:] = sd * rng.standard_t(nu, (n, horizon)) * math.sqrt((nu - 2.0) / nu)
        sig[:] = sd
    else:
        p = model.params
        nu = p["nu"]
        scale = math.sqrt((nu - 2.0) / nu)
        var = np.full(n, p["next_var"])
        for k in range(horizon):
            sig[:, k] = np.sqrt(var)
            r = sig[:, k] * rng.standard_t(nu, n) * scale
            out[:, k] = r
            var = p["omega"] + (p["alpha"] + p["gamma"] * (r < 0)) * r * r + p["beta"] * var
    out += mu
    return np.maximum(out, -1.0), sig


def draw_returns(model: Model, n: int, horizon: int, rng, drift_annual: float = 0.0) -> np.ndarray:
    """(n, horizon) simple daily returns (see draw_paths)."""
    return draw_paths(model, n, horizon, rng, drift_annual)[0]


def instrument_value(R: np.ndarray, leverage: float) -> np.ndarray:
    """Exact daily-reset recursion V_t = V_{t-1} (1 + L r_t), floored at 0 (and absorbing there). V_0 = 1 is not included."""
    return np.cumprod(np.maximum(1.0 + leverage * R, 0.0), axis=1)


# ----------------------------------------------------------------------------------------------- path statistics
def _first_hit(mask: np.ndarray) -> np.ndarray:
    """1-based index of the first True per row, 0 when none."""
    anyhit = mask.any(axis=1)
    return np.where(anyhit, mask.argmax(axis=1) + 1, 0)


def _close_cross(level: Level, S: np.ndarray, V: np.ndarray, entry: float) -> np.ndarray:
    """Close-based trigger matrix of a race level."""
    if level.kind == "ret":
        x = level.value
        return (V - 1.0 >= x) if x > 0 else (V - 1.0 <= x)
    return (S >= level.value) if level.value > entry else (S <= level.value)


def _bridge_touch(Sprev, S, sig, Sstar, cross, u) -> np.ndarray:
    """Hit within each day: the close crosses the barrier, or (no close-crossing: both closes on the same side) the Brownian bridge of the
    log price touches it with probability exp(-2 ln(b/S_prev) ln(b/S) / sigma_day^2), decided by the uniform draw u."""
    with np.errstate(all="ignore"):
        p = np.exp(-2.0 * np.log(Sstar / Sprev) * np.log(Sstar / S) / (sig * sig))
    p = np.where(np.isfinite(p) & (Sstar > 0) & (Sprev > 0) & (S > 0), p, 0.0)
    return cross | (~cross & (u < p))


def simulate_trade(model: Model, *, last_close: float, entry: float, target: Level | None, stop: Level | None, leverage: float,
                   horizon: int, n_paths: int, seed: int, drift_annual: float = 0.0, touch_levels: dict | None = None,
                   close_only: bool = False) -> dict:
    """Run the Monte Carlo in chunks. touch_levels: {'entry': price, ...} underlying prices for the touch table (from last_close).

    close_only=False (default): levels are limit / stop orders. A level is hit when a close crosses it or, between two closes on the same side
    of it, when the Brownian bridge of the underlying's log price touches it (see _bridge_touch; instrument-return barriers are mapped into the
    underlying within each day); the fill is AT the level. close_only=True: levels are checked on closes only and the fill is the instrument's
    value at the trigger close (so a driftless stock keeps a zero expected trade return).

    Returns per-path arrays: ret (final trade return), exit_day (0 = no exit by the horizon, horizon close used), outcome (+1 target, -1 stop,
    0 neither), shortfall (L log(S_T/S_0) - log V_T, NaN where V_T = 0), ruin, plus touch/above probabilities and mean r^2."""
    rng = np.random.default_rng(seed)
    touch_levels = touch_levels or {}
    per = max(1000, CHUNK_CELLS // max(horizon, 1))
    keys = ("ret", "exit_day", "outcome", "shortfall", "ruin")
    acc = {k: [] for k in keys}
    touch = {k: 0 for k in touch_levels}
    above = {k: 0 for k in touch_levels}
    m_r2, done = 0.0, 0
    L = leverage
    while done < n_paths:
        n = min(per, n_paths - done)
        R, SIG = draw_paths(model, n, horizon, rng, drift_annual)
        m_r2 += float((R * R).sum())
        S_last = last_close * np.cumprod(1.0 + R, axis=1)
        Sp_last = np.concatenate([np.full((n, 1), last_close), S_last[:, :-1]], axis=1)
        for k, lv in touch_levels.items():
            cross = (S_last >= lv) if lv > last_close else (S_last <= lv)
            hit = cross if close_only else _bridge_touch(Sp_last, S_last, SIG, lv, cross, rng.random((n, horizon)))
            touch[k] += int(hit.any(axis=1).sum())
            above[k] += int((S_last[:, -1] > lv).sum())
        S = entry * np.cumprod(1.0 + R, axis=1)
        V = instrument_value(R, L)
        Sprev = np.concatenate([np.full((n, 1), entry), S[:, :-1]], axis=1)
        Vprev = np.concatenate([np.ones((n, 1)), V[:, :-1]], axis=1)
        Sstar, hits = {}, {}
        for name, lvl in (("target", target), ("stop", stop)):
            if lvl is None:
                hits[name] = np.zeros((n, horizon), bool)
                continue
            cross = _close_cross(lvl, S, V, entry)
            if lvl.kind == "price":
                Sstar[name] = np.full((n, horizon), lvl.value)
            else:
                with np.errstate(all="ignore"):
                    Sstar[name] = np.where(Vprev > 0, Sprev * (1.0 + ((1.0 + lvl.value) / np.where(Vprev > 0, Vprev, 1.0) - 1.0) / L), np.nan)
            if close_only:
                hits[name] = cross
                continue
            u = rng.random((n, horizon))
            if name == "stop" and "target" in Sstar:     # barriers on the same side of the previous close share the draw (far hit => near hit)
                u = np.where((Sstar["stop"] > Sprev) == (Sstar["target"] > Sprev), u_target, u)
            else:
                u_target = u
            hits[name] = _bridge_touch(Sprev, S, SIG, Sstar[name], cross, u)
        t_hit, s_hit = _first_hit(hits["target"]), _first_hit(hits["stop"])
        big = horizon + 1
        td, sd = np.where(t_hit > 0, t_hit, big), np.where(s_hit > 0, s_hit, big)
        stop_first = (sd <= td) & (sd < big)
        tgt_first = (td < sd) & (td < big)
        day = np.where(stop_first, sd, np.where(tgt_first, td, 0))
        ret = V[:, -1] - 1.0
        for name, lvl, mask, dd in (("target", target, tgt_first, td), ("stop", stop, stop_first, sd)):
            if lvl is None or not mask.any():
                continue
            idx = np.nonzero(mask)[0]
            j = dd[idx] - 1
            if close_only:
                ret[idx] = V[idx, j] - 1.0
            elif lvl.kind == "ret":
                ret[idx] = lvl.value
            elif L == 1.0:
                ret[idx] = lvl.value / entry - 1.0
            else:   # instrument value when the underlying is at the level, from the previous close
                ret[idx] = np.maximum(Vprev[idx, j] * (1.0 + L * (lvl.value / Sprev[idx, j] - 1.0)), 0.0) - 1.0
        VT = V[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            sf = np.where(VT > 0, L * np.log(S[:, -1] / entry) - np.log(VT), np.nan)
        for k, v in zip(keys, (ret, day, np.where(stop_first, -1, np.where(tgt_first, 1, 0)), sf, VT <= 0)):
            acc[k].append(v)
        done += n
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["touch"] = {k: v / n_paths for k, v in touch.items()}
    out["above"] = {k: v / n_paths for k, v in above.items()}
    out["mean_r2"] = m_r2 / (n_paths * horizon)
    out["n_paths"], out["horizon"], out["close_only"] = n_paths, horizon, close_only
    return out


# ----------------------------------------------------------------------------------------------- analytic / account arithmetic
def drag_analytic(L: float, sigma_annual: float, horizon_days: int) -> float:
    """(L^2 - L) sigma^2 T / 2 in log terms, T = horizon / 252."""
    return (L * L - L) * sigma_annual ** 2 * (horizon_days / 252.0) / 2.0


def stop_return_estimate(stop: Level | None, leverage: float, entry: float, sim: dict) -> float:
    """Instrument return at the stop: exact for a return level or a price level with L = 1, else the simulated mean at stop exits."""
    if stop is None:
        return float("nan")
    if stop.kind == "ret":
        return stop.value
    if leverage == 1.0:
        return stop.value / entry - 1.0
    m = sim["outcome"] == -1
    return float(sim["ret"][m].mean()) if m.any() else float("nan")


def size_check(size_frac: float, account: float, account_stop: float, stop_ret: float, final_ret: np.ndarray,
               worst_day: float | None = None) -> dict:
    """Account arithmetic for one trade of `size_frac` of the account. Loss fractions are of the whole account."""
    loss_share = size_frac * abs(stop_ret)
    thr = account_stop / size_frac if size_frac > 0 else float("inf")           # trade return that costs exactly the account stop
    out = dict(size_frac=size_frac, position_usd=size_frac * account, loss_at_stop_share=loss_share,
               loss_at_stop_usd=loss_share * account, account_stop_usd=account_stop * account, trade_loss_threshold=thr,
               p_account_stop=float(np.mean(final_ret < -thr - EPS)),
               max_size=min(1.0, account_stop / abs(stop_ret)) if stop_ret and abs(stop_ret) > 0 else float("nan"),
               worst_day=worst_day, max_size_gap=float("nan"))
    if worst_day is not None and worst_day < 0:
        out["max_size_gap"] = min(1.0, account_stop / abs(worst_day))
    return out


def worst_day_moves(returns, leverage: float) -> tuple[float, float]:
    """(worst, best) one-day instrument return in the history: max(1 + L r, 0) - 1."""
    v = np.maximum(1.0 + leverage * np.asarray(returns, float), 0.0) - 1.0
    return float(v.min()), float(v.max())


# ----------------------------------------------------------------------------------------------- orchestration / report
def run(returns, *, ticker: str = "?", last_close: float, last_date=None, leverage: float = 1.0, entry: float | None = None,
        target=None, stop=None, horizon: int = 21, n_paths: int = 100000, drift: float = 0.0, seed: int = 12345,
        size_frac: float | None = None, account: float = 45000.0, account_stop: float = 0.15, close_only: bool = False,
        etf: dict | None = None) -> dict:
    """returns: the underlying's daily returns (what is simulated). etf (optional): dict(ticker, underlying, returns=Series of the listed
    leveraged ETF's own history, underlying_returns=Series, note) when the instrument is a listed leveraged ETF simulated from its underlying;
    its history feeds the one-day-move statistic and a tracking line."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    tgt = target if isinstance(target, Level) or target is None else parse_level(target)
    stp = stop if isinstance(stop, Level) or stop is None else parse_level(stop)
    entry = float(entry) if entry is not None else float(last_close)
    model = build_model(r)
    touch_levels = {k: lv.value for k, lv in (("entry", Level("price", entry) if entry != last_close else None), ("target", tgt), ("stop", stp))
                    if lv is not None and lv.kind == "price" and lv.value != last_close}
    kw = dict(last_close=last_close, entry=entry, target=tgt, stop=stp, leverage=leverage, horizon=horizon, n_paths=n_paths, seed=seed,
              drift_annual=drift)
    sim = simulate_trade(model, touch_levels=touch_levels, close_only=close_only, **kw)
    res = dict(ticker=ticker, last_close=last_close, last_date=last_date, leverage=leverage, entry=entry, target=tgt, stop=stp,
               horizon=horizon, drift=drift, seed=seed, model=model, sim=sim, touch_levels=touch_levels, n_hist=len(r),
               close_only=close_only, etf=etf)
    if not close_only:
        res["sim_close"] = simulate_trade(model, close_only=True, **kw)
    sigma_sim = math.sqrt(252 * sim["mean_r2"])
    res["sigma_sim"] = sigma_sim
    res["drag_analytic"] = drag_analytic(leverage, sigma_sim, horizon)
    ok = np.isfinite(sim["shortfall"])
    res["drag_sim"] = float(sim["shortfall"][ok].mean()) if ok.any() else float("nan")
    res["worst_day"], res["best_day"] = worst_day_moves(r, leverage)
    if etf is not None:
        er = np.asarray(etf["returns"], float)
        er = er[np.isfinite(er)]
        if len(er):
            res["worst_day"], res["best_day"] = float(er.min()), float(er.max())
        res["tracking"] = px.leveraged_check(etf["returns"], etf["underlying_returns"], leverage)
    res["stop_ret"] = stop_return_estimate(stp, leverage, entry, sim)
    if size_frac is not None:
        res["size"] = size_check(size_frac, account, account_stop, res["stop_ret"], sim["ret"], res["worst_day"])
        res["account"], res["account_stop"] = account, account_stop
    return res


def resolve_instrument(ticker: str, meta: dict | None, available, leverage_arg: float | None) -> dict:
    """Decide what to simulate. meta: {ticker: (underlying, leverage)} as read by journal.load_meta; available: tickers in the price file.

    Returns dict(sim_ticker, leverage, etf_ticker or None, note). A ticker that meta lists with leverage != 1 is a listed leveraged ETF:
    its UNDERLYING is simulated with that leverage; passing --leverage as well is an error (the instrument would be levered twice)."""
    t = ticker.upper()
    avail = {a.upper() for a in available}
    und, lev = (meta or {}).get(t, (None, 1.0))
    if und is not None and abs(lev - 1.0) > 1e-9:
        if leverage_arg is not None:
            raise ValueError(f"{t} is already {lev:+g}x {und}; run level_calc.py {und} --leverage {lev:+g}, or drop --leverage")
        if und in avail:
            return dict(sim_ticker=und, leverage=float(lev), etf_ticker=t, underlying=und, note="")
        return dict(sim_ticker=t, leverage=1.0, etf_ticker=None, underlying=None,
                    note=f"NOTE {t} is listed in meta as {lev:+g}x {und}, but {und} is not in the price file: simulating {t}'s own returns at leverage 1")
    return dict(sim_ticker=t, leverage=1.0 if leverage_arg is None else float(leverage_arg), etf_ticker=None, underlying=None, note="")


def _pc(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def _spc(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.{d}f}%"


def _lvl(lv: Level | None, lev: float) -> str:
    if lv is None:
        return "none"
    if lv.kind == "ret":
        return f"{_spc(lv.value)} on the instrument"
    return f"price {lv.value:g} of the underlying" + ("" if lev == 1.0 else " (trigger on the underlying close)")


def format_report(res: dict) -> str:
    m: Model = res["model"]
    s = res["sim"]
    L, h, n = res["leverage"], res["horizon"], s["n_paths"]
    tk = res["ticker"]
    etf = res.get("etf")
    out = ["=" * 100, f"LEVEL CALCULATOR for {res['ticker']} (decision support from a Monte Carlo; NOT a forecast, no directional edge)", "=" * 100,
           f"  last close {res['last_close']:.4f}" + (f" on {res['last_date']:%Y-%m-%d}" if res["last_date"] is not None else "")
           + f"   history {res['n_hist']} daily total returns",
           "  instrument: " + (f"{etf['ticker']} = {L:+g}x daily-reset product on {etf['underlying']} (from meta); levels are prices / returns as "
                              f"described below, history of {etf['underlying']} simulated" if etf else
                              'the stock itself' if L == 1 else f'daily-reset {L:+g}x product on {tk}') +
           f"   entry {res['entry']:.4f}   horizon {h} trading days   paths {n:,}   seed {res['seed']}"]
    if m.kind == "gjr":
        p = m.params
        out.append(f"  model: {m.note}: alpha {p['alpha']:.4f} gamma {p['gamma']:.4f} beta {p['beta']:.4f} nu {p['nu']:.2f} "
                   f"persistence {p['persistence']:.4f}")
        out.append(f"  annual vol: unconditional {m.sigma_annual:.1%}, current conditional (next day) {math.sqrt(252 * p['next_var']):.1%}, "
                   f"simulated over the horizon {res['sigma_sim']:.1%}")
    else:
        out.append(f"  {m.note}")
        out.append(f"  annual vol: {m.sigma_annual:.1%} (simulated {res['sigma_sim']:.1%})")
    out.append(f"  drift: {res['drift']:.1%} per year" if res["drift"] else
               "  drift: ZERO. With zero drift this calculator carries NO view on direction; every probability below comes from volatility only.")
    co = res["close_only"]
    out += ["", f"  rules: target {_lvl(res['target'], L)}; stop {_lvl(res['stop'], L)}",
            ("  mode: CLOSE-ONLY: levels checked on daily closes, fill at the instrument's value at the trigger close; "
             if co else "  mode: limit / stop orders filled AT the level; a level is hit on a close crossing or by a Brownian-bridge intraday touch; ")
            + "target and stop on the same day count as the stop"]
    if etf and etf.get("note"):
        out.append("  " + etf["note"])
    if res["touch_levels"]:
        out += ["", f"UNDERLYING touch probabilities within {h} trading days (first passage on {'daily closes' if co else 'closes plus bridge intraday touches'}, from the last close {res['last_close']:.4f})"]
        for k, lv in res["touch_levels"].items():
            out.append(f"  {k:<7} {lv:>10.4f} ({100 * (lv / res['last_close'] - 1):+.1f}% from last): touched {_pc(s['touch'][k])}   "
                       f"close at day {h} above {_pc(s['above'][k])} / below {_pc(1 - s['above'][k])}")
    out += ["", f"INSTRUMENT race and trade-return distribution (hold at most {h} days; exits {'at the trigger close' if co else 'at the level'}, else at the horizon close)"]
    if res["target"] is not None or res["stop"] is not None:
        o = s["outcome"]
        d = s["exit_day"]
        ex = d > 0
        out.append(f"  target hit first {_pc(float((o == 1).mean()))}   stop hit first {_pc(float((o == -1).mean()))}   "
                   f"neither by day {h} {_pc(float((o == 0).mean()))}")
        out.append("  days to exit (exited paths): " + (f"mean {d[ex].mean():.1f}, median {np.median(d[ex]):.0f}" if ex.any() else "no exits")
                   + f"   including paths held to the horizon: mean {np.where(ex, d, h).mean():.1f}")
    out.append(f"  expected return per trade {_spc(float(s['ret'].mean()), 2)}   (mean of exits {'at the trigger close' if co else 'at the level'} or at the horizon close; "
               "before costs)")
    sc = res.get("sim_close")
    if sc is not None:
        oc = sc["outcome"]
        out.append(f"  close-only (no intraday touches, fill at trigger close): target first {_pc(float((oc == 1).mean()))}   stop first "
                   f"{_pc(float((oc == -1).mean()))}   neither {_pc(float((oc == 0).mean()))}   expected return per trade "
                   f"{_spc(float(sc['ret'].mean()), 2)}")
    q = np.percentile(s["ret"], QUANTILES)
    out.append("  final trade return quantiles: " + "  ".join(f"{p}%: {_spc(v)}" for p, v in zip(QUANTILES, q)))
    out.append(f"  largest one-day moves of the instrument in the history ({'ETF itself' if etf else 'from the simulated series'}): worst "
               f"{_spc(res['worst_day'])}, best {_spc(res['best_day'])} (gaps of that size jump through a stop)")
    if L != 1:
        out.append(f"  paths wiped out (instrument value 0): {_pc(float(s['ruin'].mean()), 2)}")
        out.append(f"  drag over {h} days (log): analytic (L^2-L) sigma^2 T/2 = {_pc(res['drag_analytic'], 2)} with sigma {res['sigma_sim']:.1%}; "
                   f"simulated mean of L log(S_T/S_0) - log(V_T) = {_pc(res['drag_sim'], 2)} (surviving paths)")
    tr = res.get("tracking")
    if tr is not None:
        out.append(f"  tracking sanity ({etf['ticker']} vs {L:+g} x {etf['underlying']}, {tr['n']} common days"
                   + (f" {tr['start']:%Y-%m-%d}..{tr['end']:%Y-%m-%d}" if tr["n"] else "") + f"): slope {tr['slope']:+.2f}, R2 {tr['r2']:.3f}, tracking error "
                   f"{tr['te_ann']:.1%}/yr, realised log drag {_pc(tr['drag_realised'], 2)} vs formula {_pc(tr['drag_expected'], 2)}")
    sz = res.get("size")
    if sz is not None:
        out += ["", "POSITION SIZE AND ACCOUNT STOP"]
        if math.isnan(res["stop_ret"]):
            out.append("  no stop given: cannot size against the account stop")
        else:
            out.append(f"  position {_pc(sz['size_frac'])} of the account = {sz['position_usd']:,.0f} USD; loss at the stop {_spc(res['stop_ret'])} "
                       f"of the position = {sz['loss_at_stop_usd']:,.0f} USD = {_pc(sz['loss_at_stop_share'], 2)} of the account")
            out.append(f"  account stop {_pc(res['account_stop'])} of the deposit = {sz['account_stop_usd']:,.0f} USD; probability that ONE trade of this "
                       f"size ends at a loss beyond it: {_pc(sz['p_account_stop'], 2)}")
            out.append(f"  rule: size <= account_stop / |stop| = {_pc(res['account_stop'])} / {_pc(abs(res['stop_ret']))} = {_pc(sz['max_size'])} of the account "
                       f"(a stopped-out trade at or below that size cannot breach the account stop, if the stop fills at its level)")
        out.append(f"  CAUTION: overnight gaps can exceed a stop. Largest one-day move of the instrument in the history: worst {_spc(res['worst_day'])}, "
                   f"best {_spc(res['best_day'])}."
                   + (f" Sizing for a repeat of the worst day: size <= {_pc(sz['max_size_gap'])}." if not math.isnan(sz["max_size_gap"]) else ""))
    out += ["", "WHAT THIS DOES NOT TELL YOU",
            "  - It has no directional edge: with zero drift the odds of target vs stop reflect distance and volatility only, not a view on the stock.",
            "  - The zero-drift assumption is a choice; a positive or negative expected return would shift every probability.",
            "  - Intraday touches use a Brownian-bridge approximation between closes (--close-only drops it); fills are exactly at the level: overshoot, slippage and",
            "    costs are not modelled, and gaps (earnings, news) jump through stops and fill worse: see the worst one-day move statistic.",
            "  - Tail risk: a fitted model understates events outside its history; leveraged products can lose the whole position in one bad day sequence."]
    if m.kind == "fallback":
        out.append("  - Fallback model: flat volatility with no leverage effect and no volatility clustering; it cannot widen after a drop, so risk after "
                   "a sell-off is understated.")
    out.append("  - Volatility is estimated from this ticker's own recent history and can change; the probabilities are conditional on it.")
    return "\n".join(out) + "\n"
