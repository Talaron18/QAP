"""Study H (pre-registration v15): simulated 2x products, the band / momentum state machines, placebo, universe, replication check, licence rule,
trials, sealed report. SYNTHETIC data only; nothing is written to the real data/ or the real trials file."""
import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd
import pytest

from qap import bandsim as B
from qap import volfc as V

from test_pipeline import make_world

REAL_DATA = pathlib.Path(__file__).parent.parent / "data"
YEAR2020 = re.compile(r"(?<!\d\.)(?<!\d)20[2-9]\d(?!\d)")


# ------------------------------------------------------------------------------------------------------------ constants
def test_constants_are_the_pre_registered_values():
    assert (B.TOP_N, B.MIN_VOL, B.MIN_OBS, B.VOL_DAYS) == (300, 0.45, 200, 252)
    assert (B.FEE, B.COST, B.STOP, B.TP_R2, B.BAND_D, B.BAND_N) == (0.015, 0.0010, 0.20, 0.15, 0.059, 21)
    assert (B.R3_BUY, B.R3_TP, B.NW_LAGS, B.DRAWS, B.MIN_SHIFT) == (0.967, 1.133, 10, 200, 63)
    assert B.DEV == ("1995-01-01", "2014-12-31") and B.VAL == ("2015-01-01", "2019-12-31") and B.HOLD == ("2020-01-01", "2025-12-31")
    assert B.MARKER == "data/holdout_v15.done" and B.PREREG == "v15"
    p1, p2, p3 = B.primary("R1"), B.primary("R2"), B.primary("R3")
    assert (p1.lag, p1.fee, p1.cost, p1.n, p1.d, p1.stop, p1.lev) == (1, 0.015, 0.001, 21, 0.059, 0.2, 2.0)
    assert (p2.tp, p2.stop) == (0.15, 0.2) and (p3.lev, p3.fee, p3.buy, p3.sell) == (1.0, 0.0, 0.967, 1.133)
    assert B.PAIRS == (("SNXX", "SNDK", 2.0), ("LABX", "ALAB", 2.0), ("LITX", "LITE", 2.0), ("SKHX", "SKHY", 2.0),
                       ("SNDQ", "SNDK", -2.0), ("DAMD", "AMD", -2.0), ("SKDD", "SKHY", -2.0), ("AXTQ", "AXTI", -2.0))
    assert B.USER_NAMES == ("SNDK", "ALAB", "LITE", "SKHY", "AMD", "INTU", "STX", "ORCL", "CELH", "BE", "AXTI")


# ------------------------------------------------------------------------------------------------------------ instrument
def test_instrument_formula_against_hand_values_including_the_financing_term():
    r, rf, f = 0.01, 0.0002, 0.015
    long2 = B.instrument_returns([r], [rf], 2.0, f)[0]
    assert long2 == pytest.approx(2 * 0.01 + (1 - 2) * 0.0002 - 0.015 / 252)               # = 0.02 - 0.0002 - 0.0000595238
    assert long2 == pytest.approx(0.0197404762, abs=1e-10)
    short2 = B.instrument_returns([r], [rf], -2.0, f)[0]
    assert short2 == pytest.approx(-2 * 0.01 + (1 + 2) * 0.0002 - 0.015 / 252)             # the inverse EARNS 3 rf
    assert short2 == pytest.approx(-0.0194595238, abs=1e-10)
    assert B.instrument_returns([0.0], [0.0003], 2.0, 0.0)[0] == pytest.approx(-0.0003)     # long pays financing, inverse earns it
    assert B.instrument_returns([0.0], [0.0003], -2.0, 0.0)[0] == pytest.approx(0.0009)
    assert B.instrument_returns([0.02], [0.0003], 1.0, 0.0)[0] == pytest.approx(0.02)       # L = 1: the stock itself
    assert B.instrument_returns([np.nan], [np.nan], 2.0, 0.0)[0] == 0.0                     # missing -> 0 (callers count)


def test_floor_at_zero_and_a_dead_product_stays_dead():
    r = np.array([0.01, -0.6, 0.30, 0.10])
    x = B.instrument_returns(r, np.zeros(4), 2.0, 0.0)
    assert x[1] == -1.0 and x[2] == 0.0 and x[3] == 0.0 and x[0] == pytest.approx(0.02)
    assert B.instrument_value(x)[-1] == 0.0
    y = B.instrument_returns(r, np.zeros(4), 2.0, 0.0, dead_stays=False)
    assert y[1] == -1.0 and y[2] == pytest.approx(0.6)                                       # per-day floor only (the engine's stop ends a holding)
    z = B.instrument_returns([0.7, 0.0], [0.0, 0.0], -2.0, 0.0)                             # an inverse dies on a +50% day
    assert z.tolist() == [-1.0, 0.0]


# -------------------------------------------------------------------------------------------------------- band machine
P_BAND = np.array([100, 100, 100, 100, 70, 70, 70, 100, 100, 100, 140, 140, 140, 100, 100, 100, 100], dtype=float)
Z_BAND = [0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1]       # hand-derived: N = 3, d = 0.1 (see the comments below)


def _band_params(**kw):
    return B.Params(rule="R1", n=3, d=0.1, **kw)


def test_band_states_with_hysteresis_against_a_hand_derivation():
    # M_t = mean(P[t-2..t]): t4 M=90 (bands 81/99, P=70 < 81 -> +1); t6 M=70 (63/77, P=70 inside -> KEEP +1); t7 M=80 (72/88, P=100 >= 88 -> -1);
    # t9 M=100 (90/110, P=100 inside -> KEEP -1); t12 M=140 (126/154, inside -> KEEP -1); t13 M=126.67 (114/139.3, P=100 < 114 -> +1)
    M = B.rolling_mean(P_BAND, 0, len(P_BAND) - 1, 3)
    assert np.isnan(M[:2]).all() and M[2] == pytest.approx(100) and M[4] == pytest.approx(90) and M[13] == pytest.approx(380 / 3)
    Z = B.sleeve_signals(P_BAND, np.zeros(len(P_BAND)), 0, len(P_BAND) - 1, _band_params())["Z"]
    assert Z.tolist() == Z_BAND
    assert B.ffill_state([0, 0, 1, 0, 0, -1, 0]).tolist() == [0, 0, 1, 1, 1, -1, -1]
    # the boundary: P >= M (1 + d) is a SELL, P < M (1 - d) a BUY
    Pb = np.array([100.0, 100.0, 100.0, 109.0, 1.0])
    Zb = B.sleeve_signals(Pb, np.zeros(5), 0, 4, B.Params(rule="R1", n=1 + 0, d=0.1))["Z"]            # n = 1: M = P itself -> no breach ever
    assert Zb.tolist() == [0, 0, 0, 0, 0]


def test_band_anchor_uses_prior_year_closes_and_includes_the_current_close():
    P = np.arange(1.0, 11.0)                      # closes 1..10; the sleeve starts at index 6 (value 7)
    M = B.rolling_mean(P, 6, 9, 4)
    assert M.tolist() == [(4 + 5 + 6 + 7) / 4, (5 + 6 + 7 + 8) / 4, (6 + 7 + 8 + 9) / 4, (7 + 8 + 9 + 10) / 4]
    assert np.isnan(B.rolling_mean(P, 0, 3, 4)[:3]).all() and B.rolling_mean(P, 0, 3, 4)[3] == 2.5          # needs 4 closes
    pi = B.price_index([np.nan, 0.1, np.nan, -0.5])
    assert np.isnan(pi[0]) and pi[1:].tolist() == pytest.approx([1.1, 1.1, 0.55])


def test_band_flip_entry_and_year_end_close_with_costs():
    n = len(P_BAND)
    zero = np.zeros(n)
    X, tr = B.simulate(_band_params(), zero, zero, Z=np.array(Z_BAND, dtype=np.int8))
    # lag 1: want[u] = Z[u-1]; Z[4] = +1 -> enter at 5; Z[7] = -1 -> flip at 8; Z[13] = +1 -> flip at 14; last close 16 -> forced exit
    assert X.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 0]
    assert [(t[0], t[1], t[2], t[3]) for t in tr] == [(1, 5, 8, "flip"), (-1, 8, 14, "flip"), (1, 14, 16, "yearend")]
    c = 0.001
    ex, net = B.evaluate_positions(X, zero, zero, zero, c)
    assert net[0, 5] == pytest.approx(-c)                                    # entry day: flat the day before, one side
    assert net[0, 8] == pytest.approx((1 - c) ** 2 - 1)                      # flip day: two sides
    assert net[0, 16] == pytest.approx(-c)                                   # year-end close: one exit side, nothing entered
    assert net[0, 14] == pytest.approx((1 - c) ** 2 - 1)
    assert (net[0, [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15]] == 0).all()
    # lag 0 (information variant): the same signals trade one close earlier
    X0, _ = B.simulate(_band_params(lag=0), zero, zero, Z=np.array(Z_BAND, dtype=np.int8))
    assert X0.tolist() == [0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 0]


def test_year_end_close_suppresses_a_flip_at_the_final_close_and_charges_one_side_only():
    Z = np.array([1, 1, 1, -1, -1], dtype=np.int8)
    zero = np.zeros(5)
    X, tr = B.simulate(_band_params(), zero, zero, Z=Z)
    assert X.tolist() == [0, 1, 1, 1, 0]                                      # at the last close Z[3] = -1 wants a flip: only the exit happens
    assert [(t[1], t[2], t[3]) for t in tr] == [(1, 4, "yearend")]
    assert B.evaluate_positions(X, zero, zero, zero, 0.001)[1][0, 4] == pytest.approx(-0.001)


def test_lag_convention_the_position_earns_from_day_t_plus_two():
    # Z flips to +1 at close 1 (signal), lag 1: trade at close 2, first earned return is day 3; lag 0: trade at close 1, first earned day 2
    Z = np.array([0, 1, 1, 1, 1, 1], dtype=np.int8)
    ret = np.array([0.0, 0.0, 0.10, 0.20, 0.0, 0.0])
    for lag, first_day in ((1, 3), (0, 2)):
        X, _ = B.simulate(_band_params(lag=lag), ret, ret, Z=Z)
        assert int(np.argmax(X != 0)) == first_day - 1                         # trade close = first earning day - 1
        ex, net = B.evaluate_positions(X, ret, ret, np.zeros(6), 0.0)
        earned = np.flatnonzero(net[0] != 0)
        assert earned[0] == first_day
    X1, _ = B.simulate(_band_params(lag=1), ret, ret, Z=Z)
    assert B.evaluate_positions(X1, ret, ret, np.zeros(6), 0.0)[1][0].tolist() == pytest.approx([0, 0, 0, 0.2, 0, 0.0])   # misses day 2's +10%


# --------------------------------------------------------------------------------------------------------- R2 and stops
def test_r2_entry_take_profit_exit_and_signals_ignored_while_in_the_trade():
    n = 10
    ret = np.array([0, 0, 0, 0, 0.10, 0.05, 0.02, 0, 0, 0.0])
    entry = np.zeros(n, bool)
    entry[[2, 4, 5, 6]] = True                  # 4, 5 fall inside the trade (ignored); 6 is the exit close: flat after the exit -> a NEW entry
    X, tr = B.simulate(B.Params(rule="R2"), ret, np.zeros(n), entry=entry)
    # signal at 2 -> entry at 3; q = 1.10 at 4, 1.155 >= 1.15 at 5 -> take-profit SIGNAL at 5 -> exit executes at 6 (q = 1.155 x 1.02);
    # the signal at 6 (flat after the exit) -> entry at 7, closed at the last close
    assert X.tolist() == [0, 0, 0, 1, 1, 1, 0, 1, 1, 0]
    assert tr[0][:4] == (1, 3, 6, "tp") and tr[0][4] == pytest.approx(1.10 * 1.05 * 1.02)
    assert tr[1][:4] == (1, 7, 9, "yearend")
    X0, tr0 = B.simulate(B.Params(rule="R2", lag=0), ret, np.zeros(n), entry=entry)
    assert tr0[0][:4] == (1, 2, 5, "tp") and X0[2] == 1 and X0[5] == 0         # lag 0: enters at the signal close, exits at the take-profit close


def test_r2_signal_definition_two_consecutive_up_closes_using_the_prior_close():
    r = np.array([0.01, 0.01, -0.01, 0.02, 0.03, 0.0, 0.01, np.nan, 0.01])
    e = B.sleeve_signals(None, r, 1, 8, B.Params(rule="R2"))["entry"]
    assert e.tolist() == [True, False, False, True, False, False, False, False]      # index 0 = close 1 (uses r[0] > 0)
    assert B.sleeve_signals(None, r, 0, 8, B.Params(rule="R2"))["entry"][0] == False  # no prior return at all


def test_stop_is_filled_at_the_breaching_close_then_one_day_pause_then_reentry():
    n = 10
    ret = np.array([0, 0, 0, 0, -0.25, 0, 0, 0, 0, 0.0])
    entry = np.zeros(n, bool)
    entry[[2, 4, 5]] = True
    X, tr = B.simulate(B.Params(rule="R2"), ret, np.zeros(n), entry=entry)
    # entry at 3; day 4 return -25% -> q = 0.75 <= 0.8: stop FILLED AT close 4 (overshoot included); signal at 4 would execute at 5 = the pause close:
    # lost; signal at 5 executes at 6: the earliest new entry
    assert X.tolist() == [0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
    assert tr[0][:4] == (1, 3, 4, "stop") and tr[0][4] == pytest.approx(0.75)
    only4 = entry.copy()
    only4[5] = False
    X2, _ = B.simulate(B.Params(rule="R2"), ret, np.zeros(n), entry=only4)
    assert X2.tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]                  # the signal at the stop close does NOT execute at the pause close
    ex, net = B.evaluate_positions(X, ret, np.zeros(n), np.zeros(n), 0.001)
    assert net[0, 4] == pytest.approx(0.75 * 0 + (1 - 0.25) * (1 - 0.001) - 1)     # stop-day return less one exit side
    assert net[0, 3] == pytest.approx(-0.001) and net[0, 6] == pytest.approx(-0.001)   # entries at 3 and 6


def test_stop_threshold_is_inclusive_and_blocks_the_flip_too_in_the_band_rule():
    n = 9
    ret = np.array([0, 0, 0, -0.2, 0, 0, 0, 0, 0.0])
    Z = np.ones(n, dtype=np.int8)
    X, tr = B.simulate(_band_params(), ret, ret, Z=Z)
    # enter at 1 (Z[0]); day 3: q = 0.8 exactly -> stop at 3; pause close 4 (want = Z[3] = +1 blocked); re-entry at 5 from Z[4]
    assert X.tolist() == [0, 1, 1, 0, 0, 1, 1, 1, 0]
    assert tr[0][3] == "stop" and tr[0][1:3] == (1, 3)
    # inverse leg stopped on a rally: Z = -1, the short product loses 25% on day 2
    Zs = -np.ones(6, dtype=np.int8)
    rs = np.array([0, 0, -0.25, 0, 0, 0.0])
    Xs, trs = B.simulate(_band_params(), np.zeros(6), rs, Z=Zs)
    assert Xs.tolist() == [0, -1, 0, 0, -1, 0] and trs[0][3] == "stop"


def test_stop_on_the_final_close_is_a_stop_exit_and_no_entry_on_the_pause_leaks_past_it():
    ret = np.array([0, 0, 0, -0.5, 0.0])
    Z = np.ones(5, dtype=np.int8)
    X, tr = B.simulate(_band_params(), ret, ret, Z=Z)
    assert X.tolist() == [0, 1, 1, 0, 0] and tr[0][3] == "stop"


def test_r3_stock_band_buys_below_and_takes_profit_above_the_anchor():
    P = np.array([100.0, 100, 100, 95, 96, 100, 110, 120, 125, 100])
    p = B.r3_params(n=3)
    sig = B.sleeve_signals(P, np.zeros(10), 0, 9, p)
    M = B.rolling_mean(P, 0, 9, 3)
    assert sig["entry"][3] == (95 < 0.967 * M[3]) and sig["exit_u"][8] == (125 >= 1.133 * M[8])
    X, tr = B.simulate(p, np.zeros(10), np.zeros(10), **sig)
    assert (X != 0).any() and tr[0][3] in ("tp", "yearend")


# ---------------------------------------------------------------------------------------------------------------- placebo
def _x_series(n=300, seed=1):
    rng = np.random.default_rng(seed)
    X = np.zeros(n, dtype=np.int8)
    i = 5
    while i < n - 30:
        L = int(rng.integers(5, 25))
        X[i:i + L] = rng.choice([-1, 1])
        i += L + int(rng.integers(3, 20))
    X[-1] = 0
    return X


def _cyc_changes(Y):
    return int((Y != np.roll(Y, 1)).sum())


def _cyc_runs(Y):
    k = int(np.flatnonzero(Y != np.roll(Y, 1))[0]) if (Y != np.roll(Y, 1)).any() else 0
    Z = np.roll(Y, -k)
    runs, cur, val = [], 1, Z[0]
    for v in Z[1:]:
        if v == val:
            cur += 1
        else:
            runs.append((int(val), cur))
            cur, val = 1, v
    runs.append((int(val), cur))
    return sorted(runs)


def test_placebo_preserves_fractions_transitions_persistence_and_costs_by_construction():
    X = _x_series()
    n = len(X)
    ks = B.placebo_shifts(1234, 2001, n, 50)
    assert ks.min() >= 63 and ks.max() <= n - 63 and len(set(ks.tolist())) > 10
    Yall = X[(np.arange(n)[None, :] + ks[:, None]) % n]
    for Y in Yall:
        assert [(Y == v).mean() for v in (-1, 0, 1)] == [(X == v).mean() for v in (-1, 0, 1)]
        assert _cyc_changes(Y) == _cyc_changes(X)                           # equal number of transitions
        assert _cyc_runs(Y) == _cyc_runs(X)                                  # equal persistence (cyclic run lengths)
    # costs: with zero instrument returns the day nets are only costs; their sum is the same in every draw (equal sides)
    z = np.zeros(n)
    ex, net = B.evaluate_positions(X, z, z, z, 0.001, ks)
    assert np.allclose(net.sum(axis=1), net.sum(axis=1)[0]) and net.sum(axis=1)[0] < 0
    assert B.evaluate_positions(X, z, z, z, 0.001)[1].sum() == pytest.approx(net.sum(axis=1)[0])      # and equal to the real series
    # timing really is broken: with a return pattern aligned to the real positions the shifted draws earn less
    ret = np.where(np.roll(X, 1) == 1, 0.01, np.where(np.roll(X, 1) == -1, -0.01, 0.0))
    real = B.evaluate_positions(X, ret, -ret, z, 0.0)[0].sum()
    shifted = B.evaluate_positions(X, ret, -ret, z, 0.0, ks)[0].sum(axis=1)
    assert real > shifted.max()


def test_placebo_shift_zero_reproduces_the_real_series_and_short_sleeves_use_a_smaller_range():
    X = _x_series(120)
    rng = np.random.default_rng(3)
    rL, rS, rf = rng.normal(0, 0.03, 120), rng.normal(0, 0.03, 120), np.full(120, 1e-4)
    a = B.evaluate_positions(X, rL, rS, rf, 0.001)[1]
    b = B.evaluate_positions(X, rL, rS, rf, 0.001, np.array([0]))[1]
    assert np.array_equal(a, b)
    ks = B.placebo_shifts(5, 2000, 100, 40)                                  # n < 126: [n//2, n - n//2]
    assert ks.min() == 50 and ks.max() == 50
    ks = B.placebo_shifts(5, 2000, 130, 100)
    assert ks.min() >= 63 and ks.max() <= 67
    assert B.placebo_shifts(5, 2000, 1, 3).tolist() == [0, 0, 0]


def test_placebo_seeds_depend_on_permno_year_draw_only():
    a = B.placebo_shifts(10, 1999, 250, 200)
    assert np.array_equal(a, B.placebo_shifts(10, 1999, 250, 200))
    assert np.array_equal(a[:7], B.placebo_shifts(10, 1999, 250, 7))                  # draw d does not depend on the number of draws
    assert not np.array_equal(a, B.placebo_shifts(11, 1999, 250, 200)) and not np.array_equal(a, B.placebo_shifts(10, 2000, 250, 200))


def test_a_held_position_that_hits_zero_stays_at_zero_in_a_shifted_series():
    X = np.array([0, 1, 1, 1, 1, 0], dtype=np.int8)
    rL = np.array([0, 0, -1.0, 0.5, 0.5, 0.0])
    z = np.zeros(6)
    ex, net = B.evaluate_positions(X, rL, z, z, 0.0)
    assert net[0].tolist() == pytest.approx([0, 0, -1.0, 0, 0, 0])             # day 2: first held day with ret = -1 -> value 0, then stays 0 (days 3-4 held)
    assert B.evaluate_positions(X, np.array([0, 0, 0, -1.0, 0.5, 0.5]), z, z, 0.0)[1][0].tolist() == pytest.approx([0, 0, 0, -1.0, 0, 0])


# ------------------------------------------------------------------------------------------------------ statistics, licence
def test_window_stat_is_the_newey_west_t_of_the_daily_cross_sectional_mean():
    cal = np.datetime64("2000-01-03") + np.arange(400).astype("timedelta64[D]")
    rng = np.random.default_rng(0)
    C = rng.integers(1, 4, 400)
    d = rng.normal(0.001, 0.01, 400)
    S = d * C
    st = B.window_stat(S, C, cal, ("2000-02-01", "2000-12-31"))
    m = (cal >= np.datetime64("2000-02-01")) & (cal <= np.datetime64("2000-12-31"))
    mu, t = V.newey_west_t(d[m], 10)
    assert st["mean"] == pytest.approx(mu) and st["t"] == pytest.approx(t) and st["n"] == int(m.sum())
    C0 = C.copy()
    C0[:60] = 0
    assert B.window_stat(S, C0, cal, ("2000-01-01", "2000-12-31"))["n"] == int(((cal <= np.datetime64("2000-12-31")) & (C0 > 0)).sum())
    SP = np.vstack([S, 2 * S])
    assert B.placebo_stats(SP, C, cal, ("2000-02-01", "2000-12-31")) == pytest.approx([mu * 0 + d[m].mean(), 2 * d[m].mean()])
    s = B.placebo_summary(np.arange(101) / 100)
    assert s["median"] == pytest.approx(0.5) and s["p95"] == pytest.approx(0.95) and s["n"] == 101


def _lic(dm=0.001, dt=3.0, vm=0.001, p95=0.0005, hm=None, hmed=0.0, open_holdout=False):
    hold = None if hm is None else {"mean": hm, "t": 1.0}
    return B.licence_verdict({"mean": dm, "t": dt}, {"mean": vm, "t": 1.0}, {"p95": p95, "median": 0.0, "n": 200}, hold,
                             {"median": hmed, "p95": 1.0, "n": 200}, open_holdout)


def test_licence_rule_all_conditions_and_the_holdout_is_hidden_without_the_flag():
    v = _lic()
    assert v["dev_val"] and not v["opened"] and v["stands"] is None and v["status"].startswith("PASSED IN DEV/VALIDATION")
    for kw, name in ((dict(dm=-0.001), "dev mean > 0"), (dict(dt=1.99), "dev t >= 2.0"), (dict(p95=0.002), "dev > placebo 95th pct"), (dict(vm=-1e-6), "validation mean > 0")):
        w = _lic(**kw)
        assert not w["dev_val"] and name in w["status"] and w["status"].startswith("NOT LICENSED")
    assert _lic(dt=2.0)["dev_val"] and not _lic(dm=0.0005)["dev_val"]          # t >= 2 is inclusive, dev > p95 is strict (0.0005 == p95)
    assert not _lic(dm=float("nan"))["dev_val"] and not _lic(p95=float("nan"))["dev_val"]
    # holdout supplied but the flag is absent -> stays hidden
    assert not _lic(hm=0.01)["opened"] and _lic(hm=0.01)["stands"] is None
    # with the flag: stands only if mean > 0 and above the placebo median
    assert _lic(hm=0.002, hmed=0.001, open_holdout=True)["stands"] is True and _lic(hm=0.002, hmed=0.001, open_holdout=True)["status"] == "LICENSED (holdout stands)"
    assert _lic(hm=0.0005, hmed=0.001, open_holdout=True)["stands"] is False
    assert _lic(hm=-0.002, hmed=-0.01, open_holdout=True)["stands"] is False
    # a rule that failed dev / validation never opens the holdout even with the flag
    assert not _lic(dm=-1, hm=0.5, open_holdout=True)["opened"]


def test_trial_specs_name_the_rule_parameters():
    s1, s2 = B.trial_spec("R1"), B.trial_spec("R2")
    assert s1["prereg"] == s2["prereg"] == "v15" and s1["N"] == 21 and s1["d"] == 0.059 and s2["take_profit"] == 0.15 and s1 != s2
    assert s1["fee"] == 0.015 and s1["cost_per_side"] == 0.001 and s1["lag"] == 1 and s1["stop"] == 0.2


# ---------------------------------------------------------------------------------------------------------------- universe
def _panel_world():
    m, d, _, _ = make_world(n=8, years=(1996, 2000))
    m = m.copy()
    dec = lambda y: m["mthcaldt"] == pd.Timestamp(f"{y}-12-31")
    caps = {1998: {1: 100.0, 2: 900.0, 3: 500.0, 4: 400.0, 5: 300.0, 6: 50.0, 7: 40.0, 8: 30.0},
            1999: {1: 100.0, 2: 90.0, 3: 20.0, 4: 400.0, 5: 700.0, 6: 650.0, 7: 40.0, 8: 30.0}}
    for y, cs in caps.items():
        for p, c in cs.items():
            m.loc[dec(y) & (m["permno"] == p), "mthcap"] = c * 1000.0
    m = m[~((m["permno"] == 3) & (m["mthcaldt"] > pd.Timestamp("1999-03-31")))]       # permno 3 vanishes during 1999: still selected on Dec-1998 ME
    from qap.panel import build_monthly_panel
    return build_monthly_panel(m, d)


def _ret_world():
    cal = np.array(pd.bdate_range("1997-01-01", "2000-12-29"), dtype="datetime64[D]")
    n = len(cal)
    sig = {1: 0.030, 2: 0.0125, 3: 0.030, 4: 0.030, 5: 0.0299, 6: 0.0306, 7: 0.030, 8: 0.030}        # daily; annual = x sqrt(252)
    R = np.full((8, n), np.nan)
    sgn = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    for p, s in sig.items():
        R[p - 1] = s * sgn
    i98 = B.dec_index(cal, 1999)
    R[3, :i98 - 100] = np.nan                                  # permno 4: only ~100 returns in the trailing year -> fails the 200-observation rule
    R[2, i98 + 1:] = np.nan                                    # permno 3 delisted in 1999 (data ends), still selected (no survivor filter)
    return cal, R, {p: p - 1 for p in range(1, 9)}


def test_universe_ranks_on_prior_december_company_me_and_matches_volfc_top_universe():
    panel = _panel_world()
    for n in (3, 4):
        uni = V.top_universe(panel, [1999, 2000], n)
        for y in (1999, 2000):
            assert sorted(B.ranked_candidates(panel, y, n).tolist()) == sorted(uni[y].tolist())
    assert B.ranked_candidates(panel, 1999, 3).tolist() == [2, 3, 4]              # rank order by company ME: 1+2 share a company (1000, rep. permno 2), 500, 400


def test_universe_vol_filter_observation_rule_prior_december_timing_and_no_survivor_filter():
    panel = _panel_world()
    cal, R, row_of = _ret_world()
    u = B.select_universe(panel, [1999, 2000], cal, R, row_of, top_n=3, min_vol=0.45)
    ann = lambda s: s * np.sqrt(252) * np.sqrt(252 / 251)
    assert ann(0.030) > 0.45 and ann(0.0299) > 0.45 > 0.0125 * np.sqrt(252)
    # 1999: top 3 by Dec-1998 ME = [2 (vol 19.8%: too low), 3 (delisted LATER in 1999: kept), 4 (only ~100 returns: <200 obs)]
    assert u[1999]["permno"].tolist() == [3] and u[1999]["cands"] == 3 and u[1999]["obs_ok"] == 2
    # 2000: top 3 by Dec-1999 ME = [5 (vol ~47.5%), 6 (48.6%), 4]; permno 3 left the data in 1999
    assert u[2000]["permno"].tolist() == [5, 6, 4]
    assert 4 not in u[1999]["permno"] and 1 not in u[1999]["permno"]
    # threshold: both sides of 45%
    u2 = B.select_universe(panel, [2000], cal, R, row_of, top_n=3, min_vol=0.4795)
    assert u2[2000]["permno"].tolist() == [6]
    assert B.select_universe(panel, [2000], cal, R, row_of, top_n=3, assets=1)[2000]["permno"].tolist() == [5]
    # the volatility is measured at the LAST DECEMBER trading day of the prior year: later returns do not matter
    R2 = R.copy()
    R2[4, B.dec_index(cal, 2000) + 1:] = 0.5 * np.where(np.arange(R.shape[1] - B.dec_index(cal, 2000) - 1) % 2 == 0, 1, -1)
    assert B.select_universe(panel, [2000], cal, R2, row_of, top_n=3)[2000]["permno"].tolist() == u[2000]["permno"].tolist()


def test_trailing_vol_matches_numpy_sample_std_with_ddof_one():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.02, (3, 600))
    R[1, 400:420] = np.nan
    vol, cnt = B.trailing_vol(R, [0, 1, 2], 500, 252)
    for i in range(3):
        w = R[i, 249:501]
        w = w[np.isfinite(w)]
        assert vol[i] == pytest.approx(np.std(w, ddof=1) * np.sqrt(252)) and cnt[i] == len(w)
    v, c = B.trailing_vol(R, [0], -1)
    assert np.isnan(v[0]) and c[0] == 0
    assert B.terciles({1: {"vol": np.arange(1.0, 10.0)}})[0] == pytest.approx(np.quantile(np.arange(1.0, 10.0), 1 / 3))


# ----------------------------------------------------------------------------------------------------------- replication
def _secd_frame(tickers_returns: dict, start="2022-01-03", gap_drift=0.0):
    """prepare_prices-ready raw frame with prices built from the given return series."""
    rows = []
    for tic, r in tickers_returns.items():
        dates = pd.bdate_range(start, periods=len(r) + 1)
        px = 100.0 * np.concatenate([[1.0], np.cumprod(1 + np.asarray(r))])
        rows.append(pd.DataFrame({"gvkey": f"g{tic}", "iid": "01", "datadate": dates, "tic": tic, "prccd": px}))
    return pd.concat(rows, ignore_index=True)


def _ff_const(rf=0.0001, start="2021-12-01", periods=500):
    return pd.DataFrame({"date": pd.bdate_range(start, periods=periods), "mktrf": 0.0, "rf": rf})


def _etf_world(L_by_pair=None, drift=0.0, n=130, seed=4, rf=0.0001):
    from qap import prices as PR
    rng = np.random.default_rng(seed)
    out = {}
    for etf, und, L in B.PAIRS:
        if und not in out:
            out[und] = rng.normal(0.002, 0.03, n)
        L_use = L if L_by_pair is None else L_by_pair.get(etf, L)
        out[etf] = B.instrument_returns(out[und], np.full(n, rf), L_use, B.FEE) + drift
    return out


def test_replication_check_passes_on_a_simulated_etf_with_the_same_formula():
    raw = _secd_frame(_etf_world())
    from qap import prices as PR
    frame, _ = PR.prepare_prices(raw)
    rep = B.replication_check(frame, _ff_const())
    assert rep["ok"], rep["reasons"]
    used = [r for r in rep["rows"] if r["used"]]
    assert len(used) == 8 and all(r["slope_ok"] for r in used) and rep["median_abs_gap"] < 1e-6
    assert all(abs(r["slope"] - r["L"]) < 1e-6 for r in used)
    for r in used:
        assert abs(r["te_ann"]) < 1e-6 and r["years"][0]["rank"] == 1 and sum(y["days"] for y in r["years"]) == 130


def test_replication_check_aborts_on_a_corrupted_slope_and_on_a_large_gap_and_with_no_usable_pair():
    from qap import prices as PR
    frame, _ = PR.prepare_prices(_secd_frame(_etf_world(L_by_pair={"SNXX": 1.0})))                  # an "ETF" with slope 1 instead of 2
    rep = B.replication_check(frame, _ff_const())
    assert not rep["ok"] and any("SNXX/SNDK" in x and "slope" in x for x in rep["reasons"])
    frame, _ = PR.prepare_prices(_secd_frame(_etf_world(drift=0.004)))                              # +0.4%/day: slope fine, gap ~0.5 per 130 days
    rep = B.replication_check(frame, _ff_const())
    assert all(r["slope_ok"] for r in rep["rows"] if r["used"]) and not rep["ok"] and any("median absolute" in x for x in rep["reasons"])
    frame, _ = PR.prepare_prices(_secd_frame(_etf_world(n=40)))                                     # < 60 common days: pairs skipped
    rep = B.replication_check(frame, _ff_const())
    assert not rep["ok"] and not any(r["used"] for r in rep["rows"]) and any("no pair" in x for x in rep["reasons"])


def test_replication_rf_gaps_before_the_ff_file_are_zero_and_counted_but_days_after_it_carry_the_last_rf():
    from qap import prices as PR
    frame, _ = PR.prepare_prices(_secd_frame(_etf_world()))
    rep = B.replication_check(frame, _ff_const(start="2022-03-01", periods=20))                     # FF covers only a few days in the middle/start
    used = [r for r in rep["rows"] if r["used"]]
    assert used and all(0 < r["rf_missing"] < r["n"] for r in used)                                 # some gaps counted, the days after the FF file are NOT counted as missing


# --------------------------------------------------------------------------------------------------- loaders / names
def test_load_daily_equals_volfc_subset_and_never_reads_files_after_max_year(tmp_path):
    raw = tmp_path / "d"
    raw.mkdir()
    cal = pd.bdate_range("2018-06-01", "2021-06-30")
    rng = np.random.default_rng(1)
    d = pd.concat([pd.DataFrame({"permno": np.int32(p), "dlycaldt": cal, "dlyret": rng.normal(0, 0.02, len(cal)).astype("float32")}) for p in (1, 2, 3)])
    for y, g in d.groupby(d["dlycaldt"].dt.year):
        g.to_parquet(raw / f"{y}.parquet", index=False)
    R, keep, c1, rows = B.load_daily(raw, [3, 1])
    sub, rows2, c2 = V.load_daily_subset(raw, [3, 1])
    assert np.array_equal(c1, c2) and np.array_equal(R, V.return_matrix(sub, c2, [1, 3])) and keep.tolist() == [1, 3] and rows == rows2
    R19, k19, c19, r19 = B.load_daily(raw, [1], 2019)
    assert str(c19[-1]) == "2019-12-31" and max(r19) == 2019 and R19.shape == (1, len(c19))
    (raw / "2021.parquet").write_text("not a parquet file")                              # a sealed run must not even open it
    assert B.load_daily(raw, [1], 2020)[2][-1] <= np.datetime64("2020-12-31")


def test_rf_series_aligns_on_exact_dates_and_flags_missing():
    ff = pd.DataFrame({"date": ["2000-01-04", "2000-01-05", "2000-01-07"], "mktrf": [0.1, 0.2, 0.3], "rf": [0.0001] * 3})
    out = B.rf_series(ff, np.array(["2000-01-03", "2000-01-04", "2000-01-06", "2000-01-07", "2000-01-10"], dtype="datetime64[D]"))
    assert np.isnan(out[[0, 2]]).all() and out[[1, 3]].tolist() == [0.0001, 0.0001] and out[4] == 0.0001   # 2000-01-10 is after the last FF row: carried forward
    pct = pd.DataFrame({"date": ["2000-01-04"], "mktrf": [1.0], "rf": [5.0]})                    # percent units are converted by ff_decimals
    assert B.rf_series(pct, np.array(["2000-01-04"], dtype="datetime64[D]"))[0] == pytest.approx(0.05)


def test_name_sleeve_reports_trades_win_rate_and_total_returns():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.03, 300)
    rfv = np.full(300, 1e-4)
    out = B.name_sleeve(r, rfv, B.primary("R2"))
    assert out["trades"] > 0 and 0 <= out["win"] <= 1 and out["bh"] == pytest.approx(np.prod(1 + r) - 1) and np.isfinite(out["total"])
    assert B.name_sleeve(r[:10], rfv[:10], B.primary("R1")) is None


# --------------------------------------------------------------------------------------------------------- end to end
def _load_script():
    spec = importlib.util.spec_from_file_location("run_bandsim", pathlib.Path(__file__).parent.parent / "scripts" / "run_bandsim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in path.rglob("*"))


def _write_world(root, secd=True, drift=0.0, seed=21):
    """Synthetic data/raw: 10 stocks on a business-day calendar 1994-2023 (dev, validation, holdout all crossed); permnos 1-5 volatile (>45%),
    6-10 calm; 3 delists in 2010; a secd file with the user's names and 8 simulated ETFs (exact formula) in 2022."""
    raw = pathlib.Path(root) / "data" / "raw"
    (raw / "crsp_daily_ret").mkdir(parents=True)
    m, d, _, _ = make_world(n=10, years=(1993, 2022))
    m.to_parquet(raw / "crsp_monthly.parquet", index=False)
    d.to_parquet(raw / "crsp_delistings.parquet", index=False)
    cal = pd.bdate_range("1994-01-03", "2023-12-29")
    n = len(cal)
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(1, 11):
        vol = (0.045 if p <= 5 else 0.011) * (1 + 0.25 * np.sin(np.arange(n) / (50 + 9 * p) + p))
        r = vol * rng.standard_t(5, n) * np.sqrt(3 / 5)
        keep = np.ones(n, bool)
        if p == 3:
            keep[cal > pd.Timestamp("2010-06-30")] = False
        frames.append(pd.DataFrame({"permno": np.int32(p), "dlycaldt": cal[keep], "dlyret": r[keep].astype("float32")}))
    daily = pd.concat(frames, ignore_index=True)
    for y, g in daily.groupby(daily["dlycaldt"].dt.year):
        g.sort_values(["permno", "dlycaldt"]).to_parquet(raw / "crsp_daily_ret" / f"{y}.parquet", index=False)
    pd.DataFrame({"date": cal, "mktrf": rng.standard_t(5, n) * 0.008, "rf": 0.0001}).iloc[3:].to_parquet(raw / "ff_daily.parquet", index=False)
    if secd:
        rs = _etf_world(drift=drift, n=200)
        for extra in ("INTU", "STX", "ORCL", "CELH", "BE"):
            rs[extra] = rng.normal(0.001, 0.025, 200)
        _secd_frame(rs).to_parquet(raw / "secd_tickers.parquet", index=False)
    return raw


def _run(root, monkeypatch, open_holdout=False, jobs=1, assets=None, draws=B.DRAWS, skip=False, fresh=True):
    mod = _load_script()
    monkeypatch.chdir(root)
    lines = []
    mod._run(lines.append, open_holdout, jobs, assets, draws, skip)
    return lines


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w")
    _write_world(root)
    return root


def _copy_world(world, tmp_path, name):
    import shutil
    dst = tmp_path / name
    shutil.copytree(world, dst)
    return dst


def test_smoke_sealed_run_checks_first_two_trials_no_2020_numbers_and_nothing_in_the_real_data(world, tmp_path, monkeypatch):
    before = _tree(REAL_DATA)
    root = _copy_world(world, tmp_path, "s")
    t0 = time.time()
    lines = _run(root, monkeypatch)
    elapsed = time.time() - t0
    txt = "\n".join(lines)
    order = [txt.index(s) for s in ("== CHECKS", "replication check (simulated", "universe per year", "volatility distribution", "sleeve-years:",
                                    "== RESULTS (Study H", "== LICENCE", "== INFORMATION ONLY")]
    assert order == sorted(order)
    for s in ("transitions (position changes", "share of sleeve-days held flat / long / short", "rf coverage", "per-trade statistics", "sleeve-year maximum drawdown",
              "TIMEOUT, not a loss limit", "benchmarks", "buy-and-hold 2x inverse", "by trailing-volatility tercile", "by calendar year", "R1_N63", "R2_lag0",
              "R3 (INTU-style", "top-100 universe", "WITHHELD", "placebo (200 draws", "BY CONSTRUCTION", "R1 is one trial", "R2 is one trial",
              "unique trials logged so far: 2", "=> PASS"):
        assert s in txt, s
    assert not YEAR2020.findall(txt), YEAR2020.findall(txt)
    assert "HOLDOUT OPENED" not in txt and not (root / "data" / "holdout_v15.done").exists()
    tr = [json.loads(x) for x in (root / "data" / "trials.jsonl").read_text().splitlines()]
    assert [t["name"] for t in tr] == ["studyH:R1_band", "studyH:R2_momentum"] and all(t["spec"]["prereg"] == "v15" for t in tr)
    lines2 = _run(root, monkeypatch)                                          # rerun: identical report, no duplicate trial
    assert len((root / "data" / "trials.jsonl").read_text().splitlines()) == 2 and "already in the registry" in "\n".join(lines2)
    strip = lambda ls: [x for x in ls if "registry" not in x and "logged" not in x and "engine:" not in x]
    assert strip(lines) == strip(lines2)
    assert _tree(REAL_DATA) == before
    print(f"smoke sealed run: {elapsed:.1f}s")


def test_universe_in_the_smoke_world_uses_only_the_volatile_stocks_and_ends_with_the_delisting(world, tmp_path, monkeypatch):
    root = _copy_world(world, tmp_path, "u")
    lines = _run(root, monkeypatch, draws=5)
    txt = "\n".join(lines)
    row = {int(m.group(1)): int(m.group(4)) for m in re.finditer(r"^\s+(\d{4})\s+(\d+)\s+(\d+)\s+(\d+)$", txt, re.M)}
    assert row[1996] == 4 and row[2005] == 4 and row[2010] == 4 and row[2011] == 3 and row[2015] == 3     # permnos 1+2 are one company; permno 3 is out once it has no returns in the trailing year
    assert "partial run" in txt and not (root / "data" / "trials.jsonl").exists()


def test_jobs_one_and_two_give_identical_reports(world, tmp_path, monkeypatch):
    r1 = _run(_copy_world(world, tmp_path, "j1"), monkeypatch, jobs=1, draws=20)
    r2 = _run(_copy_world(world, tmp_path, "j2"), monkeypatch, jobs=2, draws=20)
    strip = lambda ls: [x for x in ls if "engine:" not in x]
    assert strip(r1) == strip(r2)


def test_the_sealed_run_ignores_2020_plus_data_entirely(world, tmp_path, monkeypatch):
    a = _copy_world(world, tmp_path, "a")
    b = _copy_world(world, tmp_path, "b")
    for y in (2020, 2021, 2022, 2023):                     # destroy every sealed daily file in b: the sealed report must not change
        (b / "data" / "raw" / "crsp_daily_ret" / f"{y}.parquet").write_text("garbage")
    la, lb = _run(a, monkeypatch, draws=10), _run(b, monkeypatch, draws=10)
    strip = lambda ls: [x for x in ls if "engine:" not in x]
    assert strip(la) == strip(lb)


def test_partial_runs_log_no_trials_and_trials_are_not_duplicated(world, tmp_path, monkeypatch):
    root = _copy_world(world, tmp_path, "p")
    lines = _run(root, monkeypatch, assets=2, draws=5)
    assert "NO trials logged" in "\n".join(lines) and not (root / "data" / "trials.jsonl").exists() and "unique trials logged so far: 0" in lines[-1]
    _run(root, monkeypatch)
    _run(root, monkeypatch)
    assert len((root / "data" / "trials.jsonl").read_text().splitlines()) == 2


def test_missing_secd_aborts_with_the_hint_unless_skip_replication_and_prints_skipped(tmp_path, monkeypatch):
    root = tmp_path / "n"
    root.mkdir()
    _write_world(root, secd=False)
    mod = _load_script()
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit, match="extract_secd.py"):
        mod.main(False, 1, 2, 5, False)
    rep = (root / "data" / "study_h_report.txt").read_text()
    assert "SKIPPED" not in rep and "== RESULTS" not in rep
    mod.main(False, 1, 2, 5, True)
    rep = (root / "data" / "study_h_report.txt").read_text()
    assert "replication check SKIPPED (no secd file)" in rep and "== RESULTS" in rep and rep.index("SKIPPED") < rep.index("universe per year")
    assert "n/a (no secd file)" in rep


def test_replication_failure_aborts_before_any_result_but_keeps_the_checks_in_the_report(tmp_path, monkeypatch):
    root = tmp_path / "r"
    root.mkdir()
    _write_world(root, drift=0.004)
    monkeypatch.chdir(root)
    with pytest.raises(AssertionError, match="replication check failed"):
        _load_script().main(False, 1, 2, 5, False)
    rep = (root / "data" / "study_h_report.txt").read_text()
    assert "== CHECKS" in rep and "ABORT" in rep and "FAIL" in rep and "== RESULTS" not in rep and not (root / "data" / "trials.jsonl").exists()
    assert not YEAR2020.findall(rep)                       # relative years of common data: no secd calendar year in a sealed report


def test_replication_slope_failure_aborts_the_script_before_any_result(tmp_path, monkeypatch):
    root = tmp_path / "sl"
    root.mkdir()
    _write_world(root)
    from qap import prices as PR
    raw = _secd_frame(_etf_world(L_by_pair={"SNXX": 1.0}, n=200))
    raw.to_parquet(root / "data" / "raw" / "secd_tickers.parquet", index=False)
    monkeypatch.chdir(root)
    with pytest.raises(AssertionError, match="replication check failed"):
        _load_script().main(False, 1, 2, 5, False)
    rep = (root / "data" / "study_h_report.txt").read_text()
    assert "SLOPE OUT OF RANGE" in rep and "ABORT" in rep and "== RESULTS" not in rep and not (root / "data" / "trials.jsonl").exists()


def _fake_pass(monkeypatch):
    real = B.licence_verdict

    def fake(dev, val, plc, hold=None, plc_hold=None, open_holdout=False):
        if hold is None:
            return real({"mean": 1.0, "t": 5.0}, {"mean": 1.0, "t": 5.0}, {"p95": 0.0, "median": 0.0, "n": 1}, None, None, open_holdout)
        return real({"mean": 1.0, "t": 5.0}, {"mean": 1.0, "t": 5.0}, {"p95": 0.0, "median": 0.0, "n": 1}, hold, plc_hold, open_holdout)
    monkeypatch.setattr(B, "licence_verdict", fake)


def test_open_holdout_without_a_passing_rule_does_not_open_it(world, tmp_path, monkeypatch):
    root = _copy_world(world, tmp_path, "h0")
    lines = _run(root, monkeypatch, open_holdout=True, draws=5)
    txt = "\n".join(lines)
    if "PASSED IN DEV/VALIDATION" not in txt:
        assert "NOT opened" in txt and "HOLDOUT OPENED" not in txt and not (root / "data" / "holdout_v15.done").exists()
    assert "IN-SAMPLE" in txt                                                  # the secd names print with the flag (v13 precedent)


def test_holdout_is_opened_once_marker_written_before_printing_and_a_second_opening_is_refused(world, tmp_path, monkeypatch):
    root = _copy_world(world, tmp_path, "h1")
    sealed = _run(root, monkeypatch, draws=B.DRAWS)
    stxt = "\n".join(sealed)
    assert "WITHHELD" in stxt and not YEAR2020.findall(stxt)
    root2 = _copy_world(world, tmp_path, "h2")
    _fake_pass(monkeypatch)
    sealed_p = _run(root2, monkeypatch, draws=B.DRAWS)
    assert "PASSED IN DEV/VALIDATION" in "\n".join(sealed_p) and "HOLDOUT OPENED" not in "\n".join(sealed_p) and not (root2 / "data" / "holdout_v15.done").exists()
    seen, opened = [], []
    real_mark = V.mark_holdout_opened
    monkeypatch.setattr(V, "mark_holdout_opened", lambda text, marker=None: (seen.append(len(opened)), real_mark(text, marker)))
    mod = _load_script()
    monkeypatch.chdir(root2)
    mod._run(opened.append, True, 1, None, B.DRAWS, False)
    otxt = "\n".join(opened)
    assert "HOLDOUT OPENED (once)" in otxt and "2020-01..2025-12" in otxt and ("LICENSED (holdout stands)" in otxt or "FAILED IN THE HOLDOUT" in otxt)
    assert len(seen) == 1 and seen[0] < len(opened) and "HOLDOUT OPENED" not in "\n".join(opened[:seen[0]])      # marker before the holdout numbers
    marker = root2 / "data" / "holdout_v15.done"
    assert marker.exists() and marker.read_text().startswith("v15 holdout opened") and "R1:" in marker.read_text()
    k = next(i for i, x in enumerate(sealed_p) if "== INFORMATION ONLY" in x)
    j = next(i for i, x in enumerate(opened) if "== INFORMATION ONLY" in x)
    pre = lambda ls, n: [x for x in ls[:n] if "engine:" not in x and "registry" not in x and "logged" not in x and "calendar:" not in x and "candidate" not in x and not re.search(r"y1\(\d+d\)", x)]
    assert pre(sealed_p, next(i for i, x in enumerate(sealed_p) if "== LICENCE" in x)) == pre(opened, next(i for i, x in enumerate(opened) if "== LICENCE" in x))
    with pytest.raises(SystemExit, match="ALREADY OPENED"):
        mod._run(lambda *_: None, True, 1, None, B.DRAWS, False)
    (root2 / "data" / "study_h_report.txt").write_text("KEEP\n")
    with pytest.raises(SystemExit, match="ALREADY OPENED"):
        mod.main(True, 1, None, B.DRAWS, False)
    assert (root2 / "data" / "study_h_report.txt").read_text() == "KEEP\n"
    assert len((root2 / "data" / "trials.jsonl").read_text().splitlines()) == 2


def test_script_requires_its_inputs_with_a_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="extract_earnings.py"):
        _load_script()._run(lambda *_: None, False)


def test_marker_helpers_use_the_v15_marker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    V.assert_holdout_unopened(B.MARKER, B.STUDY)
    V.mark_holdout_opened("x", B.MARKER)
    with pytest.raises(V.HoldoutRefused, match="v15 .*ALREADY OPENED"):
        V.assert_holdout_unopened(B.MARKER, B.STUDY)


def test_rf_after_the_last_ff_row_is_carried_forward_but_interior_gaps_stay_nan():
    import numpy as np
    import pandas as pd
    from qap.bandsim import rf_series
    ff = pd.DataFrame({"date": pd.to_datetime(["2025-12-01", "2025-12-02", "2025-12-04"]), "mktrf": [0.0, 0.0, 0.0], "rf": [0.00010, 0.00011, 0.00012]})
    d = np.array(["2025-12-01", "2025-12-03", "2025-12-04", "2026-01-05", "2026-01-06"], dtype="datetime64[D]")
    out = rf_series(ff, d)
    assert out[0] == 0.00010 and np.isnan(out[1]) and out[2] == 0.00012 and out[3] == 0.00012 and out[4] == 0.00012
