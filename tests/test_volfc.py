import importlib.util
import os
import pathlib
import re
import time

import numpy as np
import pandas as pd
import pytest

from qap import volfc as V

from test_pipeline import make_world

from scipy.integrate import trapezoid as _trapz

REAL_DATA = pathlib.Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------------------------------------- helpers
def sim_gjr(n, a, g, b, nu=None, seed=0, s2=1e-4):
    """Simulated GJR(1,1) returns; standardised-t shocks (nu) or Gaussian (None)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) if nu is None else rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    om = (1 - (a + g / 2 + b)) * s2
    r, v = np.empty(n), s2
    for k in range(n):
        r[k] = np.sqrt(v) * z[k]
        v = om + (a + (g if r[k] < 0 else 0.0)) * r[k] ** 2 + b * v
    return r


# ------------------------------------------------------------------------------------------ pre-registered constants
def test_constants_are_the_pre_registered_values():
    assert (V.WINDOW, V.MIN_OBS, V.REFIT, V.HIST_DAYS, V.EWMA_LAMBDA) == (1000, 950, 21, 252, 0.94)
    assert V.WINS_Q == (0.5, 99.5) and V.NU_BOUNDS == (2.1, 100.0) and V.P_BREACH == 0.999 and V.HAR_FLOOR == 0.1
    assert V.MC_PATHS == 20000 and V.NW_LAGS == {1: 5, 21: 30} and V.MATERIALITY == -0.01 and V.DM_T == -2.0
    assert V.DEV == ("1995-01-01", "2014-12-31") and V.VAL == ("2015-01-01", "2019-12-31") and V.HOLD == ("2020-01-01", "2025-12-31")
    assert V.TOP_N == 100 and V.MODELS == ("hist", "ewma", "garch", "gjr_t", "gjr_t_wins", "har")
    assert V.SIZING == ("hist", "ewma", "garch", "gjr_t", "har") and V.TICKERS == ("ORCL", "INTU", "LITE", "BE", "STX", "AMD", "CELH", "ALAB", "SNDK")


# ------------------------------------------------------------------------------------------------------ recursion
@pytest.mark.parametrize("a,g,b", [(0.05, 0.0, 0.93), (0.03, 0.08, 0.90), (0.0, 0.0, 0.0), (0.2, 0.3, 0.5), (0.02, 0.0, 0.9999 - 0.02)])
def test_vectorised_recursion_equals_the_slow_loop(a, g, b):
    r = np.random.default_rng(1).standard_t(4, 3000) * 0.01
    s2 = float(np.mean(r * r))
    v, w = V.sigma2_path(r, a, g, b, s2), V.recursion_slow(r, a, g, b, s2)
    assert len(v) == len(r) + 1 and v[0] == s2
    assert np.max(np.abs(v - w) / w) < 1e-10
    assert V.verify_recursion(r) < 1e-10


def test_verify_recursion_aborts_when_the_tolerance_is_missed():
    r = np.random.default_rng(2).standard_normal(500) * 0.01
    with pytest.raises(AssertionError, match="reference loop"):
        V.verify_recursion(r, tol=-1.0)


def test_recursion_value_at_t_depends_only_on_returns_up_to_t():
    r = np.random.default_rng(3).standard_normal(400) * 0.01
    r2 = r.copy()
    r2[300:] = 5.0
    a, b = V.sigma2_path(r, 0.05, 0.05, 0.88, 1e-4), V.sigma2_path(r2, 0.05, 0.05, 0.88, 1e-4)
    assert np.array_equal(a[:301], b[:301]) and a[301] != b[301]


# ---------------------------------------------------------------------------------------------------- densities
def test_standardised_t_density_integrates_to_one_with_unit_variance_and_check_helper():
    chk = V.t_density_check()
    assert V.t_density_ok(chk)
    for nu, (one, var) in chk.items():
        assert abs(one - 1) < 1e-8, nu
        if np.isfinite(var):
            assert abs(var - 1) < 1e-6, nu
    assert not V.t_density_ok({5.0: (1.01, 1.0)})
    x = np.linspace(-0.2, 0.2, 400001)
    assert abs(_trapz(np.exp(V.normal_logpdf(x, 0.0004)), x) - 1) < 1e-6
    assert abs(_trapz(np.exp(V.std_t_logpdf(x, 0.0004, 6.0)), x) - 1) < 2e-3        # truncated tails
    # large nu -> Gaussian
    assert abs(float(V.std_t_logpdf(0.01, 1e-4, 1e6)) - float(V.normal_logpdf(0.01, 1e-4))) < 1e-4


# ----------------------------------------------------------------------------------------------------- estimators
def test_parameter_recovery_on_a_simulated_gjr_t_process():
    r = sim_gjr(12000, 0.03, 0.06, 0.91, nu=6.0, seed=11)
    f = V.fit_gjr_t(r)
    assert f["status"] == "ok"
    assert abs(f["p"] - 0.97) < 0.025 and abs(f["alpha"] - 0.03) < 0.02 and abs(f["gamma"] - 0.06) < 0.04 and abs(f["beta"] - 0.91) < 0.03
    assert 4.5 < f["nu"] < 8.5
    assert f["s2"] == pytest.approx(np.mean(r * r))


def test_garch_normal_qmle_recovers_a_gaussian_garch():
    r = sim_gjr(12000, 0.06, 0.0, 0.91, nu=None, seed=5)
    f = V.fit_garch_normal(r)
    assert f["status"] == "ok" and abs(f["p"] - 0.97) < 0.025 and abs(f["alpha"] - 0.06) < 0.025 and np.isnan(f["nu"]) and f["gamma"] == 0.0


def test_fits_are_deterministic_and_scale_free():
    r = sim_gjr(1500, 0.04, 0.05, 0.9, nu=7.0, seed=2)
    a, b, c = V.fit_gjr_t(r), V.fit_gjr_t(r), V.fit_gjr_t(r * 37.0)
    assert a == b
    for k in ("alpha", "gamma", "beta", "nu"):
        assert a[k] == pytest.approx(c[k], rel=1e-5, abs=1e-6)


def test_near_integrated_window_is_a_persistence_breach_not_a_crash():
    r = sim_gjr(975, 0.12, 0.0, 0.88, nu=None, seed=202)
    assert V.fit_gjr_t(r)["status"] == "breach"
    assert V.fit_garch_normal(np.zeros(500))["status"] == "failed"                  # degenerate: zero second moment
    assert V.fit_gjr_t(np.r_[np.random.default_rng(0).normal(size=300), np.nan])["status"] == "failed"


def test_winsorisation_uses_only_the_windows_own_percentiles():
    w = np.random.default_rng(4).standard_t(3, 975) * 0.02
    c, lo, hi = V.winsorise(w)
    assert lo == np.percentile(w, 0.5) and hi == np.percentile(w, 99.5)
    assert c.min() == lo and c.max() == hi and np.array_equal(c[(w > lo) & (w < hi)], w[(w > lo) & (w < hi)])
    assert (c != w).sum() >= 8
    w2 = np.r_[w, 10.0]                                                            # a later return does not enter an earlier window's percentiles
    c2, lo2, hi2 = V.winsorise(w)
    assert (lo2, hi2) == (lo, hi)
    assert V.winsorise(w2)[2] != hi


def test_analytic_h_step_mean_and_har_pieces():
    assert V.avg_var_forecast(3.0, 1.0, 0.9, 1) == pytest.approx(3.0)
    h = 21
    ex = np.mean([1.0 + 0.9 ** (k - 1) * 2.0 for k in range(1, h + 1)])
    assert V.avg_var_forecast(3.0, 1.0, 0.9, h) == pytest.approx(ex)
    assert V.avg_var_forecast(1.0, 1.0, 0.99, h) == pytest.approx(1.0)
    # HAR design: only rows whose target lies inside the window; matches a direct loop
    rng = np.random.default_rng(6)
    w = rng.standard_normal(80) * 0.01
    s2 = np.mean(w * w)
    for hh in (1, 5):
        X, y = [], []
        for t in range(21, len(w) - hh):
            u = w * w / s2
            X.append([1, u[t], u[t - 4:t + 1].mean(), u[t - 21:t + 1].mean()])
            y.append(u[t + 1:t + hh + 1].mean())
        beta, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
        assert np.allclose(V.har_fit(w, hh), beta)
    assert V.har_fit(w[:30], 21) is None


def test_har_forecast_is_floored_at_a_tenth_of_the_window_second_moment_and_hits_are_flagged():
    x = np.random.default_rng(7).standard_normal(60) * 0.01
    s2 = 1e-4
    f, hit = V.har_forecast(x, np.array([30, 40]), np.array([-1.0, 0.0, 0.0, 0.0]) , s2)
    assert np.allclose(f, 0.1 * s2) and hit.all()
    f, hit = V.har_forecast(x, np.array([30, 40]), np.array([1.0, 0.0, 0.0, 0.0]), s2)
    assert np.allclose(f, s2) and not hit.any()


# ------------------------------------------------------------------------------------------- targets, losses, DM
def test_h21_targets_require_a_complete_calendar_window():
    r = np.random.default_rng(8).standard_normal(100) * 0.01
    r[30] = np.nan
    y1, r1, y21, r21 = V.targets(r)
    assert y1[10] == pytest.approx(r[11] ** 2) and r1[10] == r[11]
    assert np.isnan(y1[29]) and np.isnan(r1[29])                                   # calendar t+1 missing
    assert y21[5] == pytest.approx(np.mean(r[6:27] ** 2)) and r21[5] == pytest.approx(r[6:27].sum())
    for j in range(9, 30):                                                        # windows j+1..j+21 contain index 30
        assert np.isnan(y21[j]) and np.isnan(r21[j])
    assert y21[8] == pytest.approx(np.mean(r[9:30] ** 2))                          # window 9..29 stops just before the gap
    assert np.isnan(y21[79:]).all() and np.isfinite(y21[31]) and np.isnan(y21[-1]) and np.isnan(y1[-1])


def test_qlike_definition_and_zero_proxy_is_defined():
    assert V.qlike(2.0, 3.0) == pytest.approx(np.log(2.0) + 1.5)
    assert np.isfinite(V.qlike(1e-4, 0.0)) and V.mse_loss(2.0, 5.0) == 9.0


def test_newey_west_matches_a_hand_computed_case():
    m, t = V.newey_west_t([1.0, 2.0, 3.0, 4.0], 1)
    # e = [-1.5,-.5,.5,1.5]; g0 = 1.25; g1 = .3125; long-run var = 1.25 + 2 (1 - 1/2) .3125 = 1.5625; t = 2.5 / sqrt(1.5625 / 4) = 4
    assert m == 2.5 and t == pytest.approx(4.0)
    m0, t0 = V.newey_west_t([1.0, 2.0, 3.0, 4.0], 0)
    assert t0 == pytest.approx(2.5 / np.sqrt(1.25 / 4))
    assert np.isnan(V.newey_west_t([1.0, 1.0, 1.0, 1.0], 2)[1])
    m3, t3 = V.newey_west_t([1.0, 2.0, 3.0, 4.0, 5.0], 10)                          # lags are capped at T - 1
    assert np.isfinite(t3)


def test_kupiec_and_compare_use_the_window_dates():
    assert V.kupiec_pvalue(10, 1000, 0.01) == pytest.approx(1.0, abs=1e-6)
    assert V.kupiec_pvalue(40, 1000, 0.01) < 1e-6 and 0 <= V.kupiec_pvalue(0, 100, 0.01) <= 1
    cal = pd.date_range("1994-06-01", periods=120, freq="MS").to_numpy().astype("datetime64[D]")
    dl = pd.DataFrame({"a": np.r_[np.zeros(60), np.ones(60)], "b": 0.0}, index=np.arange(120))
    m, t, nd = V.compare(dl, cal, "a", "b", 1, ("1995-01-01", "1995-12-31"))
    assert nd == 12 and m == 0.0
    m, t, nd = V.compare(dl, cal, "a", "b", 1, ("1999-06-01", "1999-12-31"))
    assert m == 1.0 and nd == 7


# --------------------------------------------------------------------------------------------------- licence rule
def _comps(dev=(-0.02, -3.0), val=(-0.01, -1.0), override=None):
    c = {(a, b, h): {"dev": dev, "val": val} for a, b in V.COMPARISONS for h in V.HORIZONS}
    c.update(override or {})
    return c


def test_licence_rule_all_four_must_pass_and_holdout_is_hidden_unless_opened():
    hold = {(a, b, h): -0.001 for a, b in V.COMPARISONS for h in V.HORIZONS}
    v = V.licence_verdict(_comps())
    assert v["dev_val"] and not v["opened"] and "sealed" in v["status"] and v["holdout_all_negative"] is None
    assert V.licence_verdict(_comps(), hold, open_holdout=False) == v                         # holdout numbers are ignored without the flag
    k = ("gjr_t", "har", 21)
    for bad in ({"dev": (-0.005, -3.0), "val": (-0.01, -1.0)}, {"dev": (-0.02, -1.5), "val": (-0.01, -1.0)}, {"dev": (-0.02, -3.0), "val": (0.0, 1.0)},
                {"dev": (np.nan, np.nan), "val": (-0.01, -1.0)}):
        out = V.licence_verdict(_comps(override={k: bad}), hold, True)
        assert not out["dev_val"] and not out["opened"] and out["status"].startswith("NOT LICENSED")
    edge = V.licence_verdict(_comps(dev=(-0.01, -2.0), val=(-1e-9, 0.0)))
    assert edge["dev_val"]                                                                    # thresholds are inclusive on dev, strict on validation
    assert not V.licence_verdict(_comps(val=(0.0, 0.0)))["dev_val"]
    stands = V.licence_verdict(_comps(), hold, True)
    assert stands["opened"] and stands["holdout_all_negative"] and stands["status"].startswith("LICENSED (")
    hold2 = dict(hold)
    hold2[("gjr_t", "garch", 1)] = 0.0
    fail = V.licence_verdict(_comps(), hold2, True)
    assert fail["opened"] and not fail["holdout_all_negative"] and fail["status"] == "licensed in dev/validation, failed in the holdout"
    with pytest.raises(KeyError):
        V.licence_verdict({k: v for k, v in _comps().items() if k[2] == 1})
    with pytest.raises(ValueError):
        V.licence_verdict(_comps(), None, True)


def test_default_sizing_model_is_the_lowest_dev_qlike_at_h21_and_excludes_m4():
    assert V.default_sizing_model({"hist": 1.0, "ewma": 0.9, "garch": 0.8, "gjr_t": 0.85, "har": 0.95}) == "garch"
    assert V.default_sizing_model({"hist": 1.0, "ewma": 0.9, "garch": 0.8, "gjr_t": 0.7, "har": np.nan}) == "gjr_t"


# ---------------------------------------------------------------------------------------------------- Monte Carlo
def test_mc_mean_variance_equals_analytic_within_sampling_error_and_error_shrinks_with_paths():
    a, g, b, nu, s2, nxt = 0.04, 0.06, 0.90, 7.0, 1.0, 2.5
    p = a + g / 2 + b
    omega = (1 - p) * s2
    exact = s2 + p ** np.arange(21) * (nxt - s2)
    errs = {}
    for n in (400, 40000):
        e = []
        for seed in range(12):
            _, m_r2, m_var = V.mc_cumret(nxt, a, g, b, omega, nu, np.random.default_rng(seed), n)
            e.append(np.mean(np.abs(m_var / exact - 1)))
        errs[n] = np.mean(e)
    assert errs[40000] < 0.02 and errs[40000] < errs[400] / 4
    _, m_r2, m_var = V.mc_cumret(nxt, a, g, b, omega, nu, np.random.default_rng(99), 200000)
    assert np.allclose(m_var, exact, rtol=0.02)
    assert abs(np.mean(m_r2) / np.mean(exact) - 1) < 0.05                       # r^2 itself is noisier (heavy tails)
    # Gaussian garch, and the analytic formula used by the forecasters
    _, _, mv = V.mc_cumret(nxt, 0.05, 0.0, 0.9, 0.05 * s2, None, np.random.default_rng(1), 100000)
    assert np.mean(mv) == pytest.approx(float(V.avg_var_forecast(nxt, s2, 0.95, 21)), rel=0.01)


def test_mc_is_seeded_deterministically():
    a = V.mc_cumret(1.0, 0.05, 0.05, 0.9, 0.0, 6.0, np.random.default_rng([4, 21]), 1000)[0]
    b = V.mc_cumret(1.0, 0.05, 0.05, 0.9, 0.0, 6.0, np.random.default_rng([4, 21]), 1000)[0]
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------------------------------- rolling engine
@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(V, "WINDOW", 300)
    monkeypatch.setattr(V, "MIN_OBS", 285)
    monkeypatch.setattr(V, "HIST_DAYS", 100)
    monkeypatch.setattr(V, "MC_PATHS", 300)


def _ret(n=900, seed=3, offset=0):
    r = np.full(n + offset, np.nan)
    r[offset:] = sim_gjr(n, 0.04, 0.06, 0.90, nu=6.0, seed=seed)
    return r


def test_engine_rows_models_flags_and_proxies(small):
    ret = _ret()
    out = V.run_asset(ret, [(400, 700)], 7)
    rows = out["rows"]
    assert set(V.ROW_FLOATS) | set(V.ROW_BOOLS) | {"aid", "tidx", "refit"} == set(rows)
    n = len(rows["tidx"])
    assert n == 301 and rows["tidx"].min() == 400 and rows["tidx"].max() == 700 and all(len(v) == n for v in rows.values())
    assert (rows["refit"] % 21 == 0).all() and (rows["tidx"] - rows["refit"] <= 20).all() and (rows["tidx"] >= rows["refit"]).all()
    for m in V.MODELS:
        for h in V.HORIZONS:
            assert np.isfinite(rows[f"f{h}_{m}"]).all() and (rows[f"f{h}_{m}"] > 0).all()
    assert (rows["y1"][:-1] == ret[rows["tidx"][:-1] + 1] ** 2).all()
    assert np.isfinite(rows["mc01_gjr_t"]).sum() == np.isfinite(rows["mc05_garch"]).sum() > 5          # MC only at refit dates
    refit_rows = rows["tidx"] == rows["refit"]
    assert np.isfinite(rows["mc01_gjr_t"][~refit_rows]).sum() == 0
    assert not rows["fb_gjr_t"].any() and out["blocks"][:, 1].all()
    assert len(out["fits"]) == 3 * len(out["blocks"])
    # forecasts at the refit date: h=1 equals the filtered one-step variance from the window with the fitted parameters
    i = int(np.flatnonzero(refit_rows)[2])
    t_r = rows["tidx"][i]
    w = ret[t_r - 299:t_r + 1]
    f = V.fit_garch_normal(w[np.isfinite(w)])
    s2 = np.mean(w[np.isfinite(w)] ** 2)
    assert rows["f1_garch"][i] == pytest.approx(V.sigma2_path(w[np.isfinite(w)], f["alpha"], 0.0, f["beta"], s2)[-1], rel=1e-9)
    assert rows["f21_garch"][i] == pytest.approx(V.avg_var_forecast(rows["f1_garch"][i], s2, f["p"], 21), rel=1e-9)
    assert rows["f1_hist"][i] == pytest.approx(np.mean(w[np.isfinite(w)][-100:] ** 2))


def test_no_look_ahead_forecasts_at_t_do_not_change_when_later_returns_change(small):
    ret = _ret()
    T = 640                                                                         # inside a refit block
    ret2 = ret.copy()
    ret2[T + 1:] = np.random.default_rng(0).standard_normal(len(ret) - T - 1) * 0.2
    a = V.run_asset(ret, [(400, 800)], 3, mc=False)["rows"]
    b = V.run_asset(ret2, [(400, 800)], 3, mc=False)["rows"]
    keep = a["tidx"] <= T
    assert keep.sum() > 200
    cols = [c for c in a if c not in ("y1", "r1", "y21", "r21s") and not c.startswith("mc")]
    for c in cols:
        assert np.array_equal(a[c][keep], b[c][keep], equal_nan=True), c
    assert not np.array_equal(a["f1_garch"][~keep], b["f1_garch"][~keep])           # sanity: later forecasts do change
    assert not np.array_equal(a["y1"][keep], b["y1"][keep])                          # targets are the only thing allowed to look forward


def test_shock_after_the_window_is_winsorised_at_the_windows_own_bounds_and_leaves_fits_unchanged(small):
    ret = _ret()
    t_r = 21 * 25
    shock = ret.copy()
    shock[t_r + 3] = -0.9
    a = V.run_asset(ret, [(t_r, t_r + 20)], 1, mc=False)
    b = V.run_asset(shock, [(t_r, t_r + 20)], 1, mc=False)
    assert pd.DataFrame(a["fits"]).equals(pd.DataFrame(b["fits"]))                                                    # the window (ends at t_r) does not see the shock
    ra, rb = a["rows"], b["rows"]
    after = ra["tidx"] > t_r + 3
    assert (rb["f1_gjr_t"][after] > ra["f1_gjr_t"][after]).all()
    assert (rb["f1_gjr_t"][after] / ra["f1_gjr_t"][after]).max() > 1.2 * (rb["f1_gjr_t_wins"][after] / ra["f1_gjr_t_wins"][after]).max()
    w = ret[t_r - 299:t_r + 1]
    _, lo, hi = V.winsorise(w)
    f = V.fit_gjr_t(np.clip(w, lo, hi))
    x = np.concatenate([w, shock[t_r + 1:t_r + 21]])
    path = V.sigma2_path(np.clip(x, lo, hi), f["alpha"], f["gamma"], f["beta"], np.mean(np.clip(w, lo, hi) ** 2))
    j = int(np.flatnonzero(rb["tidx"] == t_r + 10)[0])
    assert rb["f1_gjr_t_wins"][j] == pytest.approx(path[len(w) + 10], rel=1e-9)


def test_degenerate_window_triggers_the_fallback_chain_is_counted_and_scored(small):
    ret = np.full(900, np.nan)
    ret[:] = sim_gjr(900, 0.12, 0.0, 0.88, nu=None, seed=202)                       # near-integrated: persistence >= 0.999
    out = V.run_asset(ret, [(400, 700)], 9, mc=True)
    rows = pd.DataFrame(out["rows"])
    fits = pd.DataFrame(out["fits"], columns=V.FIT_COLS)
    assert (fits["status"] == "breach").sum() >= 3
    assert rows["fb_gjr_t"].any() and rows["fb_gjr_t"].mean() == pytest.approx(fits.loc[fits["model"] == "gjr_t", "status"].ne("ok").mean(), abs=0.1)
    fb = rows[rows["fb_gjr_t"]]
    assert np.array_equal(fb["f1_gjr_t"], fb["f1_garch"]) and np.array_equal(fb["f21_gjr_t"], fb["f21_garch"])
    fg = rows[rows["fb_garch"]]
    assert len(fg) > 0 and np.array_equal(fg["f1_garch"], fg["f1_ewma"]) and np.array_equal(fg["f21_garch"], fg["f21_ewma"])
    assert fb["nu_gjr_t"].isna().all()
    # fallback rows STAY in the evaluation
    masks = V.valid_masks(rows)
    assert masks[1][rows["fb_gjr_t"].to_numpy()].mean() > 0.9
    q = V.date_losses(rows, masks[1], 1)
    assert np.isfinite(q["gjr_t"]).all() and len(q) >= rows.loc[masks[1], "tidx"].nunique()
    # log score and VaR treat a fallback as Normal
    ls = V.row_logscore(rows, masks[1] & np.isfinite(rows["r1"].to_numpy()))
    j = rows["fb_gjr_t"].to_numpy()[masks[1] & np.isfinite(rows["r1"].to_numpy())]
    assert np.allclose(ls["gjr_t"][j], V.normal_logpdf(rows["r1"].to_numpy()[masks[1] & np.isfinite(rows["r1"].to_numpy())][j],
                                                       rows["f1_gjr_t"].to_numpy()[masks[1] & np.isfinite(rows["r1"].to_numpy())][j]))


def test_history_rule_skips_short_windows_and_counts_them(small):
    ret = _ret(n=700, offset=300)
    out = V.run_asset(ret, [(300, 600)], 2, mc=False)
    blocks = out["blocks"]
    assert (blocks[:, 1] == 0).any() and blocks[:, 1].any()
    assert out["rows"]["tidx"].min() >= 21 * int(np.ceil(584 / 21)) - 20
    ok_refits = set(blocks[blocks[:, 1] == 1, 0])
    assert set(out["rows"]["refit"]) == ok_refits
    empty = V.run_asset(np.full(500, np.nan), [(0, 499)], 1)
    assert len(empty["rows"]["tidx"]) == 0 and empty["blocks"].shape == (0, 2)


def test_missing_days_in_the_window_are_dropped_and_forecast_dates_need_a_return(small):
    ret = _ret()
    ret[500:510] = np.nan
    out = V.run_asset(ret, [(480, 560)], 4, mc=False)
    t = out["rows"]["tidx"]
    assert not np.isin(np.arange(500, 510), t).any() and np.isin(np.arange(480, 500), t).all()
    assert (out["blocks"][:, 1] == 1).all()


def test_engine_results_are_identical_across_runs(small):
    ret = _ret()
    a, b = V.run_asset(ret, [(450, 650)], 5), V.run_asset(ret, [(450, 650)], 5)
    for c in a["rows"]:
        assert np.array_equal(a["rows"][c], b["rows"][c], equal_nan=True)


# -------------------------------------------------------------------------------------------------------- universe
def _panel(n_years=4):
    m, d, _, _ = make_world(n=8, years=(1996, 1996 + n_years))
    from qap.panel import build_monthly_panel
    return m, d, build_monthly_panel(m, d)


def test_universe_is_rolling_by_prior_december_company_me_with_largest_security_per_company_and_no_survivor_filter():
    m, d, _ = _panel()
    m = m.copy()
    dec = lambda y: m["mthcaldt"] == pd.Timestamp(f"{y}-12-31")
    caps = {1998: {1: 100.0, 2: 900.0, 3: 500.0, 4: 400.0, 5: 300.0, 6: 50.0, 7: 40.0, 8: 30.0},
            1999: {1: 100.0, 2: 90.0, 3: 20.0, 4: 400.0, 5: 700.0, 6: 650.0, 7: 40.0, 8: 30.0}}
    for y, cs in caps.items():
        for p, c in cs.items():
            m.loc[dec(y) & (m["permno"] == p), "mthcap"] = c * 1000.0
    m = m[~((m["permno"] == 3) & (m["mthcaldt"] > pd.Timestamp("1999-03-31")))]     # permno 3 vanishes during 1999 (a delisting): still selected for 1999 on Dec-1998 ME
    from qap.panel import build_monthly_panel
    p = build_monthly_panel(m, d)
    uni = V.top_universe(p, [1999, 2000], 3)
    # permnos 1 and 2 are one company (permco 1): ME 1000 -> its representative is the larger security, permno 2 (900)
    assert list(uni[1999]) == [2, 3, 4]
    assert 3 in uni[1999] and 6 not in uni[1999] and 8 not in uni[1999]              # stock outside the prior-December top-N is excluded
    assert list(uni[2000]) == [4, 5, 6]                                              # Dec-1999 ranking: 700, 650, 400 (permno 2 = 90 fell out)
    sy = V.stock_years(uni)
    assert sy[3] == [1999] and sy[2] == [1999]
    rng = V.year_ranges(np.array(["1999-12-30", "2000-01-03", "2000-06-01", "2001-01-02"], dtype="datetime64[D]"), [1999, 2000, 2002])
    assert rng == [(0, 0), (1, 2)]


def test_delisted_stock_stops_being_forecast(small):
    ret = _ret(n=900)
    ret[760:] = np.nan                                                                # delists at index 760
    out = V.run_asset(ret, [(400, 899)], 6, mc=False)
    t = out["rows"]["tidx"]
    assert t.max() == 759 and np.isfinite(out["rows"]["f1_garch"]).all()
    assert np.isnan(out["rows"]["y21"][t > 738]).all() and np.isfinite(out["rows"]["y21"][t < 700]).all()      # the last 21 days lack a window


# ---------------------------------------------------------------------------------------------------- ticker block
def _names():
    return pd.DataFrame({
        "permno": [1, 2, 3, 3, 4, 5, 6, 7], "ticker": ["ORCL", "INTU", "AMD", "AMD", "STX", "STX", "CELH", "BE"],
        "namedt": ["1990-01-01", "1990-01-01", "1990-01-01", "2010-01-01", "1990-01-01", "2015-01-01", "1990-01-01", "2000-01-01"],
        "nameenddt": ["2099-12-31", "2000-12-31", "2009-12-31", "2099-12-31", "2099-12-31", "2099-12-31", "2099-12-31", "2099-12-31"],
        "securitynm": ["ORACLE", "INTUIT", "AMD", "AMD INC", "SEAGATE", "SEAGATE NEW", "CELSIUS", "BLOOM"]})


def test_ticker_mapping_uses_the_record_in_force_and_refuses_ambiguous_or_missing():
    out = V.map_tickers(_names(), ["ORCL", "INTU", "AMD", "STX", "CELH", "LITE", "BE"], np.datetime64("2025-12-31"))
    assert out["ORCL"] == ("ok", 1)
    assert out["INTU"] == ("missing", None)                                           # name record ended in 2000
    assert out["AMD"] == ("ok", 3)                                                    # the 2010+ record, not the ended one (same permno either way)
    assert out["STX"] == ("ambiguous", [4, 5])
    assert out["LITE"] == ("missing", None) and out["BE"] == ("ok", 7)
    assert V.map_tickers(_names(), ["INTU"], np.datetime64("1999-06-30"))["INTU"] == ("ok", 2)
    with pytest.raises(ValueError, match="crsp_names"):
        V.map_tickers(_names().drop(columns="ticker"), ["ORCL"], np.datetime64("2025-12-31"))


def test_ticker_summary_ratio_and_drag(small):
    ret = _ret(n=1500)
    cal = (np.datetime64("2019-01-02") + np.arange(1500)).astype("datetime64[D]")
    rows = V.run_asset(ret, [(0, 1499)], 5, mc=False)["rows"]
    s = V.ticker_summary(rows, cal)
    assert s["rows"] > 500 and set(s["qlike21"]) == set(V.MODELS) and s["recent_rows"] > 100
    assert s["drag"]["realised"] == pytest.approx(252 * np.mean(ret[1000:1480] ** 2), rel=0.5)      # L=2 -> (L^2-L)/2 = 1
    assert all(np.isfinite(v) for v in s["ratio_mean"].values()) and all(0.2 < v < 5 for v in s["ratio_of_means"].values())
    assert V.ticker_summary(V._empty_rows(), cal) == {"rows": 0}


# ------------------------------------------------------------------------------------------------ end to end smoke
def _load_script():
    spec = importlib.util.spec_from_file_location("run_volfc", pathlib.Path(__file__).parent.parent / "scripts" / "run_volfc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in path.rglob("*"))


def _write_world(root):
    """Synthetic data/raw: 8 stocks on a weekly calendar 1995-2024 (dev, validation and holdout all crossed)."""
    raw = pathlib.Path(root) / "data" / "raw"
    (raw / "crsp_daily_ret").mkdir(parents=True)
    m, d, _, _ = make_world(n=8, years=(1994, 2022))
    m.to_parquet(raw / "crsp_monthly.parquet", index=False)
    d.to_parquet(raw / "crsp_delistings.parquet", index=False)
    cal = pd.date_range("1995-01-04", "2024-12-25", freq="7D")
    n = len(cal)
    rng = np.random.default_rng(21)
    frames = []
    for p in range(1, 9):
        vol = 0.012 * (1 + 0.6 * np.sin(np.arange(n) / (40 + 7 * p) + p)) ** 1.0
        r = vol * rng.standard_t(5, n) * np.sqrt(3 / 5)
        keep = np.ones(n, bool)
        if p == 3:
            keep[cal > pd.Timestamp("2010-06-30")] = False                           # delists mid-2010
        if p == 4:
            keep[900:940] = False                                                    # a gap
        if p == 5:
            keep[:n - 400] = False                                                   # short history (< 1000 days)
        frames.append(pd.DataFrame({"permno": np.int32(p), "dlycaldt": cal[keep], "dlyret": r[keep].astype("float32")}))
    daily = pd.concat(frames, ignore_index=True)
    for y, g in daily.groupby(daily["dlycaldt"].dt.year):
        g.sort_values(["permno", "dlycaldt"]).to_parquet(raw / "crsp_daily_ret" / f"{y}.parquet", index=False)
    ffd = cal[::1]
    pd.DataFrame({"date": ffd, "mktrf": rng.standard_t(5, n) * 0.008, "rf": 0.0001}).iloc[3:].to_parquet(raw / "ff_daily.parquet", index=False)
    pd.DataFrame({"permno": [1, 2, 3, 4, 5, 6, 7, 8, 8], "ticker": ["ORCL", "AMD", "STX", "INTU", "BE", "CELH", "SNDK", "SNDK", "LITE"],
                  "namedt": ["1990-01-01"] * 9, "nameenddt": ["2099-12-31"] * 8 + ["2001-01-01"],
                  "securitynm": ["ORACLE", "AMD", "SEAGATE", "INTUIT", "BLOOM", "CELSIUS", "SANDISK", "SANDISK NEW", "LUMENTUM"]}) \
        .to_parquet(raw / "crsp_names.parquet", index=False)
    return raw


@pytest.fixture
def small_world(monkeypatch):
    monkeypatch.setattr(V, "WINDOW", 200)
    monkeypatch.setattr(V, "MIN_OBS", 190)
    monkeypatch.setattr(V, "HIST_DAYS", 100)
    monkeypatch.setattr(V, "MC_PATHS", 300)


def _run(tmp_path, monkeypatch, name, jobs, open_holdout=False, assets=5, patch_licence=False, save=False):
    mod = _load_script()
    root = tmp_path / name
    root.mkdir()
    _write_world(root)
    monkeypatch.chdir(root)
    if patch_licence:
        real = V.licence_verdict

        def fake(comps, holdout=None, open_holdout=False):
            return real({k: {"dev": (-0.05, -5.0), "val": (-0.05, -5.0)} for k in comps}, holdout, open_holdout)
        monkeypatch.setattr(V, "licence_verdict", fake)
    lines = []
    mod._run(lines.append, open_holdout, jobs, assets, save)
    return root, lines


def test_smoke_end_to_end_sealed_run_checks_first_no_holdout_no_registry_and_jobs_identical(tmp_path, monkeypatch, small_world):
    before = _tree(REAL_DATA)
    t0 = time.time()
    root, l1 = _run(tmp_path, monkeypatch, "j1", 1, save=True)
    elapsed = time.time() - t0
    _, l3 = _run(tmp_path, monkeypatch, "j3", 3)
    # --save-losses is the only difference between the two runs
    assert [x for x in l1 if "saved" not in x] == [x for x in l3 if "saved" not in x]
    txt = "\n".join(l1)
    i = [txt.index(s) for s in ("== CHECKS", "== RESULTS (Study F", "== LICENCE VERDICT", "== DEFAULT SIZING MODEL", "== DESCRIPTIVE TABLES",
                                "== INFORMATION ONLY (no decision weight", "== INFORMATION ONLY: the user's tickers")]
    assert i == sorted(i)
    for s in ("daily returns file:", "calendar (sorted union", "ff coverage:", "universe size per year", "stock-years forecast:", "history rule",
              "fit gjr_t", "fit garch", "fit gjr_t_wins", "optimiser", "incomplete h=21 windows dropped", "[PASS] vectorised GARCH/GJR variance recursion",
              "[PASS] standardised-t density", "[PASS] forecasts produced", "the trial count stays at 79", "gjr_t vs garch h=1", "gjr_t vs har h=21",
              "default (dev ranking):", "winsorisation ablation", "market series", "predictive log score", "VaR violation rate", "VaR coverage by Monte Carlo",
              "mean QLIKE by calendar year", "stress 2008-09", "implied annualised 2x daily-reset drag", "ORCL", "REFUSED", "not a CRSP security",
              "insufficient history", "REFUSED: ambiguous", "REFUSED: missing"):
        assert s in txt, s
    assert "HOLDOUT OPENED" not in txt and "HOLDOUT (opened" not in txt and "stress 2020" not in txt
    assert not re.search(r"^\s+(202\d)\s", txt, re.M) and re.search(r"^\s+2019\s", txt, re.M)
    assert "holdout rows" in txt and "NOT computed (sealed)" in txt
    assert "note: sealed - this block ends on 2019-" in txt and "last 3 years (" in txt
    assert not any("trials" in x and "count" not in x for x in l1)
    # nothing leaks into the real data directory or the trials registry
    assert _tree(REAL_DATA) == before and not (root / "data" / "trials.jsonl").exists()
    # the saved losses contain no sealed rows
    sv = pd.read_parquet(root / "data" / "study_f_losses.parquet")
    assert len(sv) > 100 and sv["date"].max() <= pd.Timestamp("2019-12-31") and "qlike21_gjr_t" in sv.columns
    print(f"smoke sealed run: {elapsed:.1f}s")


def test_smoke_main_writes_the_report_file_only_under_the_working_directory(tmp_path, monkeypatch, small_world):
    mod = _load_script()
    root = tmp_path / "m"
    root.mkdir()
    _write_world(root)
    monkeypatch.chdir(root)
    mod.main(False, 1, 5, False)
    assert (root / "data" / "study_f_report.txt").read_text().startswith("== CHECKS")
    assert not (root / "data" / "study_f_losses.parquet").exists()


def test_smoke_open_holdout_shows_the_holdout_only_with_the_flag_and_a_dev_validation_licence(tmp_path, monkeypatch, small_world):
    _, sealed = _run(tmp_path, monkeypatch, "s", 1, patch_licence=True)               # licensed in dev/validation, holdout sealed (no flag)
    stxt = "\n".join(sealed)
    assert "LICENSED IN DEV/VALIDATION, holdout sealed" in stxt and "licence pending the one-time holdout" in stxt
    assert "HOLDOUT OPENED" not in stxt and "stress 2020" not in stxt and not re.search(r"^\s+2020\s", stxt, re.M)
    root, opened = _run(tmp_path, monkeypatch, "o", 2, open_holdout=True, patch_licence=True, save=True)
    txt = "\n".join(opened)
    assert "HOLDOUT OPENED (once)" in txt and "HOLDOUT (opened; information only)" in txt and ("licence STANDS" in txt or "licence FAILS" in txt)
    assert "stress 2020" in txt and re.search(r"^\s+2020\s", txt, re.M) and re.search(r"^\s+2023\s", txt, re.M)
    for h in (1, 21):
        assert re.search(rf"gjr_t vs garch h={h:<2d}: [-\d.]+ \(t", txt)
    sv = pd.read_parquet(root / "data" / "study_f_losses.parquet")
    assert sv["date"].max() >= pd.Timestamp("2023-01-01")
    assert "NOT computed" not in txt and "computed (--open-holdout)" in txt
    assert "note: sealed - this block" not in txt and "up to 2024-" in txt
    assert "licence pending" not in txt
    # the dev / validation numbers are the same whether or not the holdout is computed (everything before the licence verdict)
    k = next(i for i, x in enumerate(sealed) if "== LICENCE VERDICT" in x)
    cmp_s = [x for x in sealed[:k] if "holdout rows" not in x]
    cmp_o = [x for x in opened[:k] if "holdout rows" not in x]
    assert cmp_s == cmp_o


def test_smoke_open_holdout_without_a_licence_does_not_open_it(tmp_path, monkeypatch, small_world):
    _, lines = _run(tmp_path, monkeypatch, "u", 1, open_holdout=True)
    txt = "\n".join(lines)
    if "NOT LICENSED" in txt:
        assert "the holdout is NOT opened" in txt and "HOLDOUT OPENED" not in txt and "stress 2020" not in txt
        assert not re.search(r"^\s+2020\s", txt, re.M)
    else:                                                                               # licensed in dev/validation on synthetic noise: the holdout opened
        assert "HOLDOUT OPENED (once)" in txt


def test_script_requires_its_inputs_with_a_hint(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="extract_earnings.py"):
        mod._run(lambda *_: None, False, 1)


# ------------------------------------------------------------------------------------- v13 one-shot holdout marker (v14 gate)
def test_holdout_marker_helpers_are_one_shot(tmp_path):
    m = str(tmp_path / "data" / "holdout_v13.done")
    V.assert_holdout_unopened(m)                                       # absent: fine
    V.mark_holdout_opened("v13 holdout opened: test", m)
    assert pathlib.Path(m).read_text() == "v13 holdout opened: test\n"
    with pytest.raises(SystemExit, match="ALREADY OPENED.*v13 holdout opened: test"):
        V.assert_holdout_unopened(m)
    assert V.HOLDOUT_MARKER == "data/holdout_v13.done" and issubclass(V.HoldoutRefused, SystemExit)


def _fake_licence(monkeypatch, passing=True):
    real = V.licence_verdict
    sign = -0.05 if passing else 0.05

    def fake(comps, holdout=None, open_holdout=False):
        return real({k: {"dev": (sign, sign * 100), "val": (sign, sign * 100)} for k in comps}, holdout, open_holdout)
    monkeypatch.setattr(V, "licence_verdict", fake)


def test_smoke_v13_marker_is_written_only_when_the_holdout_is_opened_before_printing_and_a_second_opening_is_refused(tmp_path, monkeypatch, small_world):
    mod = _load_script()
    root = tmp_path / "w"
    root.mkdir()
    _write_world(root)
    monkeypatch.chdir(root)
    marker = root / "data" / "holdout_v13.done"
    _fake_licence(monkeypatch)
    sealed = []
    mod._run(sealed.append, False, 1, 5, False)                                   # no flag: behaves as before, no marker
    assert not marker.exists()
    state = []

    def log(x):
        if "HOLDOUT OPENED (once)" in str(x):
            state.append(marker.exists())                                         # the marker exists BEFORE the holdout numbers are printed
        opened.append(x)
    opened = []
    mod._run(log, True, 1, 5, False)
    assert state == [True] and marker.exists() and marker.read_text().startswith("v13 holdout opened")
    # the sealed part of the report is unchanged by the marker logic
    k = next(i for i, x in enumerate(sealed) if "== LICENCE VERDICT" in x)
    assert [x for x in sealed[:k] if "holdout rows" not in x] == [x for x in opened[:k] if "holdout rows" not in x]
    with pytest.raises(SystemExit, match="ALREADY OPENED"):
        mod._run(lambda *_: None, True, 1, 5, False)
    rep = root / "data" / "study_f_report.txt"
    rep.write_text("KEEP\n")
    with pytest.raises(SystemExit, match="ALREADY OPENED"):
        mod.main(True, 1, 5, False)
    assert rep.read_text() == "KEEP\n"                                            # a refusal never overwrites the report
    again = []
    mod._run(again.append, False, 1, 5, False)                                    # without the flag a rerun is still fine after the opening
    assert again == sealed


def test_smoke_v13_marker_is_not_written_when_the_dev_validation_rule_fails(tmp_path, monkeypatch, small_world):
    mod = _load_script()
    root = tmp_path / "f"
    root.mkdir()
    _write_world(root)
    monkeypatch.chdir(root)
    _fake_licence(monkeypatch, passing=False)
    lines = []
    mod._run(lines.append, True, 1, 5, False)
    assert "the holdout is NOT opened" in "\n".join(lines) and not (root / "data" / "holdout_v13.done").exists()
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "holdout_v13.done").write_text("earlier\n")
    mod._run(lambda *_: None, True, 1, 5, False)                                  # marker exists but nothing passes: no opening, no refusal


def _load_head_script(tmp_path):
    """The run_volfc.py from before the marker edit (frozen copy of commit ada5b76, tests/fixtures/run_volfc_premarker.py), loaded as a separate module."""
    src = (pathlib.Path(__file__).parent / "fixtures" / "run_volfc_premarker.py").read_text()
    f = tmp_path / "run_volfc_head.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location("run_volfc_head", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_is_identical_to_the_previous_commit_s_script_with_and_without_the_flag(tmp_path, monkeypatch, small_world):
    head, new = _load_head_script(tmp_path), _load_script()
    for flag in (False, True):
        out = {}
        for tag, mod in (("head", head), ("new", new)):
            root = tmp_path / f"{tag}{int(flag)}"
            root.mkdir()
            _write_world(root)
            monkeypatch.chdir(root)
            _fake_licence(monkeypatch)
            out[tag] = []
            mod._run(out[tag].append, flag, 1, 5, False)
        assert out["head"] == out["new"] and len(out["new"]) > 50
        assert (tmp_path / f"new{int(flag)}" / "data" / "holdout_v13.done").exists() is flag
        assert not (tmp_path / f"head{int(flag)}" / "data" / "holdout_v13.done").exists()
