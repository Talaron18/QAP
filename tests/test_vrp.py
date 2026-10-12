import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd
import pytest

from qap import volfc as V
from qap import vrp as VR

REAL_DATA = pathlib.Path(__file__).parent.parent / "data"


# ------------------------------------------------------------------------------------------------------------ units
def test_unit_conversions_against_hand_values():
    assert VR.iv2_win(20.0) == pytest.approx(0.04 * 30 / 365)
    assert VR.iv2_win(np.array([10.0, 40.0])) == pytest.approx([0.01 * 30 / 365, 0.16 * 30 / 365])
    assert VR.rv_win(1e-4) == pytest.approx(21e-4)
    assert VR.f_win(np.array([2e-4])) == pytest.approx([42e-4])
    assert VR.vol_points(VR.iv2_win(23.5)) == pytest.approx(23.5)               # VIX maps back to itself
    assert VR.vol_points(0.04 * 30 / 365 * 4) == pytest.approx(40.0)


def test_constants_are_the_pre_registered_values():
    assert (VR.NW_PRIMARY, VR.NW_DAILY, VR.C_MIN, VR.T_MIN, VR.T3_T_MIN) == (3, 30, 0.10, 2.0, 2.0)
    assert VR.REPRO_TOL == 5e-4 and V.REFIT == 21 and V.FIRST_FC_YEAR == 1995
    assert VR.REPRO_EXPECTED == {(1, "gjr_t"): (-8.2590, -8.8061), (1, "garch"): (-8.2144, -8.7425),
                                 (21, "gjr_t"): (-8.1053, -8.5679), (21, "garch"): (-8.0991, -8.5666)}
    assert VR.MARKER_V13 == "data/holdout_v13.done" and VR.MARKER_V14 == "data/holdout_v14.done" and VR.TRIAL_NAME == "studyG:vrp_payoff"
    assert VR.EX_CRASH == ("2008-09-01", "2008-12-31")


# ----------------------------------------------------------------------------------------------------------- HAC OLS
X5 = np.array([1.0, 2, 3, 4, 5])
Y5 = np.array([2.0, 1, 4, 3, 6])        # hand case: intercept 0.2, slope 1.0, residuals .8 -1.2 .8 -1.2 .8, Sxx 10


def test_hac_ols_against_a_hand_computed_case():
    X = np.column_stack([np.ones(5), X5])
    r0 = VR.ols_hac(Y5, X, 0)
    assert r0["coef"] == pytest.approx([0.2, 1.0]) and r0["n"] == 5
    assert r0["r2"] == pytest.approx(1 - 4.8 / 14.8)                               # RSS 0.64*3 + 1.44*2 = 4.8, TSS 14.8
    # hand values: u = (x - xbar) e = [-1.6, 1.2, 0, -1.2, 1.6]; G0 = 8.0, G1 = -3.84, G2 = -1.44; var(slope) = S / Sxx^2, Sxx = 10
    assert r0["se"][1] == pytest.approx(math_sqrt(8.0 / 100))
    r1 = VR.ols_hac(Y5, X, 1)
    assert r1["se"][1] == pytest.approx(math_sqrt((8.0 + 2 * 0.5 * -3.84) / 100))                     # = sqrt(0.0416)
    r2 = VR.ols_hac(Y5, X, 2)
    assert r2["se"][1] == pytest.approx(math_sqrt((8.0 + 2 * (2 / 3) * -3.84 + 2 * (1 / 3) * -1.44) / 100))   # = sqrt(0.0192)
    assert r2["t"][1] == pytest.approx(1.0 / math_sqrt(0.0192))
    assert r1["coef"] == pytest.approx(r0["coef"])


def math_sqrt(x):
    return float(np.sqrt(x))


def test_hac_with_zero_lags_has_plain_ols_coefficients_and_white_standard_errors():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(60), rng.normal(size=60), rng.normal(size=60)])
    y = X @ [0.5, 1.0, -0.5] + rng.normal(size=60) * (1 + np.abs(X[:, 1]))
    r = VR.ols_hac(y, X, 0)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    assert r["coef"] == pytest.approx(beta)
    e = y - X @ beta
    inv = np.linalg.inv(X.T @ X)
    hc0 = inv @ (X.T * e ** 2) @ X @ inv                                           # White HC0 sandwich
    assert r["se"] == pytest.approx(np.sqrt(np.diag(hc0)))


def test_hac_intercept_only_equals_newey_west_t_of_the_reviewed_helper():
    rng = np.random.default_rng(2)
    d = rng.normal(size=80) + np.cumsum(rng.normal(size=80)) * 0.1
    for lags in (0, 3, 30):
        m, t = V.newey_west_t(d, lags)
        r = VR.ols_hac(d, np.ones((80, 1)), lags)
        assert r["coef"][0] == pytest.approx(m) and r["t"][0] == pytest.approx(t)


def test_hac_edge_cases_nan_rows_and_too_few_observations():
    r = VR.ols_hac([1.0, 2.0], np.column_stack([np.ones(2), [0.0, 1.0]]), 3)
    assert r["n"] == 2 and np.isnan(r["coef"]).all()
    y = np.array([2.0, 1, 4, 3, 6, np.nan])
    X = np.column_stack([np.ones(6), np.append(X5, 7.0)])
    assert VR.ols_hac(y, X, 1)["coef"] == pytest.approx([0.2, 1.0]) and VR.ols_hac(y, X, 1)["n"] == 5
    assert VR.ols_hac(Y5, np.column_stack([np.ones(5), np.ones(5)]), 1)["n"] == 5          # singular design: no crash
    r = VR.ols_hac(Y5, np.column_stack([np.ones(5), np.ones(5)]), 1)
    assert np.isnan(r["coef"]).all()


# ------------------------------------------------------------------------------------------------------ VIX / samples
def test_prepare_vix_counts_nulls_mismatches_duplicates_and_the_seal():
    cal = np.array(["2019-12-02", "2019-12-03", "2019-12-04", "2019-12-05", "2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    df = pd.DataFrame({"date": ["2019-12-02", "2019-12-03", "2019-12-03", "2019-12-04", "2019-12-07", "2019-12-05", "2020-01-02", "bad"],
                       "vix": [15.0, 16.0, 17.0, None, 18.0, -1.0, 30.0, 20.0]})
    out = VR.prepare_vix(df, cal, before="2020-01-01")
    st = out["stats"]
    assert st["bad_date"] == 1 and st["duplicates"] == 1 and st["null_dropped"] == 2 and st["not_in_calendar"] == 1
    assert st["matched"] == 2 and st["rows_in_scope"] == 6 and st["last"] == "2019-12-07"
    assert out["vix"][:4].tolist()[:2] == [15.0, 17.0] and np.isnan(out["vix"][2:]).all()               # last duplicate kept; sealed 2020 row gone
    assert VR.prepare_vix(df, cal)["vix"][4] == 30.0


def _fake_rows(n=210, seed=0):
    rng = np.random.default_rng(seed)
    d = {"tidx": np.arange(n, dtype="int64"), "y1": rng.uniform(1e-5, 3e-4, n), "y21": rng.uniform(1e-5, 3e-4, n)}
    for m in V.MODELS:
        for h in V.HORIZONS:
            d[f"f{h}_{m}"] = rng.uniform(1e-5, 3e-4, n)
    return pd.DataFrame(d)


def test_primary_sampler_returns_exactly_the_multiples_of_21_with_complete_windows_vix_and_common_mask():
    n = 210
    cal = np.arange(np.datetime64("2000-01-03"), np.datetime64("2000-01-03") + np.timedelta64(n, "D"))
    df = _fake_rows(n)
    df.loc[[21, 63], "y21"] = np.nan                         # incomplete windows
    df.loc[84, "f21_har"] = -1.0                             # one model non-positive -> out of the COMMON mask for every model
    df.loc[105, "f21_ewma"] = np.nan
    vix = np.full(n, 20.0)
    vix[126] = np.nan                                        # null VIX
    vix[147] = 0.0
    s = VR.select_samples(df, cal, vix)
    prim = s["primary"]
    expect = [i for i in range(0, n, 21) if i not in (21, 63, 84, 105, 126, 147)]
    assert prim["tidx"].tolist() == expect
    assert (prim["tidx"] % 21 == 0).all()
    daily = s["daily"]
    assert set(prim["tidx"]) <= set(daily["tidx"]) and 21 not in set(daily["tidx"]) and 84 not in set(daily["tidx"]) and 126 not in set(daily["tidx"])
    assert len(daily) == n - 6 and s["counts"]["primary"] == len(expect) and s["counts"]["daily"] == len(daily)
    r = prim.iloc[2]
    assert r["iv2"] == pytest.approx(0.04 * 30 / 365) and r["rv"] == pytest.approx(21 * df["y21"].iloc[int(r["tidx"])])
    assert r["F_gjr_t"] == pytest.approx(21 * df["f21_gjr_t"].iloc[int(r["tidx"])]) and r["date"] == pd.Timestamp(cal[int(r["tidx"])])
    assert all(f"F_{m}" in prim.columns for m in V.MODELS)
    w = VR.in_window(prim, ("2000-02-01", "2000-04-30"))
    assert w["date"].min() >= pd.Timestamp("2000-02-01") and w["date"].max() <= pd.Timestamp("2000-04-30") and 0 < len(w) < len(prim)


# -------------------------------------------------------------------------------------------------------- T1 / T3 rules
def _t1(c, t, val_c=0.2):
    return VR.t1_rule({"c": c, "t_c": t}, {"c": val_c, "t_c": 0.0})


@pytest.mark.parametrize("c,t,vc,expect", [(0.10, 2.0, 0.01, True), (0.0999, 3.0, 0.5, False), (0.5, 1.99, 0.5, False), (0.5, 2.5, 0.0, False),
                                           (0.5, 2.5, -0.1, False), (0.5, 2.5, 1e-9, True), (np.nan, np.nan, 0.5, False), (0.3, 2.5, np.nan, False)])
def test_t1_rule_thresholds_and_validation_sign(c, t, vc, expect):
    assert _t1(c, t, vc)["passed"] is expect


@pytest.mark.parametrize("t,vs,expect", [(2.0, 0.001, True), (1.99, 1.0, False), (3.0, 0.0, False), (3.0, -0.2, False), (np.nan, 1.0, False), (3.0, np.nan, False)])
def test_t3_rule_thresholds_and_validation_sign(t, vs, expect):
    assert VR.t3_rule({"t": t}, {"slope": vs})["passed"] is expect


def test_decision_requires_both_tests_and_consults_the_holdout_only_when_opened():
    ok, bad = {"passed": True}, {"passed": False}
    for a, b in ((bad, ok), (ok, bad), (bad, bad)):
        d = VR.decision(a, b)
        assert not d["dev_val"] and "NO CASE FOR BUYING STOCK-LEVEL DATA" in d["status"] and "encompasses" in d["status"]
        assert VR.decision(a, b, True)["opened"] is False                              # flag given but not both passed: not opened, holdout=None accepted
    d = VR.decision(ok, ok)
    assert d["dev_val"] and not d["opened"] and "holdout sealed" in d["status"]
    with pytest.raises(ValueError):
        VR.decision(ok, ok, True, None)
    assert VR.decision(ok, ok, True, {"c": 0.1, "slope": 0.2})["stands"] is True
    for h in ({"c": -0.1, "slope": 0.2}, {"c": 0.1, "slope": -0.2}, {"c": 0.0, "slope": 0.2}, {"c": np.nan, "slope": 1.0}):
        d = VR.decision(ok, ok, True, h)
        assert d["stands"] is False and d["status"] == "passed dev/validation, failed in the holdout"


def test_t1_and_t3_fits_recover_a_designed_relationship():
    rng = np.random.default_rng(5)
    n = 400
    liv = rng.normal(-5, 0.5, n)
    lf = 0.6 * liv + rng.normal(0, 0.3, n)
    lrv = 0.2 + 0.3 * liv + 0.7 * lf + rng.normal(0, 0.2, n)
    s = pd.DataFrame({"iv2": np.exp(liv), "F_gjr_t": np.exp(lf), "F_garch": np.exp(lf), "rv": np.exp(lrv)})
    f = VR.t1_fit(s, "gjr_t")
    assert f["n"] == n and f["b"] == pytest.approx(0.3, abs=0.12) and f["c"] == pytest.approx(0.7, abs=0.12) and f["t_c"] > 5
    assert f["r2"] > f["r2_iv"] and f["r2"] > f["r2_f"] > 0
    y = VR.payoff(s)
    assert y == pytest.approx((s["rv"] - s["iv2"]) / s["iv2"])
    t3 = VR.t3_fit(s)
    d = np.log(s["F_gjr_t"]) - np.log(s["iv2"])
    assert t3["slope"] == pytest.approx(np.polyfit(d, y, 1)[0]) and VR.t3_fit(s, level=True)["slope"] == pytest.approx(
        np.polyfit(s["F_gjr_t"] - s["iv2"], y, 1)[0])


def test_tercile_breakpoints_are_frozen_on_development():
    dev = pd.DataFrame({"iv2": np.ones(9), "F_gjr_t": np.exp(np.arange(9.0)), "rv": np.arange(1.0, 10.0) + 1.0})
    d = VR.signal_d(dev)
    br = VR.tercile_breaks(d)
    assert br == pytest.approx(np.quantile(np.arange(9.0), [1 / 3, 2 / 3]))
    val = pd.DataFrame({"iv2": np.ones(6), "F_gjr_t": np.exp([-5.0, 0.5, 2.0, 4.0, 7.0, 20.0]), "rv": np.array([1.0, 2, 3, 4, 5, 6])})
    tb = VR.tercile_table(val, br)
    assert [r["n"] for r in tb] == [3, 1, 2]                                           # breaks 2.67 / 5.33: D = -5, .5, 2 | 4 | 7, 20
    other = val.copy()
    other["F_gjr_t"] = np.exp(np.linspace(-50, 50, 6))
    assert VR.tercile_breaks(d) == pytest.approx(br)                                    # breaks depend on development only
    assert VR.tercile_table(other, br)[0]["n"] == 3 and VR.tercile_table(val, br)[0]["n"] == 3
    assert np.isnan(VR.tercile_table(val.iloc[:0], br)[0]["mean"])


def test_ex_crash_windows_touching_sep_dec_2008():
    cal = np.arange(np.datetime64("2008-06-02"), np.datetime64("2009-03-02"))
    s = pd.DataFrame({"tidx": [0, 60, 85, 87, 150, 210]})
    # t=60 -> window days 61..81 = 2008-08-02..08-22 (no); t=85 -> 08-27..09-16 (touches); t=87 -> 09-?.. ; t=210 -> 2008-12-30.. (touches)
    got = VR.touches(s, cal, "2008-09-01", "2008-12-31")
    first, last = cal[s["tidx"] + 1], cal[s["tidx"] + 21]
    want = (last >= np.datetime64("2008-09-01")) & (first <= np.datetime64("2008-12-31"))
    assert got.tolist() == want.tolist() and got.tolist() == [False, False, True, True, True, True]


def test_t2_t4_and_scaled_iv_qlike():
    s = pd.DataFrame({"vix": [20.0, 20.0, 20.0, 20.0], "iv2": VR.iv2_win(np.full(4, 20.0)), "rv": VR.iv2_win(np.full(4, 20.0)) * [0.5, 0.8, 1.5, 0.9],
                      "F_gjr_t": [0.01, 0.02, 0.03, 0.02], "F_garch": [0.01, 0.02, 0.03, 0.02]})
    t = VR.t2_stats(s)
    assert t["share_pos"] == 0.75 and t["mean_var"] == pytest.approx(float(np.mean(s["iv2"] - s["rv"])))
    assert t["mean_vol_pts"] == pytest.approx(float(np.mean(20.0 - VR.vol_points(s["rv"]))))
    k = VR.scaled_iv_k(s)
    assert k == pytest.approx(s["rv"].mean() / s["iv2"].mean())
    q = VR.qlike_vs_scaled_iv(s, k)
    assert q["gjr_t"][0] == pytest.approx(float(np.mean(V.qlike(s["F_gjr_t"], s["rv"]) - V.qlike(k * s["iv2"], s["rv"])))) and q["gjr_t"][2] == 4
    r21 = np.array([0.01, -0.02, 0.03, 0.0])
    f = VR.t4_fit(s, r21, "iv_minus_f")
    assert f["slope"] == pytest.approx(np.polyfit(s["iv2"] - s["F_gjr_t"], r21, 1)[0]) and f["n"] == 4


def test_trial_logging_is_idempotent_per_unique_spec(tmp_path):
    p = str(tmp_path / "trials.jsonl")
    spec = VR.trial_spec()
    assert spec["prereg"] == "v14" and "log F_win(gjr_t) - log IV2_win" in spec["D"] and "(RV_win - IV2_win) / IV2_win" in spec["Y"]
    assert VR.log_trial_once(VR.TRIAL_NAME, spec, {"a": 1}, p) is True
    assert VR.log_trial_once(VR.TRIAL_NAME, dict(spec), {"a": 2}, p) is False
    assert VR.log_trial_once(VR.TRIAL_NAME, dict(spec, regression="other"), {"a": 3}, p) is True
    assert len(open(p).read().strip().splitlines()) == 2
    from qap.trials import trial_count
    assert trial_count(p) == 2
    assert json.loads(open(p).readline())["name"] == "studyG:vrp_payoff"


# ------------------------------------------------------------------------------------ engine, calendar, reproduction
@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(V, "WINDOW", 200)
    monkeypatch.setattr(V, "MIN_OBS", 190)
    monkeypatch.setattr(V, "HIST_DAYS", 100)
    monkeypatch.setattr(V, "MC_PATHS", 300)


def _vol(n):
    return 0.008 * (1 + 0.7 * np.sin(np.arange(n) / 30.0))


def _write_world(root, vix=True, with_sp500=False, seed=11):
    """Synthetic data/raw: daily files (calendar every 5th day 1992-2024), FF market, and a VIX correlated with the simulated volatility."""
    raw = pathlib.Path(root) / "data" / "raw"
    (raw / "crsp_daily_ret").mkdir(parents=True)
    cal = pd.date_range("1992-01-01", "2024-12-31", freq="5D")
    n = len(cal)
    rng = np.random.default_rng(seed)
    vol = _vol(n)
    frames = []
    for p in range(1, 4):
        frames.append(pd.DataFrame({"permno": np.int32(p), "dlycaldt": cal, "dlyret": (vol * rng.standard_t(5, n) * np.sqrt(3 / 5)).astype("float32")}))
    daily = pd.concat(frames, ignore_index=True)
    for y, g in daily.groupby(daily["dlycaldt"].dt.year):
        g.to_parquet(raw / "crsp_daily_ret" / f"{y}.parquet", index=False)
    mk = vol * rng.standard_t(5, n) * np.sqrt(3 / 5)
    pd.DataFrame({"date": cal, "mktrf": mk, "rf": 0.0001}).iloc[2:].to_parquet(raw / "ff_daily.parquet", index=False)
    if vix:
        v = 100 * np.sqrt(252) * vol * 1.15 * np.exp(0.05 * rng.normal(size=n))
        df = pd.DataFrame({"date": cal, "vix": v, "vxo": v, "vxn": v, "vxd": v})
        df.loc[[300, 301, 700], "vix"] = np.nan
        df = pd.concat([df, pd.DataFrame({"date": [cal[400] + pd.Timedelta(days=1)], "vix": [18.0], "vxo": [18.0], "vxn": [18.0], "vxd": [18.0]})], ignore_index=True)
        df.to_parquet(raw / "cboe_vix.parquet", index=False)
    if with_sp500:
        pd.DataFrame({"caldt": cal, "sprtrn": mk}).to_parquet(raw / "sp500_daily.parquet", index=False)
    return raw, cal, mk


def _load_script(name="run_vrp"):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).parent.parent / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in path.rglob("*"))


def test_calendar_from_the_date_column_only_equals_the_volfc_calendar(tmp_path):
    raw, cal, _ = _write_world(tmp_path)
    c1, rows = VR.load_calendar(raw / "crsp_daily_ret")
    _, rows2, c2 = V.load_daily_subset(raw / "crsp_daily_ret", [1])
    assert np.array_equal(c1, c2) and rows == rows2 and len(c1) == len(cal)


def test_reproduction_check_passes_on_engine_output_with_injected_constants_and_aborts_on_a_corrupted_series(small, tmp_path):
    _, cal, mk = _write_world(tmp_path)
    cal64 = cal.to_numpy().astype("datetime64[D]")
    dm = VR.market_rows(mk, cal64, 2019)
    assert len(dm) > 500 and (dm["tidx"] >= 0).all()
    exp = VR.repro_values(dm, cal64)
    wanted = {k: exp[k] for k in ((1, "gjr_t"), (1, "garch"), (21, "gjr_t"), (21, "garch"))}
    ok, lines = VR.repro_check(dm, cal64, wanted)
    assert ok and len(lines) == 4 and all("[PASS]" in x for x in lines)
    bad = mk.copy()
    bad[300:900] *= 1.5                                                 # a corrupted / misaligned series changes the QLIKE
    ok2, lines2 = VR.repro_check(VR.market_rows(bad, cal64, 2019), cal64, wanted)
    assert not ok2 and any("[FAIL]" in x for x in lines2)
    ok3, _ = VR.repro_check(dm, cal64)                                   # the real v13 constants do not hold for a synthetic market
    assert not ok3
    shifted = dm.copy()
    shifted["tidx"] = shifted["tidx"] + 1                                # misaligned calendar index
    ok4, _ = VR.repro_check(shifted, cal64, wanted)
    assert not ok4


# ------------------------------------------------------------------------------------------------------- end-to-end smoke
def _run(root, monkeypatch, open_holdout=False, patch_pass=False, tol=True):
    mod = _load_script()
    monkeypatch.chdir(root)
    if tol:
        monkeypatch.setattr(VR, "REPRO_TOL", 1e9)                          # the synthetic market is not the CRSP series: inject a lax tolerance
    if patch_pass:
        monkeypatch.setattr(VR, "t1_rule", lambda d, v: {"dev_ok": True, "val_ok": True, "passed": True})
        monkeypatch.setattr(VR, "t3_rule", lambda d, v: {"dev_ok": True, "val_ok": True, "passed": True})
    lines = []
    mod._run(lines.append, open_holdout)
    return lines


def test_smoke_sealed_run_checks_first_one_trial_no_2020_and_nothing_in_the_real_data(small, tmp_path, monkeypatch):
    before = _tree(REAL_DATA)
    root = tmp_path / "w"
    root.mkdir()
    _write_world(root, with_sp500=True)
    t0 = time.time()
    lines = _run(root, monkeypatch)
    elapsed = time.time() - t0
    txt = "\n".join(lines)
    idx = [txt.index(s) for s in ("== CHECKS", "Study F reproduction check", "== RESULTS (Study G", "-- T1 (information content", "-- T3 (economic payoff",
                                  "== DECISION", "== INFORMATION ONLY")]
    assert idx == sorted(idx)
    for s in ("cboe_vix:", "VIX days dropped: null / non-finite / non-positive 3", "date not in the calendar 1", "calendar overlap:", "[PASS] median VIX",
              "[PASS] mean IV2_win", "[PASS] correlation of log IV2_win with log F_win(gjr_t)", "primary sample (calendar index multiple of 21",
              "incomplete RV windows", "reproduction h=1  gjr_t", "reproduction h=21 garch", "M3 gjr_t (PRIMARY)", "M2 garch (beside it)", "T3 is one trial",
              "unique trials logged so far: 1", "-- T2 unconditional premium", "-- T4 market return", "constant-scaled implied variance", "daily-overlap versions",
              "tercile", "level version", "excluding windows touching 2008-09..2008-12", "S&P 500 daily returns", "by year:", "NO CASE FOR BUYING",
              "compensation for crash risk, not an edge", "licenses no trade of any kind"):
        assert s in txt, s
    assert "skipped" not in txt.split("S&P 500 daily returns")[1].split("-- by year")[0]
    assert not re.search(r"(?<!\d\.)(?<!\d)20[2-9]\d(?!\d)", txt), re.findall(r"(?<!\d\.)(?<!\d)20[2-9]\d(?!\d)", txt)
    assert "HOLDOUT OPENED" not in txt and not (root / "data" / "holdout_v14.done").exists()
    tr = [json.loads(x) for x in (root / "data" / "trials.jsonl").read_text().splitlines()]
    assert len(tr) == 1 and tr[0]["name"] == "studyG:vrp_payoff" and tr[0]["spec"]["prereg"] == "v14" and "dev_t" in tr[0]["metrics"]
    lines2 = _run(root, monkeypatch)                                       # rerun: identical report, no duplicate trial
    assert len((root / "data" / "trials.jsonl").read_text().splitlines()) == 1 and "already in the registry" in "\n".join(lines2)
    assert [x for x in lines if "registry" not in x and "logged" not in x] == [x for x in lines2 if "registry" not in x and "logged" not in x]
    assert _tree(REAL_DATA) == before
    print(f"smoke sealed run: {elapsed:.1f}s")


def test_smoke_missing_sp500_file_is_skipped_cleanly_and_main_writes_the_report(small, tmp_path, monkeypatch):
    root = tmp_path / "m"
    root.mkdir()
    _write_world(root)
    monkeypatch.chdir(root)
    monkeypatch.setattr(VR, "REPRO_TOL", 1e9)
    _load_script().main(False)
    rep = (root / "data" / "study_g_report.txt").read_text()
    assert rep.startswith("== CHECKS") and "skipped: data/raw/sp500_daily.parquet is absent" in rep


def test_smoke_reproduction_failure_aborts_before_any_result_but_keeps_the_checks_in_the_report(small, tmp_path, monkeypatch):
    root = tmp_path / "r"
    root.mkdir()
    _write_world(root)
    monkeypatch.chdir(root)
    with pytest.raises(AssertionError, match="reproduction check failed"):
        _load_script().main(False)                                          # real v13 constants vs a synthetic market
    rep = (root / "data" / "study_g_report.txt").read_text()
    assert "[FAIL] reproduction" in rep and "ABORT" in rep and "== RESULTS" not in rep and not (root / "data" / "trials.jsonl").exists()


def test_open_holdout_is_refused_without_the_v13_marker_and_leaves_the_report_alone(small, tmp_path, monkeypatch):
    root = tmp_path / "g"
    root.mkdir()
    _write_world(root)
    monkeypatch.chdir(root)
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "study_g_report.txt").write_text("OLD REPORT\n")
    with pytest.raises(SystemExit, match="v13 .*holdout has not been opened"):
        _load_script().main(True)
    assert (root / "data" / "study_g_report.txt").read_text() == "OLD REPORT\n"
    assert not (root / "data" / "holdout_v14.done").exists() and not (root / "data" / "trials.jsonl").exists()


def test_open_holdout_without_passing_dev_validation_does_not_open_it(small, tmp_path, monkeypatch):
    root = tmp_path / "n"
    root.mkdir()
    _write_world(root)
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "holdout_v13.done").write_text("v13 opened\n")
    lines = _run(root, monkeypatch, open_holdout=True)
    txt = "\n".join(lines)
    if "NO CASE FOR BUYING" in txt:
        assert "the holdout is NOT opened" in txt and "HOLDOUT OPENED" not in txt and not (root / "data" / "holdout_v14.done").exists()
        assert not re.search(r"(?<!\d\.)(?<!\d)20[2-9]\d(?!\d)", txt)


def test_open_holdout_after_v13_is_one_shot_with_a_marker_written_before_the_numbers_are_printed(small, tmp_path, monkeypatch):
    root = tmp_path / "o"
    root.mkdir()
    _write_world(root)
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "holdout_v13.done").write_text("v13 opened\n")
    sealed = _run(root, monkeypatch, patch_pass=True)
    stxt = "\n".join(sealed)
    assert "PASSED IN DEV/VALIDATION" in stxt and "holdout sealed" in stxt and "HOLDOUT OPENED" not in stxt
    assert not re.search(r"(?<!\d\.)(?<!\d)20[2-9]\d(?!\d)", stxt) and not (root / "data" / "holdout_v14.done").exists()
    seen = []
    real_mark = V.mark_holdout_opened
    monkeypatch.setattr(V, "mark_holdout_opened", lambda text, marker=None: (seen.append(len(opened)), real_mark(text, marker)))
    opened = []
    mod = _load_script()
    monkeypatch.chdir(root)
    monkeypatch.setattr(VR, "REPRO_TOL", 1e9)
    monkeypatch.setattr(VR, "t1_rule", lambda d, v: {"dev_ok": True, "val_ok": True, "passed": True})
    monkeypatch.setattr(VR, "t3_rule", lambda d, v: {"dev_ok": True, "val_ok": True, "passed": True})
    mod._run(opened.append, True)
    otxt = "\n".join(opened)
    assert "HOLDOUT OPENED (once)" in otxt and ("CASE STANDS" in otxt or "FAILED IN THE HOLDOUT" in otxt) and "2020-01..2026-12" in otxt
    assert len(seen) == 1 and seen[0] < len(opened)                                                              # marker written before the holdout lines
    assert "HOLDOUT OPENED" not in "\n".join(opened[:seen[0]])
    marker = root / "data" / "holdout_v14.done"
    assert marker.exists() and marker.read_text().startswith("v14 holdout opened")
    # the dev / validation part is identical to the sealed run
    k = next(i for i, x in enumerate(sealed) if "== DECISION" in x)
    assert [x for x in sealed[:k] if "T3 is one trial" not in x] == [x for x in opened[:k] if "T3 is one trial" not in x]
    # second opening refused, report untouched
    with pytest.raises(SystemExit, match="ALREADY OPENED"):
        mod._run(lambda *_: None, True)
    (root / "data" / "study_g_report.txt").write_text("KEEP\n")
    with pytest.raises(SystemExit, match="ALREADY OPENED"):
        mod.main(True)
    assert (root / "data" / "study_g_report.txt").read_text() == "KEEP\n"
    assert len((root / "data" / "trials.jsonl").read_text().splitlines()) == 1


def test_script_requires_its_inputs_with_a_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="wrds_extract.py cboe_vix"):
        _load_script()._run(lambda *_: None, False)
    with pytest.raises(SystemExit, match="extract_earnings.py"):
        _load_script()._run(lambda *_: None, False)
