"""Level calculator tests (qap.levels, scripts/level_calc.py). SYNTHETIC series only; nothing is written outside tmp_path."""
import importlib.util
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from qap import levels as lv

ROOT = pathlib.Path(__file__).resolve().parents[1]


def hist(n, sigma=0.02, seed=1, nu=5.0):
    return np.random.default_rng(seed).standard_t(nu, n) * sigma * math.sqrt((nu - 2) / nu)


def flat_model(sigma_annual=0.02 * math.sqrt(252), nu=5.0):
    return lv.Model("fallback", 0, sigma_annual, "t", dict(nu=nu, label="ewma94", window=0))


# ------------------------------------------------------------------------------------------------ arguments
def test_parse_level_percent_vs_price():
    assert lv.parse_level("15%") == lv.Level("ret", 0.15)
    assert lv.parse_level("-20%") == lv.Level("ret", -0.2)
    assert lv.parse_level("$1,250.5") == lv.Level("price", 1250.5)
    assert lv.parse_level(None) is None
    with pytest.raises(ValueError):
        lv.parse_level("-5")                      # a negative plain number is not a price


# ------------------------------------------------------------------------------------------------ instrument recursion
def test_daily_reset_recursion_hand_numbers_and_ruin():
    R = np.array([[0.10, -0.05, 0.02], [-0.30, 0.50, 0.50], [0.60, 0.1, 0.1]])
    V2 = lv.instrument_value(R, 2.0)
    assert V2[0] == pytest.approx([1.2, 1.2 * 0.9, 1.2 * 0.9 * 1.04])
    assert V2[1] == pytest.approx([0.4, 0.8, 1.6])
    Vm = lv.instrument_value(R, -2.0)                       # 2x inverse: +60% underlying wipes it out, and 0 is absorbing
    assert Vm[0] == pytest.approx([0.8, 0.88, 0.8448])
    assert Vm[2] == pytest.approx([0.0, 0.0, 0.0]) and Vm[1][0] == pytest.approx(1.6)
    assert lv.instrument_value(R, 1.0)[0] == pytest.approx(np.cumprod(1 + R[0]))


# ------------------------------------------------------------------------------------------------ first passage
def test_first_passage_matches_driftless_brownian_closed_form():
    """Log-driftless walk (drift = sigma^2/2 offsets the simple-return convexity). P(+a first) = b/(a+b) in log distance, with the usual
    discrete-monitoring overshoot of about 0.58 sigma on each barrier."""
    sig = 0.02
    a, b = math.log(1.20), -math.log(0.90)
    n = 30000
    sim = lv.simulate_trade(flat_model(), last_close=100.0, entry=100.0, target=lv.Level("ret", 0.20), stop=lv.Level("ret", -0.10),
                            leverage=1.0, horizon=800, n_paths=n, seed=3, drift_annual=252 * sig ** 2 / 2)
    p_up = float((sim["outcome"] == 1).mean())
    p_dn = float((sim["outcome"] == -1).mean())
    assert p_up + p_dn > 0.995                                       # horizon long enough that almost every path exits
    plain = b / (a + b)
    corrected = (b + 0.5826 * sig) / (a + b + 2 * 0.5826 * sig)
    se = math.sqrt(plain * (1 - plain) / n)
    assert abs(p_up - corrected) < 4 * se + 0.01
    assert abs(p_up - plain) < 0.03
    # exit returns are exactly the levels
    assert set(np.round(sim["ret"][sim["outcome"] == 1], 12)) == {0.2} and set(np.round(sim["ret"][sim["outcome"] == -1], 12)) == {-0.1}


def test_symmetric_race_is_a_coin_flip_and_stop_wins_ties():
    n = 20000
    sim = lv.simulate_trade(flat_model(), last_close=100.0, entry=100.0, target=lv.Level("ret", 0.10), stop=lv.Level("ret", -0.10),
                            leverage=1.0, horizon=400, n_paths=n, seed=4, drift_annual=252 * 0.02 ** 2 / 2)
    assert abs((sim["outcome"] == 1).mean() - 0.5) < 0.03
    # same trigger for both: every exit is a stop (conservative tie rule)
    t = lv.simulate_trade(flat_model(), last_close=100.0, entry=100.0, target=lv.Level("ret", -0.05), stop=lv.Level("ret", -0.05),
                          leverage=1.0, horizon=50, n_paths=2000, seed=4)
    assert (t["outcome"] == 1).sum() == 0 and (t["outcome"] == -1).sum() > 0


def test_price_levels_touch_probabilities_and_exit_at_level():
    sim = lv.simulate_trade(flat_model(), last_close=100.0, entry=100.0, target=lv.Level("price", 110.0), stop=lv.Level("price", 95.0),
                            leverage=1.0, horizon=60, n_paths=20000, seed=5, touch_levels={"target": 110.0, "stop": 95.0})
    assert set(np.round(sim["ret"][sim["outcome"] == 1], 12)) == {0.1} and set(np.round(sim["ret"][sim["outcome"] == -1], 12)) == {-0.05}
    # touching a level within the horizon is at least as likely as being first to it, and at least the final-close probability
    assert sim["touch"]["target"] >= (sim["outcome"] == 1).mean() and sim["touch"]["stop"] >= (sim["outcome"] == -1).mean()
    assert sim["touch"]["target"] >= sim["above"]["target"] and sim["touch"]["stop"] >= 1 - sim["above"]["stop"]
    # reflection principle for a driftless walk: P(touch) is about 2 P(end beyond) (discrete monitoring makes it a bit smaller)
    assert sim["touch"]["target"] == pytest.approx(2 * sim["above"]["target"], rel=0.2)


def test_leveraged_price_level_exits_at_instrument_close():
    sim = lv.simulate_trade(flat_model(), last_close=100.0, entry=100.0, target=lv.Level("price", 105.0), stop=None, leverage=2.0,
                            horizon=30, n_paths=5000, seed=6)
    r = sim["ret"][sim["outcome"] == 1]
    assert len(r) > 0 and r.min() > 0.0 and len(set(np.round(r, 9))) > 10      # not a single level: 2x of a closing move above +5%


# ------------------------------------------------------------------------------------------------ drag
def test_drag_formula_vs_simulation_long_horizon():
    L, sig = 2.0, 0.02
    sim = lv.simulate_trade(flat_model(), last_close=100.0, entry=100.0, target=None, stop=None, leverage=L, horizon=1000, n_paths=2000,
                            seed=7)
    sigma_sim = math.sqrt(252 * sim["mean_r2"])
    analytic = lv.drag_analytic(L, sigma_sim, 1000)
    assert lv.drag_analytic(2.0, 0.5, 252) == pytest.approx(0.25)               # (4-2) * 0.25 * 1 / 2
    assert lv.drag_analytic(-2.0, 0.5, 252) == pytest.approx(0.75)
    assert lv.drag_analytic(1.0, 0.5, 21) == 0.0
    simulated = float(np.nanmean(sim["shortfall"]))
    assert simulated == pytest.approx(analytic, rel=0.05)
    sim_m = lv.simulate_trade(flat_model(), last_close=100.0, entry=100.0, target=None, stop=None, leverage=-2.0, horizon=1000, n_paths=500,
                              seed=7)
    assert float(np.nanmean(sim_m["shortfall"])) == pytest.approx(lv.drag_analytic(-2.0, math.sqrt(252 * sim_m["mean_r2"]), 1000), rel=0.05)


# ------------------------------------------------------------------------------------------------ size / account stop arithmetic
def test_size_and_account_stop_arithmetic():
    final = np.array([-0.30, -0.25, -0.20, -0.10, 0.0, 0.15])
    s = lv.size_check(0.3, 45000.0, 0.15, -0.20, final, worst_day=-0.25)
    assert s["position_usd"] == pytest.approx(13500) and s["loss_at_stop_share"] == pytest.approx(0.06)
    assert s["loss_at_stop_usd"] == pytest.approx(2700) and s["account_stop_usd"] == pytest.approx(6750)
    assert s["max_size"] == pytest.approx(0.75) and s["trade_loss_threshold"] == pytest.approx(0.5)
    assert s["p_account_stop"] == 0.0 and s["max_size_gap"] == pytest.approx(0.6)
    big = lv.size_check(0.9, 45000.0, 0.15, -0.20, final)                    # threshold 1/6: -0.20, -0.25, -0.30 breach
    assert big["p_account_stop"] == pytest.approx(3 / 6) and big["max_size"] == pytest.approx(0.75)
    at_max = lv.size_check(0.75, 45000.0, 0.15, -0.20, np.array([-0.20]))   # exactly at the limit: not a breach
    assert at_max["p_account_stop"] == 0.0
    assert lv.size_check(1.0, 1000.0, 0.5, -0.1, final)["max_size"] == 1.0   # capped at the whole account
    assert lv.worst_day_moves([-0.6, 0.1], 2.0) == (-1.0, pytest.approx(0.2))   # 2x of -60% is a wipe-out
    assert lv.worst_day_moves([-0.1, 0.3], -2.0) == (pytest.approx(-0.6), pytest.approx(0.2))


# ------------------------------------------------------------------------------------------------ model choice
def test_fallback_selected_by_history_length():
    short = lv.build_model(hist(420))
    assert short.kind == "fallback" and "fallback, no leverage effect" in short.note and "only 420 returns" in short.note
    assert short.sigma_annual == pytest.approx(0.02 * math.sqrt(252), rel=0.2)
    assert lv.build_model(hist(749)).kind == "fallback"
    long = lv.build_model(hist(1300))
    assert long.kind == "gjr" and long.params["window"] == 1000 and 2.1 <= long.params["nu"] <= 100 and long.params["next_var"] > 0
    with pytest.raises(ValueError, match="at least 20"):
        lv.build_model(hist(10))


def test_determinism_and_draw_properties():
    kw = dict(last_close=50.0, entry=50.0, target=lv.Level("ret", 0.1), stop=lv.Level("ret", -0.1), leverage=2.0, horizon=20, n_paths=3000)
    m = lv.build_model(hist(1100, seed=2))
    a, b = lv.simulate_trade(m, seed=11, **kw), lv.simulate_trade(m, seed=11, **kw)
    c = lv.simulate_trade(m, seed=12, **kw)
    assert np.array_equal(a["ret"], b["ret"]) and not np.array_equal(a["ret"], c["ret"])
    R = lv.draw_returns(m, 4000, 21, np.random.default_rng(0))
    assert R.shape == (4000, 21) and abs(R.mean()) < 5e-3 and R.min() >= -1.0
    assert lv.draw_returns(flat_model(), 4000, 5, np.random.default_rng(0), drift_annual=2.52).mean() == pytest.approx(0.01, abs=0.002)


def test_drift_changes_direction_odds():
    kw = dict(last_close=100.0, entry=100.0, target=lv.Level("ret", 0.1), stop=lv.Level("ret", -0.1), leverage=1.0, horizon=60,
              n_paths=8000, seed=1)
    up = lv.simulate_trade(flat_model(), drift_annual=1.0, **kw)
    zero = lv.simulate_trade(flat_model(), drift_annual=0.0, **kw)
    assert (up["outcome"] == 1).mean() > (zero["outcome"] == 1).mean() + 0.03


# ------------------------------------------------------------------------------------------------ report / CLI
def test_run_report_contents():
    res = lv.run(hist(500), ticker="ZZ", last_close=100.0, leverage=2.0, target="15%", stop="-20%", n_paths=4000, size_frac=0.3)
    t = lv.format_report(res)
    for must in ("model: fallback, no leverage effect", "drift: ZERO", "NO view on direction", "target hit first", "stop hit first", "neither by day",
                 "final trade return quantiles", "5%:", "95%:", "analytic (L^2-L) sigma^2 T/2", "POSITION SIZE AND ACCOUNT STOP",
                 "size <= account_stop / |stop|", "overnight gaps", "WHAT THIS DOES NOT TELL YOU", "Fallback model"):
        assert must in t, must
    assert "75.0% of the account" in t
    res = lv.run(hist(500), ticker="ZZ", last_close=100.0, target="110", stop="90", n_paths=2000)
    t = lv.format_report(res)
    assert "UNDERLYING touch probabilities" in t and "110.0000 (+10.0% from last)" in t
    assert "POSITION SIZE" not in t and "drift: 0" not in t


def load_script():
    spec = importlib.util.spec_from_file_location("level_calc", ROOT / "scripts" / "level_calc.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cli_end_to_end_writes_only_to_tmp(tmp_path, capsys):
    days = pd.bdate_range("2021-01-04", periods=900)
    p = 50 * np.cumprod(1 + hist(900, seed=5))
    pd.DataFrame(dict(gvkey="1", iid="01", tic="ZZQX", datadate=days, prccd=p)).to_parquet(tmp_path / "p.parquet")
    rep = tmp_path / "out" / "level_report.txt"
    argv = ["zzqx", "--prices-file", str(tmp_path / "p.parquet"), "--leverage", "-2", "--target", "10%", "--stop", "-15%", "--horizon", "10",
            "--paths", "3000", "--seed", "3", "--size-frac", "0.25", "--report", str(rep), "--meta", str(tmp_path / "none.csv")]
    mod = load_script()
    assert mod.main(argv) == 0
    out = capsys.readouterr().out
    assert out.strip() == rep.read_text().strip() and "daily-reset -2x product on ZZQX" in out and f"{p[-1]:.4f}" in out
    assert "model: GJR-GARCH(1,1)-t fitted on the last 899 returns" in out          # 899 >= 750: the volfc estimator
    assert mod.main(argv) == 0 and capsys.readouterr().out.strip() == rep.read_text().strip()
    with pytest.raises(SystemExit, match="not in"):
        mod.main(["NOPE", "--prices-file", str(tmp_path / "p.parquet"), "--report", str(rep)])


# ------------------------------------------------------------------------------------------------ intraday bridge / martingale
SNDK_LIKE = lv.Model("fallback", 417, 0.73, "t", dict(nu=5.0, label="ewma94", window=417))


def test_driftless_stock_with_level_exits_is_a_martingale_with_the_bridge():
    """Spot 1482.33, target 1800, stop 1450, vol 73%: price martingale => P(target first) = (S0-stop)/(target-stop) = 9.2%, E[trade return] = 0."""
    n = 100000
    kw = dict(last_close=1482.33, entry=1482.33, target=lv.Level("price", 1800.0), stop=lv.Level("price", 1450.0), leverage=1.0, horizon=21,
              n_paths=n, seed=1)
    br = lv.simulate_trade(SNDK_LIKE, **kw)
    co = lv.simulate_trade(SNDK_LIKE, close_only=True, **kw)
    ideal = (1482.33 - 1450.0) / (1800.0 - 1450.0)
    assert abs((br["outcome"] == 1).mean() - ideal) < 0.012
    se = br["ret"].std() / math.sqrt(n)
    assert abs(br["ret"].mean()) < max(4 * se, 0.003)                                  # about 0, not the +1.9% of the closes-only fills at the level
    # closes-only: the stop is hit far less often (pierced intraday but closed above is never stopped)
    assert (co["outcome"] == -1).mean() < (br["outcome"] == -1).mean() - 0.05
    assert (co["outcome"] == 1).mean() > (br["outcome"] == 1).mean()
    assert abs(co["ret"].mean()) < 0.003                                               # fills at the trigger close keep the martingale too
    # return-level barriers are the same barriers for L = 1
    rl = lv.simulate_trade(SNDK_LIKE, **{**kw, "target": lv.Level("ret", 1800 / 1482.33 - 1), "stop": lv.Level("ret", 1450 / 1482.33 - 1)})
    assert np.array_equal(rl["outcome"], br["outcome"])


def test_bridge_first_passage_long_horizon_martingale():
    m = flat_model()
    sim = lv.simulate_trade(m, last_close=100.0, entry=100.0, target=lv.Level("price", 120.0), stop=lv.Level("price", 90.0), leverage=1.0,
                            horizon=600, n_paths=30000, seed=2)
    p = float((sim["outcome"] == 1).mean())
    assert abs(p - 1 / 3) < 4 * math.sqrt(p * (1 - p) / 30000) + 0.01 and (sim["outcome"] == 0).mean() < 0.005
    assert abs(sim["ret"].mean()) < 0.006


@pytest.mark.parametrize("L", [2.0, -2.0])
def test_leveraged_return_levels_are_martingale_too(L):
    sim = lv.simulate_trade(SNDK_LIKE, last_close=100.0, entry=100.0, target=lv.Level("ret", 0.25), stop=lv.Level("ret", -0.15), leverage=L,
                            horizon=42, n_paths=60000, seed=3)
    assert abs(sim["ret"].mean()) < 0.01                      # E[V_t] = 1 for a zero-mean underlying; limit fills at the levels keep it
    co = lv.simulate_trade(SNDK_LIKE, last_close=100.0, entry=100.0, target=lv.Level("ret", 0.25), stop=lv.Level("ret", -0.15), leverage=L,
                           horizon=42, n_paths=60000, seed=3, close_only=True)
    assert abs(co["ret"].mean()) < 0.01 and ((sim["outcome"] != 0).mean() >= (co["outcome"] != 0).mean() - 0.01)


def test_report_shows_close_only_comparison_unless_close_only():
    kw = dict(ticker="ZZ", last_close=1482.33, target="1800", stop="1450", n_paths=3000)
    r = lv.run(hist(417, sigma=0.046), **kw)
    t = lv.format_report(r)
    assert "close-only (no intraday touches, fill at trigger close):" in t and "limit / stop orders filled AT the level" in t
    assert "largest one-day moves" in t and "Brownian-bridge approximation" in t
    t2 = lv.format_report(lv.run(hist(417, sigma=0.046), close_only=True, **kw))
    assert "close-only (no intraday touches" not in t2 and "mode: CLOSE-ONLY" in t2 and "sim_close" not in lv.run(hist(417), close_only=True, **kw)
    assert r["sim"]["close_only"] is False and r["sim_close"]["close_only"] is True


# ------------------------------------------------------------------------------------------------ ETF awareness
META = {"SNDQ": ("SNDK", -2.0), "SNXX": ("SNDK", 2.0), "INTU": ("INTU", 1.0)}


def test_resolve_instrument_branches():
    avail = ["SNDK", "SNDQ", "INTU", "AMD"]
    r = lv.resolve_instrument("sndq", META, avail, None)
    assert (r["sim_ticker"], r["leverage"], r["etf_ticker"]) == ("SNDK", -2.0, "SNDQ")
    assert lv.resolve_instrument("SNXX", META, avail, None)["leverage"] == 2.0
    with pytest.raises(ValueError, match=r"SNDQ is already -2x SNDK; run level_calc.py SNDK --leverage -2, or drop --leverage"):
        lv.resolve_instrument("SNDQ", META, avail, 2.0)
    fb = lv.resolve_instrument("SNDQ", META, ["SNDQ"], None)                      # underlying not in the file: own returns at leverage 1
    assert (fb["sim_ticker"], fb["leverage"], fb["etf_ticker"]) == ("SNDQ", 1.0, None) and "SNDK is not in the price file" in fb["note"]
    assert lv.resolve_instrument("AMD", META, avail, 2.0)["leverage"] == 2.0       # plain stock with --leverage
    assert lv.resolve_instrument("AMD", META, avail, None)["leverage"] == 1.0
    assert lv.resolve_instrument("INTU", META, avail, None)["leverage"] == 1.0      # meta leverage 1: not an ETF
    assert lv.resolve_instrument("INTU", META, avail, 2.0)["leverage"] == 2.0
    assert lv.resolve_instrument("AMD", None, avail, None)["etf_ticker"] is None


def etf_world(tmp_path, with_underlying=True, n=900):
    days = pd.bdate_range("2021-01-04", periods=n)
    u = hist(n, sigma=0.03, seed=8)
    e = np.maximum(1 - 2 * u, 0) - 1                                              # exact -2x daily-reset ETF
    rows = [pd.DataFrame(dict(gvkey="2", iid="01", tic="SNDQ", datadate=days, prccd=20 * np.cumprod(1 + e)))]
    if with_underlying:
        rows.append(pd.DataFrame(dict(gvkey="1", iid="01", tic="SNDK", datadate=days, prccd=100 * np.cumprod(1 + u))))
    pd.concat(rows).to_parquet(tmp_path / "p.parquet")
    (tmp_path / "meta.csv").write_text("ticker,underlying,leverage\nSNDQ,SNDK,-2\n")
    return u, e


def test_cli_listed_etf_simulates_the_underlying(tmp_path, capsys):
    u, e = etf_world(tmp_path)
    mod = load_script()
    base = ["SNDQ", "--prices-file", str(tmp_path / "p.parquet"), "--meta", str(tmp_path / "meta.csv"), "--report", str(tmp_path / "r.txt"),
            "--paths", "2000", "--horizon", "10", "--target", "10%", "--stop", "-15%"]
    assert mod.main(base) == 0
    out = capsys.readouterr().out
    assert "instrument: SNDQ = -2x daily-reset product on SNDK (from meta)" in out and "LEVEL CALCULATOR for SNDK" in out
    assert "tracking sanity (SNDQ vs -2 x SNDK, 899 common days" in out and "slope -2.00" in out and "tracking error 0.0%/yr" in out
    assert f"worst {100 * e[1:].min():+.1f}%" in out                                 # the ETF's own worst day, not -2x of the stock's
    assert "annual vol: unconditional" in out
    with pytest.raises(SystemExit, match="SNDQ is already -2x SNDK; run level_calc.py SNDK --leverage -2, or drop --leverage"):
        mod.main(base + ["--leverage", "-2"])
    assert mod.main(["SNDK", "--leverage", "-2"] + base[1:]) == 0                    # the stock with --leverage still works
    assert "daily-reset -2x product on SNDK" in capsys.readouterr().out


def test_cli_etf_without_underlying_in_file_falls_back(tmp_path, capsys):
    etf_world(tmp_path, with_underlying=False)
    mod = load_script()
    assert mod.main(["SNDQ", "--prices-file", str(tmp_path / "p.parquet"), "--meta", str(tmp_path / "meta.csv"), "--report", str(tmp_path / "r.txt"),
                     "--paths", "1000", "--horizon", "5"]) == 0
    out = capsys.readouterr().out
    assert "SNDK is not in the price file: simulating SNDQ's own returns at leverage 1" in out and "LEVEL CALCULATOR for SNDQ" in out
    assert "from meta" not in out
    # a missing meta file is simply not used
    assert mod.main(["SNDQ", "--prices-file", str(tmp_path / "p.parquet"), "--meta", str(tmp_path / "nope.csv"), "--report", str(tmp_path / "r.txt"),
                     "--paths", "1000", "--horizon", "5"]) == 0
