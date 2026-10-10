import numpy as np
import pandas as pd
import pytest

from qap.costs import (abdi_ranaldo_spread, corwin_schultz, liquidity_inputs, sanity_report, trade_cost,
                       yyyymm_to_mi)
from qap.portfolio import costed_books, leg_costs, leg_trades, period_stats

from test_pipeline import make_world
from qap.panel import build_monthly_panel, to_mi


def test_corwin_schultz_known_pair():
    # hand-computed: beta=1.998e-4, gamma=1.2087e-4, k=3-2*sqrt(2) -> alpha ~= 0.00758, S ~= alpha
    assert corwin_schultz(100.6, 99.6, 100.5, 99.5) == pytest.approx(0.00758, rel=0.02)
    assert corwin_schultz(100, 100, 100, 100) == 0.0                      # no range -> no spread
    assert corwin_schultz(102, 100, 101, 99) == 0.0                       # negative estimate floored at 0


def test_abdi_ranaldo_floor():
    assert abdi_ranaldo_spread([0.0001, -0.5, 0.0])[0] == pytest.approx(0.02)
    assert abdi_ranaldo_spread([-0.5])[0] == 0.0


def test_trade_cost_impact_scales_with_three_halves_power():
    kw = dict(spread=0.0, sigma_d=0.02, adv=1e7, aum=1e8)
    c1, c4 = trade_cost(0.01, **kw), trade_cost(0.04, **kw)
    assert c4 / c1 == pytest.approx(4 ** 1.5)
    # fixed part is linear in |dw|; stress multiplier is linear overall
    f1, f4 = trade_cost(0.01, 0.002, 0.0, 1e7, 1e8), trade_cost(0.04, 0.002, 0.0, 1e7, 1e8)
    assert f4 / f1 == pytest.approx(4.0) and f1 == pytest.approx(0.01 * 0.001)
    assert trade_cost(0.01, 0.002, 0.02, 1e7, 1e8, mult=2.0) == pytest.approx(2 * trade_cost(0.01, 0.002, 0.02, 1e7, 1e8))
    assert trade_cost(-0.01, **kw) == pytest.approx(c1)


def _two_month_example():
    # long leg: month 1 weights A=.5, B=.5. Month 2 returns: A +100%, B 0 -> drifted A=2/3, B=1/3.
    # new weights month 2: A=.5, B=.25, C=.25 -> dw: A=1/6, B=1/12, C=1/4, sum=0.5, one-way turnover 0.25
    W = pd.DataFrame({"permno": [1, 2, 1, 2, 3], "mi": [100, 100, 101, 101, 101], "w": [.5, .5, .5, .25, .25]})
    panel = pd.DataFrame({"permno": [1, 2, 3, 1, 2, 3], "mi": [100, 100, 100, 101, 101, 101],
                          "mthret_adj": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]})
    inputs = pd.DataFrame({"permno": [1, 2, 3], "mi": [101, 101, 101], "spread": 0.002, "sigma_d": 0.0, "adv": 1e9})
    return W, panel, inputs


def test_turnover_and_cost_hand_computed():
    W, panel, inputs = _two_month_example()
    t = leg_trades(W, panel).set_index("permno")
    assert t.loc[1, "dw"] == pytest.approx(2 / 3 - 0.5) and t.loc[2, "dw"] == pytest.approx(1 / 3 - 0.25)
    assert t.loc[3, "dw"] == pytest.approx(0.25)
    c = leg_costs(W, panel, inputs, aum=1e6)
    assert c.loc[101, "turnover"] == pytest.approx(0.25)
    assert c.loc[101, "cost"] == pytest.approx(0.5 * 0.5 * 0.002)         # sum|dw| * half-spread = .5 * .001
    assert list(c.index) == [101]                                          # first formation month is not a rebalance


def test_exited_name_is_traded_out_at_monthly_median_cost():
    W = pd.DataFrame({"permno": [1, 2, 1], "mi": [100, 100, 101], "w": [.5, .5, 1.0]})   # name 2 leaves the leg
    panel = pd.DataFrame({"permno": [1, 2, 1], "mi": [100, 100, 101], "mthret_adj": [0.0, 0.0, 0.0]})
    inputs = pd.DataFrame({"permno": [1], "mi": [101], "spread": 0.004, "sigma_d": 0.0, "adv": 1e9})
    c = leg_costs(W, panel, inputs, aum=1e6)
    assert c.loc[101, "turnover"] == pytest.approx(0.5)                    # 0.5 sold + 0.5 bought of name 1
    assert c.loc[101, "cost"] == pytest.approx(1.0 * 0.002)                # exited name uses the month median spread


def _agg_for(panel, seed=0):
    rng = np.random.default_rng(seed)
    a = panel[["permno", "mi"]].copy()
    a["yyyymm"] = (a["mi"] // 12) * 100 + a["mi"] % 12 + 1
    n = len(a)
    a["n_ret"], a["n_q"], a["n_cs"], a["n_ar"] = 21, 21, 21, 21
    a["sigma_d"] = rng.uniform(.01, .03, n)
    a["qspread"] = rng.uniform(.001, .01, n)
    a["cs_spread"] = a["qspread"] * rng.uniform(.8, 1.2, n)
    a["ar_x_mean"] = (a["qspread"] / 2) ** 2 / 4
    a["adv_a"] = rng.uniform(1e5, 1e7, n)
    a["adv_b"] = a["adv_a"]
    return a.drop(columns="mi")


def test_liquidity_inputs_fallback_uses_size_decile_median():
    m, d, *_ = make_world(n=120)
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    agg = _agg_for(p)
    drop = agg.sample(frac=0.2, random_state=1).index
    agg.loc[drop, ["n_q", "n_cs"]] = 0                                      # no usable estimate -> fallback
    inp = liquidity_inputs(p, agg)
    assert set(inp["spread_src"]) == {"quoted", "fallback"}
    assert inp[["spread", "sigma_d", "adv"]].notna().all().all()
    assert inp["spread"].between(1e-4, 0.5).all()


def test_sanity_report_runs_and_flags_shapes():
    m, d, *_ = make_world(n=240, years=(1995, 2008))
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    inp = liquidity_inputs(p, _agg_for(p))
    lines, checks = sanity_report(inp)
    assert checks["dollar volume columns agree (>=90% within 10%)"]
    assert any("decimalization" in l for l in lines) and len(checks) == 3   # size-monotone check is False on random data
    bad = inp.copy()
    bad["adv_a"] = bad["adv_b"] * 1000
    assert not sanity_report(bad)[1]["dollar volume columns agree (>=90% within 10%)"]


def test_costed_books_end_to_end_and_period_stats():
    from qap.evaluate import add_composite, build_research_signals
    from qap.fundamentals import prepare_annual
    m, d, funda, links = make_world(n=240)
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    sig = build_research_signals(p, prepare_annual(funda), links)
    inp = liquidity_inputs(p, _agg_for(p))
    res = costed_books(sig, p, inp, "gpa", aum=1e7)
    for book in ("LS", "L_minus_U"):
        g, c = res[book]["gross"], res[book]["cost"]
        assert len(g) > 50 and (c > 0).all() and c.index.isin(g.index).all()
    assert (res["turnover_L"].between(0, 1)).all()
    # cost grows with AUM
    big = costed_books(sig, p, inp, "gpa", aum=1e10)
    assert big["LS"]["cost"].mean() > res["LS"]["cost"].mean()
    ps = period_stats(res["LS"]["gross"])
    assert set(ps) == {"pre1990", "dev1990_2014", "valid2015_2019", "pre2020", "post1990"}


def test_sql_matches_python_reference():
    duckdb = pytest.importorskip("duckdb")
    from qap.extract import crsp_daily_agg_sql
    con = duckdb.connect()
    con.execute("create schema crsp")
    con.execute("""create table crsp.dsf_v2(permno integer, dlycaldt date, dlyret double, dlyprc double, dlyclose double,
                   dlyvol double, dlyprcvol double, dlybid double, dlyask double, dlyhigh double, dlylow double)""")
    rng = np.random.default_rng(0)
    rows, ref = [], []
    px = 50.0
    for day in pd.bdate_range("2019-12-02", "2020-02-28"):
        r = rng.normal(0, .02)
        px *= 1 + r
        hi, lo = px * (1 + abs(rng.normal(0, .01))), px * (1 - abs(rng.normal(0, .01)))
        rows.append((1, day.date(), r, px, px, 1e5, px * 1e5, px * .999, px * 1.001, hi, lo))
        ref.append((day, hi, lo, r))
    con.executemany("insert into crsp.dsf_v2 values (?,?,?,?,?,?,?,?,?,?,?)", rows)
    out = con.execute(crsp_daily_agg_sql("2020-01-01", "2020-01-31")).df()
    assert out["yyyymm"].tolist() == [202001]
    df = pd.DataFrame(ref, columns=["d", "hi", "lo", "r"])
    cs = []
    for i in range(1, len(df)):
        if df.d[i].month == 1 and (df.d[i] - df.d[i - 1]).days <= 5:
            cs.append(corwin_schultz(df.hi[i], df.lo[i], df.hi[i - 1], df.lo[i - 1]))
    jan = df[df.d.dt.month == 1]
    assert out["n_cs"].iloc[0] == len(cs)
    assert out["cs_spread"].iloc[0] == pytest.approx(np.mean(cs), rel=1e-9)
    assert out["qspread"].iloc[0] == pytest.approx(0.002 / 1.0, rel=1e-6)
    assert out["sigma_d"].iloc[0] == pytest.approx(jan["r"].std(ddof=1), rel=1e-9)
    assert out["adv_a"].iloc[0] == pytest.approx(out["adv_b"].iloc[0])


def test_market_scaled_aum_is_constant_share_of_market():
    from qap.portfolio import market_scaled_aum
    mis = [2018 * 12 + k for k in range(12)] + [2019 * 12 + k for k in range(12)] + [1970 * 12]
    rows = []
    for mi in mis:
        cap = 100.0 if mi >= 2019 * 12 else (10.0 if mi == 1970 * 12 else 50.0)   # market cap: 10 in 1970, 100 in 2019
        rows.append({"permno": 1, "mi": mi, "me": cap, "eq": True})
    a = market_scaled_aum(pd.DataFrame(rows), 1e7)
    assert a[2019 * 12 + 3] == pytest.approx(1e7) and a[1970 * 12] == pytest.approx(1e6) and a[2018 * 12] == pytest.approx(5e6)


def test_leg_costs_accepts_aum_series_and_larger_early_aum_costs_more():
    W, panel, inputs = _two_month_example()
    inputs = inputs.assign(sigma_d=0.02, adv=1e6)
    small = leg_costs(W, panel, inputs, aum=pd.Series({101: 1e6}))
    big = leg_costs(W, panel, inputs, aum=pd.Series({101: 1e8}))
    assert big.loc[101, "cost"] > small.loc[101, "cost"]
    assert leg_costs(W, panel, inputs, aum=1e6).loc[101, "cost"] == pytest.approx(small.loc[101, "cost"])
