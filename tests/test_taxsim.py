import numpy as np
import pandas as pd
import pytest

from qap.panel import build_monthly_panel
from qap.taxsim import (TaxLedger, TaxParams, annualised, build_inputs, decision_label, exit_mask_from_returns, is_long_term,
                        net_year, run_windows, simulate_window, summarize_windows, tracking_error, waterfill, window_me0)
from test_pipeline import make_world

SM = 2000 * 12
P0 = TaxParams(contrib=0.0, cost_bp=0.0, outside_lt_gains=0.0, n_names=3)


def sim(pr, dy=None, params=P0, me0=None, exit_mask=None, mkt=None):
    pr = np.asarray(pr, float)
    T, N = pr.shape
    dy = np.zeros((T, N)) if dy is None else np.asarray(dy, float)
    me0 = np.ones(N) if me0 is None else np.asarray(me0, float)
    m = np.zeros(T) if mkt is None else np.asarray(mkt, float)
    return simulate_window(pr, dy, me0, m, np.zeros(T), params, start_mi=SM, exit_mask=exit_mask)


# ---------------------------------------------------------------- (i) ledger

def test_ledger_two_year_netting_cap_and_carryforward():
    p = TaxParams()
    L = TaxLedger(p)
    # year 1: ST net -2000, LT 0 + 3000 outside -> cross-net 2000 -> LT gain 1000 -> 238
    L.add_gain(2000, -2000.0, False)
    assert L.close_year(2000) == pytest.approx(1000 * 0.238)
    assert (L.carry_st, L.carry_lt) == (0.0, 0.0)
    # year 2: ST -10000, LT -1000 + 3000 outside = 2000 -> cross 2000 -> ST loss 8000, deduct 3000 at 40.8%, carry 5000 ST
    L.add_gain(2001, -10000.0, False)
    L.add_gain(2001, -1000.0, True)
    assert L.close_year(2001) == pytest.approx(-3000 * 0.408)
    assert L.carry_st == pytest.approx(5000.0) and L.carry_lt == 0.0
    # year 3: ST +1000 less 5000 carry = -4000, LT 3000 -> cross 3000 -> ST -1000 deducted fully
    L.add_gain(2002, 1000.0, False)
    assert L.close_year(2002) == pytest.approx(-1000 * 0.408)
    assert L.carry_st == 0.0
    # dividends are taxed at div_rate, outside gain alone is 3000 * 23.8%
    L.add_div(2003, 1000.0)
    assert L.close_year(2003) == pytest.approx(1000 * 0.238 + 3000 * 0.238)


def test_net_year_rules():
    p = TaxParams(outside_lt_gains=0.0)
    r = net_year(5000.0, -2000.0, 0.0, 0.0, p)           # ST gain absorbs LT loss first: 3000 ST taxed at 40.8%
    assert r["tax"] == pytest.approx(3000 * 0.408)
    r = net_year(-500.0, -4000.0, 0.0, 0.0, p)            # both losses: ST used first against the cap, then LT
    assert r["tax"] == pytest.approx(-3000 * 0.408)
    assert r["carry_st"] == 0.0 and r["carry_lt"] == pytest.approx(1500.0)
    r = net_year(0.0, 2000.0, 0.0, 1500.0, p)             # LT carryforward used against LT gain
    assert r["tax"] == pytest.approx(500 * 0.238) and r["carry_lt"] == 0.0


# ---------------------------------------------------------------- (iv) long vs short

def test_holding_period_boundary():
    p = TaxParams()
    assert not is_long_term(11, -1, p)      # bought at end of month -1, sold at end of month 11: exactly 12 months
    assert is_long_term(12, -1, p)          # 13 months
    for t_exit, key in ((11, "realised_st"), (12, "realised_lt")):
        pr = np.zeros((14, 2))
        pr[0, 0] = 1.0                       # name A doubles in month 0
        em = np.zeros((14, 2), bool)
        em[t_exit, 0] = True
        r = sim(pr, params=P0.with_(n_names=2), exit_mask=em, me0=[1, 1])["nohv"]
        other = "realised_lt" if key == "realised_st" else "realised_st"
        assert r[key] == pytest.approx(17500.0) and r[other] == 0.0


# ---------------------------------------------------------------- (v) delisting

def test_delisting_is_taxable_sale_and_proceeds_reinvested():
    T = 12
    pr = np.zeros((T, 2))
    pr[0, 0] = 1.0                           # A: 17500 -> 35000
    em = np.zeros((T, 2), bool)
    em[5, 0] = True
    for arm in ("nohv", "hv"):
        r = sim(pr, params=P0.with_(n_names=2), exit_mask=em, me0=[1, 1])[arm]
        assert r["mv"] == pytest.approx(52500.0)
        assert r["n_names"] == 1
        lots = r["lots"]
        assert (lots["name"] == 1).all() and sorted(lots["mon"].tolist()) == [-1, 5]
        assert lots["basis"][lots["mon"] == 5][0] == pytest.approx(35000.0)
        assert r["realised_st"] == pytest.approx(17500.0)          # held 6 months
        assert r["wealth_hold"] == pytest.approx(52500.0 - 17500.0 * 0.408)
    pr2 = np.zeros((T, 2))
    pr2[3, 0] = -1.0                         # bankruptcy: total loss realised at the delisting month
    em2 = np.zeros((T, 2), bool)
    em2[3, 0] = True
    r = sim(pr2, params=P0.with_(n_names=2), exit_mask=em2, me0=[1, 1])["nohv"]
    assert r["realised_st"] == pytest.approx(-17500.0) and r["mv"] == pytest.approx(17500.0)
    assert r["tax_hold_final"] == pytest.approx(-3000 * 0.408)       # capped; 14500 carries forward
    assert r["carry_st_end"] == pytest.approx(0.0)                   # carry_* is the pre-final-year state
    assert set(r["lots"]["name"].tolist()) == {1}


# ---------------------------------------------------------------- (ii) hand-calculated harvest

def test_harvest_one_name_falls_then_recovers_hand_calc():
    T = 12
    pr = np.zeros((T, 3))
    pr[0, 0], pr[1, 0], pr[2, 0] = -0.08, -0.12, 0.25
    res = sim(pr)
    a0 = 35000 / 3
    va = a0 * 0.92 * 0.88
    loss = a0 - va
    hv, nh = res["hv"], res["nohv"]
    assert hv["harvest_lots"] == 1 and nh["harvest_lots"] == 0
    assert hv["harv_st"][0] == pytest.approx(loss) and hv["harv_lt"][0] == 0.0
    assert hv["mv"] == pytest.approx(35000 - loss)
    assert sorted(hv["lots"]["name"].tolist()) == [1, 1, 2, 2]       # A gone; proceeds went pro rata to B and C
    for n in (1, 2):
        assert hv["lots"]["val"][hv["lots"]["name"] == n].sum() == pytest.approx(a0 + va / 2)
    assert hv["tax_hold_final"] == pytest.approx(-loss * 0.408)      # loss < 3000 cap, fully deducted
    assert hv["wealth_hold"] == pytest.approx(35000 - loss + loss * 0.408)
    assert nh["mv"] == pytest.approx(2 * a0 + va * 1.25) and nh["wealth_hold"] == nh["mv"]
    # liquidating later gives back the tax benefit: B and C gained nothing, so liquidation tax is 0 for harvest too
    assert hv["wealth_liq"] == pytest.approx(hv["wealth_hold"])
    # just under the threshold: no harvest
    pr[1, 0] = -0.0
    pr[0, 0] = -0.095
    assert sim(pr)["hv"]["harvest_lots"] == 0


def test_harvest_blocks_sold_name_two_months_and_reinvests_via_dividends():
    T = 8
    pr = np.zeros((T, 3))
    pr[0, 0], pr[1, 0] = -0.08, -0.12
    dy = np.zeros((T, 3))
    dy[1:, 1:] = 0.01                         # B, C pay dividends from month 1: cash that wants to buy the (blocked) underweight A
    hv = sim(pr, dy)["hv"]
    a_mon = sorted(hv["lots"]["mon"][hv["lots"]["name"] == 0].tolist())
    assert 1 not in a_mon and 2 not in a_mon        # blocked in the month of the sale and the next
    assert a_mon[0] == 3                              # first month it can be bought again
    assert hv["harvest_lots"] == 1


# ---------------------------------------------------------------- (iii) wash-sale / recent purchase

def test_recent_purchase_rule_and_wash_sale_block():
    T = 6
    pr = np.zeros((T, 3))
    pr[0, 1] = 0.10
    pr[1, 0] = pr[1, 1] = -0.25                       # A and B both fall 25% in month 1 (A fell no earlier)
    pk = P0.with_(contrib=500.0)
    r = sim(pr[:2], params=pk)                        # two months: only the month-1 decision
    hv = r["hv"]
    # month 0 contribution (500) goes to the underweight A and C (B rose), 250 each; so A has a purchase in month 0
    assert hv["harvest_lots"] == 1                    # only B's initial lot (no purchase in month 0)
    assert hv["skipped_recent"] == 2                  # A's initial lot and A's month-0 lot qualify but were bought-in-previous-month
    assert hv["harv_st"][0] == pytest.approx(35000 / 3 - 35000 / 3 * 1.1 * 0.75)
    r = sim(pr, params=pk)
    hv = r["hv"]
    b_mon = hv["lots"]["mon"][hv["lots"]["name"] == 1].tolist()
    assert 1 not in b_mon and 2 not in b_mon           # B blocked from the contributions of months 1 and 2
    assert 3 in b_mon                                  # and bought again from month 3 (most underweight)
    # the no-harvest arm never sells
    assert r["nohv"]["harvest_lots"] == 0 and r["nohv"]["sold_notional"] == 0.0


def test_waterfill_lifts_most_underweight_first():
    v = np.array([10.0, 30.0, 20.0])
    w = np.array([0.25, 0.5, 0.25])
    b = waterfill(v, w, np.ones(3, bool), 20.0)        # levels v/w = 40, 60, 80
    assert b.sum() == pytest.approx(20.0) and b[2] == 0 and b[0] > 0
    assert (v[:2] + b[:2]) / w[:2] == pytest.approx([(v[0] + b[0]) / w[0]] * 2)
    b2 = waterfill(v, w, np.array([False, True, True]), 20.0)
    assert b2[0] == 0 and b2.sum() == pytest.approx(20.0)
    assert waterfill(v, w, np.ones(3, bool), 0.0).sum() == 0.0


# ---------------------------------------------------------------- random worlds

def rand_world(T=48, N=12, seed=1, sd=0.12, dyv=0.003):
    rng = np.random.default_rng(seed)
    pr = rng.normal(0.004, sd, (T, N))
    dy = np.full((T, N), dyv)
    em = np.zeros((T, N), bool)
    em[20, 3] = em[33, 7] = True
    pr[20, 3] = -0.6
    pr[33, 7] = 0.4
    mkt = pr.mean(1)
    return pr, dy, em, mkt


def eq_dict(a, b):
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, dict):
            eq_dict(x, y)
        elif isinstance(x, np.ndarray):
            assert np.array_equal(x, y, equal_nan=True), k
        else:
            assert x == y or (x != x and y != y), k


def test_impossible_threshold_makes_harvest_equal_no_harvest_exactly():
    pr, dy, em, mkt = rand_world()
    p = TaxParams(n_names=10, harvest_pct=2.0)
    r = simulate_window(pr, dy, np.arange(1, 13, dtype=float), mkt, np.zeros(48), p, start_mi=SM, exit_mask=em)
    a, b = dict(r["hv"]), dict(r["nohv"])
    for d in (a, b):
        d.pop("skipped_recent")
    eq_dict(a, b)
    assert r["hv"]["harvest_lots"] == 0
    # sanity: with the real threshold the harvest arm does differ on the same world
    r2 = simulate_window(pr, dy, np.arange(1, 13, dtype=float), mkt, np.zeros(48), p.with_(harvest_pct=0.10), start_mi=SM, exit_mask=em)
    assert r2["hv"]["harvest_lots"] > 0 and r2["hv"]["wealth_hold"] != r2["nohv"]["wealth_hold"]


def test_wealth_conservation_and_zero_tax_cost_equivalence():
    pr, dy, em, mkt = rand_world(seed=3)
    for p in (TaxParams(n_names=10), TaxParams(n_names=10, contrib=0.0, cost_bp=15.0)):
        r = simulate_window(pr, dy, np.arange(1, 13, dtype=float), mkt, np.zeros(48), p, start_mi=SM, exit_mask=em)
        for arm in ("etf", "nohv", "hv"):
            a = r[arm]
            # independent identity: ledger-realised + lot-unrealised gains (costs are inside basis/proceeds) + dividends
            lhs = a["mv"] + a["taxes_prior"]
            rhs = p.start_capital + a["contrib_total"] + a["dividends"] + a["realised_st"] + a["realised_lt"] + a["unrealised"]
            assert lhs == pytest.approx(rhs, abs=1e-6)
            bp = 0.0 if arm == "etf" else p.cost_bp / 1e4
            assert a["costs"] == pytest.approx(bp * (a["buy_notional"] + a["sold_notional"]))
            assert a["contrib_total"] == p.contrib * 48
    # With all taxes and costs zero: harvest-disabled run IS the no-harvest arm; with harvest on, arms differ only through the
    # substitution effect (sell-and-reinvest changes weights), so equality of pre-tax wealth is NOT expected for different returns...
    z = TaxParams(n_names=10, st_rate=0.0, lt_rate=0.0, div_rate=0.0, outside_lt_gains=0.0, cost_bp=0.0)
    r = simulate_window(pr, dy, np.arange(1, 13, dtype=float), mkt, np.zeros(48), z, start_mi=SM, exit_mask=em)
    assert r["nohv"]["wealth_hold"] == r["nohv"]["mv"] and r["hv"]["taxes"].sum() == 0.0
    rz = simulate_window(pr, dy, np.arange(1, 13, dtype=float), mkt, np.zeros(48), z.with_(harvest_pct=2.0), start_mi=SM, exit_mask=em)
    assert rz["hv"]["mv"] == r["nohv"]["mv"]
    assert r["hv"]["mv"] != r["nohv"]["mv"]
    # ... but when every name has the same return the substitution is value-neutral: harvest must not change pre-tax wealth.
    # (3 big names are harvested; 9 tiny ones never reach the USD 500 floor and so always remain eligible receivers: with every
    # name blocked the proceeds would sit idle in cash, which is a real, intended feature but not value-neutral)
    common = np.repeat(pr[:, :1], 12, axis=1) - 0.02          # persistent decline so that lots do go >10% under water
    me_s = np.array([100.0, 100.0, 100.0] + [1.0] * 9)
    rc = simulate_window(common, np.zeros_like(dy), me_s, mkt, np.zeros(48), z.with_(contrib=0.0, harvest_min_usd=500.0),
                         start_mi=SM, exit_mask=np.zeros((48, 12), bool))
    assert rc["hv"]["harvest_lots"] > 0
    assert rc["hv"]["mv"] == pytest.approx(rc["nohv"]["mv"], rel=1e-12)


def test_etf_benchmark_is_vw_compounding():
    rng = np.random.default_rng(5)
    T = 36
    mp, md = rng.normal(0.006, 0.04, T), np.full(T, 0.002)
    p = TaxParams(st_rate=0.0, lt_rate=0.0, div_rate=0.0, outside_lt_gains=0.0, contrib=0.0, n_names=3)
    r = simulate_window(np.zeros((T, 3)), np.zeros((T, 3)), np.ones(3), mp, md, p, start_mi=SM)["etf"]
    assert r["mv"] == pytest.approx(35000 * np.prod(1 + mp + md), rel=1e-12)
    assert np.allclose(r["ret"], mp + md)
    assert r["costs"] == 0.0 and r["n_lots"] <= T + 1
    # with taxes the dividend tax is paid out of the account and no gain is ever realised
    r2 = simulate_window(np.zeros((T, 3)), np.zeros((T, 3)), np.ones(3), mp, md, TaxParams(contrib=0.0, n_names=3), start_mi=SM)["etf"]
    assert r2["realised_st"] == 0.0 and r2["realised_lt"] == 0.0
    assert r2["taxes"][:3].sum() > 0 and r2["dividends"] > 0 and r2["costs"] == 0.0
    # outside gain taxed in every year: 3 years x 3000 x 23.8% when nothing else happens
    r3 = simulate_window(np.zeros((T, 3)), np.zeros((T, 3)), np.ones(3), np.zeros(T), np.zeros(T), TaxParams(contrib=0.0, n_names=3),
                         start_mi=SM)["etf"]
    assert r3["taxes"].tolist() == pytest.approx([714.0] * 3)


# ---------------------------------------------------------------- (ix) summary and decision rule

def fake(sy, w_hv, w_nh, w_etf, liq_hv=None, liq_nh=None, ret_gap=0.0):
    rr = np.tile([0.01, -0.01], 60)
    arm = lambda wh, wl, r: {"wealth_hold": wh, "wealth_liq": wl, "ret": r, "harv_st": np.zeros(10), "harv_lt": np.zeros(10),
                             "carry_st_end": 0.0, "carry_lt_end": 0.0, "turnover": 0.1, "n_names": 100, "n_lots": 500,
                             "skipped_recent": 3, "harvest_lots": 20}
    return {"start_year": sy, "years": 10, "etf": arm(w_etf, w_etf, rr), "nohv": arm(w_nh, liq_nh or w_nh, rr + ret_gap),
            "hv": arm(w_hv, liq_hv or w_hv, rr + ret_gap * 2 * np.tile([1, -1], 60))}


def test_decision_rule_boundaries():
    ok = dict(median_benefit=0.5, share_pos=0.75, te=2.0, liq_median=0.1)
    assert decision_label(**ok) == "WORTH PURSUING"
    assert decision_label(**{**ok, "median_benefit": 0.4999}).startswith("NOT WORTH THE COMPLEXITY")
    assert decision_label(**{**ok, "share_pos": 0.7499}).startswith("NOT WORTH THE COMPLEXITY")
    assert decision_label(**{**ok, "te": 2.0001}).startswith("NOT WORTH THE COMPLEXITY")
    assert decision_label(**{**ok, "liq_median": 0.0}) == "WORTH PURSUING (deferral only: liquidate-at-end median benefit <= 0)"
    assert "deferral only" in decision_label(**{**ok, "median_benefit": 0.0, "liq_median": -1.0})
    assert "deferral only" not in decision_label(**{**ok, "liq_median": 1e-9})


def test_summarize_windows():
    ratio = 1.01 ** 10
    rs = [fake(1963 + i, 100.0 * ratio, 100.0, 100.0) for i in range(4)]
    s = summarize_windows(rs)
    assert s["primary"]["median"] == pytest.approx(1.0) and s["primary"]["share_pos"] == 1.0 and s["n_windows"] == 4
    assert s["decision"] == "WORTH PURSUING"
    assert set(s["by_decade"]) == {1960}
    assert annualised(110.0, 100.0, 1.0) == pytest.approx(10.0)
    rs[0] = fake(1963, 90.0, 100.0, 100.0)
    rs[1] = fake(1964, 90.0, 100.0, 100.0)
    assert summarize_windows(rs)["primary"]["share_pos"] == 0.5
    assert summarize_windows(rs)["decision"].startswith("NOT WORTH")
    # liquidate-at-end figure <= 0 while the hold figure is positive -> deferral only
    rs = [fake(1963 + i, 100.0 * ratio, 100.0, 100.0, liq_hv=100.0, liq_nh=100.0) for i in range(4)]
    assert "deferral only" in summarize_windows(rs)["decision"]
    assert tracking_error(np.tile([0.01, -0.01], 60), np.zeros(120)) == pytest.approx(
        np.std(np.tile([0.01, -0.01], 60), ddof=1) * np.sqrt(12) * 100)


def test_exit_mask_from_returns():
    r = np.ones((5, 3))
    r[3:, 0] = np.nan
    r[:, 2] = np.nan
    m = exit_mask_from_returns(r)
    assert m[2, 0] and m.sum() == 1


# ---------------------------------------------------------------- inputs and end-to-end

def nullable_world(n=40, years=(1990, 2003)):
    m, d, *_ = make_world(n=n, years=years)
    rng = np.random.default_rng(11)
    m = m[~((m.permno.isin([5, 6, 7])) & (m.mthcaldt > pd.Timestamp("1995-06-30")))].copy()   # last observations (delistings)
    m["mthretx"] = m["mthret"] - 0.002
    for c in ("mthret", "mthretx", "mthcap"):
        m[c] = m[c].astype("Float64")
    m["primaryexch"] = m["primaryexch"].astype("string")
    m.loc[m.sample(30, random_state=1).index, "mthretx"] = pd.NA
    return m, d


def test_share_class_deduplicated_and_inputs():
    m, d = nullable_world()
    panel = build_monthly_panel(m, d)
    inp = build_inputs(panel)
    assert inp["pr"].dtype == np.float64 and inp["mkt_pr"].dtype == np.float64
    mi0 = 1992 * 12
    me0 = window_me0(inp, mi0)
    p1, p2 = [int(np.nonzero(inp["permnos"] == k)[0][0]) for k in (1, 2)]
    assert np.isfinite(me0[p1]) != np.isfinite(me0[p2])          # permnos 1 and 2 share a permco: one is kept
    kept = p1 if np.isfinite(me0[p1]) else p2
    row = mi0 - 1 - inp["first_mi"]
    assert me0[kept] == inp["me_co"][row, kept]
    assert inp["me"][row, kept] >= inp["me"][row, p1 + p2 - kept]
    # market series is the lag-weighted mean of total returns: price part + dividend part
    t = mi0 - inp["first_mi"] + 5
    w = np.where(inp["eq"][t - 1] & (inp["me"][t - 1] > 0), inp["me"][t - 1], 0.0)
    v = np.isfinite(inp["pr"][t])
    assert inp["mkt_pr"][t] == pytest.approx((w[v] * inp["pr"][t][v]).sum() / w[v].sum())
    assert (inp["dy"] >= 0).all()


def test_end_to_end_jobs_1_vs_3_identical():
    m, d = nullable_world()
    inp = build_inputs(build_monthly_panel(m, d))
    p = TaxParams(n_names=10)
    a = run_windows(inp, 1990, 1994, p, jobs=1)
    b = run_windows(inp, 1990, 1994, p, jobs=3)
    assert len(a) == 5 and len(b) == 5
    for x, y in zip(a, b):
        eq_dict(x, y)
    s = summarize_windows(a)
    assert s["n_windows"] == 5 and s["decision"]
    for r in a:
        assert r["nohv"]["n_names"] <= 10
    assert any(r["nohv"]["realised_st"] + r["nohv"]["realised_lt"] != 0 for r in a)       # the delisted names were sold
