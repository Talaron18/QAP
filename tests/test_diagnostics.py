import numpy as np
import pandas as pd
import pytest

from qap.diagnostics import (diagnose_signal, format_report, quintile_table, reading, size_tercile_stats,
                             split_summary)
from qap.study import SEALED_CUT, net_lu_series

from test_costs import _agg_for
from test_pipeline import make_world

F = 100   # formation month; returns are earned in F + 1


def _world(extra_post=False):
    """16 names: permno 1 is an NYSE microcap (dropped); permnos 2..6 NYSE, 7..16 Nasdaq; me = 10 * permno, signal = permno."""
    p = np.arange(1, 17)
    sig = pd.DataFrame({"permno": p, "mi": F, "me": 10.0 * p, "me_co": 10.0 * p, "primaryexch": np.where(p <= 6, "N", "Q"),
                        "sig": p.astype(float)})
    sig.loc[0, ["me", "me_co"]] = 1.0
    r = {2: .10, 3: .08, 4: .06, 5: .04, 6: .02, 7: .01, 8: .02, 9: .03, 10: .04, 11: .05,
         12: .03, 13: np.nan, 14: .02, 15: .05, 16: .04, 1: .50}
    panel = pd.DataFrame({"permno": p, "mi": F + 1, "mthret_adj": [r[k] for k in p]})
    panel = pd.concat([panel, pd.DataFrame({"permno": p, "mi": F, "mthret_adj": 0.0})], ignore_index=True)
    if extra_post:   # poisoned months at/after the seal: must change nothing
        post = sig.assign(mi=SEALED_CUT + 3, sig=-sig["sig"])
        sig = pd.concat([sig, sig.assign(mi=SEALED_CUT - 1, sig=-sig["sig"]), post], ignore_index=True)
        panel = pd.concat([panel, panel.assign(mi=panel["mi"] + SEALED_CUT - F, mthret_adj=9.0)], ignore_index=True)
    return sig, panel, r


def test_quintile_table_hand_checked_vw_vs_ew():
    sig, panel, r = _world()
    t = quintile_table(sig, panel, "sig")
    assert list(t.index) == [F + 1]
    row = t.iloc[0]
    # eligible permnos 2..16 (15 names); overall quintile j holds permnos 3j-1..3j+1
    assert row["Q1_vw"] == pytest.approx((20 * .10 + 30 * .08 + 40 * .06) / 90)
    assert row["Q1_ew"] == pytest.approx((.10 + .08 + .06) / 3)
    # Q5 = permnos 14, 15, 16; permno 13 (missing return) is in Q4 and counts as 0
    assert row["Q5_vw"] == pytest.approx((140 * .02 + 150 * .05 + 160 * .04) / 450)
    assert row["Q5_ew"] == pytest.approx((.02 + .05 + .04) / 3)
    assert row["Q4_ew"] == pytest.approx((.05 + .03 + 0.0) / 3)
    me = 10.0 * np.arange(2, 17)
    ret = np.array([r[k] if np.isfinite(r[k]) else 0.0 for k in range(2, 17)])
    assert row["U_vw"] == pytest.approx((me * ret).sum() / me.sum()) and row["U_ew"] == pytest.approx(ret.mean())
    assert row["D10_vw"] == pytest.approx((150 * .05 + 160 * .04) / 310)   # decile 10 = permnos 15, 16
    assert row["D10_ew"] == pytest.approx((.05 + .04) / 2)
    assert row["Q5_vw"] != pytest.approx(row["Q5_ew"], abs=1e-6)


def test_quintile_table_matches_portfolio_legs():
    from qap.portfolio import leg_gross_return, quintile_legs
    sig, panel, _ = _world()
    t = quintile_table(sig, panel, "sig")
    legs = quintile_legs(sig, "sig")
    for leg, col in (("L", "Q5_vw"), ("S", "Q1_vw"), ("U", "U_vw")):
        g = leg_gross_return(legs[leg], panel)
        assert t[col].iloc[0] == pytest.approx(g.iloc[0], abs=1e-12)


def test_size_tercile_stats_hand_checked_with_nan_return():
    sig, panel, _ = _world()
    t = size_tercile_stats(sig, panel, "sig")
    row = t.iloc[0]
    assert row["ic_T1"] == pytest.approx(-1.0) and row["ic_T2"] == pytest.approx(1.0)
    # T3 = permnos 12..16, returns .03, NaN, .02, .05, .04: IC uses the four present names -> Spearman 0.6
    assert row["ic_T3"] == pytest.approx(0.6)
    assert row["ew_T1"] == pytest.approx(.02 - .10) and row["ew_T2"] == pytest.approx(.05 - .01)
    assert row["ew_T3"] == pytest.approx(.04 - .03)       # within-tercile quintiles: one name each; missing return counted 0


def test_dtypes_nullable_give_identical_tables():
    sig, panel, _ = _world()
    s2 = sig.astype({"permno": "Int64", "mi": "Int64", "me": "Float64", "me_co": "Float64", "sig": "Float64"})
    p2 = panel.astype({"permno": "Int64", "mi": "Int64", "mthret_adj": "Float64"})
    pd.testing.assert_frame_equal(quintile_table(sig, panel, "sig"), quintile_table(s2, p2, "sig"))
    pd.testing.assert_frame_equal(size_tercile_stats(sig, panel, "sig"), size_tercile_stats(s2, p2, "sig"))


def test_nothing_from_2020_on_is_used():
    sig, panel, _ = _world()
    sigp, panelp, _ = _world(extra_post=True)
    a, b = quintile_table(sig, panel, "sig"), quintile_table(sigp, panelp, "sig")
    pd.testing.assert_frame_equal(a, b)
    assert (b.index < SEALED_CUT).all()
    c, d = size_tercile_stats(sig, panel, "sig"), size_tercile_stats(sigp, panelp, "sig")
    pd.testing.assert_frame_equal(c, d)
    assert (d.index < SEALED_CUT).all()


def test_split_summary_windows_and_t():
    idx = np.arange(1989 * 12, 2021 * 12)
    s = pd.Series(1.0, index=idx)
    s[(idx >= 1990 * 12) & (idx <= 2014 * 12 + 11)] = np.tile([0.01, 0.03], 150)[:300]
    s[(idx >= 2015 * 12) & (idx <= 2019 * 12 + 11)] = 0.02
    s[idx >= 2020 * 12] = 99.0
    out = split_summary({"x": s})["x"]
    v = np.tile([0.01, 0.03], 150)
    assert out["dev"][0] == pytest.approx(2.0) and out["dev"][1] == pytest.approx(v.mean() / (v.std(ddof=1) / np.sqrt(300)))
    assert out["val"][0] == pytest.approx(2.0)
    assert split_summary({"x": s}, {"post": (2020, 2020)}, scale=1.0)["x"]["post"][0] == pytest.approx(99.0)
    assert split_summary({"x": s.iloc[:5]})["x"]["dev"] == (None, None)


def test_reading_flags():
    def res(vw, ew, ics, q1u, q5u, g, c):
        mk = lambda x: {"dev": (x, 2.0), "val": (x, 2.0)}
        return {"returns": {"Q5-Q1_vw": mk(vw), "Q5-Q1_ew": mk(ew), "Q1-U_vw": mk(q1u), "Q5-U_vw": mk(q5u)},
                "tercile_ic": {f"ic_T{k}": mk(ics[k - 1]) for k in (1, 2, 3)}, "costs": {"gross_LU": mk(g), "cost_L": mk(c)}}
    assert "SMALLCAP_EW_EFFECT" in reading(res(0.1, 0.5, [.06, .01, .01], -.02, .12, .12, .02), "dev")
    assert "SHORT_LEG_CARRIES" in reading(res(0.4, 0.4, [.02, .02, .02], -.3, .1, .1, .02), "dev")
    assert "COSTS_EXPLAIN_GAP" in reading(res(0.4, 0.4, [.02, .02, .02], -.1, .3, .3, .25), "dev")
    assert reading(res(0.4, 0.45, [.02, .02, .02], -.1, .3, .3, .05), "dev").endswith("flags: NONE")


def test_end_to_end_matches_net_lu_series_and_logs_no_trials(tmp_path, monkeypatch):
    import qap.trials
    from qap.costs import liquidity_inputs
    from qap.evaluate import build_research_signals
    from qap.fundamentals import prepare_annual
    from qap.panel import build_monthly_panel
    from qap.portfolio import market_scaled_aum

    def boom(*a, **k):
        raise AssertionError("log_trial must not be called")
    monkeypatch.setattr(qap.trials, "log_trial", boom)
    monkeypatch.chdir(tmp_path)
    m, d, funda, links = make_world(n=120, years=(2010, 2021))
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    sig = build_research_signals(p, prepare_annual(funda), links)
    inp = liquidity_inputs(p, _agg_for(p))
    aum_t = market_scaled_aum(p, 1e7)
    r = diagnose_signal(sig, p, inp, aum_t, "gpa")
    ref = split_summary({"x": net_lu_series(sig, p, inp, aum_t, "gpa")})["x"]
    for sp in ("dev", "val"):
        assert ref[sp][0] is not None
        assert r["costs"]["net_LU"][sp][0] == pytest.approx(ref[sp][0], abs=1e-12)
        assert r["costs"]["net_LU"][sp][1] == pytest.approx(ref[sp][1], abs=1e-12)
        assert r["costs"]["cost_L"][sp][0] > 0 and r["costs"]["cost_S"][sp][0] > 0
        assert 0 < r["costs"]["turnover_L"][sp][0] < 100
        assert r["returns"]["Q5-Q1_vw"][sp][0] == pytest.approx(r["returns"]["Q5_vw"][sp][0] - r["returns"]["Q1_vw"][sp][0])
    assert r["months"]["dev"] > 30
    lines = format_report([r])
    assert lines[0].startswith("== gpa") and any(l.strip().startswith("reading [dev]") for l in lines)
    assert not (tmp_path / "data" / "trials.jsonl").exists()
