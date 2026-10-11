import numpy as np
import pandas as pd
import pytest

from qap.innovation import (build_innovation_signals, june30_after, kpss_value_36m, patent_tables, rd_capital, rq_table)
from qap.jkp import add_composites, rank_corr_by_month, theme_scores
from qap.links import filter_ccm_links
from qap.panel import to_mi

THEMES = {"A": [("a1", 1), ("a2", -1), ("a3", 1), ("a4", 1)], "B": [("b1", 1), ("b2", 1), ("b3", 1)]}


def test_theme_score_sign_rank_and_availability():
    df = pd.DataFrame({"mi": [1] * 5 + [2] * 5, "a1": list(range(5)) * 2, "a2": list(range(5)) * 2,
                       "a3": [0, 1, 2, 3, np.nan] * 2, "a4": np.nan,
                       "b1": np.nan, "b2": 0.0, "b3": 1.0})
    s = theme_scores(df, THEMES)                       # A needs max(3, 2) = 3 members; has a1, a2, a3 present (a4 all NaN)
    # a1 (+) and a2 (-) cancel exactly; a3 adds a positive rank tilt -> row 4 has only a1,a2 -> too few available members
    assert s["A"].iloc[:4].gt(s["A"].iloc[0] - 1e-9).all()
    assert np.isnan(s["A"].iloc[4])                    # only 2 of 4 members available (< 3) -> missing
    assert s["B"].isna().all()                         # b1 all NaN: only 2 members present, needs 3
    flipped = theme_scores(df.assign(a1=-df["a1"]), THEMES)["A"]
    assert not np.allclose(flipped.iloc[:4], s["A"].iloc[:4])


def test_theme_score_uses_within_month_ranks():
    df = pd.DataFrame({"mi": [1, 1, 1, 2, 2, 2], "b1": [1, 2, 3, 100, 200, 300.0], "b2": [1, 2, 3, 100, 200, 300.0],
                       "b3": [1, 2, 3, 100, 200, 300.0]})
    s = theme_scores(df, {"B": THEMES["B"]})["B"]
    assert s.iloc[:3].tolist() == pytest.approx(s.iloc[3:].tolist())      # same ranks in both months regardless of level


def test_composites_need_enough_themes_and_exclude_slow_ones():
    n = 6
    names = ["Accruals", "Debt Issuance", "Investment", "Low Leverage", "Low Risk", "Momentum", "Profit Growth", "Profitability",
             "Quality", "Seasonality", "Size", "Short-Term Reversal", "Value"]
    rng = np.random.default_rng(0)
    sc = pd.DataFrame(rng.normal(size=(n, 13)), columns=names)
    mi = pd.Series([1] * n)
    out = add_composites(sc, mi)
    assert out["jkp_all13"].notna().all() and out["jkp_slow10"].notna().all()
    sc2 = sc.copy()
    sc2.loc[0, names[:4]] = np.nan                     # 9 of 13 themes available -> all13 missing (needs 10); slow10 has 7 of 10
    out2 = add_composites(sc2, mi)
    assert np.isnan(out2["jkp_all13"].iloc[0]) and np.isnan(out2["jkp_slow10"].iloc[0]) is np.True_
    r = add_composites(sc.assign(Size=1000.0, Seasonality=-1000.0), mi)       # excluded themes cannot move slow10
    assert r["jkp_slow10"].tolist() == pytest.approx(out["jkp_slow10"].tolist())


def test_rank_corr_by_month_perfect_and_inverse():
    d = pd.DataFrame({"mi": np.repeat([1, 2], 60), "x": np.tile(np.arange(60.0), 2)})
    d["y"] = d["x"] ** 3
    assert rank_corr_by_month(d, "x", "y").tolist() == pytest.approx([1.0, 1.0])
    d["z"] = -d["x"]
    assert rank_corr_by_month(d, "x", "z").tolist() == pytest.approx([-1.0, -1.0])


def test_rd_capital_weights_and_observation_rule():
    a = pd.DataFrame({"gvkey": "000001", "fyear": [2001, 2002, 2003, 2004, 2005], "xrd": 10.0})
    r = rd_capital(a).set_index("fyear")["rd_capital"]
    assert r.loc[2005] == pytest.approx(10 * (1 + .8 + .6 + .4 + .2))
    assert r.loc[2003] == pytest.approx(10 * (1 + .8 + .6)) and np.isnan(r.loc[2002])      # 2 observed years -> missing
    b = pd.DataFrame({"gvkey": "000001", "fyear": [2001, 2003, 2005], "xrd": [10.0, 20.0, 30.0]})
    assert rd_capital(b).set_index("fyear")["rd_capital"].loc[2005] == pytest.approx(30 * 1 + 20 * .6 + 10 * .2)  # gaps count as 0 (3 seen)


def test_patent_tables_timing_and_zero_patent_rule():
    rd = pd.DataFrame({"gvkey": ["000001"] * 3, "fyear": [2000, 2001, 2002], "rd_capital": [10.0, 10.0, 10.0]})
    grants = pd.DataFrame({"gvkey": ["000001", "000001", "000002"], "grant_year": [2000, 1997, 2002],
                           "n_patents": [5, 4, 1], "cites3y_sum": [10.0, 12.0, 1.0], "n_cite3y": [5, 4, 1]})
    flow, qual = patent_tables(grants, rd)
    f = flow.set_index("avail")["pat_flow_rd"]
    assert f.loc[pd.Timestamp("2001-06-30")] == pytest.approx(0.5)             # 5 patents / 10, available June 30 2001
    assert f.loc[pd.Timestamp("2002-06-30")] == 0.0                            # R&D firm with no grants that year -> 0
    assert pd.Timestamp("2003-06-30") not in f.index                           # 2002 is the last grant year: not fully covered
    q = qual.set_index(["gvkey", "avail"])["pat_quality3"]
    assert q.loc[("000001", pd.Timestamp("2001-06-30"))] == pytest.approx(3.0)  # cohort 1997 (12/4), complete end-2000, avail 2001-06-30
    assert q.loc[("000001", pd.Timestamp("2004-06-30"))] == pytest.approx(2.0)  # cohort 2000 (10/5)


def test_kpss_trailing_36_months_by_hand():
    kp = pd.DataFrame({"permno": [1.0, 1.0, 2.0], "issue_date": pd.to_datetime(["1999-01-15", "2000-06-10", "2000-01-05"]),
                       "xi_nominal": [2.0, 3.0, 7.0]})
    keys = pd.DataFrame({"permno": [1, 1, 1, 1, 2, 3], "mi": [to_mi(pd.Series([d])).iloc[0] for d in
                         ["1998-12-31", "2001-12-31", "2002-01-31", "2003-07-31", "2000-12-31", "2000-12-31"]]})
    v = kpss_value_36m(keys, kp)
    assert np.isnan(v.iloc[0])                       # before the first patent
    assert v.iloc[1] == pytest.approx(5.0)           # window 1999-01..2001-12 holds both (1999-01 is month t-35)
    assert v.iloc[2] == pytest.approx(3.0)           # 1999-01 patent dropped one month later
    assert v.iloc[3] == pytest.approx(0.0)           # 2000-06 patent drops out after 36 months; firm has patented before -> 0
    assert v.iloc[4] == pytest.approx(7.0) and np.isnan(v.iloc[5])       # permno 3 never patented -> missing
    capped = kpss_value_36m(keys, kp, end_mi=to_mi(pd.Series(["2002-01-31"])).iloc[0])
    assert np.isnan(capped.iloc[3]) and capped.iloc[2] == pytest.approx(3.0)


def test_innovation_signals_are_point_in_time_and_end_to_end():
    links = filter_ccm_links(pd.DataFrame({"gvkey": ["000001"], "lpermno": [10.0], "linktype": ["LC"], "linkprim": ["P"],
                                           "linkdt": ["1990-01-01"], "linkenddt": [None]}))
    months = pd.date_range("2000-01-31", "2003-12-31", freq="ME")
    keys = pd.DataFrame({"permno": 10, "mthcaldt": months, "me_co": 100.0})
    keys["mi"] = to_mi(keys["mthcaldt"]).values
    annual = pd.DataFrame({"gvkey": "000001", "fyear": range(1996, 2004), "xrd": 10.0})
    rq = pd.DataFrame({"gvkey": ["000001"], "fyear": [2000], "rawrq": [0.2]})
    grants = pd.DataFrame({"gvkey": ["000001", "000001", "000001"], "grant_year": [1997, 2000, 2003],
                           "n_patents": [4, 5, 1], "cites3y_sum": [12.0, 10.0, 1.0], "n_cite3y": [4, 5, 1]})
    kpss = pd.DataFrame({"permno": [10.0], "issue_date": pd.to_datetime(["2000-06-10"]), "xi_nominal": [3.0]})
    s = build_innovation_signals(keys, annual, links, rq, grants, kpss).set_index("mthcaldt")
    assert np.isnan(s.loc["2001-05-31", "rq"]) and s.loc["2001-06-30", "rq"] == pytest.approx(0.2)   # fyear 2000 -> June 30 2001
    assert s.loc["2001-06-30", "pat_flow_rd"] == pytest.approx(5 / 30.0)                                  # 5 patents / R&D capital 30 (10 x (1+.8+.6+.4+.2))
    assert s.loc["2001-06-30", "pat_quality3"] == pytest.approx(3.0)
    assert s.loc["2000-06-30", "kpss_xi_me"] == pytest.approx(3.0 / 100.0) and np.isnan(s.loc["2000-05-31", "kpss_xi_me"])
    assert june30_after(pd.Series([1999, 2000])).tolist() == [pd.Timestamp("2000-06-30"), pd.Timestamp("2001-06-30")]
    assert rq_table(rq).loc[0, "avail"] == pd.Timestamp("2001-06-30")
