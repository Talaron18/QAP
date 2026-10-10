import numpy as np
import pandas as pd
import pytest

from qap.evaluate import build_research_signals, evaluate_signal
from qap.fundamentals import attach_pit, book_equity, prepare_annual
from qap.links import filter_ccm_links
from qap.panel import (assert_month_grid, attach_gvkey, build_monthly_panel, calibrate_me_scale,
                       lag_one_month, to_mi)
from qap.replicate import (expand_holding, form_portfolios, portfolio_returns, smb_hml, vw_market_return)


def make_world(n=240, years=(1990, 1999), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-31", f"{years[1]}-12-31", freq="ME")
    permnos = np.arange(1, n + 1)
    permco = permnos.copy()
    permco[1] = permco[0]  # permnos 1 and 2 share a company (two share classes)
    rows = []
    for p, pc in zip(permnos, permco):
        cap = 1000.0 * np.exp(rng.normal(5, 1)) * np.exp(np.cumsum(rng.normal(0, 0.05, len(dates))))
        ret = rng.normal(0.01, 0.08, len(dates))
        for d, c, r in zip(dates, cap, ret):
            rows.append((p, pc, d, r, c))
    m = pd.DataFrame(rows, columns=["permno", "permco", "mthcaldt", "mthret", "mthcap"])
    m["mthdelflg"] = "N"
    m["primaryexch"] = np.where(m.permno % 2 == 0, "N", "Q")
    for c, v in dict(securitytype="EQTY", securitysubtype="COM", sharetype="NS", usincflg="Y",
                     issuertype="CORP", conditionaltype="RW", tradingstatusflg="A").items():
        m[c] = v
    delists = pd.DataFrame({"permno": [], "delistingdt": [], "delret": [], "delretmisstype": [], "delreasontype": []})
    f = []
    for p in permnos:
        for y in range(years[0], years[1] + 1):
            dd = pd.Timestamp(f"{y}-12-31")
            cap = m[(m.permno == p) & (m.mthcaldt == dd)].mthcap.iloc[0] / 1000.0  # $ millions
            f.append(dict(gvkey=f"{p:06d}", datadate=dd, fyear=y, fyr=12, at=cap * 2, lt=cap, seq=cap * rng.uniform(.3, 1.2),
                          ceq=np.nan, pstk=0.0, pstkrv=np.nan, pstkl=np.nan, txditc=0.0, revt=cap * 3, cogs=cap * 2,
                          csho=1.0, prcc_f=cap))
    funda = pd.DataFrame(f)
    links = pd.DataFrame({"gvkey": [f"{p:06d}" for p in permnos], "lpermno": permnos.astype(float), "linktype": "LC",
                          "linkprim": "P", "linkdt": "1980-01-01", "linkenddt": None})
    return m, delists, funda, filter_ccm_links(links)


def test_book_equity_branches():
    f = pd.DataFrame({
        "seq": [100, np.nan, np.nan, 100, 100, -5],
        "ceq": [np.nan, 80, np.nan, np.nan, np.nan, np.nan],
        "pstk": [0, 10, 0, 0, 7, 0],
        "pstkrv": [np.nan, np.nan, np.nan, 20, 30, np.nan],
        "pstkl": [np.nan, np.nan, np.nan, 25, 40, np.nan],
        "at": [np.nan, np.nan, 500, np.nan, np.nan, np.nan],
        "lt": [np.nan, np.nan, 300, np.nan, np.nan, np.nan],
        "txditc": [5, np.nan, np.nan, 0, 0, 0],
    })
    be = book_equity(f)
    assert be.iloc[0] == 105                      # seq + txditc - 0
    assert be.iloc[1] == 80                       # seq = ceq + pstk = 90, txditc missing -> 0, ps = pstk 10
    assert be.iloc[2] == 200                      # at - lt
    assert be.iloc[3] == 80                       # ps = pstkrv 20
    assert be.iloc[4] == 70                       # ps = pstkrv 30 (not pstkl)
    assert np.isnan(be.iloc[5])                   # non-positive dropped


def test_month_grid_asserts():
    m, d, *_ = make_world(n=5)
    p = build_monthly_panel(m, d)
    assert_month_grid(p, "1990-01", "1999-12")
    with pytest.raises(AssertionError, match="unique months"):
        assert_month_grid(p, "1990-01", "2000-12")
    dup = pd.concat([p, p.iloc[:1]])
    with pytest.raises(AssertionError, match="duplicate"):
        assert_month_grid(dup, "1990-01", "1999-12")


def test_company_me_sums_share_classes():
    m, d, *_ = make_world(n=5)
    p = build_monthly_panel(m, d)
    x = p[p.mi == p.mi.min()].set_index("permno")
    assert x.loc[1, "me_co"] == pytest.approx(x.loc[1, "me"] + x.loc[2, "me"])
    assert x.loc[1, "me_co"] == x.loc[2, "me_co"]
    assert x.loc[3, "me_co"] == x.loc[3, "me"]


def test_unit_calibration_finds_thousands():
    m, d, funda, links = make_world(n=120)
    p = build_monthly_panel(m, d)
    scale, ratio = calibrate_me_scale(p, funda, links)
    assert scale == 1000.0 and ratio == pytest.approx(1000.0, rel=0.01)


def test_attach_gvkey_tie_break_and_uniqueness():
    df = pd.DataFrame({"permno": [1], "mi": [to_mi(pd.Series(["2000-06-30"])).iloc[0]], "mthcaldt": pd.to_datetime(["2000-06-30"])})
    links = pd.DataFrame({"gvkey": ["000001", "000002", "000003"], "permno": [1, 1, 1], "linkprim": ["C", "P", "P"],
                          "linkdt": pd.to_datetime(["1990-01-01", "1995-01-01", "1999-01-01"]),
                          "linkenddt": pd.to_datetime(["2020-01-01"] * 3)})
    out = attach_gvkey(df, links)
    assert out["gvkey"].tolist() == ["000003"]     # P over C, then latest linkdt


def test_holding_window_is_july_to_next_june():
    a = pd.DataFrame({"permno": [1], "t": [1995], "port": ["SL"]})
    h = expand_holding(a)
    months = sorted(h["mi"] - (1995 * 12))
    assert months[0] == 6 and months[-1] == 17     # July 1995 (index 6) .. June 1996 (index 17)
    assert len(h) == 12


def test_june_timing_return_in_june_not_in_that_years_portfolio():
    m, d, funda, links = make_world(n=240)
    p = build_monthly_panel(m, d)
    annual = prepare_annual(funda)
    assign = form_portfolios(p, annual, links, start_year=1992, end_year=1996)
    assert set(assign["t"]) == {1992, 1993, 1994, 1995, 1996}
    # spike returns only in June 1994: portfolios formed June 1994 must not hold June 1994
    p2 = p.copy()
    p2["mthret_adj"] = 0.0
    p2.loc[p2["mi"] == 1994 * 12 + 5, "mthret_adj"] = 0.5
    r = portfolio_returns(p2, assign)
    june94 = r.loc[1994 * 12 + 5]
    assert (june94.dropna() == 0.5).all()          # held by 1993 formation
    only_94 = assign[assign.t == 1994].copy()
    r94 = portfolio_returns(p2, only_94)
    assert (1994 * 12 + 5) not in r94.index        # first held month is July 1994


def test_labels_use_unfiltered_returns_no_survivorship():
    m, d, funda, links = make_world(n=240)
    m.loc[(m.permno == 10) & (m.mthcaldt >= "1995-07-31"), "tradingstatusflg"] = "H"   # leaves the filter
    p = build_monthly_panel(m, d)
    annual = prepare_annual(funda)
    sig = build_research_signals(p, annual, links)
    jun = to_mi(pd.Series(["1995-06-30"])).iloc[0]
    row = sig[(sig.permno == 10) & (sig.mi == jun)]
    assert len(row) == 1
    from qap.evaluate import _with_next_return
    lab = _with_next_return(row, p)
    expected = p[(p.permno == 10) & (p.mi == jun + 1)].mthret_adj.iloc[0]
    assert lab["fwd"].iloc[0] == pytest.approx(expected) and not np.isnan(lab["fwd"].iloc[0])


def test_research_signals_and_evaluation_run():
    m, d, funda, links = make_world(n=240)
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    sig = build_research_signals(p, prepare_annual(funda), links)
    assert sig.attrs["link_hit_rate"] == 1.0
    assert {"bm_timely", "gpa", "neg_ag"} <= set(sig.columns)
    res = evaluate_signal(sig, p, "bm_timely")
    assert res["months"] > 50 and np.isfinite(res["rank_ic_mean"]) and np.isfinite(res["ls_vw_t"])


def test_market_return_uses_previous_month_weights():
    m, d, *_ = make_world(n=30)
    p = build_monthly_panel(m, d)
    mk = vw_market_return(p)
    assert mk.index.min() == p.mi.min() + 1 and len(mk) == p.mi.nunique() - 1


def test_momentum_skips_formation_month_and_uses_11_months():
    from qap.replicate import momentum_signal
    m, d, *_ = make_world(n=3)
    p = build_monthly_panel(m, d)
    p["mthret_adj"] = 0.0
    base = to_mi(pd.Series(["1995-12-31"])).iloc[0]
    p.loc[p.mi == base, "mthret_adj"] = 0.9            # month f itself: must NOT enter mom at f
    p.loc[p.mi == base - 1, "mthret_adj"] = 0.10       # f-1: enters
    p.loc[p.mi == base - 11, "mthret_adj"] = 0.10      # f-11: enters
    p.loc[p.mi == base - 12, "mthret_adj"] = 0.90      # f-12: must NOT enter
    mom = momentum_signal(p).query("permno == 1 and mi == @base")["mom"].iloc[0]
    assert mom == pytest.approx(1.1 * 1.1 - 1)


def test_umd_runs_and_period_summary_shapes():
    from qap.replicate import umd
    m, d, funda, links = make_world(n=240)
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    u = umd(p)
    assert len(u) > 90 and np.isfinite(u).all()
    sig = build_research_signals(p, prepare_annual(funda), links)
    assert set(sig.attrs["link_diag"]) >= {"formation_rows", "with_gvkey", "with_annual_record", "with_positive_be"}
    res = evaluate_signal(sig, p, "gpa")
    assert set(res["by_period"]) == {"pre1990", "dev1990_2014", "valid2015_2019", "pre2020", "post1990"}   # holdout sealed
    opened = evaluate_signal(sig, p, "gpa", include_holdout=True)
    assert {"holdout2020+", "full"} <= set(opened["by_period"])


def test_composite_and_momentum_attach():
    from qap.evaluate import add_composite, add_momentum
    m, d, funda, links = make_world(n=240)
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    sig = add_momentum(add_composite(build_research_signals(p, prepare_annual(funda), links)), p)
    assert sig["composite_vpi"].between(-0.5, 0.5).all() or sig["composite_vpi"].isna().any()
    assert sig["composite_vpi"].dropna().between(-0.5, 0.5).all()
    assert "mom" in sig and sig["mom"].notna().any()
    assert evaluate_signal(sig, p, "composite_vpi")["months"] > 30
