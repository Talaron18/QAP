import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from qap.earnings import (NAMES, TIE_EPS, announcement_days, break_ties, build_earnings_signals, build_events, crsp_daily_ret_sql,
                          earnings_checks, ff_daily_columns, ff_daily_sql, ff_decimals, ic_by_bucket, link_events, load_daily,
                          prepare_daily, prepare_quarters, quarterly_events, window_excess)
from qap.links import filter_ccm_links
from qap.panel import to_mi

from test_costs import _agg_for
from test_pipeline import make_world

D = pd.Timestamp
CAL = pd.bdate_range("2010-01-04", "2010-12-31")              # no holidays: every weekday trades
MONTH_ENDS = pd.date_range("2010-01-31", "2010-12-31", freq="ME")


# ------------------------------------------------------------------------------------------------------- fixtures
def _r(p, i):
    return 0.001 * (i % 7) + 0.0005 * p                       # deterministic stock return of permno p on calendar day i


def _mkt(i):
    return 0.0004 * (i % 5)


RF = 0.00001


def _daily(permnos=(1, 2), drop=(), change=None):
    rows = []
    for p in permnos:
        for i, d in enumerate(CAL):
            if (p, d) in drop:
                continue
            r = _r(p, i)
            if change and (p, d) in change:
                r = change[(p, d)]
            rows.append((p, d, r))
    return pd.DataFrame(rows, columns=["permno", "dlycaldt", "dlyret"])


def _ff(pct=False):
    f = pd.DataFrame({"date": CAL, "mktrf": [_mkt(i) for i in range(len(CAL))], "rf": RF})
    if pct:
        f.loc[3, "mktrf"] = 0.05                            # keep the decimal series in range ...
        f[["mktrf", "rf"]] = f[["mktrf", "rf"]] * 100.0     # ... then deliver it in percent (max > 1.5 is not needed: 5.0 > 1.5)
    return f


def _expected_ear(p, days):
    idx = [CAL.get_loc(D(d)) for d in days]
    return sum(_r(p, i) - _mkt(i) - RF for i in idx)


def _qrows(gvkey, ibq, saleq=None, start_year=2018, lag=40, rdq_missing=()):
    rows = []
    for k, x in enumerate(ibq):
        fy, fq = start_year + k // 4, k % 4 + 1
        dd = D(f"{fy}-{3 * fq:02d}-01") + pd.offsets.MonthEnd(0)
        rows.append((gvkey, dd, fy, fq, pd.NaT if k in rdq_missing else dd + pd.Timedelta(days=lag), x,
                     (saleq[k] if saleq is not None else 10.0 * x)))
    return pd.DataFrame(rows, columns=["gvkey", "datadate", "fyearq", "fqtr", "rdq", "ibq", "saleq"])


def _links(rows):
    return filter_ccm_links(pd.DataFrame(rows, columns=["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]))


LINK1 = _links([("000001", 1, "LC", "P", "1990-01-01", None)])


def _event_q(rdq, gvkey="000001", fy=2009, fq=4, ibq=1.0):
    """A single quarter that has a computable ear but no sue (no history)."""
    return pd.DataFrame({"gvkey": [gvkey], "datadate": [D("2009-12-31")], "fyearq": [fy], "fqtr": [fq], "rdq": [D(rdq)], "ibq": [ibq], "saleq": [ibq]})


def _keys(permnos=(1,)):
    k = pd.concat([pd.DataFrame({"permno": p, "mthcaldt": MONTH_ENDS, "me_co": 100.0}) for p in permnos], ignore_index=True)
    k["mi"] = to_mi(k["mthcaldt"]).values
    return k


# ------------------------------------------------------------------------------------------------------------ SUE
def test_seasonal_change_and_sue_are_exact_with_the_min_six_rule():
    x = [1.0, 2.0, 1.5, 3.0, 2.0, 3.5, 1.0, 4.0, 2.5, 3.0, 2.0, 5.5, 3.0, 4.0]
    q = _qrows("1", x).sample(frac=1, random_state=3)                    # row order must not matter; gvkey is zero-padded internally
    p = prepare_quarters(q).set_index("qidx")
    assert p["gvkey"].iloc[0] == "000001"
    base = p.index.min()
    chg = {k: x[k] - x[k - 4] for k in range(4, 14)}
    for k in range(14):
        q_ = base + k
        w = [chg[j] for j in range(k - 7, k + 1) if j in chg]
        if k in chg and len(w) >= 6:
            assert p.loc[q_, "sue"] == pytest.approx(chg[k] / np.std(w, ddof=1), rel=1e-12), k
        else:
            assert np.isnan(p.loc[q_, "sue"]), k
    assert p.loc[base + 9, "sue"] == pytest.approx(chg[9] / np.std([chg[j] for j in range(4, 10)], ddof=1))       # exactly 6 changes (incl. current)
    assert np.isnan(p.loc[base + 8, "sue"])                                                                          # 5 changes: not enough
    assert p.loc[base + 13, "sue"] == pytest.approx(chg[13] / np.std([chg[j] for j in range(6, 14)], ddof=1))      # window slides: 8 changes
    assert p.loc[base + 4, "dibq"] == pytest.approx(1.0) and np.isnan(p.loc[base + 3, "dibq"])                       # needs the quarter a year earlier


def test_rsue_uses_sales_and_a_constant_history_gives_missing_not_inf():
    x = list(range(1, 15))
    s = [3.0 * v ** 2 for v in x]
    p = prepare_quarters(_qrows("1", x, saleq=s)).set_index("qidx")
    k = p.index.max()
    cs = {j: s[j] - s[j - 4] for j in range(4, 14)}
    assert p.loc[k, "rsue"] == pytest.approx(cs[13] / np.std([cs[j] for j in range(6, 14)], ddof=1))
    assert np.isnan(p.loc[k, "sue"]) and np.isfinite(p["sue"].fillna(0)).all() and np.isfinite(p["rsue"].fillna(0)).all()     # ibq changes are all 4 -> sd 0 -> missing


def test_missing_quarter_breaks_the_comparison_and_duplicates_keep_the_latest_datadate():
    x = [1.0, 2.0, 1.5, 3.0, 2.0, 3.5, 1.0, 4.0, 2.5, 3.0, 2.0, 5.5]
    q = _qrows("1", x)
    gone = q.drop(index=7)                                                           # quarter 8 absent: no change at 8 and none at 12 (no q-4)
    p = prepare_quarters(gone).set_index("qidx")
    base = p.index.min()
    assert (base + 7) not in p.index and np.isnan(p.loc[base + 11, "dibq"]) and np.isnan(p.loc[base + 11, "sue"])
    dup = pd.concat([q, q.iloc[[5]].assign(datadate=q.iloc[5]["datadate"] + pd.Timedelta(days=5), ibq=99.0)], ignore_index=True)
    d = prepare_quarters(dup).set_index("qidx")
    assert len(d) == 12 and d.loc[base + 5, "ibq"] == 99.0                           # latest datadate wins


def test_rows_without_rdq_are_not_events_but_still_feed_the_history():
    x = [1.0, 2.0, 1.5, 3.0, 2.0, 3.5, 1.0, 4.0, 2.5, 3.0, 2.0, 5.5]
    full = prepare_quarters(_qrows("1", x)).set_index("qidx")
    part = _qrows("1", x, rdq_missing=(5, 11))
    ev = quarterly_events(part).set_index("qidx")
    base = full.index.min()
    assert (base + 5) not in ev.index and (base + 11) not in ev.index and len(ev) == 10
    assert ev.loc[base + 9, "sue"] == pytest.approx(full.loc[base + 9, "sue"])       # its change at q5 is still one of the 6 used by q9


# ------------------------------------------------------------------------------------------------------------ EAR
def test_ear_is_the_exact_three_day_excess_return_and_weekend_rdq_moves_day_zero_to_monday():
    prep = prepare_daily(_daily(), _ff())
    d0, d1 = announcement_days(pd.Series([D("2010-01-13"), D("2010-01-23"), D("2010-01-24")]), prep["cal"])
    assert d0.tolist() == [CAL.get_loc(D("2010-01-13")), CAL.get_loc(D("2010-01-25")), CAL.get_loc(D("2010-01-25"))]
    assert pd.to_datetime(d1).tolist() == [D("2010-01-14"), D("2010-01-26"), D("2010-01-26")]
    ear = window_excess(np.array([1, 2, 1]), d0, prep)
    assert ear[0] == pytest.approx(_expected_ear(1, ["2010-01-12", "2010-01-13", "2010-01-14"]), abs=1e-12)
    assert ear[1] == pytest.approx(_expected_ear(2, ["2010-01-22", "2010-01-25", "2010-01-26"]), abs=1e-12)        # Saturday rdq -> Monday
    assert ear[2] == pytest.approx(_expected_ear(1, ["2010-01-22", "2010-01-25", "2010-01-26"]), abs=1e-12)


def test_ear_requires_all_three_days_and_ff_in_percent_gives_the_same_answer():
    ev = _event_q("2010-02-10")
    for gone in (D("2010-02-09"), D("2010-02-10"), D("2010-02-11")):
        daily = _daily(drop={(1, gone)})                                     # permno 2 still trades that day, so the calendar keeps it
        e, _ = build_events(ev, LINK1, daily, _ff())
        assert np.isnan(e["ear"].iloc[0]), gone
    ok, _ = build_events(ev, LINK1, _daily(), _ff())
    assert ok["ear"].iloc[0] == pytest.approx(_expected_ear(1, ["2010-02-09", "2010-02-10", "2010-02-11"]), abs=1e-12)
    pct, _ = build_events(ev, LINK1, _daily(), _ff(pct=True))
    f = ff_decimals(_ff(pct=True))
    assert f["mktrf"].abs().max() < 1.5 and f["rf"].iloc[0] == pytest.approx(RF)
    # the percent file has one altered day (index 3) that is outside this window
    assert pct["ear"].iloc[0] == pytest.approx(ok["ear"].iloc[0], abs=1e-12)
    nan_ff = _ff()
    nan_ff.loc[CAL.get_loc(D("2010-02-10")), "mktrf"] = np.nan
    assert np.isnan(build_events(ev, LINK1, _daily(), nan_ff)[0]["ear"].iloc[0])
    nan_ret = build_events(ev, LINK1, _daily(change={(1, D("2010-02-11")): np.nan}), _ff())[0]
    assert np.isnan(nan_ret["ear"].iloc[0])


def test_events_at_the_calendar_edges_get_no_ear_but_keep_an_availability_date():
    e, f = build_events(pd.concat([_event_q("2010-01-04"), _event_q("2010-12-31", fq=3)], ignore_index=True)
                        .assign(datadate=[D("2009-12-31"), D("2010-09-30")]), LINK1, _daily(), _ff())
    assert e["ear"].isna().all()                                             # day -1 is before the file starts / day +1 after it ends
    assert e["avail"].notna().all() and not f["complete_window"].any()
    assert e["avail"].tolist() == [D("2010-01-05"), D("2011-01-03")]        # first: calendar day +1; last: second weekday on/after rdq (fallback)


def test_no_lookahead_returns_after_day_plus_one_are_irrelevant_returns_inside_the_window_matter():
    ev = _event_q("2010-03-10")
    base, _ = build_events(ev, LINK1, _daily(), _ff())
    sig0 = build_earnings_signals(_keys(), base)
    for day, same in (("2010-03-12", True), ("2010-03-31", True), ("2010-04-20", True), ("2010-03-08", True),
                      ("2010-03-09", False), ("2010-03-10", False), ("2010-03-11", False)):
        e, _ = build_events(ev, LINK1, _daily(change={(1, D(day)): 0.5}), _ff())
        assert (e["ear"].iloc[0] == base["ear"].iloc[0]) == same, day
        if same:
            pd.testing.assert_frame_equal(build_earnings_signals(_keys(), e), sig0)       # nothing observable changes either


# ---------------------------------------------------------------------------------------------------------- linking
def test_link_is_date_bounded_at_rdq_and_prefers_primary_links_and_serves_every_share_class():
    links = _links([("000001", 1, "LC", "P", "1990-01-01", "2010-06-30"), ("000001", 2, "LC", "P", "2010-07-01", None),
                    ("000001", 5, "LU", "C", "1990-01-01", None), ("000003", 5, "LC", "P", "1990-01-01", None),
                    ("000009", 9, "NR", "P", "1990-01-01", None)])
    ev = pd.DataFrame({"gvkey": ["000001", "000001", "000009"], "rdq": [D("2010-03-10"), D("2010-08-10"), D("2010-03-10")],
                       "ibq": 1.0}).assign(sue=1.0)
    out = link_events(ev, links)
    got = sorted(zip(out["rdq"].dt.strftime("%Y-%m-%d"), out["permno"]))
    assert got == [("2010-03-10", 1), ("2010-03-10", 5), ("2010-08-10", 2), ("2010-08-10", 5)]       # linktype NR never links; the C-link share class gets it too
    ev2 = pd.concat([ev.iloc[[0]], ev.iloc[[0]].assign(gvkey="000003")], ignore_index=True)
    two = link_events(ev2, links)                                                                      # permno 5 reached by gvkey 1 (C) and 3 (P) on the same rdq
    assert two[two["permno"] == 5]["gvkey"].tolist() == ["000003"]
    assert two["permno"].tolist().count(5) == 1


def test_two_quarters_reported_on_one_date_keep_the_later_quarter_and_both_count_as_linked():
    x = [1.0, 2.0, 1.5, 3.0, 2.0, 3.5, 1.0, 4.0, 2.5, 3.0, 2.0, 5.5]
    q = _qrows("1", x, start_year=2009)
    q.loc[[10, 11], "rdq"] = D("2010-03-10")                                        # a late filer: fiscal quarters 11 and 12 announced together
    e, f = build_events(q, LINK1, _daily(), _ff())
    kept = e[e["rdq"] == D("2010-03-10")]
    full = prepare_quarters(q).set_index("qidx")
    assert len(kept) == 1 and kept["qidx"].iloc[0] == full.index.max()
    assert kept["sue"].iloc[0] == pytest.approx(full.loc[full.index.max(), "sue"])
    assert f["linked"].all()                                                         # the losing quarter was linked too


def test_events_follow_the_link_in_force_on_rdq():
    links = _links([("000001", 1, "LC", "P", "1990-01-01", "2010-06-30"), ("000001", 2, "LC", "P", "2010-07-01", None)])
    q = pd.concat([_event_q("2010-03-10"), _event_q("2010-08-10", fq=1, fy=2010)], ignore_index=True).assign(datadate=[D("2009-12-31"), D("2010-06-30")])
    e, f = build_events(q, links, _daily(), _ff())
    assert e.sort_values("rdq")["permno"].tolist() == [1, 2]
    assert f["linked"].all() and f["complete_window"].all()
    none, f2 = build_events(q, _links([("000077", 1, "LC", "P", "1990-01-01", None)]), _daily(), _ff())
    assert none.empty and not f2["linked"].any()


# --------------------------------------------------------------------------------------------------------- signals
def _one_event_signals(rdq, ibq_ear_nan=False):
    e, _ = build_events(_event_q(rdq), LINK1, _daily(), _ff())
    return e


def test_availability_is_the_first_month_end_on_or_after_the_calendar_date_of_day_plus_one():
    for rdq, first in (("2010-04-28", "2010-04-30"), ("2010-04-29", "2010-04-30"),            # day +1 = Apr 29 / Apr 30: known at the April month-end
                       ("2010-04-30", "2010-05-31"), ("2010-04-26", "2010-04-30")):           # day +1 = May 3: only the May month-end
        e = _one_event_signals(rdq)
        s = build_earnings_signals(_keys(), e).set_index("mthcaldt")["ear"]
        assert s.dropna().index.min() == D(first), rdq
    e = _one_event_signals("2010-04-29")
    assert e["avail"].iloc[0] == D("2010-04-30")


def test_staleness_is_counted_from_rdq_and_exactly_130_days_still_counts():
    e = _one_event_signals("2010-01-13")
    k = pd.DataFrame({"permno": 1, "mthcaldt": [D("2010-05-23"), D("2010-05-24"), D("2010-04-30")]})
    k["mi"] = to_mi(k["mthcaldt"]).values
    s = build_earnings_signals(k, e)
    assert (D("2010-05-23") - D("2010-01-13")).days == 130
    assert s["ear"].notna().tolist() == [True, False, True]
    assert s.loc[s["ear"].isna(), ["sue", "rsue", "ear_sue", "msa", "msa_bucket"]].isna().all().all()
    assert build_earnings_signals(k, e, max_age_days=100)["ear"].notna().tolist() == [False, False, False]


def test_latest_event_hides_an_older_valid_one_even_if_its_own_value_is_missing():
    q = pd.concat([_event_q("2010-02-10"), _event_q("2010-03-10", fq=1, fy=2010)], ignore_index=True).assign(datadate=[D("2009-12-31"), D("2010-03-31")])
    e, _ = build_events(q, LINK1, _daily(drop={(1, D("2010-03-11"))}), _ff())          # second event: incomplete window
    s = build_earnings_signals(_keys(), e).set_index("mthcaldt")["ear"]
    assert s.loc["2010-02-28"] == pytest.approx(_expected_ear(1, ["2010-02-09", "2010-02-10", "2010-02-11"]), abs=1e-12)
    assert np.isnan(s.loc["2010-03-31"]) and np.isnan(s.loc["2010-04-30"])


def test_months_since_announcement_buckets_and_ear_sue_needs_both():
    e = _one_event_signals("2010-01-13")
    s = build_earnings_signals(_keys(), e).set_index("mthcaldt")
    assert s["msa"].loc["2010-01-31":"2010-04-30"].tolist() == [0, 1, 2, 3] and np.isnan(s["msa"].loc["2010-05-31"])     # May: 138 days, stale
    assert s["msa_bucket"].loc["2010-01-31":"2010-04-30"].tolist() == [0, 0, 2, 3]       # 0-1, 2, 3 or more
    assert s["ear_sue"].isna().all()                                                      # no history -> sue missing -> ear_sue missing


def _signal_world():
    """Three permnos, one announcement each in the same month with chosen ear / sue (set directly on an events frame)."""
    ev = pd.DataFrame({"permno": [1, 2, 3, 4], "gvkey": "x", "rdq": D("2010-02-10"), "datadate": D("2009-12-31"), "qidx": 1, "day0": 0,
                       "avail": D("2010-02-11"), "avail_mi": to_mi(pd.Series([D("2010-02-11")])).iloc[0],
                       "ear": [0.10, 0.20, 0.30, np.nan], "sue": [3.0, 1.0, 2.0, 5.0], "rsue": [np.nan, 1.0, 2.0, 3.0]})
    return ev


def test_ear_sue_is_the_mean_of_within_month_percentile_ranks_and_needs_both():
    s = build_earnings_signals(_keys((1, 2, 3, 4)), _signal_world())
    m = s[s["mthcaldt"] == D("2010-02-28")].set_index("permno")
    r_e = {1: 1 / 3, 2: 2 / 3, 3: 1.0}                                  # ranks among the three names with an ear
    r_s = {1: 3 / 4, 2: 1 / 4, 3: 2 / 4, 4: 1.0}                          # ... and among the four with a sue
    for p in (1, 2, 3):
        assert m.loc[p, "ear_sue"] == pytest.approx((r_e[p] + r_s[p]) / 2)
    assert np.isnan(m.loc[4, "ear_sue"]) and m.loc[4, "sue"] == 5.0 and np.isnan(m.loc[4, "ear"])
    assert m.loc[1, "rsue"] != m.loc[1, "rsue"]                         # rsue missing is carried as missing
    assert s[s["mthcaldt"] == D("2010-01-31")][NAMES].isna().all().all()


def test_tie_break_is_deterministic_keeps_order_and_makes_ties_distinct():
    s = build_earnings_signals(_keys((1, 2, 3, 4)), _signal_world().assign(sue=[0.0, 0.0, 0.0, -1.0]))
    a, b = break_ties(s), break_ties(s)
    pd.testing.assert_frame_equal(a, b)
    m = a[a["mthcaldt"] == D("2010-02-28")].set_index("permno")["sue"]
    assert m.loc[4] < m.loc[[1, 2, 3]].min() and m.loc[[1, 2, 3]].nunique() == 3
    assert (a["sue"] - s["sue"]).dropna().abs().max() < TIE_EPS * 1.0001
    assert (a["ear"] - s["ear"]).dropna().gt(0).all()


# ------------------------------------------------------------------------------------------------ dtypes and checks
def test_nullable_dtypes_give_identical_events():
    x = [1.0, 2.0, 1.5, 3.0, 2.0, 3.5, 1.0, 4.0, 2.5, 3.0]
    q = _qrows("1", x)
    base, _ = build_events(q, LINK1, _daily().assign(dlycaldt=lambda d: d["dlycaldt"]), _ff())
    qn = q.astype({"fyearq": "Int64", "fqtr": "Int64", "ibq": "Float64", "saleq": "Float64", "gvkey": "string"})
    dn = _daily().astype({"permno": "Int32", "dlyret": "Float32"})
    fn = _ff().astype({"mktrf": "Float64", "rf": "Float64"})
    nullable, _ = build_events(qn, LINK1, dn, fn)
    assert len(base) == len(nullable) == len(x)
    pd.testing.assert_frame_equal(base.drop(columns="ear"), nullable.drop(columns="ear"))
    assert nullable["ear"].isna().equals(base["ear"].isna())
    assert np.allclose(nullable["ear"].dropna(), base["ear"].dropna(), atol=1e-6)         # float32 returns


def test_checks_lines_cover_the_pre_registered_items():
    x = [1.0, 2.0, 1.5, 3.0, 2.0, 3.5, 1.0, 4.0, 2.5, 3.0]
    q = _qrows("1", x, start_year=2009, rdq_missing=(2,))
    e, f = build_events(q, LINK1, _daily(), _ff())
    s = build_earnings_signals(_keys(), e)
    txt = "\n".join(earnings_checks(prepare_quarters(q), f, s, {2010: len(_daily())}, len(CAL)))
    for needle in ("daily returns file: 520 rows", "daily FF file: 260 rows", "share with rdq", "2010s=", "merge hit rate to permno",
                   "complete 3-day window", "SUE available", "signal availability"):
        assert needle in txt, needle
    assert "events (rdq present): 9" in txt and all(n in txt for n in NAMES)


def test_sql_builders_run_in_duckdb_and_fail_clearly_on_unknown_ff_columns():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("create schema crsp")
    con.execute("create table crsp.dsf_v2 (permno integer, dlycaldt date, dlyret double, dlyprc double, dlyvol double)")
    con.execute("insert into crsp.dsf_v2 values (1, date '2010-12-31', 0.01, 5, 5), (1, date '2011-01-03', 0.02, 5, 5), "
                "(2, date '2010-06-01', null, 5, 5), (3, date '2010-06-01', -0.5, 5, 5)")
    got = con.execute(crsp_daily_ret_sql(2010)).df()
    assert list(got.columns) == ["permno", "dlycaldt", "dlyret"] and got["permno"].tolist() == [1, 3]       # year bounds and null returns respected
    con.execute("create schema ff")
    con.execute("create table ff.factors_daily (date date, mktrf double, smb double, hml double, rf double, umd double)")
    con.execute("insert into ff.factors_daily values (date '2010-01-05', 0.5, 0, 0, 0.001, 0), (date '2010-01-04', -0.2, 0, 0, 0.001, 0)")
    cols = ff_daily_columns(["date", "mktrf", "smb", "hml", "rf", "umd"])
    df = con.execute(ff_daily_sql(cols)).df()
    assert list(df.columns) == ["date", "mktrf", "rf"] and df["mktrf"].tolist() == [-0.2, 0.5]
    assert ff_daily_columns(["Date", "Mkt-RF", "RF"]) == {"date": "Date", "mktrf": "Mkt-RF", "rf": "RF"}
    with pytest.raises(KeyError) as ei:
        ff_daily_columns(["dateff", "excess_mkt", "riskfree"])
    assert "excess_mkt" in str(ei.value) and "available columns" in str(ei.value)


# ------------------------------------------------------------------------------------------ extraction script (stub db)
def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).parent.parent / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StubDb:
    def __init__(self, con):
        self.con, self.sql = con, []

    def raw_sql(self, sql, date_cols=None):
        self.sql.append(sql)
        return self.con.execute(sql).df()

    def describe_table(self, library, table):
        return self.con.execute(f"select column_name as name from information_schema.columns where table_schema = '{library}' "
                                f"and table_name = '{table}'").df()


def test_extraction_is_per_year_resumable_and_atomic(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    mod = _load_script("extract_earnings")
    con = duckdb.connect()
    con.execute("create schema crsp")
    con.execute("create table crsp.dsf_v2 (permno bigint, dlycaldt date, dlyret decimal(10,6))")
    con.execute("insert into crsp.dsf_v2 values (10, date '2001-03-01', 0.0125), (11, date '2001-03-01', -0.5), (10, date '2002-03-01', 0.01), "
                "(10, date '2002-03-04', null)")
    con.execute("create schema ff")
    con.execute("create table ff.factors_daily (date date, mktrf double, rf double, smb double)")
    con.execute("insert into ff.factors_daily values (date '2001-03-01', 1.5, 0.01, 0), (date '2001-03-02', -2.0, 0.01, 0)")
    db = _StubDb(con)
    mod.extract_crsp_daily_ret(db, tmp_path, 2001, 2003)
    d = tmp_path / "crsp_daily_ret"
    assert sorted(p.name for p in d.iterdir()) == ["2001.parquet", "2002.parquet", "2003.parquet"]          # no temp files left
    y1 = pd.read_parquet(d / "2001.parquet")
    assert str(y1["permno"].dtype) == "int32" and str(y1["dlyret"].dtype) == "float32" and str(y1["dlycaldt"].dtype).startswith("datetime64")
    assert len(y1) == 2 and len(pd.read_parquet(d / "2002.parquet")) == 1 and len(pd.read_parquet(d / "2003.parquet")) == 0
    n = len(db.sql)
    (d / "2002.parquet").unlink()                                                                           # resume: only the missing year is fetched
    mod.extract_crsp_daily_ret(db, tmp_path, 2001, 2003)
    assert len(db.sql) == n + 1 and "'2002-01-01'" in db.sql[-1]
    mod.extract_ff_daily(db, tmp_path)
    ff = pd.read_parquet(tmp_path / "ff_daily.parquet")
    assert list(ff.columns) == ["date", "mktrf", "rf"] and len(ff) == 2
    daily, rows = load_daily(d)
    assert rows == {2001: 2, 2002: 1, 2003: 0} and len(daily) == 3
    con.execute("drop table ff.factors_daily")
    con.execute("create table ff.factors_daily (dt date, excess double, rf double)")
    with pytest.raises(SystemExit) as ei:
        mod.extract_ff_daily(db, tmp_path)
    assert "available columns" in str(ei.value) and "excess" in str(ei.value)


# ---------------------------------------------------------------------------------------------------- end to end
def _world():
    from qap.costs import liquidity_inputs
    from qap.panel import build_monthly_panel
    from qap.portfolio import market_scaled_aum
    m, d, funda, links = make_world(n=90, years=(1998, 2022))
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    return p, links, funda, liquidity_inputs(p, _agg_for(p)), market_scaled_aum(p, 1e7)


def _raw(world, tmp_path):
    p, links, funda, inp, aum = world
    rng = np.random.default_rng(5)
    raw = tmp_path / "data" / "raw"
    (raw / "crsp_daily_ret").mkdir(parents=True)
    cal = pd.bdate_range("2000-01-03", "2022-12-30")
    permnos = np.arange(1, 91)
    mk = rng.normal(0.0003, 0.01, len(cal))
    for y in range(2000, 2023):
        c = cal[cal.year == y]
        r = rng.normal(0.0, 0.015, (len(permnos), len(c))) + mk[cal.year == y]
        df = pd.DataFrame({"permno": np.repeat(permnos, len(c)).astype("int32"), "dlycaldt": np.tile(c.values, len(permnos)),
                           "dlyret": r.ravel().astype("float32")})
        df.to_parquet(raw / "crsp_daily_ret" / f"{y}.parquet", index=False)
    pd.DataFrame({"date": cal, "mktrf": mk * 100, "rf": 0.01}).to_parquet(raw / "ff_daily.parquet", index=False)      # percent units
    rows = []
    for g in sorted(funda["gvkey"].unique()):
        for k in range(24 * 4):
            fy, fq = 1999 + k // 4, k % 4 + 1
            dd = D(f"{fy}-{3 * fq:02d}-01") + pd.offsets.MonthEnd(0)
            rows.append((g, dd, fy, fq, dd + pd.Timedelta(days=int(rng.integers(20, 60))) if rng.random() > 0.05 else pd.NaT,
                         float(rng.normal(1, 1)), float(rng.normal(50, 10))))
    pd.DataFrame(rows, columns=["gvkey", "datadate", "fyearq", "fqtr", "rdq", "ibq", "saleq"]).to_parquet(raw / "compustat_quarterly.parquet", index=False)


@pytest.fixture(scope="module")
def world():
    return _world()


def _script():
    return _load_script("run_earnings")


def _patch(mod, monkeypatch, world, tmp_path):
    p, links, funda, inp, aum = world
    _raw(world, tmp_path)
    monkeypatch.setattr(mod, "require_inputs", lambda *a, **k: None)
    monkeypatch.setattr(mod, "load_context", lambda: (p, links, funda, inp, aum))


def _run_in(tmp_path, monkeypatch, world, jobs, open_holdout=False, promoted=None):
    mod = _script()
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, world, tmp_path)
    if promoted is not None:
        monkeypatch.setattr(mod, "run_promotion_report", lambda *a, **k: ({}, promoted))
    lines = []
    mod._run(lines.append, open_holdout, jobs)
    return lines


def test_script_end_to_end_checks_first_exactly_four_trials_info_block_and_jobs_identical(tmp_path, monkeypatch, world):
    a, b = tmp_path / "j1", tmp_path / "j3"
    a.mkdir(), b.mkdir()
    l1 = _run_in(a, monkeypatch, world, 1)
    l3 = _run_in(b, monkeypatch, world, 3)
    assert l1 == l3
    txt = "\n".join(l1)
    assert txt.index("== CHECKS") < txt.index("== RESULTS (Study D)") < txt.index("== INFORMATION ONLY")
    assert "[PASS] every signal produced values" in txt and "daily returns file:" in txt and "merge hit rate to permno" in txt
    assert "HOLDOUT OPENED" not in txt and "PROMOTED (provisional; sealed holdout opens once for these): " in txt
    assert "IC by months since announcement" in txt and "IC by size" in txt and "EW Q5-Q1" in txt
    for n in NAMES:
        assert f"{n}: " in txt
    trials = [json.loads(x) for x in (a / "data" / "trials.jsonl").read_text().splitlines()]
    assert sorted(t["name"] for t in trials) == sorted(f"studyD:{n}" for n in NAMES) and len(trials) == 4
    assert {t["spec"]["prereg"] for t in trials} == {"v11"}
    assert json.loads((a / "data" / "study_d_promoted.json").read_text()) == json.loads((b / "data" / "study_d_promoted.json").read_text())


def test_script_holdout_opens_only_for_promoted_signals(tmp_path, monkeypatch, world):
    none = _run_in(tmp_path, monkeypatch, world, 1, open_holdout=True, promoted=[])
    assert not any(f"  {n}: {{'months'" in l for l in none for n in NAMES)
    d = tmp_path / "p"
    d.mkdir()
    some = "\n".join(_run_in(d, monkeypatch, world, 1, open_holdout=True, promoted=["sue"]))
    assert "HOLDOUT OPENED (once) for promoted signals only" in some and "  sue: {'months'" in some and "  ear: {" not in some
    assert json.loads((d / "data" / "study_d_promoted.json").read_text()) == ["sue"]


def test_script_applies_60_month_minimum_only_inside_the_run(tmp_path, monkeypatch, world):
    import qap.candidates
    import qap.study
    mod = _script()
    monkeypatch.setattr(mod, "MIN_DEV_MONTHS", 10_000)
    lines = []
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, world, tmp_path)
    mod._run(lines.append, False, 1)
    promo = [l for l in lines if l.startswith("  [") and "PROMOTE" in l]
    assert len(promo) == 4 and all(l.startswith("  [FAIL]") and ("fewer than 10000 overlapping dev months" in l or "too few overlapping" in l) for l in promo)
    assert qap.study.promotion_stats is qap.candidates.promotion_stats


def test_main_writes_the_report_file_even_when_the_run_fails(tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_run", lambda log, o, j: (log("== CHECKS"), (_ for _ in ()).throw(RuntimeError("boom"))))
    with pytest.raises(RuntimeError):
        mod.main(False, 1)
    assert (tmp_path / "data" / "study_d_report.txt").read_text().startswith("== CHECKS")


def test_ic_by_bucket_returns_months_before_the_cut_only(world):
    p, links, funda, inp, aum = world
    from qap.study import eligible_universe
    k = eligible_universe(p)
    rng = np.random.default_rng(1)
    sig = k.assign(msa_bucket=rng.choice([0.0, 2.0, 3.0, np.nan], len(k)), x=rng.normal(size=len(k)))
    ic = ic_by_bucket(sig, p, "x", 2010 * 12)
    assert list(ic.columns) == ["ic_m01", "ic_m2", "ic_m3p"] and ic.index.max() < 2010 * 12 and ic.notna().all().all() and ic.abs().max().max() < 1
