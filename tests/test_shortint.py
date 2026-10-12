import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from qap.extract import QUERIES, short_interest_sql
from qap.links import filter_ccm_links
from qap.panel import to_mi
from qap.shortint import (NAMES, STALE_DAYS, attach_shortint, break_ties, build_shortint_signals, holdout_improvement, link_records, monthly_ic,
                          shortint_checks, sir_dtc, veto_floor, veto_mask)

from test_costs import _agg_for
from test_pipeline import make_world

D = pd.Timestamp


def _links(rows):
    """rows: (gvkey, permno, iid/liid)."""
    return filter_ccm_links(pd.DataFrame({"gvkey": [r[0] for r in rows], "lpermno": [float(r[1]) for r in rows], "liid": [r[2] for r in rows],
                                          "linktype": "LC", "linkprim": "P", "linkdt": "1990-01-01", "linkenddt": None}))


def _si(rows):
    """rows: (gvkey, iid, datadate, shortint[, shortintadj])."""
    df = pd.DataFrame(rows, columns=["gvkey", "iid", "datadate", "shortint", "shortintadj"][:len(rows[0])])
    df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def _keys(permnos, dates):
    k = pd.DataFrame([(p, D(d)) for p in permnos for d in dates], columns=["permno", "mthcaldt"])
    k["mi"] = to_mi(k["mthcaldt"]).values
    return k


def _at(frame, permno, date, col="shortint"):
    return frame.loc[(frame["permno"] == permno) & (frame["mthcaldt"] == D(date)), col].iloc[0]


# ------------------------------------------------------------------------------------------------------------------ SQL
def test_short_interest_sql_columns_filter_and_order():
    sql = QUERIES["short_interest"]()
    assert sql == short_interest_sql()
    assert sql == ("select gvkey, iid, datadate, shortint, shortintadj from comp.sec_shortint where datadate >= '2006-01-01' "
                   "order by gvkey, iid, datadate")


def test_short_interest_sql_runs():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("create schema comp")
    con.execute("create table comp.sec_shortint (gvkey varchar, iid varchar, shortint double, shortintadj double, datadate date, splitadjdate date)")
    con.execute("insert into comp.sec_shortint values ('1','01',5,5,date '2005-12-30',null),('1','01',7,7,date '2006-01-13',null),('0','01',1,1,date '2007-01-01',null)")
    out = con.execute(QUERIES["short_interest"]()).df()
    assert list(out.columns) == ["gvkey", "iid", "datadate", "shortint", "shortintadj"] and out["gvkey"].tolist() == ["0", "1"]


# --------------------------------------------------------------------------------------------------- (a) PIT boundary
def test_record_is_invisible_before_datadate_plus_14_days_and_visible_at_the_first_month_end_on_or_after():
    links = _links([("000001", 10, "01")])
    dates = ["2012-02-29", "2012-03-30", "2012-04-30"]
    # datadate 2012-03-16 -> available 2012-03-30 (== month-end: visible); 2012-03-17 -> 2012-03-31 (after the 03-30 month-end)
    for dd, first_visible in (("2012-03-16", "2012-03-30"), ("2012-03-17", "2012-04-30"), ("2012-03-15", "2012-03-30")):
        out, _ = attach_shortint(_keys([10], dates), links, _si([("000001", "01", dd, 500.0)]))
        vis = [d for d in dates if np.isfinite(_at(out, 10, d))]
        assert vis[0] == first_visible and vis == dates[dates.index(first_visible):], dd
    out, _ = attach_shortint(_keys([10], dates), links, _si([("000001", "01", "2012-03-16", 500.0)]))
    assert out["shortint"].isna().tolist() == [True, False, False] and len(out) == 3


def test_latest_available_record_wins_and_unavailable_newer_record_is_ignored():
    links = _links([("000001", 10, "01")])
    si = _si([("000001", "01", "2012-02-29", 100.0), ("000001", "01", "2012-03-15", 200.0), ("000001", "01", "2012-03-30", 300.0)])
    out, _ = attach_shortint(_keys([10], ["2012-03-30", "2012-04-30"]), links, si)
    assert out["shortint"].tolist() == [200.0, 300.0]                    # 03-30 record is available 04-13, not yet at the 03-30 month-end


# --------------------------------------------------------------------------------------------------------- (b) dual class
def test_dual_class_each_permno_gets_its_own_short_interest_and_unlinked_iid_is_dropped_and_counted():
    links = _links([("000001", 10, "01"), ("000001", 11, "02")])           # one gvkey, two share classes
    si = _si([("000001", "01", "2012-03-15", 111.0), ("000001", "02", "2012-03-15", 222.0), ("000001", "03", "2012-03-15", 999.0),
              ("000009", "01", "2012-03-15", 5.0)])
    out, stats = attach_shortint(_keys([10, 11], ["2012-04-30"]), links, si)
    assert _at(out, 10, "2012-04-30") == 111.0 and _at(out, 11, "2012-04-30") == 222.0
    assert stats["records"] == 4 and stats["linked"] == 2 and stats["dropped_no_link"] == 2          # iid '03' and the unknown gvkey
    rec, _ = link_records(si, links)
    assert 999.0 not in rec["shortint"].tolist()


def test_liid_and_iid_compare_as_stripped_strings_and_links_are_date_bounded():
    links = filter_ccm_links(pd.DataFrame({"gvkey": ["000001", "000001"], "lpermno": [10.0, 20.0], "liid": ["01 ", "01"], "linktype": "LC",
                                           "linkprim": "P", "linkdt": ["1990-01-01", "2013-01-01"], "linkenddt": ["2012-12-31", None]}))
    si = _si([("000001", " 01", "2012-03-15", 1.0), ("000001", "01", "2013-03-15", 2.0)])
    rec, stats = link_records(si, links)
    assert sorted(zip(rec["permno"], rec["shortint"])) == [(10, 1.0), (20, 2.0)] and stats["dropped_no_link"] == 0
    with pytest.raises(KeyError):
        link_records(si, links.drop(columns="liid"))


# ------------------------------------------------------------------------------------------------------- (c) staleness
def test_stale_record_is_missing_and_missing_is_never_zero():
    links = _links([("000001", 10, "01")])
    si = _si([("000001", "01", "2012-01-02", 100.0)])                     # available 2012-01-16
    avail = D("2012-01-16")
    k = _keys([10], [avail + pd.Timedelta(days=d) for d in (STALE_DAYS - 1, STALE_DAYS, STALE_DAYS + 1)])
    out, _ = attach_shortint(k, links, si)
    assert out["shortint"].isna().tolist() == [False, False, True]        # exactly 62 days after availability is still used; 63 is missing
    assert not (out["shortint"] == 0).any()
    out2, _ = attach_shortint(_keys([10, 12], ["2012-01-31"]), links, si)  # permno 12 has no link: missing, not zero
    assert out2["shortint"].isna().tolist() == [False, True]
    assert np.isnan(sir_dtc([np.nan], [1000.0], [100.0])[0][0])


def test_nan_latest_record_does_not_hide_an_older_valid_one_and_nan_is_not_zero():
    links = _links([("000001", 10, "01")])
    si = _si([("000001", "01", "2012-03-01", 100.0), ("000001", "01", "2012-03-15", np.nan)])
    out, _ = attach_shortint(_keys([10], ["2012-03-30", "2012-05-31"]), links, si)
    assert out["shortint"].iloc[0] == 100.0 and np.isnan(out["shortint"].iloc[1])      # older valid record used while fresh; then stale -> NaN


# ----------------------------------------------------------------------------------------------- (d) SIR, DTC, units
def _panel_for(permnos, mis, shrout, mthvol):
    return pd.DataFrame([(p, m, s, v) for p in permnos for m, s, v in zip(mis, shrout, mthvol)], columns=["permno", "mi", "shrout", "mthvol"])


def _base(permnos, dates, gpa):
    b = _keys(permnos, dates)
    b["gpa"] = gpa
    b["me"] = 100.0
    return b


def test_sir_uses_formation_month_shrout_and_never_shortintadj_and_dtc_uses_formation_month_volume():
    links = _links([("000001", 10, "01")])
    si = _si([("000001", "01", "2012-03-01", 1000.0, 7e9)])               # shortintadj is absurd on purpose
    dates = ["2012-03-30", "2012-04-30"]
    base = _base([10], dates, 0.5)
    mi = base["mi"].tolist()
    panel = _panel_for([10], mi, [2000.0, 5000.0], [630.0, 6300.0])       # shrout (thousands) changes between the two months
    elig = base[["permno", "mi"]]
    sig, _ = build_shortint_signals(elig, links, si, base, panel)
    assert sig["sir"].tolist() == pytest.approx([1000.0 / 2_000_000.0, 1000.0 / 5_000_000.0])
    assert sig["sir_neg"].tolist() == pytest.approx([-1000.0 / 2_000_000.0, -1000.0 / 5_000_000.0])
    assert sig["dtc"].tolist() == pytest.approx([1000.0 / (630.0 / 21), 1000.0 / (6300.0 / 21)])
    sir, dtc = sir_dtc([1.0, 1.0, 1.0], [0.0, -5.0, np.nan], [0.0, 10.0, np.nan])
    assert np.isnan(sir).all() and np.isnan(dtc[[0, 2]]).all() and dtc[1] == pytest.approx(2.1)       # never inf


# ---------------------------------------------------------------------------------------------------------- (e) gpa_veto
def _veto_world(n=20, with_sir=15):
    permnos = list(range(1, n + 1))
    dates = ["2012-04-30"]
    links = _links([(f"{p:06d}", p, "01") for p in permnos])
    si = _si([(f"{p:06d}", "01", "2012-04-02", 100.0 * p) for p in permnos[:with_sir]])   # SIR rises with permno (equal shrout)
    base = _base(permnos, dates, np.linspace(0.1, 0.9, n))
    panel = _panel_for(permnos, base["mi"].unique().tolist(), [1000.0], [100.0])
    return permnos, links, si, base, panel


def test_gpa_veto_sets_exactly_the_top_sir_decile_to_nan_and_does_not_veto_names_without_sir():
    permnos, links, si, base, panel = _veto_world()
    sig, _ = build_shortint_signals(base[["permno", "mi"]], links, si, base, panel)
    # 15 names have a SIR: percentile ranks 14/15 = 0.93 and 15/15 exceed 0.9 -> the two highest (permnos 14, 15)
    assert sig["sir"].notna().sum() == 15
    assert sig.loc[sig["vetoed"], "permno"].tolist() == [14, 15]
    assert sig.loc[sig["gpa_veto"].isna(), "permno"].tolist() == [14, 15]
    assert sig.loc[sig["permno"] > 15, "gpa_veto"].notna().all()                     # names without a SIR keep their gpa
    ok = ~sig["permno"].isin([14, 15])
    assert (sig.loc[ok, "gpa_veto"] == sig.loc[ok, "gpa"]).all()
    # 20 names with a SIR -> exactly 2 of 20 are the top decile
    _, links, si, base, panel = _veto_world(with_sir=20)
    sig, _ = build_shortint_signals(base[["permno", "mi"]], links, si, base, panel)
    assert sorted(sig.loc[sig["vetoed"], "permno"]) == [19, 20] and sig["gpa_veto"].isna().sum() == 2


def test_veto_decile_is_formed_among_eligible_names_only_but_applies_to_the_whole_frame():
    permnos, links, si, base, panel = _veto_world(with_sir=20)
    elig = base.loc[base["permno"] <= 10, ["permno", "mi"]]                          # only permnos 1..10 are eligible
    sig, _ = build_shortint_signals(elig, links, si, base, panel)
    assert sig.loc[sig["vetoed"], "permno"].tolist() == [10]                          # 10 eligible names with a SIR -> the single highest
    m = veto_mask(pd.Series([1.0, 2.0, np.nan, 9.0]), pd.Series([1, 1, 1, 1]), pd.Series([True, True, True, False]))
    assert m.tolist() == [False, True, False, False]


def test_zero_vetoes_gpa_veto_equals_gpa_row_for_row_and_veto_is_applied_before_ranking():
    _, links, _, base, panel = _veto_world()
    sig0, _ = build_shortint_signals(base[["permno", "mi"]], links, _si([("000099", "01", "2012-04-02", 1.0)]), base, panel)     # nobody has a SIR
    assert not sig0["vetoed"].any() and sig0["sir_neg"].isna().all()
    pd.testing.assert_series_equal(sig0["gpa_veto"], sig0["gpa"], check_names=False)
    # ranked after the veto: the remaining 18 names fill the percentile scale
    _, links, si, base, panel = _veto_world(with_sir=20)
    s, _ = build_shortint_signals(base[["permno", "mi"]], links, si, base, panel)
    r = break_ties(s)["gpa_veto"].rank(pct=True).dropna()
    assert len(r) == 18 and r.max() == 1.0


def test_break_ties_is_study_e_local_and_keeps_missing():
    permnos, links, si, base, panel = _veto_world(with_sir=20)
    s, _ = build_shortint_signals(base[["permno", "mi"]], links, si, base, panel)
    b = break_ties(s)
    assert b["gpa_veto"].isna().sum() == s["gpa_veto"].isna().sum() and set(NAMES) <= set(b.columns)
    assert ((b["sir_neg"] - s["sir_neg"]).abs() < 1e-8).all()


# -------------------------------------------------------------------------------------------------- (f) duplicates, checks
def test_duplicate_gvkey_iid_datadate_assertion_fires():
    links = _links([("000001", 10, "01")])
    si = _si([("000001", "01", "2012-03-15", 1.0), ("000001", "01", "2012-03-15", 2.0)])
    with pytest.raises(AssertionError, match="1 duplicate"):
        attach_shortint(_keys([10], ["2012-04-30"]), links, si)
    permnos, links, si1, base, panel = _veto_world()
    sig, stats = build_shortint_signals(base[["permno", "mi"]], links, si1, base, panel)
    with pytest.raises(AssertionError, match="duplicate"):
        shortint_checks(pd.concat([si1, si1.iloc[:1]]), stats, sig)


def test_checks_lines_cover_every_pre_registered_item():
    permnos, links, si, base, panel = _veto_world(with_sir=20)
    si = si.assign(shortintadj=si["shortint"] * 2)
    sig, stats = build_shortint_signals(base[["permno", "mi"]], links, si, base, panel)
    txt = "\n".join(shortint_checks(si, stats, sig))
    for s in ("records per year: 2012=20", "cadence: median days between records", "duplicate (gvkey, iid, datadate) records: 0", "[PASS] zero duplicate",
              "iid->liid security link hit rate: 20 of 20", "share of eligible names with a SIR by year: 2012=100%", "SIR (sanity only): median",
              "p99", "share > 1", "DTC (days, sanity only): median", "share of formation rows vetoed", "median shortint", "median shortintadj"):
        assert s in txt, s


def test_veto_floor_requires_dev_improvement_of_five_bp_and_positive_validation():
    dev = np.arange(2010 * 12, 2010 * 12 + 60)
    val = np.arange(2015 * 12, 2015 * 12 + 24)
    idx = np.r_[dev, val]
    g = pd.Series(0.01, index=idx)
    noise = pd.Series(np.cos(np.arange(len(idx)) * 1.7) * 0.0001, index=idx)

    def floor(dev_imp, val_imp):
        imp = pd.Series(np.r_[np.full(60, dev_imp), np.full(24, val_imp)], index=idx) + noise
        return veto_floor(g + imp, g)

    assert floor(0.0006, 0.0001)["PASS"] and not floor(0.00045, 0.0001)["PASS"] and not floor(0.0006, -0.0001)["PASS"]
    f = floor(0.0006, 0.0001)
    assert f["dev"]["months"] == 60 and f["val"]["months"] == 24 and f["dev"]["mean"] == pytest.approx(0.0006, abs=2e-5) and f["dev"]["t"] > 2
    h = holdout_improvement(pd.Series(0.02, index=[2020 * 12 + i for i in range(5)] + [5]), pd.Series(0.01, index=[2020 * 12 + i for i in range(5)] + [5]), 2020 * 12)
    assert h["months"] == 5 and h["mean"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------------------------------------- end to end
def _load_script():
    spec = importlib.util.spec_from_file_location("run_shortint", pathlib.Path(__file__).parent.parent / "scripts" / "run_shortint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world():
    from qap.costs import liquidity_inputs
    from qap.panel import build_monthly_panel
    from qap.portfolio import market_scaled_aum
    rng = np.random.default_rng(11)
    m, d, funda, links = make_world(n=90, years=(1998, 2022))
    m["shrout"] = m["mthcap"] / 20.0                                             # price 20, thousands of shares
    m["mthvol"] = m["shrout"] * 1000.0 * rng.uniform(0.005, 0.05, len(m))
    p = build_monthly_panel(m, d)
    assert "mthvol" in p.columns
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    links = links.assign(liid="01")
    gvs = sorted(funda["gvkey"].unique())
    rows = []
    for g in gvs:
        sh = float(p.loc[p["permno"] == int(g), "shrout"].median()) * 1000.0
        f = rng.uniform(0.005, 0.15)
        for dd in pd.date_range("2006-01-15", "2022-12-31", freq="SMS-15") if hasattr(pd.offsets, "SemiMonthBegin") else []:
            rows.append((g, "01", dd, float(sh * f * rng.uniform(0.7, 1.3)), 1.0))
    si = pd.DataFrame(rows, columns=["gvkey", "iid", "datadate", "shortint", "shortintadj"])
    return p, links, funda, liquidity_inputs(p, _agg_for(p)), market_scaled_aum(p, 1e7), si


@pytest.fixture(scope="module")
def world():
    return _world()


def _patch(mod, monkeypatch, world, tmp_path):
    p, links, funda, inp, aum, si = world
    (tmp_path / "data" / "raw").mkdir(parents=True)
    si.to_parquet(tmp_path / "data" / "raw" / "short_interest.parquet", index=False)
    monkeypatch.setattr(mod, "require_inputs", lambda *a, **k: None)
    monkeypatch.setattr(mod, "load_context", lambda: (p, links, funda, inp, aum))


def _run_in(tmp_path, monkeypatch, world, jobs, open_holdout=False, promoted=None):
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, world, tmp_path)
    if promoted is not None:
        monkeypatch.setattr(mod, "run_promotion_report", lambda *a, **k: ({}, promoted))
    lines = []
    mod._run(lines.append, open_holdout, jobs)
    return lines


def test_script_end_to_end_checks_first_three_trials_floor_info_block_and_jobs_identical(tmp_path, monkeypatch, world):
    a, b = tmp_path / "j1", tmp_path / "j3"
    a.mkdir(), b.mkdir()
    l1 = _run_in(a, monkeypatch, world, 1)
    l3 = _run_in(b, monkeypatch, world, 3)
    assert l1 == l3
    txt = "\n".join(l1)
    assert txt.index("== CHECKS") < txt.index("== RESULTS (Study E)") < txt.index("== SIZE FLOOR") < txt.index("== INFORMATION ONLY")
    assert "[PASS] every signal produced values" in txt and "iid->liid security link hit rate" in txt and "improvement over gpa:" in txt
    assert "HOLDOUT OPENED" not in txt and "top-SIR decile minus universe, gross VW" in txt and "rank IC by year" in txt and "IC by size" in txt
    for n in NAMES:
        assert f"{n}: " in txt
    trials = [json.loads(x) for x in (a / "data" / "trials.jsonl").read_text().splitlines()]
    assert sorted(t["name"] for t in trials) == sorted(f"studyE:{n}" for n in NAMES) and {t["spec"]["prereg"] for t in trials} == {"v12"}
    assert json.loads((a / "data" / "study_e_promoted.json").read_text()) == json.loads((b / "data" / "study_e_promoted.json").read_text())
    assert (a / "data" / "study_e_promoted.json").exists()


def test_script_removes_gpa_veto_when_the_size_floor_fails_and_logs_it(tmp_path, monkeypatch, world):
    import qap.study
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, world, tmp_path)
    monkeypatch.setattr(mod, "run_promotion_report", lambda *a, **k: ({}, ["sir_neg", "gpa_veto"]))
    monkeypatch.setattr(mod, "veto_floor", lambda *a, **k: {"dev": {"mean": 0.0001, "t": 3.0, "months": 100}, "val": {"mean": 0.001, "t": 1.0, "months": 60}, "PASS": False})
    lines = []
    mod._run(lines.append, False, 1)
    txt = "\n".join(lines)
    assert "FAILS the size floor: REMOVED from the promoted list" in txt and "[FAIL] size floor" in txt
    assert json.loads((tmp_path / "data" / "study_e_promoted.json").read_text()) == ["sir_neg"]
    d = tmp_path / "ok"
    d.mkdir()
    monkeypatch.chdir(d)
    (d / "data" / "raw").mkdir(parents=True)
    _patch_raw = world[5]
    _patch_raw.to_parquet(d / "data" / "raw" / "short_interest.parquet", index=False)
    monkeypatch.setattr(mod, "veto_floor", lambda *a, **k: {"dev": {"mean": 0.001, "t": 3.0, "months": 100}, "val": {"mean": 0.001, "t": 1.0, "months": 60}, "PASS": True})
    lines = []
    mod._run(lines.append, True, 1)
    assert json.loads((d / "data" / "study_e_promoted.json").read_text()) == ["sir_neg", "gpa_veto"]
    t2 = "\n".join(lines)
    assert "HOLDOUT OPENED" in t2 and "  sir_neg: {'months'" in t2 and "gpa_veto holdout improvement over gpa" in t2 and "dtc_neg: {" not in t2


def test_panel_now_carries_mthvol_when_present_and_not_otherwise():
    from qap.panel import build_monthly_panel
    m, d, _, _ = make_world(n=5, years=(2000, 2001))
    assert "mthvol" not in build_monthly_panel(m, d).columns
    assert "mthvol" in build_monthly_panel(m.assign(mthvol=1.0), d).columns


def test_monthly_ic_helper_is_signed_and_bounded(world):
    p = world[0]
    e = p.loc[p["eq"] & (p["me_co"] > 0), ["permno", "mi", "me", "me_co", "primaryexch"]].copy()
    e["x"] = np.random.default_rng(0).normal(size=len(e))
    ic = monthly_ic(e, p, "x", 2010 * 12)
    assert len(ic) > 100 and ic.index.max() < 2010 * 12 and ic.abs().max() < 1
