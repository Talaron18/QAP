import decimal
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from qap.altdata import (NAMES, STALE_15M, STALE_18M, altdata_checks, arc_table, blocks_table, build_altdata_signals, cbi_table, cik10,
                         cik_gvkey_map, culture_table, litig_table, risk_table)
from qap.candidates import promotion_stats
from qap.links import filter_ccm_links
from qap.panel import to_mi

from test_costs import _agg_for
from test_pipeline import make_world

MONTHS = pd.date_range("2010-01-31", "2015-12-31", freq="ME")
D = lambda s: pd.Timestamp(s)


def _links():
    return filter_ccm_links(pd.DataFrame({"gvkey": ["000001"], "lpermno": [10.0], "linktype": ["LC"], "linkprim": ["P"],
                                          "linkdt": ["1990-01-01"], "linkenddt": [None]}))


def _keys():
    k = pd.DataFrame({"permno": 10, "mthcaldt": MONTHS, "me_co": 100.0})
    k["mi"] = to_mi(k["mthcaldt"]).values
    return k


def _funda(extra=None):
    f = pd.DataFrame({"gvkey": ["000001"], "cik": ["0000001750"]})
    return pd.concat([f, extra], ignore_index=True) if extra is not None else f


EMPTY = {
    "arc": pd.DataFrame({"cik": pd.Series(dtype="float64"), "fiscal_period_end": pd.Series(dtype="datetime64[ns]"), "arc": pd.Series(dtype="float64"),
                         "form": pd.Series(dtype="object"), "date_filed": pd.Series(dtype="datetime64[ns]")}),
    "risk": pd.DataFrame({"cik": pd.Series(dtype="object"), "fyear": pd.Series(dtype="float64"), "filing_date": pd.Series(dtype="datetime64[ns]"),
                          "item_1a_word_count": pd.Series(dtype="float64")}),
    "blocks": pd.DataFrame({c: pd.Series(dtype="float64") for c in ("company_cik", "blockholder_cik", "year", "position")}),
    "boards": pd.DataFrame({"gvkey": pd.Series(dtype="object"), "datadate": pd.Series(dtype="datetime64[ns]"), "fyear": pd.Series(dtype="float64"),
                            "cbi": pd.Series(dtype="float64")}),
    "culture": pd.DataFrame({c: pd.Series(dtype="float64") for c in ("year", "s_integrity", "s_teamwork", "s_innovation", "s_respect", "s_quality")})
    .assign(gvkey=pd.Series(dtype="object"), source=pd.Series(dtype="object")),
    "lit": pd.DataFrame({"gvkey": pd.Series(dtype="object"), "year": pd.Series(dtype="float64"), "case_p": pd.Series(dtype="float64"),
                         "case_f": pd.Series(dtype="float64")}),
}


def _signals(funda=None, **tables):
    t = {**EMPTY, **tables}
    s, _ = build_altdata_signals(_keys(), _funda() if funda is None else funda, _links(), t["arc"], t["risk"], t["blocks"], t["boards"],
                                 t["culture"], t["lit"])
    return s.set_index("mthcaldt")


def _arc(rows):
    return pd.DataFrame(rows, columns=["cik", "fiscal_period_end", "arc", "form", "date_filed"])


# --------------------------------------------------------------------------------------------------------- keys
def test_cik10_handles_varchar_double_decimal_and_missing():
    s = pd.Series(["0000000020", 1750.0, decimal.Decimal(7), None, np.nan, pd.NA, "x"], dtype=object)
    assert cik10(s).tolist() == ["0000000020", "0000001750", "0000000007", None, None, None, None]
    assert cik10(pd.Series([1750.0, np.nan], dtype="Float64")).tolist() == ["0000001750", None]
    assert cik10(pd.Series([20, None], dtype="Int64")).tolist() == ["0000000020", None]
    assert cik10(pd.Series([], dtype="float64")).tolist() == []


def test_cik_with_two_gvkeys_is_dropped_and_counted():
    f = pd.DataFrame({"gvkey": ["000001", "000001", "000002", "000003", "000004", "000005"],
                      "cik": ["0000001750", "0000001750", "0000000020", "0000000020", "0000000099", None]})
    m, stats = cik_gvkey_map(f)
    assert m.sort_values("cik").values.tolist() == [["0000000099", "000004"], ["0000001750", "000001"]]      # 0000000020 -> 2 gvkeys -> dropped
    assert stats["dropped_multi_gvkey"] == 1 and stats["ciks"] == 3 and stats["gvkeys"] == 2
    num = cik_gvkey_map(pd.DataFrame({"gvkey": ["1", "2"], "cik": [1750.0, np.nan]}))[0]
    assert num.values.tolist() == [["0000001750", "000001"]]


# --------------------------------------------------------------------------------------------------------- arc10k
def test_arc10k_available_from_the_month_end_on_or_after_filing_and_stale_after_15_months():
    arc = _arc([(1750.0, D("2011-12-31"), 100.0, "10-K", D("2012-03-15"))])
    s = _signals(arc=arc)["arc10k"]
    assert np.isnan(s.loc["2012-02-29"]) and s.loc["2012-03-31"] == -100.0         # filed 03-15: not visible in February, visible in March
    assert s.loc["2013-05-31"] == -100.0                                           # age 442 days <= 457
    assert STALE_15M == 457 and (D("2012-03-15") + pd.Timedelta(days=STALE_15M)) == D("2013-06-15")
    assert np.isnan(s.loc["2013-06-30"])                                           # stale
    nxt = _arc([(1750.0, D("2011-12-31"), 100.0, "10-K", D("2012-03-15")), (1750.0, D("2012-12-31"), 50.0, "10-K", D("2013-03-01"))])
    s2 = _signals(arc=nxt)["arc10k"]
    assert s2.loc["2013-02-28"] == -100.0 and s2.loc["2013-03-31"] == -50.0        # latest filed 10-K wins from its own month-end


def test_arc10k_forms_amendments_excluded_and_10k405_kept():
    arc = _arc([(1750.0, D("2009-12-31"), 10.0, "10-K405", D("2010-03-15")), (1750.0, D("2009-12-31"), 999.0, "10-K/A", D("2010-04-10")),
                (1750.0, D("2009-12-31"), 777.0, "10-Q", D("2010-05-10")), (1750.0, D("2009-12-31"), 555.0, "10-k/a", D("2010-05-12")),
                (1750.0, D("2009-12-31"), 333.0, "NT 10-K", D("2010-03-20")), (1750.0, D("2009-12-31"), np.nan, "10-K", D("2010-09-01"))])
    s = _signals(arc=arc)["arc10k"]
    assert s.loc["2010-03-31"] == -10.0 and s.loc["2010-05-31"] == -10.0 and s.loc["2010-09-30"] == -10.0     # amendments, 10-Q, NT, NaN arc ignored
    assert len(arc_table(arc, cik_gvkey_map(_funda())[0])) == 1


def test_arc10k_unmapped_or_ambiguous_cik_gives_nothing():
    arc = _arc([(1750.0, D("2011-12-31"), 100.0, "10-K", D("2012-03-15"))])
    dup = _funda(pd.DataFrame({"gvkey": ["000002"], "cik": ["0000001750"]}))
    assert _signals(funda=dup, arc=arc)["arc10k"].isna().all()
    assert _signals(arc=_arc([(5.0, D("2011-12-31"), 1.0, "10-K", D("2012-03-15"))]))["arc10k"].isna().all()


# ---------------------------------------------------------------------------------------------------------- rf_len
def test_rf_len_log_zero_count_is_missing_and_does_not_hide_older_value():
    rt = pd.DataFrame({"cik": ["0000001750", "0000001750", "0000001750"], "fyear": [2010.0, 2011.0, 2012.0],
                       "filing_date": [D("2011-03-01"), D("2012-03-01"), D("2013-03-01")], "item_1a_word_count": [1000.0, 0.0, 3000.0]})
    s = _signals(risk=rt)["rf_len"]
    assert s.loc["2011-03-31"] == pytest.approx(-np.log1p(1000.0)) and np.isnan(s.loc["2011-02-28"])
    assert s.loc["2012-03-31"] == pytest.approx(-np.log1p(1000.0))                 # the 0-count 2012 filing is dropped, 2011 value persists
    assert D("2011-03-01") + pd.Timedelta(days=STALE_15M) == D("2012-05-31")      # inclusive tolerance
    assert s.loc["2012-05-31"] == pytest.approx(-np.log1p(1000.0)) and np.isnan(s.loc["2012-06-30"])
    assert s.loc["2013-03-31"] == pytest.approx(-np.log1p(3000.0))
    assert len(risk_table(rt, cik_gvkey_map(_funda())[0])) == 2


# ----------------------------------------------------------------------------------------------------------- blocks
def test_blocks_sum_june30_rule_and_zero_fill_only_in_populated_years():
    cmap = pd.DataFrame({"cik": ["0000001750", "0000002000"], "gvkey": ["000001", "000002"]})
    b = pd.DataFrame({"company_cik": [1750.0, 1750.0, 2000.0, 9999.0, 2000.0],
                      "blockholder_cik": [1.0, 2.0, 1.0, 1.0, 3.0],
                      "year": [2010.0, 2010.0, 2011.0, 2011.0, 2012.0], "position": [5.0, 7.0, 4.0, 9.0, 6.0]})
    t = blocks_table(b, cmap, min_rows=3).set_index(["gvkey", "avail"])["blocks"]
    # 2010 has 2 raw rows (<3, unpopulated), 2011 has 2 (<3), 2012 has 1: nothing populated -> only real rows, no zeros
    assert t.to_dict() == {("000001", D("2011-06-30")): 12.0, ("000002", D("2012-06-30")): 4.0, ("000002", D("2013-06-30")): 6.0}
    b2 = pd.concat([b, pd.DataFrame({"company_cik": [2000.0, 2000.0], "blockholder_cik": [4.0, 5.0], "year": [2011.0, 2011.0], "position": [1.0, 1.0]})],
                   ignore_index=True)
    t2 = blocks_table(b2, cmap, min_rows=4).set_index(["gvkey", "avail"])["blocks"]
    assert t2.loc[("000002", D("2012-06-30"))] == 6.0 and t2.loc[("000001", D("2012-06-30"))] == 0.0      # 2011 populated (4 rows): gv1 has none -> 0
    assert ("000001", D("2011-06-30")) in t2.index and t2.loc[("000001", D("2011-06-30"))] == 12.0
    assert ("000001", D("2013-06-30")) not in t2.index                                                    # 2012 unpopulated: absent stays missing
    assert t2.loc[("000002", D("2012-06-30"))] == 6.0 and t2.loc[("000002", D("2013-06-30"))] == 6.0


def test_blocks_signal_timing_and_staleness():
    b = pd.DataFrame({"company_cik": [1750.0, 1750.0], "blockholder_cik": [1.0, 2.0], "year": [2010.0, 2010.0], "position": [5.0, 7.0]})
    s = _signals(blocks=b)["blocks"]
    assert np.isnan(s.loc["2011-05-31"]) and s.loc["2011-06-30"] == 12.0           # June 30 of year+1
    assert D("2011-06-30") + pd.Timedelta(days=STALE_18M) == D("2012-12-30")
    assert s.loc["2012-11-30"] == 12.0 and np.isnan(s.loc["2012-12-31"])


# ---------------------------------------------------------------------------------------------------------- cbi_neg
def test_cbi_neg_six_months_after_datadate_and_stale_after_18():
    bd = pd.DataFrame({"gvkey": ["1"], "datadate": [D("2011-05-31")], "fyear": [2010], "cbi": [1]})
    s = _signals(boards=bd)["cbi_neg"]
    assert np.isnan(s.loc["2011-10-31"]) and s.loc["2011-11-30"] == -1.0
    assert s.loc["2013-04-30"] == -1.0 and np.isnan(s.loc["2013-06-30"])           # avail 2011-11-30 + 549d = 2013-06-01
    assert cbi_table(bd)["avail"].iloc[0] == D("2011-11-30")


# ---------------------------------------------------------------------------------------------------------- culture
def test_culture_zscore_within_year_after_averaging_sources():
    sc = ["s_integrity", "s_teamwork", "s_innovation", "s_respect", "s_quality"]
    rows = [("000001", "StreetEvents", 2010, 0.0), ("000001", "Other", 2010, 2.0), ("000002", "StreetEvents", 2010, 2.0),
            ("000003", "StreetEvents", 2010, 3.0),
            ("000001", "StreetEvents", 2011, 10.0), ("000002", "StreetEvents", 2011, 20.0), ("000003", "StreetEvents", 2011, 30.0),
            ("000009", "StreetEvents", 2012, 5.0)]
    c = pd.DataFrame(rows, columns=["gvkey", "source", "year", "s_integrity"])
    for k in sc[1:]:
        c[k] = c["s_integrity"]
    t = culture_table(c).set_index(["gvkey", "avail"])["culture"]
    assert t.loc[("000001", D("2011-06-30"))] == pytest.approx(-1.0)               # (0+2)/2 = 1 -> z = (1-2)/1
    assert t.loc[("000002", D("2011-06-30"))] == pytest.approx(0.0) and t.loc[("000003", D("2011-06-30"))] == pytest.approx(1.0)
    assert t.loc[("000001", D("2012-06-30"))] == pytest.approx(-1.0) and t.loc[("000003", D("2012-06-30"))] == pytest.approx(1.0)  # level-free
    assert ("000009", D("2013-06-30")) not in t.index                                # single row in its year: std undefined -> missing
    c2 = c.copy()
    c2.loc[c2["gvkey"].eq("000002") & c2["year"].eq(2010), "s_quality"] = np.nan
    assert ("000002", D("2011-06-30")) not in culture_table(c2).set_index(["gvkey", "avail"]).index      # needs all five
    two = pd.DataFrame({"gvkey": ["000001", "000002"], "source": "x", "year": [2010, 2010], "s_integrity": [1.0, 3.0], "s_teamwork": [1.0, 3.0],
                        "s_innovation": [1.0, 3.0], "s_respect": [1.0, 3.0], "s_quality": [1.0, 3.0]})
    sig = _signals(culture=two)["culture"]                                            # permno 10 = gvkey 000001, z = -1/sqrt(2)
    assert np.isnan(sig.loc["2011-05-31"]) and sig.loc["2011-06-30"] == pytest.approx(-1 / np.sqrt(2))


# ------------------------------------------------------------------------------------------------------------- litig
def test_litig_neglog1p_june30_and_no_forward_fill_past_staleness():
    lt = pd.DataFrame({"gvkey": ["000001"], "year": [decimal.Decimal(2012)], "case_p": [decimal.Decimal(1)], "case_f": [decimal.Decimal(3)]})
    s = _signals(lit=lt)["litig"]
    assert np.isnan(s.loc["2013-05-31"]) and s.loc["2013-06-30"] == pytest.approx(-np.log(4.0))
    assert s.loc["2014-12-31"] == pytest.approx(-np.log(4.0)) and np.isnan(s.loc["2015-01-31"])      # 2013-06-30 + 549 days = 2014-12-31: exactly 18 months is kept
    zero = pd.DataFrame({"gvkey": ["000001"], "year": [2012], "case_p": [0], "case_f": [0]})
    assert _signals(lit=zero)["litig"].loc["2013-06-30"] == 0.0                     # an observed zero is a value
    assert _signals(lit=lt.assign(gvkey="000002"))["litig"].isna().all()           # no row for the firm = missing, never zero
    assert litig_table(lt)["litig"].iloc[0] == pytest.approx(-np.log(4.0))


# ------------------------------------------------------------------------------------------------- dtypes / checks
def test_nullable_and_object_dtypes_give_identical_signals():
    arc = _arc([(1750.0, D("2011-12-31"), 100.0, "10-K", D("2012-03-15")), (np.nan, D("2011-12-31"), 5.0, "10-K", D("2012-03-16"))])
    bl = pd.DataFrame({"company_cik": [1750.0], "blockholder_cik": [1.0], "year": [2010.0], "position": [5.0]})
    ca = pd.DataFrame({"gvkey": ["000001"] * 3, "source": "s", "year": [2010, 2010, 2010], "s_integrity": [1.0, 2.0, 3.0],
                       "s_teamwork": 1.0, "s_innovation": 1.0, "s_respect": 1.0, "s_quality": 1.0}).assign(gvkey=["000001", "000002", "000003"])
    lt = pd.DataFrame({"gvkey": ["000001"], "year": [2012.0], "case_p": [1.0], "case_f": [3.0]})
    plain = _signals(arc=arc, blocks=bl, culture=ca, lit=lt)

    def nullable(d):
        return d.astype({c: ("Float64" if d[c].dtype.kind == "f" else d[c].dtype) for c in d})

    fancy = _signals(arc=nullable(arc).assign(form=lambda d: d["form"].astype("string")), blocks=nullable(bl), culture=nullable(ca)
                     .astype({"year": "Int16"}), lit=lt.astype(object).map(lambda v: decimal.Decimal(str(v)) if isinstance(v, float) else v),
                     funda=_funda().astype("string"))
    pd.testing.assert_frame_equal(plain[NAMES], fancy[NAMES])
    assert plain["arc10k"].notna().any() and plain["blocks"].notna().any() and plain["litig"].notna().any()


def test_checks_block_reports_counts_hit_rate_and_decade_coverage():
    arc = _arc([(1750.0, D("2011-12-31"), 100.0, "10-K", D("2012-03-15")), (77.0, D("2011-12-31"), 1.0, "10-K", D("2012-03-15"))])
    sig = _signals(arc=arc).reset_index()
    raw = {"arc_data": arc, "risk_text": EMPTY["risk"].assign(cik=pd.Series(["0000001750"], dtype=object)).reindex([0]),
           "blockholders": EMPTY["blocks"], "class_boards": EMPTY["boards"], "corp_culture": EMPTY["culture"], "fed_litigation": EMPTY["lit"]}
    lines = altdata_checks(sig, raw, _funda(pd.DataFrame({"gvkey": ["000002", "000003"], "cik": ["0000000020", "0000000020"]})))
    txt = "\n".join(lines)
    assert "arc_data 2" in txt and "1 map to more than one gvkey" in txt
    assert "hit rate arc_data: 2 distinct ciks; unique 50.0%, ambiguous 0.0%, not in Compustat 50.0%" in txt
    assert "2010s=" in txt and all(n in txt for n in NAMES)
    assert "arc10k   median share tied at max" in txt and "empty L leg" in txt


# ------------------------------------------------------------------------------------------- promotion minimum
def test_short_or_empty_series_cannot_promote_and_explain_why():
    def series(n_dev):
        idx = np.r_[np.arange(2014 * 12 + 12 - n_dev, 2014 * 12 + 12), np.arange(2015 * 12, 2015 * 12 + 12)]       # n_dev dev months + 12 validation months
        return (pd.Series(np.cos(np.arange(len(idx)) * 1.3) * 0.01 + 0.002, index=idx), pd.Series(np.sin(np.arange(len(idx))) * 0.01, index=idx))
    rc, rg = series(59)
    st = promotion_stats(rc, rg, min_dev_months=60)
    assert st["PROMOTE"] is False and st["dev_months"] == 59 and st["val_months"] == 12 and "fewer than 60" in st["note"]
    assert "dev_alpha_t" in promotion_stats(rc, rg)                                  # default rule (24 months) still computes statistics
    assert "dev_alpha_t" in promotion_stats(*series(60), min_dev_months=60)          # exactly 60 dev months is enough
    assert "fewer than" not in promotion_stats(*series(10), min_dev_months=60).get("note", "")      # below 24: the original message
    e = promotion_stats(pd.Series(dtype=float), pd.Series(dtype=float), min_dev_months=60)
    assert e["PROMOTE"] is False and e["dev_months"] == 0 and "note" in e


# ---------------------------------------------------------------------------------------------------- end to end
def _script():
    spec = importlib.util.spec_from_file_location("run_altdata", pathlib.Path(__file__).parent.parent / "scripts" / "run_altdata.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw_tables(gvs, seed=3):
    rng = np.random.default_rng(seed)
    cik = {g: 1000 + 10 * int(g) for g in gvs}
    arc, rt, bl, cb, cu, li = [], [], [], [], [], []
    for g in gvs:
        for y in range(2004, 2021):
            fpe = pd.Timestamp(f"{y}-12-31")
            arc.append((float(cik[g]), fpe, float(rng.normal(100, 20)), "10-K", fpe + pd.Timedelta(days=int(rng.integers(40, 90)))))
            arc.append((float(cik[g]), fpe, 1e4, "10-K/A", fpe + pd.Timedelta(days=150)))
            rt.append((f"{cik[g]:010d}", float(y), (fpe + pd.Timedelta(days=60)), 0.0 if rng.random() < .05 else float(rng.integers(500, 30000))))
            cb.append((g, fpe - pd.Timedelta(days=214), y, int(rng.random() < .3)))
            li.append((g, decimal.Decimal(y), decimal.Decimal(int(rng.poisson(2))), decimal.Decimal(int(rng.poisson(1)))))
            cu.append((g, "SE", y, *rng.normal(2, 1, 5)))
            for _ in range(int(rng.integers(1, 5))):
                bl.append((float(rng.integers(1, 99)), float(cik[g]), "other", float(y), float(rng.uniform(5, 15))))
    raw = {"arc_data": pd.DataFrame(arc, columns=["cik", "fiscal_period_end", "arc", "form", "date_filed"]).astype({"cik": "Float64"}),
           "risk_text": pd.DataFrame(rt, columns=["cik", "fyear", "filing_date", "item_1a_word_count"]),
           "blockholders": pd.DataFrame(bl, columns=["blockholder_cik", "company_cik", "block_type", "year", "position"]),
           "class_boards": pd.DataFrame(cb, columns=["gvkey", "datadate", "fyear", "cbi"]).astype({"fyear": "Int16", "cbi": "Int16"}),
           "corp_culture": pd.DataFrame(cu, columns=["gvkey", "source", "year", "s_integrity", "s_teamwork", "s_innovation", "s_respect", "s_quality"])
           .astype({"year": "Int16"}),
           "fed_litigation": pd.DataFrame(li, columns=["gvkey", "year", "case_p", "case_f"])}
    return raw, cik


def _world():
    from qap.costs import liquidity_inputs
    from qap.panel import build_monthly_panel
    from qap.portfolio import market_scaled_aum
    m, d, funda, links = make_world(n=90, years=(1998, 2022))
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    gvs = sorted(funda["gvkey"].unique())
    raw, cik = _raw_tables(gvs)
    funda = funda.assign(cik=funda["gvkey"].map(lambda g: f"{cik[g]:010d}"))
    return p, links, funda, liquidity_inputs(p, _agg_for(p)), market_scaled_aum(p, 1e7), raw


def _patch(mod, monkeypatch, world, tmp_path):
    p, links, funda, inp, aum, raw = world
    (tmp_path / "data" / "raw").mkdir(parents=True)
    for n, df in raw.items():
        df.to_parquet(tmp_path / "data" / "raw" / f"{n}.parquet", index=False)
    monkeypatch.setattr(mod, "require_inputs", lambda *a, **k: None)
    monkeypatch.setattr(mod, "load_context", lambda: (p, links, funda, inp, aum))


@pytest.fixture(scope="module")
def world():
    return _world()


def _run_in(tmp_path, monkeypatch, world, jobs, open_holdout=False, promoted=None):
    mod = _script()
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, world, tmp_path)
    if promoted is not None:
        monkeypatch.setattr(mod, "run_promotion_report", lambda *a, **k: ({}, promoted))
    lines = []
    mod._run(lines.append, open_holdout, jobs)
    return lines


def test_script_end_to_end_checks_first_trials_logged_holdout_sealed_and_jobs_identical(tmp_path, monkeypatch, world):
    a, b = tmp_path / "j1", tmp_path / "j3"
    a.mkdir(), b.mkdir()
    l1 = _run_in(a, monkeypatch, world, 1)
    l3 = _run_in(b, monkeypatch, world, 3)
    assert l1 == l3                                                                  # identical report text (all stats) for --jobs 1 and 3
    txt = "\n".join(l1)
    assert txt.index("== CHECKS") < txt.index("== RESULTS (Study C)") and "cik->gvkey hit rate arc_data" in txt
    assert "HOLDOUT OPENED" not in txt and "PROMOTED (provisional; sealed holdout opens once for these): " in txt
    for n in NAMES:
        assert f"{n}: " in txt
    trials = [json.loads(x) for x in (a / "data" / "trials.jsonl").read_text().splitlines()]
    assert sorted(t["name"] for t in trials) == sorted(f"studyC:{n}" for n in NAMES) and {t["spec"]["prereg"] for t in trials} == {"v9"}
    assert json.loads((a / "data" / "study_c_promoted.json").read_text()) == json.loads((b / "data" / "study_c_promoted.json").read_text())


def test_script_holdout_opens_only_for_promoted_signals(tmp_path, monkeypatch, world):
    none = _run_in(tmp_path, monkeypatch, world, 1, open_holdout=True, promoted=[])
    assert not any(f"  {n}: {{'months'" in l for l in none for n in NAMES)
    d = tmp_path / "p"
    d.mkdir()
    some = "\n".join(_run_in(d, monkeypatch, world, 1, open_holdout=True, promoted=["litig"]))
    assert "HOLDOUT OPENED (once) for promoted signals only" in some and "  litig: {'months'" in some and "  arc10k: {" not in some
    assert json.loads((d / "data" / "study_c_promoted.json").read_text()) == ["litig"]


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
    assert len(promo) == 6 and all(l.startswith("  [FAIL]") and ("fewer than 10000 overlapping dev months" in l or "too few overlapping" in l) for l in promo)
    assert sum("fewer than 10000 overlapping dev months" in l for l in promo) >= 4
    assert qap.study.promotion_stats is qap.candidates.promotion_stats                 # patch restored


def test_tie_break_is_deterministic_and_preserves_order():
    from qap.altdata import NAMES, TIE_EPS, break_ties, tie_key
    k1, k2 = tie_key(np.array([5, 6, 7, 5])), tie_key(np.array([5, 6, 7, 5]))
    assert (k1 == k2).all() and k1[0] == k1[3] and ((k1 >= 0) & (k1 < 1)).all() and len(set(k1[:3])) == 3
    v = np.array([0.0, 0.0, -1.0, 0.0])                       # binary signal with ties at the maximum
    w = v + TIE_EPS * k1
    assert w[2] < w[[0, 1, 3]].min() - 0.5 and len(set(w[:2])) == 2      # distinct values keep their order, tied ones are separated
    df = pd.DataFrame({"permno": [5, 6, 7, 8], **{n: [0.0, 0.0, 0.0, -1.0] for n in NAMES}})
    out = break_ties(df)
    assert out["cbi_neg"].iloc[3] < out["cbi_neg"].iloc[:3].min() and out["cbi_neg"].iloc[:3].nunique() == 3
    assert (df.drop(columns="permno") == 0).sum().sum() > 0 and (out.drop(columns="permno").iloc[:3] != 0).all().all()
