import pathlib

import numpy as np
import pandas as pd
import pytest

from qap.etfcheck import (ETF_COLS, ETF_EXCLUDE_WORDS, detect_by_returns, masked_corr_beta, build_etf_table, coverage, decision, etf_tokens, etf_words, excluded_etfs,
                          find_underlying, first_last_months, format_report, ols_cluster, parse_etf_name, pick_columns,
                          prep_names, shortfall_panel, shortfall_summary)

M0 = 2022 * 12      # Jan 2022


# ---------- name parsing ----------
@pytest.mark.parametrize("name,L,cand", [
    ("Tradr 2X Short LITE Daily ETF", -2.0, True),
    ("GraniteShares 2x Long NVDA Daily ETF", 2.0, True),
    ("Direxion Daily TSLA Bull 2X Shares", 2.0, True),
    ("Direxion Daily AAPL Bear 1X Shares", -1.0, True),
    ("Leverage Shares 1.5X Long MSFT Daily ETF", 1.5, True),
    ("Defiance Daily Target 2X Inverse AMD ETF", -2.0, True),
    ("Direxion Daily S&P 500 Bull 3X Shares", 3.0, False),
    ("Direxion Daily Gold Miners Index Bull 2X Shares", 2.0, False),
    ("Direxion Daily Semiconductor Bear 3X Shares", -3.0, False),
    ("Direxion Daily NASDAQ100 Bull 2X", 2.0, False),
    ("ProShares UltraPro QQQ", None, False),
    ("GraniteShares 2x Long NVDA ETF", None, False),          # no 'daily'
    ("Direxion Daily Bull Shares", None, False),               # no multiple
    ("Direxion Daily GOLDMAN Bull 2X Shares", 2.0, True),      # 'Goldman' is not the word Gold
])
def test_parse_etf_name(name, L, cand):
    r = parse_etf_name(name)
    assert r["L"] == L and r["single_stock_candidate"] is cand


def test_parse_handles_missing_and_exclusion_word_reported():
    assert parse_etf_name(None)["L"] is None and parse_etf_name(float("nan"))["single_stock_candidate"] is False
    assert parse_etf_name("Direxion Daily S&P 500 Bull 3X Shares")["excluded"].lower() == "s&p"
    assert "Real Estate" in ETF_EXCLUDE_WORDS and parse_etf_name("Daily 2X Long Real   Estate")["single_stock_candidate"] is False


def test_tokens_and_words():
    assert etf_tokens("Tradr 2X Short LITE Daily ETF") == ["LITE"]
    assert etf_tokens("DIREXION DAILY TSLA BULL 2X SHARES") == ["TSLA"]
    assert etf_tokens("T-REX 2X Long Tesla Daily Target ETF") == []          # TREX is an issuer stopword
    assert etf_tokens("Defiance Daily Target 1.5X Long NVDA, AMD ETF") == ["NVDA", "AMD"]
    assert etf_words("T-REX 2X Long Tesla Daily Target ETF") == ["TESLA"]


# ---------- matching ----------
def _names():
    d = lambda s: pd.Timestamp(s)
    rows = [   # permno, ticker, securitynm, issuernm, namedt, nameenddt
        (1, "NVDA", "NVIDIA CORP", "NVIDIA CORP", "2000-01-01", None),
        (2, "LITE", "LUMENTUM HOLDINGS INC", "LUMENTUM HOLDINGS", "2015-01-01", None),
        (3, "TSLA", "TESLA INC", "TESLA INC", "2010-06-01", None),
        (30, "TSLX", "TESLA ENERGY CO", "TESLA ENERGY", "2010-06-01", None),          # same first word, smaller ME
        (4, "LITE", "OLD LITE CORP", "OLD LITE", "2000-01-01", "2020-12-31"),          # ticker no longer active in 2022
        (5, "LITE", "LITE NEW INC", "LITE NEW", "2021-06-01", None),
        (6, "DUP", "DUP ONE INC", "DUP ONE", "2000-01-01", None),
        (7, "DUP", "DUP TWO INC", "DUP TWO", "2000-01-01", None),
        (9001, "LITS", "Tradr 2X Short LITE Daily ETF", "Tradr ETF Trust", "2022-01-01", None),
        (9002, "NVDG", "GraniteShares 2x Long NVDA Daily ETF", "GraniteShares ETF Trust", "2022-01-01", None),
        (9003, "TSLL", "Direxion Daily TSLA Bull 2X Shares", "Direxion Shares ETF Trust", "2022-01-01", None),
        (9004, "SPXL", "Direxion Daily S&P 500 Bull 3X Shares", "Direxion Shares ETF Trust", "2022-01-01", None),
        (9005, "UPRO", "ProShares UltraPro QQQ", "ProShares Trust", "2022-01-01", None),
        (9006, "ZZZ2", "Foo 2X Long ZZZQ Daily ETF", "Foo Trust", "2022-01-01", None),
        (9007, "TSLT", "T-REX 2X Long Tesla Daily Target ETF", "REX Shares", "2022-01-01", None),
        (9008, "DUPL", "Bar 2X Long DUP Daily ETF", "Bar Trust", "2022-01-01", None),
    ]
    return pd.DataFrame({
        "permno": pd.array([r[0] for r in rows], dtype="Int64"), "ticker": pd.array([r[1] for r in rows], dtype="string"),
        "securitynm": pd.array([r[2] for r in rows], dtype="string"), "issuernm": pd.array([r[3] for r in rows], dtype="string"),
        "namedt": pd.to_datetime([r[4] for r in rows]), "nameenddt": pd.to_datetime([r[5] for r in rows])})


ME = {1: 500, 2: 100, 3: 900, 30: 50, 4: 800, 5: 60, 6: 5, 7: 9}


def _panel(extra_rows=()):
    """Stocks with eq True Jan..Jun 2022, ETFs with eq False from Jan 2022 (first return Jan 2022). Nullable dtypes."""
    rows = []
    for p, me in ME.items():
        for k in range(6):
            rows.append((p, M0 + k, 0.01 * k, float(me), True))
    for p in (9001, 9002, 9003, 9004, 9006, 9007, 9008):
        for k in range(6):
            rows.append((p, M0 + k, 0.0, np.nan, False))
    d = pd.DataFrame(rows, columns=["permno", "mi", "mthret_adj", "me_co", "eq"])
    return d.astype({"permno": "Int64", "mi": "Int64", "mthret_adj": "Float64", "me_co": "Float64", "eq": "boolean"})


def test_build_etf_table_matching_and_dtypes():
    t = build_etf_table(_names(), _panel()).set_index("etf_permno")
    assert set(t.index) == {9001, 9002, 9003, 9006, 9007, 9008}          # 9004 excluded by word, 9005 not leveraged
    assert t.loc[9001, "L"] == -2 and t.loc[9002, "L"] == 2 and t.loc[9003, "L"] == 2
    # LITE: permno 4 (ME 800) held the ticker only until 2020 (expired); 2 (ME 100) and 5 (ME 60) are active -> the larger ME, 2
    assert t.loc[9001, ["underlying_permno", "underlying_ticker", "method"]].tolist() == [2.0, "LITE", "ticker"]
    assert t.loc[9002, "underlying_permno"] == 1 and t.loc[9003, "underlying_permno"] == 3
    assert t.loc[9003, "method"] == "ticker"
    assert t.loc[9008, "underlying_permno"] == 7 and t.loc[9008, "method"] == "ticker"      # DUP tie -> larger ME
    # name fallback: first word TESLA, permnos 3 (ME 900) and 30 (ME 50) -> 3
    assert t.loc[9007, "underlying_permno"] == 3 and t.loc[9007, "method"] == "name"
    assert np.isnan(t.loc[9006, "underlying_permno"]) and t.loc[9006, "method"] == "unmatched"
    assert (t["first_mi"] == M0).all() and (t["last_mi"] == M0 + 5).all()
    ex = excluded_etfs(_names())
    assert ex["permno"].tolist() == [9004] and ex["excluded"].iloc[0].lower() == "s&p"


def test_ticker_active_date_and_largest_me():
    nm = prep_names(_names())
    eq = pd.DataFrame({"permno": [2, 4, 5], "me_co": [100.0, 800.0, 60.0]})
    assert find_underlying("2X Short LITE Daily", pd.Timestamp("2022-01-31"), nm, eq)[0] == 2        # 4 expired in 2020
    assert find_underlying("2X Short LITE Daily", pd.Timestamp("2020-06-30"), nm, eq)[0] == 4        # 4 active, biggest
    assert find_underlying("2X Short LITE Daily", pd.Timestamp("2022-01-31"), nm, eq[eq.permno != 2])[0] == 5
    # stock without an eq row that month is not eligible (e.g. an ETF or a non-common name)
    assert find_underlying("2X Short LITE Daily", pd.Timestamp("2022-01-31"), nm, eq[eq.permno == 4]) == (None, None, "unmatched")
    # name fallback only when no ticker hit
    assert find_underlying("2X Long Tesla Daily", pd.Timestamp("2022-01-31"), nm, pd.DataFrame({"permno": [3, 30], "me_co": [1.0, 2.0]}))[::2] == (30, "name")


def test_first_months_and_alternate_names_columns():
    p = _panel()
    p.loc[(p.permno == 9001) & (p.mi < M0 + 2), "mthret_adj"] = pd.NA            # returns start in Mar 2022
    fl = first_last_months(p)
    assert fl.at[9001, "first_mi"] == M0 + 2 and fl.at[9001, "last_mi"] == M0 + 5
    n = _names().rename(columns={"ticker": "tradingsymbol", "securitynm": "comnam", "issuernm": "issuer_name"})
    assert pick_columns(n) == {"permno": "permno", "ticker": "tradingsymbol", "names": ["comnam", "issuer_name"],
                               "start": "namedt", "end": "nameenddt"}
    t = build_etf_table(n, p)
    assert t.set_index("etf_permno").loc[9001, "first_mi"] == M0 + 2


def test_pick_columns_error_lists_columns():
    with pytest.raises(ValueError) as e:
        pick_columns(pd.DataFrame({"permno": [1], "symbol": ["A"], "comnam": ["x"], "namedt": [1], "nameenddt": [2]}))
    assert "available columns" in str(e.value) and "symbol" in str(e.value) and "ticker" in str(e.value)
    assert pick_columns(pd.DataFrame({"permno": [1], "ticker": ["A"], "comnam": ["x"], "namedt_x": [1], "nameenddt": [2]}))["start"] == "namedt_x"


# ---------- shortfall ----------
def _sf_world(nullable=True, drop_ru=None):
    """ETF 9001 (L=-2) and 9002 (L=+2) on underlying 5; r_etf = L * r_u - 0.001 exactly; months Jan..Jun 2022."""
    ru = [0.10, -0.05, 0.08, -0.12, 0.03, 0.06]
    rows = [(5, M0 + k, ru[k], 100.0, True) for k in range(6)]
    for etf, L in ((9001, -2.0), (9002, 2.0)):
        rows += [(etf, M0 + k, L * ru[k] - 0.001, np.nan, False) for k in range(6)]
    p = pd.DataFrame(rows, columns=["permno", "mi", "mthret_adj", "me_co", "eq"])
    if drop_ru is not None:
        p.loc[(p.permno == 5) & (p.mi == M0 + drop_ru), "mthret_adj"] = np.nan
    agg = pd.DataFrame({"permno": 5, "yyyymm": [202201 + k for k in range(6)], "n_ret": 20, "sigma_d": 0.02})
    if nullable:
        p = p.astype({"permno": "Int64", "mi": "Int64", "mthret_adj": "Float64", "me_co": "Float64", "eq": "boolean"})
        agg = agg.astype({"permno": "Int64", "yyyymm": "Int64", "n_ret": "Int64", "sigma_d": "Float64"})
    t = pd.DataFrame({"etf_permno": [9001, 9002], "etf_ticker": ["AAA", "BBB"], "name": ["x", "y"], "L": [-2.0, 2.0],
                      "underlying_permno": [5.0, 5.0], "underlying_ticker": ["U", "U"], "method": "ticker",
                      "first_mi": float(M0), "last_mi": float(M0 + 5)})
    return t, p, agg


@pytest.mark.parametrize("nullable", [True, False])
def test_shortfall_panel_exact_and_first_last_dropped(nullable):
    t, p, agg = _sf_world(nullable)
    sp = shortfall_panel(t, p, agg)
    assert sorted(sp["mi"].unique()) == [M0 + 1, M0 + 2, M0 + 3, M0 + 4] and len(sp) == 8      # Jan and Jun dropped
    assert sp["shortfall"].to_numpy() == pytest.approx(-0.001, abs=1e-12)
    r = sp[(sp.etf_permno == 9001) & (sp.mi == M0 + 2)].iloc[0]       # r_u = 0.08, v = 0.02^2 * 20 = 0.008
    assert r["r_u"] == pytest.approx(0.08) and r["v"] == pytest.approx(0.008) and r["L"] == -2
    assert r["conv"] == pytest.approx((4 + 2) / 2 * (0.08 ** 2 - 0.008))
    s = shortfall_summary(sp)
    assert s["fm"]["all"]["mean"] == pytest.approx(-0.001) and s["fm"]["all"]["n_months"] == 4 and s["fm"]["all"]["n_etfs"] == 2
    assert s["fm"]["L<=-2"]["mean"] == pytest.approx(-0.001) and s["fm"]["L>0"]["n_obs"] == 4 and s["fm"]["-2<L<0"]["n_obs"] == 0
    assert s["fm_per_unit"]["all"]["mean"] == pytest.approx(-0.0005) and s["fm"]["all"]["noise"] == pytest.approx(0, abs=1e-12)
    rg = s["regression"]["short"]
    assert rg["a"] == pytest.approx(-0.001) and rg["b"] == pytest.approx(0, abs=1e-9) and rg["a_per_unit"] == pytest.approx(-0.0005)
    assert s["regression"]["all"]["a"] == pytest.approx(-0.001) and s["regression"]["all"]["n"] == 8
    assert all(isinstance(v, float) for v in s["fm"]["all"].values() if not isinstance(v, int))


def test_shortfall_missing_underlying_return_dropped_and_unmatched_ignored():
    t, p, agg = _sf_world(drop_ru=2)
    t.loc[len(t)] = [9003, "CCC", "z", 2.0, np.nan, None, "unmatched", float(M0), float(M0 + 5)]
    sp = shortfall_panel(t, p, agg)
    assert sorted(sp["mi"].unique()) == [M0 + 1, M0 + 3, M0 + 4] and set(sp["etf_permno"]) == {9001, 9002}


def test_shortfall_with_noise_regression_slope_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200) * 0.01
    g = np.repeat(np.arange(20), 10)
    r = ols_cluster(-0.002 + 1.0 * x, x, g)
    assert r["a"] == pytest.approx(-0.002) and r["b"] == pytest.approx(1.0) and r["n_clusters"] == 20
    r = ols_cluster(-0.002 + x + rng.normal(size=200) * 0.001, x, g)
    assert 0.5 < r["b"] < 1.5 and r["se_b"] > 0 and np.isfinite(r["t_a"])
    assert np.isnan(ols_cluster([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1, 2, 3])["a"])           # constant regressor


# ---------- coverage ----------
F = M0 + 4


def _sig():
    """Months F and F+1: 16 names, permno 1 an NYSE microcap (dropped), permnos 2..6 NYSE; me = 10 p; signal = permno."""
    p = np.arange(1, 17)
    s = pd.DataFrame({"permno": p, "mi": F, "me": 10.0 * p, "me_co": 10.0 * p, "primaryexch": np.where(p <= 6, "N", "Q"), "sig": p.astype(float)})
    s.loc[0, ["me", "me_co"]] = 1.0
    return pd.concat([s, s.assign(mi=F + 1)], ignore_index=True)


def _etf(rows):
    return pd.DataFrame(rows, columns=["etf_permno", "L", "underlying_permno", "first_mi", "last_mi"]).astype("float64")


def test_coverage_by_hand():
    # eligible permnos 2..16, bottom quintile = 2, 3, 4 (me 20, 30, 40), top = 14, 15, 16 (me 140, 150, 160)
    etf = _etf([(9001, -2, 3, F, F + 1), (9002, -2, 4, F, F), (9003, -3, np.nan, F, F + 1), (9004, 2, 16, F, F + 1), (9005, 2, 2, F + 1, F + 1)])
    c = coverage(_sig(), etf, "sig", None)
    y = c["by_year"][2022]
    assert y["n_months"] == 2 and y["bot_n"] == 3
    assert y["bot_cnt_share"] == pytest.approx((2 / 3 + 1 / 3) / 2)                    # F: names 3, 4; F+1: name 3 only
    assert y["bot_vw_share"] == pytest.approx((70 / 90 + 30 / 90) / 2)
    assert y["top_cnt_share"] == pytest.approx(1 / 3)                                  # only 16 is covered; 2 is not in the top quintile
    assert y["top_vw_share"] == pytest.approx(160 / 450)
    assert y["n_inv_live"] == 2 and y["n_long_live"] == 2                              # F+1: 9001, 9003 inverse; 9004, 9005 long
    assert c["last_month"] == {"mi": F + 1, "bottom_n": 3, "bottom_covered_n": 1}


def test_coverage_nullable_inputs_and_year_filter_and_empty_table():
    s = _sig().astype({"permno": "Int64", "mi": "Int64", "me": "Float64", "me_co": "Float64", "primaryexch": "string", "sig": "Float64"})
    s = pd.concat([s, s.assign(mi=M0 - 12)], ignore_index=True)                       # a 2021 month: not reported
    c = coverage(s, _etf([]), "sig")
    assert list(c["by_year"]) == [2022] and c["by_year"][2022]["bot_vw_share"] == 0.0 and c["by_year"][2022]["n_inv_live"] == 0
    assert c["last_month"]["bottom_covered_n"] == 0


def test_coverage_uses_ranks_only():
    s = _sig()
    etf = _etf([(9001, -2, 3, F, F + 1)])
    a = coverage(s, etf, "sig")
    b = coverage(s.assign(sig=s["sig"] * 1000 + 7), etf, "sig")                         # monotone transform: same ranks, same answer
    assert a == b


# ---------- report, decision and side effects ----------
def _cov_for_report(vw25):
    return {"gpa": {"name": "gpa", "by_year": {2025: {"n_months": 12, "bot_n": 100, "bot_cnt_share": vw25, "bot_vw_share": vw25,
                                                     "top_cnt_share": 0.1, "top_vw_share": 0.1, "n_inv_live": 5, "n_long_live": 6}},
                    "last_month": {"mi": 2025 * 12 + 11, "bottom_n": 100, "bottom_covered_n": 10}}}


def test_decision_and_report_lines():
    t, p, agg = _sf_world()
    summ = shortfall_summary(shortfall_panel(t, p, agg))               # a_per_unit = -0.0005 -> drag 0.05 %/mo
    d = decision(summ, _cov_for_report(0.5), 12)
    assert d["drag_pct_mo"] == pytest.approx(0.05) and d["signals"]["gpa"]["carry_hit"] is False      # 0.05 < 0.5 * 0.221
    assert d["signals"]["slow10"]["vw_cov"] != d["signals"]["slow10"]["vw_cov"]                        # NaN: not in cov dict
    assert d["verdict"] == "PILOT CONSIDERED"
    full2 = {k: _cov_for_report(0.2)["gpa"] for k in ("gpa", "Quality", "jkp_slow10", "jkp_all13")}
    assert decision(summ, full2, 12)["verdict"] == "NOT VIABLE"                                              # coverage < 30% for every signal
    assert decision(summ, _cov_for_report(0.2), 12)["verdict"] == "PILOT CONSIDERED"                         # only one signal evaluable and failing
    summ["regression"]["short"]["a_per_unit"] = -0.0012                                                  # drag 0.12 >= 0.5 * 0.221
    assert decision(summ, _cov_for_report(0.5), 12)["signals"]["gpa"]["carry_hit"] is True
    tab = pd.concat([build_etf_table(_names(), _panel())] * 3, ignore_index=True)       # 15 matched rows
    lines = format_report(tab, excluded_etfs(_names()), _cov_for_report(0.5), summ)
    txt = "\n".join(lines)
    assert "PARSED SINGLE-STOCK" in txt and "UNMATCHED (3)" in txt and "excluded by 'S&P'" in txt and "2022-01" in txt
    assert "VERDICT: PILOT CONSIDERED" in lines[-1] and "DECISION RULE" in txt and "INCOMPLETE" in lines[-1]      # slow10 etc. absent
    full = {k: _cov_for_report(0.5)["gpa"] for k in ("gpa", "Quality", "jkp_slow10", "jkp_all13")}
    d2 = decision(summ, full, 12)
    assert d2["unevaluable"] == [] and "INCOMPLETE" not in format_report(tab, excluded_etfs(_names()), full, summ)[-1]
    assert decision(summ, {}, 12)["unevaluable"][0] == "coverage 2025: slow10" and "INCOMPLETE" in "\n".join(format_report(tab, excluded_etfs(_names()), {}, summ))
    assert "n/a (year absent)" in "\n".join(format_report(tab, excluded_etfs(_names()), {}, summ))      # no 2025 data: no crash


def test_no_trials_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t, p, agg = _sf_world()
    tab = build_etf_table(_names(), _panel())
    sp = shortfall_panel(t, p, agg)
    format_report(tab, excluded_etfs(_names()), {"sig": coverage(_sig(), t, "sig")}, shortfall_summary(sp))
    assert not pathlib.Path("data/trials.jsonl").exists() and not pathlib.Path("data").exists()


def test_name_columns_are_concatenated_without_duplicates():
    n = pd.DataFrame({"permno": [1], "ticker": ["XX"], "securitynm": ["Tradr 2X Short LITE Daily ETF"], "issuernm": ["Tradr 2X Short LITE Daily"],
                      "comnam": ["Tradr Trust"], "namedt": [pd.Timestamp("2022-01-01")], "nameenddt": [pd.NaT]})
    assert prep_names(n)["text"].iloc[0] == "Tradr 2X Short LITE Daily ETF Tradr Trust"


def test_renamed_etf_flagged_and_allcaps_names_do_not_match_issuer_words():
    n = pd.concat([_names(), pd.DataFrame({
        "permno": pd.array([9001, 6, 8, 9010], dtype="Int64"), "ticker": pd.array(["LITS", "DUP", "CAPX", "XYZ1"], dtype="string"),
        "securitynm": pd.array(["Tradr 3X Short LITE Daily ETF", "DUP ONE INC", "CAPITAL ONE FIN CORP",
                                "TRADR 2X LONG NVDA DAILY ETF"], dtype="string"),
        "issuernm": pd.array(["Tradr ETF Trust", "DUP ONE", "CAPITAL ONE", "INVESTMENT MANAGERS SERIES TRUST II"], dtype="string"),
        "namedt": pd.to_datetime(["2023-01-01", "2000-01-01", "2000-01-01", "2022-01-01"]), "nameenddt": pd.to_datetime([None] * 4)})],
        ignore_index=True)
    p = _panel()
    p = pd.concat([p, pd.DataFrame({"permno": pd.array([8] * 6 + [9010] * 6, dtype="Int64"), "mi": pd.array(list(range(M0, M0 + 6)) * 2, dtype="Int64"),
                                    "mthret_adj": pd.array([0.0] * 12, dtype="Float64"), "me_co": pd.array([9999.0] * 6 + [np.nan] * 6, dtype="Float64"),
                                    "eq": pd.array([True] * 6 + [False] * 6, dtype="boolean")})], ignore_index=True)
    t = build_etf_table(n, p).set_index("etf_permno")
    assert t.loc[9001, "variants"] == 2 and t.loc[9001, "L"] == -2 and t.loc[9002, "variants"] == 1       # active at 2022-01: the 2X row
    assert t.loc[9010, "underlying_permno"] == 1 and t.loc[9010, "method"] == "ticker"                      # NVDA, not CAPITAL / MANAGERS / II
    assert etf_tokens("TRADR 2X LONG NVDA DAILY ETF INVESTMENT MANAGERS SERIES TRUST II") == ["NVDA"]
    assert etf_words("TRADR 2X LONG NVDA DAILY ETF CAPITAL TRUST") == ["NVDA"]


def test_shortfall_is_in_excess_of_rf():
    t, p, agg = _sf_world()
    raw = shortfall_panel(t, p, agg)
    rf = pd.Series(0.004, index=range(int(p["mi"].min()), int(p["mi"].max()) + 1))
    ex = shortfall_panel(t, p, agg, rf)
    # (r_etf - rf) - L (r_u - rf) = raw - (1 - L) * rf
    np.testing.assert_allclose(ex["shortfall"].to_numpy(), (raw["shortfall"] - (1.0 - raw["L"]) * 0.004).to_numpy(), atol=1e-12)


# ---------- return-based detection ----------
def _det_world(nullable=False):
    """30 stocks (eq) x 24 months from 2020; ETFs: 8001 L=-2, 8002 L=+2, 8003 L=-1 (small noise), 8004 covered-call-like (beta .6),
    8005 non-eq share class (L=+1), 8006 only 6 months. Share class 8005 has the same permco as its stock (also excluded by permco)."""
    rng = np.random.default_rng(3)
    T, N = 24, 30
    mi = 2020 * 12 + np.arange(T)
    R = rng.normal(0, 0.08, (T, N))
    rows = [(100 + j, 100 + j, mi[t], R[t, j], True) for j in range(N) for t in range(T)]
    def etf(pid, u, L, noise, n=T, pco=None):
        r = L * R[:, u] + rng.normal(0, noise, T)
        return [(pid, pid if pco is None else pco, mi[t], r[t], False) for t in range(T - n, T)]
    rows += etf(8001, 3, -2, 0.003) + etf(8002, 7, 2, 0.003) + etf(8003, 11, -1, 0.003) + etf(8004, 5, 0.6, 0.02)
    rows += etf(8005, 9, 1.0, 0.001, pco=999) + etf(8006, 13, 2, 0.003, n=6)
    d = pd.DataFrame(rows, columns=["permno", "permco", "mi", "mthret_adj", "eq"])
    if nullable:
        d = d.astype({"permno": "Int64", "permco": "Int64", "mi": "Int64", "mthret_adj": "Float64", "eq": "boolean"})
    return d


@pytest.mark.parametrize("nullable", [False, True])
def test_detect_by_returns(nullable):
    names = pd.DataFrame({"permno": pd.array([8001, 8002, 8003, 103, 8001], dtype="Int64"),
                          "ticker": pd.array(["OLD", "UP", "INV", "STK3", "NEWT"], dtype="string"),
                          "issuernm": pd.array(["X", "Y TRUST", "Z", "STOCK 3", "TRUST A"], dtype="string"),
                          "namedt": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-01", "2000-01-01", "2022-01-01"]),
                          "nameenddt": pd.to_datetime([None] * 5)})
    t = detect_by_returns(_det_world(nullable), names).set_index("etf_permno")
    assert list(t.index) == [8001, 8002, 8003]                   # covered call (8004), L=+1 share class (8005), 6 months (8006) rejected
    assert t["L"].tolist() == [-2, 2, -1] and t["underlying_permno"].tolist() == [103, 107, 111]
    assert set(t["method"]) == {"returns"} and (t["n_months"] == 24).all() and (t["rho"].abs() > 0.99).all()
    assert t.loc[8001, "etf_ticker"] == "NEWT" and t.loc[8001, "name"] == "TRUST A" and t.loc[8001, "underlying_ticker"] == "STK3"
    assert t.loc[8002, "underlying_ticker"] == "" and t.loc[8001, "first_mi"] == 2020 * 12 and t.loc[8001, "last_mi"] == 2020 * 12 + 23
    assert t.loc[8001, "beta"] == pytest.approx(-2, abs=0.1) and set(ETF_COLS) <= set(t.reset_index().columns)
    assert detect_by_returns(_det_world(), None).shape[0] == 3         # no names frame: blank labels
    assert detect_by_returns(_det_world(), start_mi=2021 * 12 + 6).shape[0] == 0       # only 6 months left in the window: min_months 12


def test_detect_params_and_empty():
    w = _det_world()
    assert set(detect_by_returns(w, tol=0.01, rho_min=0.999)["etf_permno"]) <= {8001, 8002, 8003}
    assert detect_by_returns(w[w["eq"]], None).empty                    # no candidates
    assert list(detect_by_returns(w[w["eq"]], None).columns)[-3:] == ["rho", "beta", "n_months"] 


def test_masked_corr_matches_pandas_pairwise():
    rng = np.random.default_rng(5)
    C, X = rng.normal(size=(30, 4)), rng.normal(size=(30, 6))
    C[rng.random(C.shape) < .2] = np.nan
    X[rng.random(X.shape) < .2] = np.nan
    n, rho, beta = masked_corr_beta(C, X)
    for a in range(4):
        for b in range(6):
            d = pd.DataFrame({"c": C[:, a], "x": X[:, b]}).dropna()
            assert n[a, b] == len(d) and rho[a, b] == pytest.approx(d["c"].corr(d["x"]))
            assert beta[a, b] == pytest.approx(d.cov().iloc[0, 1] / d["x"].var())


def test_names_without_name_columns_gives_empty_inventory():
    n = pd.DataFrame({"permno": [1], "ticker": ["A"], "namedt": [pd.Timestamp("2000-01-01")], "nameenddt": [pd.NaT]})   # no name column
    t = build_etf_table(n, _panel())
    assert t.empty and list(t.columns) == ETF_COLS and excluded_etfs(n).empty
    only_issuer = _names()[["permno", "ticker", "issuernm", "namedt", "nameenddt"]]
    assert build_etf_table(only_issuer, _panel()).empty                    # issuer names carry no 'daily' / multiple


@pytest.mark.parametrize("n,invalid", [(0, True), (9, True), (10, False), (25, False)])
def test_invalid_guard(n, invalid):
    t, p, agg = _sf_world()
    summ = shortfall_summary(shortfall_panel(t, p, agg))
    full = {k: _cov_for_report(0.2)["gpa"] for k in ("gpa", "Quality", "jkp_slow10", "jkp_all13")}      # would be NOT VIABLE if valid
    d = decision(summ, full, n)
    assert d["invalid"] is invalid
    tab = pd.DataFrame({c: pd.Series(dtype="float64") for c in ETF_COLS})
    tab = tab.reindex(range(n)).assign(etf_ticker="T", name="n", method="returns", underlying_permno=1.0, underlying_ticker="U", L=-2.0,
                                       first_mi=float(M0), last_mi=float(M0 + 5), variants=1)
    lines = format_report(tab, excluded_etfs(_names()), full, summ)
    txt = "\n".join(lines)
    if invalid:
        msg = f"INVALID: only {n} matched ETFs found (< 10); coverage and shortfall are not informative"
        assert d["verdict"] == msg and lines[0] == msg and f"VERDICT: {msg}" in lines[-1] and "NOT VIABLE" not in txt.replace("NOT VIABLE only", "")
        assert "inventory invalid" in txt
    else:
        assert d["verdict"] == "NOT VIABLE" and "INVALID" not in txt
