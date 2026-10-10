import pytest
import pandas as pd
from qap.crsp import common_equity_mask, flag_delisting_returns, monthly_return_panel
from qap.links import filter_ccm_links
from qap.pit import annual_available_date, quarterly_available_date


def _row(**kw):
    base = dict(securitytype="EQTY", securitysubtype="COM", sharetype="NS", usincflg="Y",
                issuertype="CORP", primaryexch="N", conditionaltype="RW", tradingstatusflg="A")
    base.update(kw)
    return base


def test_equity_mask():
    df = pd.DataFrame([_row(), _row(securitytype="FUND", securitysubtype="ETF"), _row(sharetype="AD"),
                       _row(usincflg="N"), _row(issuertype="REIT"), _row(primaryexch="R"),
                       _row(conditionaltype="NT"), _row(tradingstatusflg="H")])
    assert df[common_equity_mask(df)].index.tolist() == [0]
    assert common_equity_mask(df, issuer_types=("CORP", "REIT")).sum() == 2


def test_delisting_flags():
    df = pd.DataFrame({"mthdelflg": list("NAPVGM")})
    out = flag_delisting_returns(df)
    assert out.delist_ret_included.tolist() == [False, True, True, True, False, False]
    assert out.delist_ret_missing.tolist() == [False, False, False, False, True, True]


def test_panel_shape():
    df = pd.DataFrame({"permno": [1, 1, 2], "mthcaldt": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-02-29"]),
                       "mthret": [0.1, 0.2, 0.3]})
    p = monthly_return_panel(df)
    assert p.shape == (2, 2) and p.loc["2020-02-29", 2] == 0.3


def test_ccm_accepts_lpermno():
    links = pd.DataFrame({"gvkey": ["1"], "lpermno": [10.0], "linktype": ["LC"], "linkprim": ["P"],
                          "linkdt": ["2000-01-01"], "linkenddt": [None]})
    out = filter_ccm_links(links)
    assert out["permno"].tolist() == [10] and out["linkenddt"].iloc[0] == pd.Timestamp.max


def test_annual_available():
    d = annual_available_date(pd.Series(["2020-12-31", "2021-06-30"]))
    assert d.tolist() == [pd.Timestamp("2021-04-30"), pd.Timestamp("2021-10-30")]


def test_quarterly_available_uses_rdq_else_fallback():
    dd = pd.Series(["2020-03-31"] * 4)
    rdq = pd.Series(["2020-05-05", None, "2020-03-01", "2021-01-01"])  # ok, missing, before period, too late
    out = quarterly_available_date(dd, rdq)
    assert out.iloc[0] == pd.Timestamp("2020-05-06")
    for i in (1, 2, 3):
        assert out.iloc[i] == pd.Timestamp("2020-05-30")


def test_adjust_delisting_returns_all_branches():
    from qap.crsp import adjust_delisting_returns
    m = pd.DataFrame({
        "permno": [1, 2, 3, 4, 5, 6, 7],
        "mthcaldt": pd.to_datetime(["2020-03-31"] * 7),
        "mthret": [0.10, 0.10, 0.10, 0.10, None, 0.10, 0.10],
        "mthdelflg": ["N", "A", "G", "M", "M", "M", "G"],
    })
    d = pd.DataFrame({
        "permno": [3, 4, 5, 6],
        "delistingdt": ["2020-03-12", "2020-03-30", "2020-03-02", "2020-03-31"],
        "delret": [-0.5, None, None, None],
        "delretmisstype": ["NA", "DM", "DM", "DM"],
        "delreasontype": ["UNAV", "BKPY", "BKPY", "CORQ"],
    })
    out = adjust_delisting_returns(m, d).set_index("permno")
    assert out.loc[1, "delist_adj_method"] == "none" and out.loc[1, "mthret_adj"] == 0.10
    assert out.loc[2, "mthret_adj"] == 0.10                                    # A: already included
    assert out.loc[3, "delist_adj_method"] == "delret"
    assert out.loc[3, "mthret_adj"] == pytest.approx(1.10 * 0.5 - 1)           # compounded delret
    assert out.loc[4, "delist_adj_method"] == "imputed"
    assert out.loc[4, "mthret_adj"] == pytest.approx(1.10 * 0.70 - 1)          # -30% imputed
    assert out.loc[5, "mthret_adj"] == pytest.approx(-0.30)                    # missing mthret treated as 0
    assert out.loc[6, "delist_adj_method"] == "zero" and out.loc[6, "mthret_adj"] == 0.10
    assert out.loc[7, "delist_adj_method"] == "unmatched"
    assert len(out) == len(m)


def test_adjust_delisting_does_not_duplicate_rows_with_repeat_delist_records():
    from qap.crsp import adjust_delisting_returns
    m = pd.DataFrame({"permno": [1], "mthcaldt": pd.to_datetime(["2020-03-31"]), "mthret": [0.0], "mthdelflg": ["M"]})
    d = pd.DataFrame({"permno": [1, 1], "delistingdt": ["2020-03-05", "2020-03-20"], "delret": [None, None],
                      "delretmisstype": ["DM", "DM"], "delreasontype": ["LP", "LP"]})
    assert len(adjust_delisting_returns(m, d)) == 1
