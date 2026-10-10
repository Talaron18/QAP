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
