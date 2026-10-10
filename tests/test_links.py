import pandas as pd
from qap.links import filter_ccm_links, links_valid_on


def _links():
    return pd.DataFrame({
        "gvkey": ["1", "1", "1", "2", "2"],
        "permno": [10, 11, 12, 20, 21],
        "linktype": ["LC", "LU", "LX", "LC", "LC"],
        "linkprim": ["P", "C", "P", "J", "P"],
        "linkdt": ["2000-01-01", "2005-01-01", "2000-01-01", "2000-01-01", "2003-01-01"],
        "linkenddt": ["2004-12-31", None, "2010-01-01", "2010-01-01", None],
    })


def test_filters_type_and_prim():
    out = filter_ccm_links(_links())
    assert sorted(out["permno"]) == [10, 11, 21]  # LX and linkprim J dropped


def test_null_enddate_is_current():
    out = filter_ccm_links(_links())
    assert out.loc[out.permno == 11, "linkenddt"].iloc[0] == pd.Timestamp.max
    assert list(links_valid_on(out, "2020-06-01")["permno"]) == [11, 21]


def test_date_bounds_inclusive_and_no_overlap_lookahead():
    out = filter_ccm_links(_links())
    assert list(links_valid_on(out, "2004-12-31")["permno"]) == [10, 21]
    assert list(links_valid_on(out, "2005-01-01")["permno"]) == [11, 21]
    assert list(links_valid_on(out, "1999-12-31")["permno"]) == []
