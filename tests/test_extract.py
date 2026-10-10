from qap import config
from qap.extract import QUERIES, ccm_links_sql, compustat_annual_sql, crsp_monthly_sql


def test_all_queries_build_and_select_from_confirmed_tables():
    for name, fn in QUERIES.items():
        sql = fn()
        assert sql.startswith("select ") and " from " in sql, name


def test_crsp_monthly_uses_ciz_names_not_legacy():
    sql = crsp_monthly_sql()
    assert "crsp.msf_v2" in sql and "mthret" in sql and "mthdelflg" in sql
    for legacy in ("shrcd", "exchcd", "dlret", " ret,"):
        assert legacy not in sql


def test_compustat_filter_and_ccm():
    assert "indfmt = 'INDL'" in compustat_annual_sql() and "datafmt = 'STD'" in compustat_annual_sql()
    assert "ccmxpf_lnkhist" in ccm_links_sql()
    assert config.CCM_LINKS == "crsp.ccmxpf_lnkhist"
