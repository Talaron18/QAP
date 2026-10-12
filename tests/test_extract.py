import pytest

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


ALT_TABLES = {
    "arc_data": ("contrib_general.arc", ["cik", "fiscal_period_end", "arc", "form", "date_filed"]),
    "risk_text": ("contrib_general.as_firm_risks", ["cik", "fyear", "filing_date", "item_1a_word_count"]),
    "blockholders": ("contrib_general.blocksv", ["company_cik", "blockholder_cik", "year", "position", "block_type"]),
    "class_boards": ("contrib_general.classified_boards", ["gvkey", "datadate", "fyear", "cbi"]),
    "corp_culture": ("contrib_corporate_culture.corp_culture", ["gvkey", "source", "year", "s_integrity", "s_teamwork", "s_innovation",
                                                              "s_respect", "s_quality"]),
    "fed_litigation": ("contrib_corp_fed_litigation.firmyear_level", ["gvkey", "year", "case_p", "case_f"]),
}


def test_alt_data_queries_select_only_needed_columns_from_confirmed_tables():
    for name, (table, cols) in ALT_TABLES.items():
        sql = QUERIES[name]()
        assert sql.startswith("select ") and f" from {table}" in sql, name
        assert "*" not in sql
        for c in cols:
            assert c in sql, (name, c)
    assert "where" not in QUERIES["arc_data"]()         # all forms are extracted; the 10-K filter happens in pandas


def test_alt_data_sql_runs_and_returns_the_named_columns():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    ddl = {"arc_data": "cik double, fiscal_period_end date, arc double, form varchar, date_filed date, adsh varchar",
           "risk_text": "cik varchar, fyear double, filing_date date, item_1a_word_count double, other double",
           "blockholders": "blockholder_cik double, company_cik double, blockholder_name varchar, block_type varchar, year double, "
                           "position double, files_13f double",
           "class_boards": "gvkey varchar, datadate date, fyear smallint, cbi smallint, cbv smallint",
           "corp_culture": "gvkey varchar, source varchar, year smallint, s_integrity double, s_teamwork double, s_innovation double, "
                           "s_respect double, s_quality double, document_length integer",
           "fed_litigation": "gvkey varchar, year numeric, case_p numeric, case_f numeric, ff12 double"}
    for name, (table, cols) in ALT_TABLES.items():
        schema = table.split(".")[0]
        con.execute(f"create schema if not exists {schema}")
        con.execute(f"create table {table}({ddl[name]})")
        con.execute(f"insert into {table} ({cols[0]}) values (NULL)")
        out = con.execute(QUERIES[name]()).df()
        assert list(out.columns) == cols, name
        assert len(out) == 1
