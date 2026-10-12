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


def test_jkp_intl_sql_text_and_guards():
    from qap.extract import jkp_intl_sql
    sql = jkp_intl_sql(["gp_at", "bidaskhl_21d"], 2001, ["JPN", "GBR"])
    assert "cast(id as bigint) as id" in sql and "excntry in ('JPN', 'GBR')" in sql and "'USA'" not in sql
    for s in ("common = 1", "primary_sec = 1", "exch_main = 1", "obs_main = 1", "size_grp in ('mega', 'large', 'small')",
              "DATE '2001-01-01'", "DATE '2001-12-31'", "ret_exc_lead1m", "gp_at", "bidaskhl_21d"):
        assert s in sql, s
    with pytest.raises(ValueError):
        jkp_intl_sql(["gp_at"], 2001, ["JPN", "USA"])
    with pytest.raises(ValueError):
        jkp_intl_sql(["gp_at"], 2001, [])
    assert len(config.DEV_EX_US) == 22 and "USA" not in config.DEV_EX_US and len(set(config.DEV_EX_US)) == 22
    assert set(config.REGIONS) == {"Developed Europe", "Japan", "Other developed"}


def test_jkp_intl_sql_screens_run_in_duckdb():
    duckdb = pytest.importorskip("duckdb")
    from qap.extract import jkp_intl_sql
    con = duckdb.connect()
    con.execute("create schema contrib_global_factor")
    con.execute("create table contrib_global_factor.global_factor(id double, excntry varchar, eom date, size_grp varchar, "
                "ret_exc_lead1m double, me double, gp_at double, common integer, primary_sec integer, exch_main integer, obs_main integer)")
    rows = [  # id, excntry, eom, size_grp, common, primary_sec, exch_main, obs_main
        (300000001, "JPN", "2001-03-31", "small", 1, 1, 1, 1),        # kept
        (300000002, "GBR", "2001-12-31", "mega", 1, 1, 1, 1),         # kept (last day of the year)
        (300000003, "USA", "2001-03-31", "small", 1, 1, 1, 1),        # US excluded
        (300000004, "IND", "2001-03-31", "small", 1, 1, 1, 1),        # not in the list
        (300000005, "JPN", "2002-01-31", "small", 1, 1, 1, 1),        # other year
        (300000006, "JPN", "2001-03-31", "micro", 1, 1, 1, 1),        # micro dropped
        (300000007, "JPN", "2001-03-31", "nano", 1, 1, 1, 1),         # nano dropped
        (300000008, "JPN", "2001-03-31", "large", 0, 1, 1, 1),        # not common
        (300000009, "JPN", "2001-03-31", "large", 1, 0, 1, 1),        # not primary
        (300000010, "JPN", "2001-03-31", "large", 1, 1, 0, 1),        # not main exchange
        (300000011, "JPN", "2001-03-31", "large", 1, 1, 1, 0),        # obs_main = 0
        (None, "JPN", "2001-03-31", "large", 1, 1, 1, 1)]             # no id
    for r in rows:
        con.execute("insert into contrib_global_factor.global_factor values (?, ?, ?, ?, 0.01, 100.0, 0.3, ?, ?, ?, ?)", list(r))
    out = con.execute(jkp_intl_sql(["gp_at"], 2001, ["JPN", "GBR"])).df()
    assert sorted(out["id"].tolist()) == [300000001, 300000002]
    assert list(out.columns) == ["id", "excntry", "eom", "size_grp", "ret_exc_lead1m", "me", "gp_at"]
