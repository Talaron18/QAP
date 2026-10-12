"""secd SQL builders, extraction script (column validation, ambiguity refusal) and probe helpers. DuckDB stub tables, SYNTHETIC rows."""
import importlib.util
import pathlib

import pandas as pd
import pytest

from qap import config
from qap.extract import QUERIES, secd_candidates_sql, secd_columns, secd_sql

duckdb = pytest.importorskip("duckdb")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ext, probe = load("extract_secd"), load("probe_secd")
ALL = ["gvkey", "iid", "datadate", "tic", "conm", "tpci", "exchg", "curcdd", "prccd", "prcod", "prchd", "prcld", "cshtrd", "ajexdi",
       "trfd", "cshoc", "fic"]


class FakeDb:
    """The slice of wrds.Connection the scripts use, backed by DuckDB."""

    def __init__(self, drop=(), fail_describe=False):
        self.con = duckdb.connect()
        self.fail_describe = fail_describe
        self.con.execute("create schema comp_na_daily_all")
        cols = [c for c in ALL if c not in drop]
        ddl = {"gvkey": "varchar", "iid": "varchar", "datadate": "date", "tic": "varchar", "conm": "varchar", "tpci": "varchar",
               "exchg": "integer", "curcdd": "varchar", "fic": "varchar"}
        self.con.execute("create table comp_na_daily_all.secd(" + ", ".join(f"{c} {ddl.get(c, 'double')}" for c in cols) + ", junk double)")
        self.cols = cols
        self.sql = []

    def add(self, gvkey, iid, tic, conm, dates, px0=10.0, cur="USD", tpci="0"):
        for k, d in enumerate(dates):
            v = dict(gvkey=gvkey, iid=iid, datadate=d, tic=tic, conm=conm, tpci=tpci, exchg=11, curcdd=cur, prccd=px0 + k, prcod=px0,
                     prchd=px0 + 2, prcld=px0 - 1, cshtrd=1000.0, ajexdi=1.0, trfd=1.0, cshoc=5e6, fic="USA")
            use = [c for c in self.cols]
            self.con.execute(f"insert into comp_na_daily_all.secd ({', '.join(use)}, junk) values ({', '.join('?' * len(use))}, 1.0)",
                             [v[c] for c in use])

    def alias(self):
        self.con.execute("create schema comp")
        self.con.execute("create view comp.secd as select * from comp_na_daily_all.secd")

    def describe_table(self, library, table):
        if self.fail_describe:
            raise RuntimeError("boom")
        return self.con.execute("select column_name as name, data_type as type from information_schema.columns "
                                f"where table_schema = '{library}' and table_name = '{table}'").df()

    def raw_sql(self, sql, date_cols=None):
        self.sql.append(sql)
        return self.con.execute(sql).df()


def world(**kw):
    db = FakeDb(**kw)
    days = ["2025-09-24", "2025-09-25", "2025-09-26"]
    db.add("001", "01", "AMD", "ADVANCED MICRO DEVICES", days, 100.0)
    db.add("001", "01", "AMD", "ADVANCED MICRO DEVICES", ["2025-09-26"], 140.0, cur="CAD")    # not USD: must be excluded
    db.add("002", "01", "ORCL", "ORACLE CORP", days, 200.0)
    db.add("010", "01", "SNDK", "SANDISK OLD", ["2024-02-01", "2024-03-01"], 50.0)
    db.add("020", "01", "SNDK", "SANDISK CORP", days, 60.0)
    db.add("030", "01", "HXSK", "SK HYNIX INC", days, 70.0)
    return db


# ------------------------------------------------------------------------------------------------ SQL builders
def test_secd_sql_text_and_guards():
    sql = secd_sql(["gvkey", "iid", "datadate", "tic", "prccd", "curcdd"], gvkey_iids=[("20", "01")], tickers=["amd", "ORCL"], start="2024-05-01")
    for s in ("from comp_na_daily_all.secd", "datadate >= DATE '2024-05-01'", "tic in ('AMD', 'ORCL')", "(gvkey = '20' and iid = '01')",
              "curcdd = 'USD'", "order by gvkey, iid, datadate"):
        assert s in sql, s
    assert "curcdd" not in secd_sql(["gvkey", "tic"], tickers=["AMD"])                      # no column, no filter
    assert config.COMP_SECD == "comp_na_daily_all.secd" and "secd" not in QUERIES
    for bad in (dict(tickers=["AMD'; drop table x; --"]), dict(gvkey_iids=[("1' or '1'='1", "01")]), dict(tickers=["AMD"], start="2024-1-1"),
                dict(tickers=[]), dict()):
        with pytest.raises(ValueError):
            secd_sql(["gvkey"], **bad)
    assert secd_candidates_sql(["gvkey", "conm"], tickers=["AMD"]).count("group by") == 1


def test_secd_columns_validation():
    cols, absent = secd_columns(ALL + ["junk"])
    assert absent == [] and cols == config.SECD_REQUIRED + config.SECD_OPTIONAL and "junk" not in cols
    cols, absent = secd_columns(["gvkey", "iid", "datadate", "tic", "ajexdi", "weird"])
    assert absent == ["prccd"] and cols[-1] == "ajexdi"


def test_secd_sql_runs_in_duckdb_usd_only():
    db = world()
    cols, _ = secd_columns(db.cols)
    out = db.con.execute(secd_sql(cols, tickers=["AMD", "SNDK"], start="2025-01-01")).df()
    assert list(out.columns) == cols
    assert set(zip(out["gvkey"], out["tic"])) == {("001", "AMD"), ("020", "SNDK")} and set(out["curcdd"]) == {"USD"} and len(out) == 6
    c = db.con.execute(secd_candidates_sql(cols, tickers=["SNDK"], start="2024-01-01")).df()
    assert sorted(c["gvkey"]) == ["010", "020"] and list(c["n"]) == [2, 3]


# ------------------------------------------------------------------------------------------------ extraction script
def test_extract_writes_one_usd_parquet_with_datetime(tmp_path, capsys):
    db = world()
    out = tmp_path / "secd.parquet"
    assert ext.main(["AMD", "orcl", "NOPE", "--start", "2025-01-01", "--out", str(out)], db=db) == 0
    df = pd.read_parquet(out)
    assert str(df["datadate"].dtype).startswith("datetime64") and set(df["tic"]) == {"AMD", "ORCL"} and len(df) == 6
    assert set(df["curcdd"]) == {"USD"} and "junk" not in df.columns
    text = capsys.readouterr().out
    assert "no match since 2025-01-01: NOPE" in text
    assert "AMD" in text and "last close 102.0000" in text and "2025-09-24 .. 2025-09-26" in text


def test_extract_refuses_ambiguous_ticker_and_accepts_disambiguation(tmp_path, capsys):
    out = tmp_path / "secd.parquet"
    with pytest.raises(SystemExit) as e:
        ext.main(["SNDK", "AMD", "--start", "2024-01-01", "--out", str(out)], db=world())
    msg = str(e.value)
    assert "REFUSED" in msg and "010:01" in msg and "020:01" in msg and "SANDISK OLD" in msg and "--gvkey-iid" in msg
    assert not out.exists()
    assert ext.main(["SNDK", "AMD", "--start", "2024-01-01", "--gvkey-iid", "020:01", "--out", str(out)], db=world()) == 0
    df = pd.read_parquet(out)
    assert set(df.loc[df["tic"] == "SNDK", "gvkey"]) == {"020"} and set(df["tic"]) == {"SNDK", "AMD"}
    with pytest.raises(SystemExit):                     # a pair that is not one of the candidates does not disambiguate
        ext.main(["SNDK", "--start", "2024-01-01", "--gvkey-iid", "999:01", "--out", str(out)], db=world())
    with pytest.raises(SystemExit):
        ext.parse_pair("020")


def test_extract_validates_required_columns_and_optional_curcdd(tmp_path, capsys):
    with pytest.raises(SystemExit, match="required columns absent.*prccd"):
        ext.main(["AMD", "--out", str(tmp_path / "x.parquet")], db=world(drop=["prccd"]))
    db = world(drop=["curcdd", "ajexdi", "trfd", "fic"])
    out = tmp_path / "y.parquet"
    ext.main(["AMD", "--start", "2025-01-01", "--out", str(out)], db=db)
    df = pd.read_parquet(out)
    assert "curcdd" not in df.columns and "ajexdi" not in df.columns and len(df) == 4       # CAD row kept: the column is absent
    assert "no curcdd column" in capsys.readouterr().out


def test_extract_default_ticker_list_and_documented():
    assert config.SECD_DEFAULT_TICKERS == ["ORCL", "INTU", "LITE", "BE", "STX", "AMD", "CELH", "ALAB", "SNDK", "SKHY", "AXTI",
                                          "SNDQ", "AXTQ", "DAMD", "SKDD"]
    assert "extract_secd.py" in (ROOT / "scripts" / "wrds_extract.py").read_text().split('"""')[1]


def test_extract_output_feeds_prices_module(tmp_path):
    from qap import prices
    out = tmp_path / "s.parquet"
    ext.main(["AMD", "ORCL", "--start", "2025-01-01", "--out", str(out)], db=world())
    d, diag = prices.load_prices(out)
    assert prices.last_price(d, "AMD") == (102.0, pd.Timestamp("2025-09-26")) and diag["ORCL"]["rows"] == 3


# ------------------------------------------------------------------------------------------------ probe
def test_probe_sql_builders_run_and_are_guarded():
    db = world()
    cols = db.cols
    r = db.con.execute(probe.lookup_sql("comp_na_daily_all.secd", ["SNDK", "AMD"], cols)).df()
    assert list(r["tic"]) == sorted(r["tic"])
    assert set(zip(r["tic"], r["gvkey"], r["curcdd"])) == {("AMD", "001", "USD"), ("AMD", "001", "CAD"), ("SNDK", "010", "USD"), ("SNDK", "020", "USD")}
    w = db.con.execute(probe.words_sql("comp_na_daily_all.secd", ["hynix", "ASTERA"], cols)).df()
    assert list(w["tic"]) == ["HXSK"]
    last = db.con.execute(probe.last_rows_sql("comp_na_daily_all.secd", "020", "01", cols, start="2025-01-01", n=2)).df()
    assert list(last.columns) == probe.ROW_COLS and len(last) == 2 and last["datadate"].iloc[0] > last["datadate"].iloc[1]
    with pytest.raises(SystemExit):
        probe.lookup_sql("comp_na_daily_all.secd", ["AMD') or ('1"], cols)
    assert "group by gvkey, iid, tic, conm, tpci, exchg, curcdd" in probe.lookup_sql("t", ["A"], cols)


def test_probe_formatting_helpers():
    df = pd.DataFrame({"tic": ["AMD", "SNDK"], "gvkey": ["1", "2"], "iid": ["01", "01"], "px": [1.23456789, float("nan")],
                       "d": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]})
    lines = probe.fmt_frame(df)
    assert lines[0].split() == ["tic", "gvkey", "iid", "px", "d"] and "1.23457" in lines[1] and "2025-01-03" in lines[2]
    assert probe.ticker_groups(pd.DataFrame({"tic": ["SNDK", "SNDK", "AMD"], "gvkey": ["1", "2", "3"], "iid": ["01"] * 3})) == \
        {"SNDK": ["1:01", "2:01"], "AMD": ["3:01"]}
    assert probe.try_sql(object(), "select 1").startswith("ERROR AttributeError")


@pytest.mark.parametrize("fail_describe", [False, True])
def test_probe_main_report(tmp_path, capsys, fail_describe):
    db = world(fail_describe=fail_describe)
    db.alias()
    out = tmp_path / "p" / "secd_probe.txt"
    assert probe.main(["AMD", "SNDK", "SKHY", "ORCL", "--words", "HYNIX"], db=db, out=str(out)) == 0
    text = out.read_text()
    assert text.strip() == capsys.readouterr().out.strip()
    for must in ("## comp_na_daily_all.secd", "## comp.secd", "max(datadate) (restricted to datadate >= 2025-06-01): 2025-09-26",
                 "min(datadate): 2024-02-01", "rows with datadate >= 2025-08-27", "AMBIGUOUS SNDK: 2 securities (010:01, 020:01)",
                 "NO MATCH for ticker SKHY", "company-name search", "SK HYNIX INC", "LAST 5 ROWS", "2024-03-01", "prccd", "ajexdi", "trfd"):
        assert must in text, must
    assert ("describe_table failed" in text) == fail_describe and "gvkey:" in text
    assert "NO MATCH for ticker AMD" not in text


def test_probe_main_survives_failing_queries(tmp_path):
    class Dead:
        def describe_table(self, **k):
            raise RuntimeError("denied")

        def raw_sql(self, sql, date_cols=None):
            raise RuntimeError("permission denied")
    out = tmp_path / "o.txt"
    assert probe.main(["AMD"], db=Dead(), out=str(out)) == 0
    t = out.read_text()
    assert "ERROR RuntimeError" in t and "no secd table answered" in t
