"""Probe the tables that matter for the next research round: columns, exact row count, date range, 3 example rows.

    WRDS_USERNAME=you python scripts/wrds_probe.py
Writes data/wrds_probe.txt (git-ignored; paste it back). Every query has a 90 s statement timeout and its own
try/except, so one slow or forbidden table cannot stop the run. Example rows are only the first 3 rows of each table.
"""
import pathlib

from qap.wrds_client import connect

TARGETS = {
    # Cboe: looks like an equity-option database (implied vols, borrow rates, historical vol, one table per year)
    "cboe": ["eqmaster", "eqprice", "eqhvol", "ivborrowrate", "ivcmdr", "ivcmpr", "ivexdr", "ivexpr", "deltanumber",
             "eqfactor", "eqsplitdiv", "ivlisted_2019", "ivlisted_2024"],
    # contributed academic data
    "contrib_global_factor": ["global_factor", "ctff_chars", "ctff_features", "ctff_daily_ret"],
    "contrib_intangible_value": ["int_xsec", "int_factors"],
    "contrib_kpss": ["kpss_patents", "kpss_permco_link"],
    "contrib_patent_firm_link": ["patent_level", "firm_year_panel"],
    "contrib_char_returns": ["char_returns"],
    "contrib_general": ["better_beta", "liva", "arc", "classified_boards", "marginal_tax", "as_firm_risks", "blocksv",
                        "common_own_industry", "factors", "shale"],
    "contrib_corporate_culture": ["corp_culture"],
    "contrib_corp_fed_litigation": ["firmyear_level"],
    "macrofin": ["q_factors_monthly", "cay", "macro_uncertainty", "financial_uncertainty"],
    "rq_all": ["rq_data"],
    "wrdssec_midas": ["security"],
    "otc_endofday": ["endofday"],
    "block_all": ["block"],
    "wrdsapps": ["bondret"],
    "contrib_bond_firm_link": ["fang_link"],
}
DATE_NAMES = ("date", "datadate", "yyyymm", "year", "fyear", "grantdate", "caldt", "mthcaldt", "eom", "month")


def q(db, sql):
    try:
        return db.raw_sql(sql)
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}: {str(e)[:160]}"


def main() -> None:
    db = connect()
    q(db, "set statement_timeout = 90000")
    out = []

    def log(m=""):
        print(m, flush=True)
        out.append(str(m))

    libs = set(db.list_libraries())
    if "cboe" in libs:
        names = sorted(db.list_tables(library="cboe"))
        log(f"# cboe: {len(names)} tables: " + ", ".join(names))
    for lib, tables in TARGETS.items():
        for t in tables:
            full = f"{lib}.{t}"
            log(f"\n## {full}")
            try:
                d = db.describe_table(library=lib, table=t)
            except Exception as e:  # noqa: BLE001
                log(f"  describe failed: {type(e).__name__}: {str(e)[:120]}")
                continue
            cols = list(d["name"])
            log(f"  {len(cols)} columns: " + ", ".join(f"{r.name}:{str(r.type).split('(')[0]}" for r in d.itertuples())[:3500])
            cnt = q(db, f"select count(*) as n from {full}")
            log("  rows: " + (f"{int(cnt['n'].iloc[0]):,}" if not isinstance(cnt, str) else cnt))
            dc = [c for c in cols if c.lower() in DATE_NAMES] or [r.name for r in d.itertuples() if "date" in str(r.type).lower()]
            if dc:
                r = q(db, f"select min({dc[0]}) as lo, max({dc[0]}) as hi from {full}")
                log(f"  {dc[0]} range: " + (f"{r['lo'].iloc[0]} .. {r['hi'].iloc[0]}" if not isinstance(r, str) else r))
            ex = q(db, f"select * from {full} limit 3")
            if not isinstance(ex, str):
                for row in ex.to_dict("records"):
                    log("  eg: " + str({k: (str(v)[:30]) for k, v in list(row.items())[:14]}))
            else:
                log("  " + ex)
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/wrds_probe.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
