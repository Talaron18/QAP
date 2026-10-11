"""SQL builders for WRDS extraction. Pure functions; run them with scripts/wrds_extract.py."""
import datetime as dt
from typing import Sequence

from qap import config


def _select(cols: Sequence[str], table: str, where: str = "", order: str = "") -> str:
    sql = f"select {', '.join(cols)} from {table}"
    if where:
        sql += f" where {where}"
    if order:
        sql += f" order by {order}"
    return sql


def crsp_monthly_sql(start: str = "1963-01-01") -> str:
    """All CIZ monthly rows (no equity filter: filter in pandas once flag codes are confirmed)."""
    return _select(config.CRSP_MONTHLY_COLS, config.CRSP_MONTHLY, f"mthcaldt >= '{start}'", "permno, mthcaldt")


def crsp_delistings_sql() -> str:
    return _select(config.CRSP_DELISTINGS_COLS, config.CRSP_DELISTINGS, order="permno, delistingdt")


def ccm_links_sql() -> str:
    return _select(config.CCM_LINK_COLS, config.CCM_LINKS, order="gvkey, linkdt")


def compustat_annual_sql(start: str = "1960-01-01") -> str:
    """Standard industrial, consolidated, domestic-population annual fundamentals."""
    where = f"indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' and datadate >= '{start}'"
    return _select(config.COMPUSTAT_ANNUAL_COLS, config.COMPUSTAT_ANNUAL, where, "gvkey, datadate")


def compustat_quarterly_sql(start: str = "1960-01-01") -> str:
    where = f"indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' and datadate >= '{start}'"
    return _select(config.COMPUSTAT_QUARTERLY_COLS, config.COMPUSTAT_QUARTERLY, where, "gvkey, datadate")


def total_q_sql() -> str:
    """Peters-Taylor Total Q and intangible capital (small table)."""
    return _select(config.TOTAL_Q_COLS, config.TOTAL_Q, order="gvkey, datadate")


def ff_monthly_sql() -> str:
    return _select(["date", "mktrf", "smb", "hml", "umd", "rf"], config.FF_FACTORS_MONTHLY, order="date")


_CS_K = "0.1715728752538097"  # 3 - 2*sqrt(2)


def crsp_daily_agg_sql(start: str, end: str, overlap_days: int = 14) -> str:
    """Per (permno, month) liquidity inputs from daily CIZ data, aggregated on the WRDS server.

    start/end are inclusive ISO dates for the OUTPUT rows. The daily scan is widened by
    `overlap_days` each side so lag()/lead() windows (Corwin-Schultz, Abdi-Ranaldo) are complete at
    chunk edges. Portable SQL (runs unchanged on Postgres and DuckDB); every ln/sqrt/division sits
    behind a CASE guard so one bad row cannot abort a chunk.

    Columns: n_ret, sigma_d (daily return stdev), n_q/qspread (mean quoted relative spread where
    0 < bid < ask and spread < 50% of mid), n_cs/cs_spread (Corwin-Schultz, negatives floored at 0),
    n_ar/ar_x_mean (mean Abdi-Ranaldo cross-product; spread = sqrt(max(4*mean, 0))),
    adv_a = mean dlyprcvol, adv_b = mean |dlyprc| * dlyvol.
    """
    d0 = dt.date.fromisoformat(start) - dt.timedelta(days=overlap_days)
    d1 = dt.date.fromisoformat(end) + dt.timedelta(days=overlap_days)
    k = _CS_K
    return f"""
with d as (
  select permno, dlycaldt, dlyret, dlybid, dlyask, dlyvol, dlyprcvol,
         abs(dlyprc) as prc,
         case when dlyclose > 0 then dlyclose when abs(dlyprc) > 0 then abs(dlyprc) end as px,
         case when dlyhigh > 0 and dlylow > 0 and dlyhigh >= dlylow then cast(dlyhigh as double precision) end as hi,
         case when dlyhigh > 0 and dlylow > 0 and dlyhigh >= dlylow then cast(dlylow as double precision) end as lo
  from crsp.dsf_v2
  where dlycaldt >= DATE '{d0.isoformat()}' and dlycaldt <= DATE '{d1.isoformat()}'
),
e as (
  select d.*,
         case when hi is not null and px > 0 then (ln(hi) + ln(lo)) / 2.0 end as eta,
         case when px > 0 then ln(cast(px as double precision)) end as lpx
  from d
),
w as (
  select e.*,
         lag(hi) over (partition by permno order by dlycaldt) as hi0,
         lag(lo) over (partition by permno order by dlycaldt) as lo0,
         lag(dlycaldt) over (partition by permno order by dlycaldt) as dt0,
         lead(eta) over (partition by permno order by dlycaldt) as eta1,
         lead(dlycaldt) over (partition by permno order by dlycaldt) as dt1
  from e
),
b as (
  select w.*,
         case when hi is not null and hi0 is not null and (dlycaldt - dt0) <= 5
              then ln(hi / lo) * ln(hi / lo) + ln(hi0 / lo0) * ln(hi0 / lo0) end as beta,
         case when hi is not null and hi0 is not null and (dlycaldt - dt0) <= 5
              then ln(greatest(hi, hi0) / least(lo, lo0)) * ln(greatest(hi, hi0) / least(lo, lo0)) end as gamma
  from w
),
c as (
  select b.*,
         case when beta is not null and gamma is not null
              then (sqrt(2.0 * beta) - sqrt(beta)) / {k} - sqrt(gamma / {k}) end as alpha
  from b
),
f as (
  select c.*,
         cast(extract(year from dlycaldt) as integer) as yr,
         cast(extract(month from dlycaldt) as integer) as mo,
         case when alpha is not null and alpha > -30 and alpha < 30
              then greatest(2.0 * (exp(alpha) - 1.0) / (1.0 + exp(alpha)), 0.0) end as cs,
         case when dlybid > 0 and dlyask > dlybid and (dlyask - dlybid) < 0.5 * ((dlyask + dlybid) / 2.0)
              then (dlyask - dlybid) / ((dlyask + dlybid) / 2.0) end as qs,
         case when lpx is not null and eta is not null and eta1 is not null and (dt1 - dlycaldt) <= 5
              then (lpx - eta) * (lpx - eta1) end as ar_x
  from c
)
select permno, yr * 100 + mo as yyyymm,
       count(dlyret) as n_ret, stddev_samp(dlyret) as sigma_d,
       count(qs) as n_q, avg(qs) as qspread,
       count(cs) as n_cs, avg(cs) as cs_spread,
       count(ar_x) as n_ar, avg(ar_x) as ar_x_mean,
       avg(dlyprcvol) as adv_a, avg(prc * dlyvol) as adv_b
from f
where dlycaldt >= DATE '{start}' and dlycaldt <= DATE '{end}'
group by permno, yr, mo
order by permno, yr, mo
"""


def jkp_us_sql(columns: Sequence[str], year: int) -> str:
    """US CRSP-sourced stock-months of one calendar year from the JKP Global Factor Data, JKP screens applied.

    columns: characteristic columns that exist in the table (validated by the caller against describe_table).
    Months are keyed by eom (information available at that month-end); size_grp micro/nano are dropped.
    """
    groups = ", ".join(f"'{g}'" for g in config.JKP_SIZE_GROUPS)
    cols = ", ".join(["cast(permno as bigint) as permno", "eom", "size_grp", "ret_exc_lead1m", "me", *columns])
    return (f"select {cols} from {config.JKP_TABLE} where excntry = 'USA' and permno is not null and common = 1 "
            f"and primary_sec = 1 and exch_main = 1 and obs_main = 1 and size_grp in ({groups}) "
            f"and eom >= DATE '{year}-01-01' and eom <= DATE '{year}-12-31' order by permno, eom")


def rq_sql() -> str:
    return _select(["gvkey", "fyear", "loc", "sic", "xrd", "rawrq", "se_rawrq", "rdintensity"], config.RQ_TABLE, order="gvkey, fyear")


def kpss_sql() -> str:
    return _select(["permno", "filing_date", "issue_date", "patent_num", "xi_real", "xi_nominal", "cites"], config.KPSS_TABLE,
                   "permno is not null and issue_date is not null", "permno, issue_date")


def patent_grants_sql() -> str:
    """Patents per (gvkey, grant year), aggregated on the server (the firm-year panel's `year` is ambiguous)."""
    return ("select gvkey, cast(grant_year as integer) as grant_year, count(*) as n_patents, sum(num_cite_3y) as cites3y_sum, "
            f"count(num_cite_3y) as n_cite3y from {config.PATENT_LEVEL_TABLE} "
            "where gvkey is not null and trim(gvkey) <> '' and trim(gvkey) <> '.' and grant_year is not null "
            "group by gvkey, cast(grant_year as integer) order by gvkey, grant_year")


def crsp_names_sql() -> str:
    """CRSP name history (ticker and security/issuer names by date range); select * because column names vary by CIZ release."""
    return f"select * from {config.CRSP_NAMES}"


def comp_secm_sql(columns: Sequence[str], start: str = "2022-01-01") -> str:
    """US common-stock (tpci '0') monthly security rows from Compustat from `start`; fic/curcdm filters only if the columns exist."""
    where = f"datadate >= DATE '{start}' and tpci = '0'"
    if "fic" in columns:
        where += " and fic = 'USA'"
    if "curcdm" in columns:
        where += " and curcdm = 'USD'"
    return _select(list(columns), config.COMP_SECM, where, "gvkey, iid, datadate")


QUERIES = {
    "crsp_names": crsp_names_sql,
    "crsp_monthly": crsp_monthly_sql,
    "crsp_delistings": crsp_delistings_sql,
    "ccm_links": ccm_links_sql,
    "compustat_annual": compustat_annual_sql,
    "compustat_quarterly": compustat_quarterly_sql,
    "totalq": total_q_sql,
    "rq_data": rq_sql,
    "kpss_patents": kpss_sql,
    "patent_grants": patent_grants_sql,
    "ff_monthly": ff_monthly_sql,
}
