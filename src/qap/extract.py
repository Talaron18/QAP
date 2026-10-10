"""SQL builders for WRDS extraction. Pure functions; run them with scripts/wrds_extract.py."""
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


def ccm_links_sql() -> str:
    return _select(config.CCM_LINK_COLS, config.CCM_LINKS, order="gvkey, linkdt")


def compustat_annual_sql(start: str = "1960-01-01") -> str:
    """Standard industrial, consolidated, domestic-population annual fundamentals."""
    where = f"indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' and datadate >= '{start}'"
    return _select(config.COMPUSTAT_ANNUAL_COLS, config.COMPUSTAT_ANNUAL, where, "gvkey, datadate")


def compustat_quarterly_sql(start: str = "1960-01-01") -> str:
    where = f"indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' and datadate >= '{start}'"
    return _select(config.COMPUSTAT_QUARTERLY_COLS, config.COMPUSTAT_QUARTERLY, where, "gvkey, datadate")


def ff_monthly_sql() -> str:
    return _select(["date", "mktrf", "smb", "hml", "umd", "rf"], config.FF_FACTORS_MONTHLY, order="date")


QUERIES = {
    "crsp_monthly": crsp_monthly_sql,
    "ccm_links": ccm_links_sql,
    "compustat_annual": compustat_annual_sql,
    "compustat_quarterly": compustat_quarterly_sql,
    "ff_monthly": ff_monthly_sql,
}
