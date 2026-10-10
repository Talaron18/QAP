"""Point-in-time availability dates for Compustat fundamentals (no Snapshot licence).

Conservative lags per report section 7.4. Join fundamentals to the monthly panel on
availability date, never on fiscal period end.
"""
import pandas as pd


def annual_available_date(datadate: pd.Series, months: int = 4) -> pd.Series:
    """Fiscal year-end + `months` (default 4, to cover 10-K deadlines)."""
    return pd.to_datetime(datadate) + pd.DateOffset(months=months)


def quarterly_available_date(datadate: pd.Series, rdq: pd.Series, fallback_days: int = 60,
                             max_gap_days: int = 180) -> pd.Series:
    """Report date plus one day when rdq is usable, else fiscal period end + fallback_days.

    rdq is unusable if missing, before the period end, or more than `max_gap_days` after it.
    """
    dd = pd.to_datetime(datadate)
    r = pd.to_datetime(rdq)
    ok = r.notna() & (r >= dd) & (r <= dd + pd.Timedelta(days=max_gap_days))
    return (r + pd.Timedelta(days=1)).where(ok, dd + pd.Timedelta(days=fallback_days))
