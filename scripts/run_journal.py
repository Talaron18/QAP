"""Personal trade-journal analyzer. DESCRIPTIVE ONLY: not a pre-registered study, no trial registry, no alpha claim.

    python scripts/run_journal.py data/journal/*.txt [--fills-csv f.csv] [--log-tz Asia/Shanghai] [--fix-years]
        [--vol SNDK=0.70 --vol AMD=0.55 ...] [--default-vol 0.60] [--corr 0.5] [--tax-st 0.408] [--tax-lt 0.238]
        [--start-capital 35000] [--prices TICKER=PRICE ...] [--meta data/journal/meta.csv]
        [--prices-file data/raw/secd_tickers.parquet]   (Compustat daily prices from scripts/extract_secd.py: current prices, realised vols, section 4b)
Inputs live under data/journal/ (git-ignored; never commit real positions). Writes data/journal_report.txt and
data/journal/roundtrips.csv; deterministic, idempotent.
"""
import sys

from qap.journal import main

if __name__ == "__main__":
    sys.exit(main())
