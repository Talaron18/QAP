# QAP

Quantitative + probabilistic US equity research stack. Spec: [docs/research_report.md](docs/research_report.md); plan: [ROADMAP.md](ROADMAP.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]   # dev extras: pytest, duckdb (checks the extraction SQL)
pytest
```

## WRDS extraction (run on your machine)

WRDS data is licensed: it stays in git-ignored `data/` and is never committed.

```bash
pip install -e .[wrds]
export WRDS_USERNAME=your_id
python scripts/wrds_inspect.py   # writes data/wrds_schema.txt; share it so loaders use the real schema
python scripts/wrds_extract.py   # writes data/raw/*.parquet (CRSP monthly, CCM links, Compustat, FF)
python scripts/check_raw.py      # writes data/raw_report.txt: counts and distributions only
python scripts/run_signals.py    # FF replication gates, then Tier-1 signals; writes data/signals_report.txt
python scripts/wrds_extract.py crsp_daily_agg   # server-side daily liquidity aggregation, per year, resumable
python scripts/run_costs.py      # cost model sanity checks, then costed backtests; writes data/costs_report.txt
python scripts/wrds_extract.py totalq     # Peters-Taylor intangible capital (small)
python scripts/run_candidates.py # pre-registered candidates vs gpa; writes data/candidates_report.txt (2020+ sealed)
python scripts/run_tilt.py       # pre-registered long-only gpa tilt (3 specs); writes data/tilt_report.txt (2020+ sealed)
```

## Python version

`pyqlib` had no installable distribution on Python 3.13 in the dev container (pip found no
matching version). Qlib is not a dependency yet; when you add it, check its wheel tags on PyPI and
use a supported Python (probably 3.12 or earlier, unverified) in your local venv.
