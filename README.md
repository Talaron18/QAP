# QAP

Quantitative + probabilistic US equity research stack. Spec: [docs/research_report.md](docs/research_report.md); plan: [ROADMAP.md](ROADMAP.md).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest
```

## WRDS extraction (run on your machine)

WRDS data is licensed: it stays in git-ignored `data/` and is never committed.

```bash
pip install -e .[wrds]
export WRDS_USERNAME=your_id
python scripts/wrds_inspect.py   # writes data/wrds_schema.txt; share it so loaders use the real schema
python scripts/wrds_extract.py   # writes data/raw/*.parquet (CRSP monthly, CCM links, Compustat, FF)
```

## Python version

`pyqlib` had no installable distribution on Python 3.13 in the dev container (pip found no
matching version). Qlib is not a dependency yet; when you add it, check its wheel tags on PyPI and
use a supported Python (probably 3.12 or earlier, unverified) in your local venv.
