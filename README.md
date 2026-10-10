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
```
