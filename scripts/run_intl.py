"""International replication of the JKP findings (pre-registration v10, ROADMAP.md). DESCRIPTIVE: no promotion rule, no trial log.

    python scripts/wrds_extract.py jkp_intl     # once (per-year, resumable); jkp_us must exist already
    python scripts/run_intl.py [--jobs N]
Writes data/intl_report.txt.
"""
import argparse
import pathlib
import time

from qap import intl
from qap.study import require_inputs

INTL_DIR = "data/raw/jkp_intl"
US_DIR = "data/raw/jkp_us"
REPORT = "data/intl_report.txt"


def _missing(directory: str) -> list:
    f = pathlib.Path(directory, "_missing_columns.txt")
    return f.read_text().split() if f.exists() else []


def main(jobs: int = 1) -> dict:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    labels = {}
    try:
        labels = _run(log, jobs)
    finally:
        pathlib.Path(REPORT).parent.mkdir(exist_ok=True)
        pathlib.Path(REPORT).write_text("\n".join(report) + "\n")
    return labels


def _run(log, jobs: int) -> dict:
    t0 = time.time()
    require_inputs([INTL_DIR, US_DIR], "Run: python scripts/wrds_extract.py jkp_intl jkp_us")
    log(intl.BANNER)
    log("Series are indexed by RETURN month (formation month + 1); periods: dev 1990-2014, validation 2015-2019, later 2020-2025.")
    slim, info = intl.load_all(INTL_DIR, US_DIR, jobs=jobs)
    for line in intl.format_checks(info, {"jkp_intl": _missing(INTL_DIR), "jkp_us": _missing(US_DIR)}):
        log(line)
    log(f"scored stock-months kept: {len(slim):,}; loaded in {time.time() - t0:.0f}s")
    log()
    keep = [c for c in (*intl.BASE_COLS, *intl.SIGNALS)]
    series = intl.run_series(slim[keep], jobs=jobs, say=log)       # raw characteristic columns are dropped before the fork
    for line in intl.format_tables(series):
        log(line)
    lines, labels = intl.questions(series)
    for line in lines:
        log(line)
    log(f"\n{intl.BANNER}")
    log(f"elapsed {time.time() - t0:.0f}s")
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (fork) for loading and the per-(region, signal) jobs")
    main(ap.parse_args().jobs)
