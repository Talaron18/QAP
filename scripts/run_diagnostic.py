"""Conversion diagnostic (pre-registration v4 b): quintile / EW-vs-VW / cost / size-tercile decomposition. INFORMATION ONLY.

    python scripts/run_diagnostic.py [--jobs N]
Needs the Study A inputs (data/raw incl. jkp_us). Writes data/diagnostic_report.txt. Uses months < 2020 only; logs no trials.
"""
import argparse
import pathlib

from qap.diagnostics import diagnose_signal, format_report
from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.jkp import build_theme_signals
from qap.parallel import context, pmap, set_context
from qap.study import eligible_universe, load_context, require_inputs

NAMES = ["gpa", "jkp_all13", "jkp_slow10", "Quality", "Profitability", "Value", "Momentum"]
JKP_DIR = "data/raw/jkp_us"
BANNER = "INFORMATION ONLY, 2020+ not used, no trials logged"


def _one(name: str) -> dict:
    c = context()
    return diagnose_signal(c["sigs"]["gpa" if name == "gpa" else "themes"], c["panel"], c["inp"], c["aum_t"], name)


def main(jobs: int) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, jobs)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/diagnostic_report.txt").write_text("\n".join(report) + "\n")


def _run(log, jobs) -> None:
    require_inputs([JKP_DIR], "Run: python scripts/wrds_extract.py jkp_us")
    log(BANNER)
    panel, links, funda, inp, aum_t = load_context()
    sig_g = build_research_signals(panel, prepare_annual(funda), links)[["permno", "mi", "me", "me_co", "primaryexch", "gpa"]]
    sig_g = sig_g.astype({"permno": "int64"})
    themes, _ = build_theme_signals(eligible_universe(panel), JKP_DIR)
    themes = themes.astype({"permno": "int64"})
    set_context(sigs={"gpa": sig_g, "themes": themes}, panel=panel[["permno", "mi", "mthret_adj"]], inp=inp, aum_t=aum_t)
    log(f"signals: {', '.join(NAMES)}; costs 1.5x, $10M (2019 market-size); dev 1990-2014, validation 2015-2019")
    log("gpa uses its own research-signal universe (as in Studies A/B); JKP composites and themes use the JKP-merged eligible universe\n")
    for line in format_report(pmap(_one, NAMES, jobs, say=log)):
        log(line)
    log(BANNER)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    main(ap.parse_args().jobs)
