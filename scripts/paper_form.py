"""Forward paper trading of the gpa tilt, formation step (ROADMAP 'Pre-registration v4', part a).

    python scripts/wrds_extract.py compustat_annual comp_secm      # refresh before every formation
    python scripts/paper_form.py [--from 2025-12] [--no-overlap]
Builds the Compustat-keyed universe and the spec-(c) tilt, prints the CRSP-overlap reproduction check FIRST (tolerance fixed in the
pre-registration: weight overlap >= 0.80 and active-return correlation >= 0.90, else nothing is written), then writes one frozen
ledger file per formation month under data/paper/forms/ (git-ignored licensed-data derivatives; existing files are never overwritten)
and appends its sha256 to paper_ledger/manifest.jsonl, which IS tracked: commit and push it before each month starts.
Writes data/paper_form_report.txt.
"""
import argparse
import datetime as dt
import pathlib
import subprocess

import numpy as np
import pandas as pd

from qap.fundamentals import prepare_annual
from qap.paper import (build_pseudo_panel, calibrate_secm_scale, complete_months, form_sha, manifest_line, manifest_path, month_form, overlap_stats,
                       research_signals)
from qap.portfolio import leg_gross_return
from qap.study import require_inputs
from qap.tilt import benchmark_weights, tilt_weights

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/paper")
OVERLAP_FROM, OVERLAP_TO = 2024 * 12, 2025 * 12 + 10          # formation months whose next-month CRSP return exists (<= 2025-12)
TOL_OVERLAP, TOL_CORR = 0.80, 0.90


def main(first: str, overlap: bool) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, first, overlap)
    finally:
        pathlib.Path("data").mkdir(exist_ok=True)
        pathlib.Path("data/paper_form_report.txt").write_text("\n".join(report) + "\n")


def _ym_to_mi(s: str) -> int:
    y, m = s.split("-")
    return int(y) * 12 + int(m) - 1


def _crsp_side(log):
    """CRSP-built tilt and benchmark for the overlap check, mapped to gvkey, plus the CRSP panel."""
    from qap.evaluate import build_research_signals
    from qap.panel import attach_gvkey
    from qap.study import load_context
    panel, links, funda, _, _ = load_context()
    sig = build_research_signals(panel, prepare_annual(funda), links)
    W, B = tilt_weights(sig, "gpa"), benchmark_weights(sig, "gpa")
    dates = panel[["permno", "mi", "mthcaldt"]]

    def to_gv(x):
        y = x.merge(dates, on=["permno", "mi"])
        return attach_gvkey(y, links, "mthcaldt")[["gvkey", "mi", "w"]]
    return panel, sig, W, B, to_gv(W), to_gv(B)


def _run(log, first, overlap) -> None:
    require_inputs([RAW / "comp_secm.parquet", RAW / "compustat_annual.parquet"],
                   "Run: python scripts/wrds_extract.py compustat_annual comp_secm")
    secm = pd.read_parquet(RAW / "comp_secm.parquet")
    funda = pd.read_parquet(RAW / "compustat_annual.parquet")
    scale, ratio = calibrate_secm_scale(secm, funda)
    pseudo = build_pseudo_panel(secm, scale)
    sig = research_signals(pseudo, prepare_annual(funda))
    W, B = tilt_weights(sig, "gpa"), benchmark_weights(sig, "gpa")
    full = complete_months(pseudo, "me")
    last_mi = max(full)
    if int(pseudo["mi"].max()) != last_mi:
        m = int(pseudo["mi"].max())
        log(f"NOTE: {m // 12}-{m % 12 + 1:02d} is a partial month in comp.secm ({int((pseudo['mi'] == m).sum())} rows) and is ignored; "
            f"last complete month {last_mi // 12}-{last_mi % 12 + 1:02d}")
    log("== CHECKS (printed before any forward result) ==")
    log(f"comp.secm rows {len(secm):,}; months {int(pseudo['mi'].min()) // 12}-{int(pseudo['mi'].min()) % 12 + 1:02d}"
        f"..{last_mi // 12}-{last_mi % 12 + 1:02d} (complete); ME scale {scale:g} (median ratio {ratio:.3f}); NYSE share of rows "
        f"{(pseudo['primaryexch'] == 'N').mean():.1%}")
    last = pseudo[pseudo["mi"] == last_mi]
    log(f"latest month: {len(last):,} issues, {int((last['eq']).sum()):,} with ME; trt1m present for {last['mthret_adj'].notna().mean():.1%}")
    g_last = sig[sig["mi"] == last_mi]
    log(f"latest month signal rows {len(g_last):,}, gpa present {g_last['gpa'].notna().mean():.1%}; tilt holdings {int((W['mi'] == last_mi).sum())}")
    ok = True
    if overlap:
        panel, csig, Wc, Bc, Wc_g, Bc_g = _crsp_side(log)
        Wg = W.merge(sig[["permno", "mi", "gvkey"]], on=["permno", "mi"])[["gvkey", "mi", "w"]]
        Bg = B.merge(sig[["permno", "mi", "gvkey"]], on=["permno", "mi"])[["gvkey", "mi", "w"]]
        months = [m for m in range(OVERLAP_FROM, OVERLAP_TO + 1) if m in set(Wg["mi"]) and m in set(Wc_g["mi"])]
        ow_w = pd.Series({m: overlap_stats(Wg, Wc_g, m) for m in months})
        ow_b = pd.Series({m: overlap_stats(Bg, Bc_g, m) for m in months})
        act_p = (leg_gross_return(W, pseudo) - leg_gross_return(B, pseudo)).reindex([m + 1 for m in months])
        act_c = (leg_gross_return(Wc, panel) - leg_gross_return(Bc, panel)).reindex([m + 1 for m in months])
        j = pd.concat([act_p.rename("p"), act_c.rename("c")], axis=1).dropna()
        corr = float(j["p"].corr(j["c"])) if len(j) > 3 else float("nan")
        j2 = csig[csig["mi"].isin(months)][["gvkey", "mi", "gpa"]].merge(sig[sig["mi"].isin(months)][["gvkey", "mi", "gpa"]],
                                                                         on=["gvkey", "mi"], suffixes=("_c", "_p"))
        rank_corr = float(j2.groupby("mi").apply(lambda t: t["gpa_c"].rank().corr(t["gpa_p"].rank()), include_groups=False).mean())
        log(f"CRSP overlap, formation months {months[0] // 12}-{months[0] % 12 + 1:02d}..{months[-1] // 12}-{months[-1] % 12 + 1:02d} ({len(months)}):")
        log(f"  tilt weight overlap (sum of min weights, by gvkey): mean {ow_w.mean():.3f}, min {ow_w.min():.3f}; benchmark overlap mean {ow_b.mean():.3f}")
        log(f"  gpa rank corr Compustat vs CRSP (shared gvkeys): mean {rank_corr:.3f}")
        log(f"  monthly tilt-vs-benchmark active return, Compustat vs CRSP: corr {corr:.3f}, mean diff {float((j['p'] - j['c']).mean()) * 100:+.3f} %/mo over {len(j)} months")
        for k, good in ((f"tilt weight overlap >= {TOL_OVERLAP}", ow_w.mean() >= TOL_OVERLAP),
                        (f"active-return correlation >= {TOL_CORR}", corr >= TOL_CORR)):
            log(f"  [{'PASS' if good else 'FAIL'}] {k}")
            ok &= bool(good)
    else:
        log("  CRSP overlap check SKIPPED (--no-overlap): the harness is NOT validated against the research")
    if not ok:
        raise SystemExit("Harness does not reproduce the CRSP research; nothing written (pre-registered tolerance).")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "forms").mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=pathlib.Path(__file__).resolve().parent).stdout.strip() or "unknown"
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    log(f"\n== LEDGER (spec commit {commit[:10]}, formed_on {today}) ==")
    for mi in range(_ym_to_mi(first), last_mi + 1):
        f = OUT / "forms" / f"{mi // 12}{mi % 12 + 1:02d}.parquet"
        form = month_form(W, B, sig, mi)
        n_spec = int((form["kind"] == "spec").sum())
        if n_spec == 0:
            log(f"  {mi // 12}-{mi % 12 + 1:02d}: no tilt holdings (skipped)")
            continue
        sha = form_sha(form)
        if f.exists():
            old = form_sha(pd.read_parquet(f))
            log(f"  {mi // 12}-{mi % 12 + 1:02d}: frozen file kept ({'identical' if old == sha else 'differs from recomputation: data revision'})")
            continue
        form.to_parquet(f, index=False)
        with open(manifest_path(), "a") as fh:
            fh.write(manifest_line(mi, sha, commit, today) + "\n")
        s = form[form["kind"] == "spec"]
        log(f"  {mi // 12}-{mi % 12 + 1:02d}: written, {n_spec} spec names (max w {s['w'].max():.3f}), "
            f"{int((form['kind'] == 'top50').sum())} top-50 names, sha {sha[:10]}")
    log("\nNext: git add paper_ledger && git commit && git push; settle with `python scripts/paper_settle.py` after the next month's comp.secm refresh.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", default="2025-12", help="first formation month (YYYY-MM)")
    ap.add_argument("--no-overlap", action="store_true", help="skip the CRSP reproduction check (harness then unvalidated)")
    a = ap.parse_args()
    main(a.first, not a.no_overlap)
