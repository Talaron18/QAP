"""Settle the frozen v8 ML ledger against realised Compustat returns (ROADMAP 'Pre-registration v8').

    python scripts/wrds_extract.py comp_secm_hist       # refresh, then
    python scripts/ml_settle.py
Verifies every ledger file against paper_ledger/ml_manifest.jsonl, settles each formation whose next month is a COMPLETE month in
comp_secm_hist, and reports per month and per group (pseudo-forward = formation 2025-12..2026-08; prospective = formed within 7 days of the
target month's start; retrospective = anything else) the rank IC and the VW top-quintile minus VW eligible universe, gross and net of 10 bp per
one-way turnover, for ML, the equal-weight composite and gpa. Checkpoints at 12/24/36 forward months use the first months in time order,
EXCLUDING retrospective ones; the DECISION is read only at 36 months. Writes data/ml_settled.csv and data/ml_report.txt.
"""
import pathlib

import numpy as np
import pandas as pd

import qap.mlcheck as ml
from qap.paper import build_pseudo_panel, complete_months
from qap.study import require_inputs

RAW = pathlib.Path("data/raw")


def _f(mean, t, scale=100.0, nd=3):
    return f"{mean * scale:+.{nd}f} (t {t:+.1f})" if np.isfinite(mean) else "n/a"


def stats_lines(m: pd.DataFrame, indent="  ") -> list:
    s = ml.summarize(m)
    out = [f"{indent}n={s['n']}"]
    for k, name in (("score_ml", "ML"), ("score_comp", "composite"), ("gpa", "gpa")):
        out.append(f"{indent}{name:<10} IC {s[k]['ic']:+.4f} (t {s[k]['ic_t']:+.1f}); L-U %/mo gross {_f(s[k]['gross'], s[k]['gross_t'])}, net {_f(s[k]['net'], s[k]['net_t'])}; "
                   f"one-way turnover {s[k]['turn']:.2f}")
    out.append(f"{indent}paired net: ML - composite {_f(s['ml_minus_comp']['mean'], s['ml_minus_comp']['t'])}; ML - gpa {_f(s['ml_minus_gpa']['mean'], s['ml_minus_gpa']['t'])}")
    return out


def main() -> None:
    require_inputs([RAW / "comp_secm_hist.parquet", ml.manifest_path()], "Run: python scripts/wrds_extract.py comp_secm_hist  and  python scripts/ml_form.py")
    out = []
    log = lambda s="": (print(s), out.append(str(s)))
    verified = ml.load_verified()                       # asserts every sha
    if not verified:
        log("No frozen formation yet.")
        return
    secm = pd.read_parquet(RAW / "comp_secm_hist.parquet")
    p = build_pseudo_panel(secm, 1.0)                  # scale irrelevant: only returns are used
    last_mi = max(complete_months(p))
    if int(p["mi"].max()) != last_mi:
        log(f"NOTE: latest comp.secm month is partial and ignored; settling through {ml.ym(last_mi)}")
    ret = ml.return_table(p, last_mi)
    info, parts = {}, []
    for man, form in verified:
        if man["mi"] + 1 > last_mi:
            continue
        info[man["mi"]] = (man["ym"], ml.form_group(man["mi"], man["formed_on"]), man["formed_on"])
        parts.append(form.assign(permno=ml.pseudo_permno(form["gvkey"], form["iid"])))
    if not parts:
        log("No formation has a realised complete next month yet.")
        return
    sc = pd.concat(parts, ignore_index=True)
    m = ml.evaluate_scores(sc, ret)
    m["ym"] = [info[i][0] for i in m.index]
    m["group"] = [info[i][1] for i in m.index]
    m["formed_on"] = [info[i][2] for i in m.index]
    pathlib.Path("data").mkdir(exist_ok=True)
    m.to_csv("data/ml_settled.csv")
    log(f"== ML CHECK SETTLEMENT (ledger hashes verified; benchmark = VW eligible universe; net = gross - 10 bp x one-way turnover; first month = full build) ==")
    for i, r in m.iterrows():
        log(f"  form {r['ym']} -> {ml.ym(i + 1)} [{r['group']}]: " + "; ".join(
            f"{n} IC {r[f'ic_{k}']:+.3f} net {r[f'net_{k}'] * 100:+.2f}%" for k, n in (("score_ml", "ML"), ("score_comp", "comp"), ("gpa", "gpa"))))
    for g in ("pseudo-forward", "prospective", "retrospective"):
        d = m[m["group"] == g]
        if len(d):
            log(f"\n{g}:")
            for ln in stats_lines(d):
                log(ln)
    fwd = m[m["group"] != "retrospective"].sort_index()
    nretro = int((m["group"] == "retrospective").sum())
    if nretro:
        log(f"\nWARNING: {nretro} retrospective month(s) (formed after the target month began) are excluded from the checkpoints and the decision.")
    log("\n== CHECKPOINTS (first k non-retrospective forward months; interim readings never trigger deployment) ==")
    for k in (12, 24, 36):
        if len(fwd) >= k:
            log(f"after {k} months ({ml.ym(fwd.index[0])}..{ml.ym(fwd.index[k - 1])}):")
            for ln in stats_lines(fwd.iloc[:k]):
                log(ln)
        else:
            log(f"after {k} months: not reached (have {len(fwd)})")
    log(f"\nDECISION (first 36 non-retrospective forward months; ML net L-U - composite >= +0.15 %/mo with paired t >= 2, and ML net mean > gpa net mean): "
        f"{ml.decision_label(fwd)}")
    log("Power note: with ~5% tracking error a 0.15 %/mo edge needs many years to confirm; 'NO EVIDENCE' is the expected outcome.")
    pathlib.Path("data/ml_report.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
