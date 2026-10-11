"""Settle the frozen paper-trading ledger against realised Compustat returns (ROADMAP 'Pre-registration v4', part a).

    python scripts/wrds_extract.py comp_secm       # refresh, then
    python scripts/paper_settle.py
Reports gross active return (spec tilt minus benchmark; the implementable top-50 variant too) per formation, split into
prospective (file formed within 7 days of the target month's start) and retrospective (formed later). Writes
data/paper/settled.csv and data/paper_report.txt. Sample sizes are small: read the t-stats with that in mind.
"""
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from qap.paper import build_pseudo_panel, form_sha, settle_form
from qap.study import require_inputs

RAW, OUT = pathlib.Path("data/raw"), pathlib.Path("data/paper")


def _stats(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) < 3:
        return f"n={len(s)}"
    return f"n={len(s)}, mean {s.mean() * 100:+.3f} %/mo, t {s.mean() / (s.std(ddof=1) / np.sqrt(len(s))):+.2f}, cum {((1 + s).prod() - 1) * 100:+.2f}%"


def main() -> None:
    require_inputs([RAW / "comp_secm.parquet", OUT / "manifest.jsonl"], "Run: python scripts/wrds_extract.py comp_secm  and  python scripts/paper_form.py")
    secm = pd.read_parquet(RAW / "comp_secm.parquet")
    p = build_pseudo_panel(secm, 1.0)                       # scale irrelevant: only returns are used
    ret = p.rename(columns={"mthret_adj": "ret"})[["gvkey", "iid", "mi", "ret"]]
    last_mi = int(ret["mi"].max())
    lines = []
    for ln in (OUT / "manifest.jsonl").read_text().splitlines():
        m = json.loads(ln)
        form = pd.read_parquet(OUT / "forms" / f"{m['ym'].replace('-', '')}.parquet")
        assert form_sha(form) == m["sha256"], f"ledger file for {m['ym']} was modified after it was written"
        if m["mi"] + 1 > last_mi:
            continue
        r = settle_form(form, ret)
        tgt_start = dt.date((m["mi"] + 1) // 12, (m["mi"] + 1) % 12 + 1, 1)
        r.update(ym=m["ym"], formed_on=m["formed_on"], group="prospective" if dt.date.fromisoformat(m["formed_on"]) <= tgt_start + dt.timedelta(days=7) else "retrospective")
        lines.append(r)
    out = []
    log = lambda s="": (print(s), out.append(str(s)))
    if not lines:
        log("No formation has a realised next month yet.")
    else:
        df = pd.DataFrame(lines).sort_values("mi")
        df.to_csv(OUT / "settled.csv", index=False)
        log("== PAPER-TRADING SETTLEMENT (gross; benchmark = VW eligible universe; ledger hashes verified) ==")
        for _, r in df.iterrows():
            log(f"  form {r['ym']} -> {(r['mi'] + 1) // 12}-{(r['mi'] + 1) % 12 + 1:02d} [{r['group']}]: spec {r['spec'] * 100:+.2f}%  bench {r['bench'] * 100:+.2f}%  "
                f"active {r['active_spec'] * 100:+.2f}%  top50 active {r['active_top50'] * 100:+.2f}%  unmatched w spec {r['spec_unmatched_w']:.3f}")
        for g in ("retrospective", "prospective"):
            d = df[df["group"] == g]
            if len(d):
                log(f"\n{g}: spec active {_stats(d['active_spec'])}; top-50 active {_stats(d['active_top50'])}; "
                    f"top-50 minus spec {_stats(d['active_top50'] - d['active_spec'])}")
        log(f"\nall: spec active {_stats(df['active_spec'])}")
        log("Power note: at ~5% annual tracking error a true +0.2%/mo needs decades for t = 2; this is a sanity/implementation check, not validation.")
    pathlib.Path("data/paper_report.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
