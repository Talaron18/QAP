"""Forward paper trading of the gpa tilt (pre-registration v4): Compustat-keyed universe, same signal and tilt code as the research.

The research modules key everything on `permno`. Here a pseudo-permno is built from (gvkey, iid) so `tilt_weights`,
`benchmark_weights` and `leg_gross_return` are reused unchanged; only the data source (comp.secm instead of CRSP) differs.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from qap.fundamentals import attach_pit_gvkey
from qap.panel import to_mi
from qap.tilt import cap_weights

NYSE_EXCHG = 11


def _iid_ord(iid: pd.Series) -> pd.Series:
    d = iid.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(d, errors="coerce").fillna(99).astype("int64")


def build_pseudo_panel(secm: pd.DataFrame, me_scale: float = 1.0) -> pd.DataFrame:
    """Month-keyed panel from comp.secm: pseudo `permno`, `permco` (= gvkey), me, company ME, NYSE flag, total return.

    me_scale divides prccm*cshom into $ millions (calibrate with `calibrate_secm_scale`). Rows need a positive price and share count
    for ME; returns (`trt1m`, percent) are kept for every row so labels are never filtered by the formation mask.
    """
    p = secm.copy()
    p["gvkey"] = p["gvkey"].astype(str).str.strip().str.zfill(6)
    p["iid"] = p["iid"].astype(str).str.strip()
    p["mthcaldt"] = pd.to_datetime(p["datadate"])
    p["mi"] = to_mi(p["mthcaldt"]).values
    p["permco"] = p["gvkey"].astype("int64")
    p["permno"] = p["permco"] * 100 + _iid_ord(p["iid"])
    assert not p.duplicated(["permno", "mi"]).any(), "(gvkey, iid-ordinal) is not unique per month"
    prc = pd.to_numeric(p["prccm"], errors="coerce").astype("float64")
    shr = pd.to_numeric(p["cshom"], errors="coerce").astype("float64")
    p["me"] = (prc * shr / me_scale).where((prc > 0) & (shr > 0))
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform(lambda s: s.sum(min_count=1))
    p["primaryexch"] = np.where(pd.to_numeric(p["exchg"], errors="coerce") == NYSE_EXCHG, "N", "O")
    p["mthret_adj"] = pd.to_numeric(p["trt1m"], errors="coerce").astype("float64") / 100.0
    p["eq"] = p["me_co"] > 0
    return p[["permno", "permco", "gvkey", "iid", "mthcaldt", "mi", "me", "me_co", "primaryexch", "mthret_adj", "eq"]]


def calibrate_secm_scale(secm: pd.DataFrame, funda: pd.DataFrame) -> tuple:
    """Median of company ME (prccm*cshom) / (csho*prcc_f) at fiscal-year-end months; returns (power-of-ten scale, ratio)."""
    raw = build_pseudo_panel(secm, 1.0)
    f = funda.dropna(subset=["csho", "prcc_f"]).copy()
    f["gvkey"] = f["gvkey"].astype(str).str.strip().str.zfill(6)
    f = f[(f.csho > 0) & (f.prcc_f > 0)]
    f["mi"] = to_mi(f["datadate"]).values
    f["cmp_me"] = f["csho"].astype("float64") * f["prcc_f"].astype("float64")
    co = raw[raw["me_co"] > 0].drop_duplicates(["permco", "mi"])[["gvkey", "mi", "me_co"]]
    j = co.merge(f[["gvkey", "mi", "cmp_me"]], on=["gvkey", "mi"])
    assert len(j) > 200, f"too few fiscal-year-end months overlapping comp.secm for unit calibration: {len(j)}"
    ratio = float((j["me_co"] / j["cmp_me"]).median())
    scale = float(10 ** round(np.log10(ratio)))
    assert 0.7 < ratio / scale < 1.4, f"secm ME / (csho*prcc_f) median {ratio:.3f} is not near a power of ten"
    return scale, ratio


def research_signals(pseudo: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """gpa on the formation universe, PIT exactly as `build_research_signals` (4-month lag, 548-day staleness)."""
    keys = pseudo.loc[pseudo["eq"] & (pseudo["me_co"] > 0), ["permno", "permco", "gvkey", "iid", "mi", "mthcaldt", "me", "me_co", "primaryexch"]]
    k = attach_pit_gvkey(keys, annual)
    k["gpa"] = k["gp"] / k["at"].where(k["at"] > 0)
    return k


def complete_months(pseudo: pd.DataFrame, col: str = "mthret_adj", frac: float = 0.5) -> set:
    """Month indices whose number of rows with `col` present is at least `frac` of the median month (a partial month is excluded).

    Settlement needs returns (col='mthret_adj'); formation needs only market equity (col='me').
    """
    n = pseudo[pseudo[col].notna()].groupby("mi").size()
    return set(n[n >= frac * n.median()].index.astype(int))


def top_n_variant(w: pd.DataFrame, n: int = 50, cap: float = 0.03) -> pd.DataFrame:
    """Information-only implementable variant: the n largest weights of one month's spec, renormalised, capped."""
    t = w.nlargest(n, "w").copy()
    if t.empty:
        return t
    t["w"] = cap_weights(t["w"].to_numpy(dtype="float64"), cap)
    return t


def month_form(W: pd.DataFrame, B: pd.DataFrame, sig: pd.DataFrame, mi: int, n_top: int = 50, cap: float = 0.03) -> pd.DataFrame:
    """One formation month as long ledger rows: kind in spec / bench / top50, keyed by (gvkey, iid)."""
    ids = sig.loc[sig["mi"] == mi, ["permno", "gvkey", "iid"]].drop_duplicates("permno")
    spec = W[W["mi"] == mi][["permno", "w"]]
    if spec.empty:
        return pd.DataFrame({"kind": [], "gvkey": [], "iid": [], "w": [], "mi": []})
    parts = {"spec": spec, "bench": B[B["mi"] == mi][["permno", "w"]], "top50": top_n_variant(spec, n_top, cap)}
    out = []
    for kind, d in parts.items():
        x = d.merge(ids, on="permno", how="left")
        assert x["gvkey"].notna().all(), "weights for names without an id"
        x["kind"] = kind
        out.append(x[["kind", "gvkey", "iid", "w"]])
    r = pd.concat(out, ignore_index=True)
    r["mi"] = mi
    return r


def form_sha(form: pd.DataFrame) -> str:
    """Order-independent sha256 of a formation's weights (rounded to 12 decimals)."""
    f = form.sort_values(["kind", "gvkey", "iid"])
    s = "\n".join(f"{k}|{g}|{i}|{w:.12f}" for k, g, i, w in zip(f["kind"], f["gvkey"], f["iid"], f["w"]))
    return hashlib.sha256(s.encode()).hexdigest()


def settle_form(form: pd.DataFrame, secm_ret: pd.DataFrame) -> dict:
    """Gross return of each kind over month mi+1 from comp.secm `trt1m`; missing returns count 0 and their weight is reported.

    secm_ret: gvkey, iid, mi, ret (decimal). Returns spec/bench/top50 returns, active returns and unmatched weights.
    """
    mi = int(form["mi"].iloc[0])
    r = secm_ret[secm_ret["mi"] == mi + 1][["gvkey", "iid", "ret"]]
    res = {"mi": mi}
    for kind, g in form.groupby("kind"):
        m = g.merge(r, on=["gvkey", "iid"], how="left")
        res[kind] = float((m["w"] * m["ret"].fillna(0.0)).sum())
        res[kind + "_unmatched_w"] = float(m.loc[m["ret"].isna(), "w"].sum())
    res["active_spec"] = res.get("spec", np.nan) - res.get("bench", np.nan)
    res["active_top50"] = res.get("top50", np.nan) - res.get("bench", np.nan)
    return res


def overlap_stats(W_a: pd.DataFrame, W_b: pd.DataFrame, mi: int) -> float:
    """Sum of min weights of two single-month weight tables keyed by gvkey (1 = identical portfolios)."""
    a = W_a[W_a["mi"] == mi].groupby("gvkey")["w"].sum()
    b = W_b[W_b["mi"] == mi].groupby("gvkey")["w"].sum()
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0.0)
    return float(np.minimum(j["a"], j["b"]).sum())


def manifest_line(mi: int, sha: str, commit: str, formed_on: str) -> str:
    return json.dumps({"mi": int(mi), "ym": f"{mi // 12}-{mi % 12 + 1:02d}", "sha256": sha, "spec_commit": commit, "formed_on": formed_on})


def manifest_path(root: str = ".") -> "pathlib.Path":
    """Tracked hash manifest (no licensed data: only shas, months, dates, commit). Migrates the old data/paper/manifest.jsonl once."""
    import pathlib
    new, old = pathlib.Path(root, "paper_ledger", "manifest.jsonl"), pathlib.Path(root, "data", "paper", "manifest.jsonl")
    new.parent.mkdir(exist_ok=True)
    if not new.exists() and old.exists():
        new.write_text(old.read_text())
        old.unlink()
    return new
