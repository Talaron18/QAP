"""Study E (pre-registration v12): short interest. Security-level link, point-in-time attachment, SIR / DTC / gpa_veto signals.

Record (gvkey, iid, datadate, shortint) is available at datadate + 14 calendar days; at month-end m the latest record with
availability <= m is used and a record more than 62 days past its availability is missing (inclusive tolerance, as in altdata).
Missing stays missing, never zero. Rows with a missing shortint are dropped BEFORE the merge_asof (a NaN latest record would
otherwise hide an older valid one; same rule as altdata).
SIR = shortint / (formation-month shrout * 1000); DTC = shortint / (formation-month mthvol / 21). `shortintadj` is never a signal input.
"""
import numpy as np
import pandas as pd

from qap.altdata import TIE_EPS, gv6, tie_key
from qap.candidates import DEV, VAL

NAMES = ["sir_neg", "dtc_neg", "gpa_veto"]
AVAIL_LAG_DAYS = 14
STALE_DAYS = 62
VETO_PCT = 0.9                      # top SIR decile: within-month percentile rank > 0.9 (k/n is correctly rounded, so exactly 0.9 is not vetoed)
TRADING_DAYS = 21
FLOOR_DEV = 0.0005                  # 0.05 %/month, as a fraction


def _istr(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), None).map(lambda v: None if v is None else str(v).strip())


def prepare_si(si: pd.DataFrame) -> pd.DataFrame:
    """Normalised copy: gvkey zero-padded string, iid stripped string, datadate datetime64[ns], numeric shortint (non-finite -> NaN)."""
    s = si.copy()
    s["gvkey"] = gv6(s["gvkey"])
    s["iid"] = _istr(s["iid"])
    s["datadate"] = pd.to_datetime(s["datadate"]).astype("datetime64[ns]")
    s["shortint"] = pd.to_numeric(s["shortint"].astype(object).where(s["shortint"].notna(), np.nan), errors="coerce").astype("float64")
    s.loc[~np.isfinite(s["shortint"]), "shortint"] = np.nan
    return s


def duplicate_count(si: pd.DataFrame) -> int:
    s = prepare_si(si)
    return int(s.duplicated(["gvkey", "iid", "datadate"]).sum())


def assert_no_duplicates(si: pd.DataFrame) -> None:
    n = duplicate_count(si)
    assert n == 0, f"short interest has {n} duplicate (gvkey, iid, datadate) records"


def security_links(links: pd.DataFrame) -> pd.DataFrame:
    """(gvkey, liid) -> permno date-bounded links from the filtered CCM links; liid is a stripped string."""
    if "liid" not in links.columns:
        raise KeyError("CCM links have no `liid` column (it is in CCM_LINK_COLS; re-extract ccm_links)")
    l = links[["gvkey", "liid", "permno", "linkdt", "linkenddt"]].copy()
    l["gvkey"] = gv6(l["gvkey"])
    l["iid"] = _istr(l["liid"])
    l["permno"] = l["permno"].astype("int64")
    return l.dropna(subset=["iid"]).drop(columns="liid")


def link_records(si: pd.DataFrame, links: pd.DataFrame) -> tuple:
    """Records -> permno at the SECURITY level (iid == liid, link in force on datadate). Returns (records with permno, avail; stats).

    A record without such a link is dropped and counted; a (permno, datadate) reached by more than one (gvkey, iid) keeps the
    last by (gvkey, iid) and the surplus is counted. A record maps to several permnos only if the links overlap (each then gets it).
    """
    assert_no_duplicates(si)
    s = prepare_si(si)
    s["rec"] = np.arange(len(s))
    m = s.merge(security_links(links), on=["gvkey", "iid"], how="left")
    m = m[(m["linkdt"] <= m["datadate"]) & (m["datadate"] <= m["linkenddt"])]
    m["permno"] = m["permno"].astype("int64")                  # the left merge made it float; only matched rows survive the date filter
    linked = m["rec"].nunique()
    m = m.sort_values(["permno", "datadate", "gvkey", "iid"], kind="mergesort")
    n_pre = len(m)
    m = m.drop_duplicates(["permno", "datadate"], keep="last")
    m["avail"] = (m["datadate"] + pd.Timedelta(days=AVAIL_LAG_DAYS)).astype("datetime64[ns]")
    stats = {"records": int(len(s)), "linked": int(linked), "dropped_no_link": int(len(s) - linked), "dup_permno_date_dropped": int(n_pre - len(m))}
    return m[["permno", "gvkey", "iid", "datadate", "avail", "shortint"]].reset_index(drop=True), stats


def attach_shortint(eligible: pd.DataFrame, links: pd.DataFrame, si: pd.DataFrame) -> tuple:
    """Add `shortint` and `si_datadate` to `eligible` (needs permno, mi, mthcaldt) point in time. Returns (frame, stats).

    stats: record counts (records, linked, dropped_no_link, dup_permno_date_dropped). Rows keep their order and count; no value = NaN.
    """
    rec, stats = link_records(si, links)
    rec = rec.dropna(subset=["shortint"]).sort_values("avail", kind="mergesort")           # NaN records dropped before the as-of merge
    k = eligible.copy()
    k["_ord"] = np.arange(len(k))
    k["permno"] = k["permno"].astype("int64")
    k["_m"] = pd.to_datetime(k["mthcaldt"]).astype("datetime64[ns]")
    left = k[["_ord", "permno", "_m"]].sort_values("_m", kind="mergesort")
    r = rec[["permno", "avail", "datadate", "shortint"]].rename(columns={"datadate": "si_datadate"})
    out = pd.merge_asof(left, r, left_on="_m", right_on="avail", by="permno", direction="backward", tolerance=pd.Timedelta(days=STALE_DAYS))
    out = out.sort_values("_ord").set_index("_ord")
    k["shortint"] = out["shortint"].to_numpy()
    k["si_datadate"] = out["si_datadate"].to_numpy()
    return k.drop(columns=["_ord", "_m"]), stats


def sir_dtc(shortint, shrout, mthvol) -> tuple:
    """(SIR, DTC); a non-positive or missing denominator gives NaN, never inf."""
    sh = pd.to_numeric(pd.Series(shrout), errors="coerce").astype("float64").to_numpy()
    vol = pd.to_numeric(pd.Series(mthvol), errors="coerce").astype("float64").to_numpy()
    si = np.asarray(shortint, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        sir = np.where(sh > 0, si / (sh * 1000.0), np.nan)
        dtc = np.where(vol > 0, si / (vol / TRADING_DAYS), np.nan)
    return sir, dtc


def veto_mask(sir: pd.Series, mi: pd.Series, is_eligible: pd.Series) -> pd.Series:
    """True for names in the top SIR decile of their month: highest SIR among ELIGIBLE names that have a SIR. No SIR -> never vetoed."""
    pool = is_eligible.to_numpy(dtype=bool) & sir.notna().to_numpy()
    pct = sir.where(pool).groupby(mi).rank(pct=True)
    return (pct > VETO_PCT).fillna(False) & pd.Series(pool, index=sir.index)


def build_shortint_signals(eligible: pd.DataFrame, links: pd.DataFrame, si: pd.DataFrame, base: pd.DataFrame, panel: pd.DataFrame) -> tuple:
    """Signals frame on the rows of `base` plus (frame, link stats).

    eligible: the formation universe (qap.study.eligible_universe) - the veto decile is formed among its names with a SIR.
    base:     the incumbent's frame (qap.evaluate.build_research_signals output, one row per linked formation row, with `gpa`). The signals
              are attached to ITS rows so that gpa_veto and gpa are ranked on the identical universe (a frame built on `eligible` would be
              microcap-filtered a second time inside the portfolio code and confound the veto with a size cut).
    panel:    supplies the formation month's shrout and mthvol.
    Columns added: sir, dtc, sir_neg, dtc_neg, gpa_veto (gpa with the vetoed names set to NaN BEFORE ranking), vetoed, elig.
    """
    for c in ("shrout", "mthvol"):
        if c not in panel.columns:
            raise KeyError(f"panel has no `{c}` column (qap.panel.build_monthly_panel keeps it when crsp_monthly has it)")
    k = base.copy()
    k["permno"] = k["permno"].astype("int64")
    k, stats = attach_shortint(k, links, si)
    p = panel[["permno", "mi", "shrout", "mthvol"]].drop_duplicates(["permno", "mi"])
    k = k.merge(p, on=["permno", "mi"], how="left")
    k["sir"], k["dtc"] = sir_dtc(k["shortint"].to_numpy(), k["shrout"], k["mthvol"])
    ek = eligible[["permno", "mi"]].drop_duplicates().assign(elig=True)
    k = k.merge(ek, on=["permno", "mi"], how="left")
    k["elig"] = k["elig"].fillna(False).astype(bool)
    k["vetoed"] = veto_mask(k["sir"], k["mi"], k["elig"]).to_numpy()
    k["sir_neg"] = -k["sir"]
    k["dtc_neg"] = -k["dtc"]
    k["gpa_veto"] = k["gpa"].where(~k["vetoed"])
    return k, stats


def break_ties(signals: pd.DataFrame) -> pd.DataFrame:
    """The v9 tie rule on the Study E signals: add TIE_EPS x the fixed per-permno hash key (altdata.break_ties is bound to Study C's names)."""
    out = signals.copy()
    key = tie_key(out["permno"])
    for c in NAMES:
        out[c] = out[c] + TIE_EPS * key
    return out


# ---------------------------------------------------------------------------------------------------------------- size floor
def veto_floor(veto: pd.Series, gpa: pd.Series) -> dict:
    """Mean net long-minus-universe improvement of gpa_veto over gpa on aligned months, unrounded fractions per month.

    PASS iff dev mean >= 0.05 %/month AND validation mean > 0.
    """
    j = pd.concat([veto.rename("v"), gpa.rename("g")], axis=1).dropna()
    d = j["v"] - j["g"]
    out = {}
    for label, (y0, y1) in (("dev", DEV), ("val", VAL)):
        v = d[(d.index >= y0 * 12) & (d.index <= y1 * 12 + 11)]
        t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if len(v) > 2 and v.std(ddof=1) > 0 else float("nan")
        out[label] = {"months": int(len(v)), "mean": float(v.mean()) if len(v) else float("nan"), "t": t}
    out["PASS"] = bool(out["dev"]["months"] > 0 and out["val"]["months"] > 0 and out["dev"]["mean"] >= FLOOR_DEV and out["val"]["mean"] > 0)
    return out


def holdout_improvement(veto_full: pd.Series, gpa_full: pd.Series, cut: int) -> dict:
    j = pd.concat([veto_full.rename("v"), gpa_full.rename("g")], axis=1).dropna()
    d = (j["v"] - j["g"])[j.index >= cut]
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 and d.std(ddof=1) > 0 else float("nan")
    return {"months": int(len(d)), "mean": float(d.mean()) if len(d) else float("nan"), "t": t}


# ------------------------------------------------------------------------------------------------------------- information
def monthly_ic(sig: pd.DataFrame, panel: pd.DataFrame, name: str, cut: int) -> pd.Series:
    """Information only: Spearman rank IC of `name` vs next-month return per formation month (index f+1 < cut), same eligibility as the quintiles."""
    from qap.evaluate import _with_next_return, exclude_microcaps
    s = sig.loc[sig["mi"] < cut, ["permno", "mi", "me", "me_co", "primaryexch", name]].copy()
    p = panel.loc[panel["mi"] < cut, ["permno", "mi", "mthret_adj"]]
    e = exclude_microcaps(s).dropna(subset=[name, "me"])
    e = _with_next_return(e[e["me"] > 0], p).dropna(subset=["fwd"])
    if e.empty:
        return pd.Series(dtype="float64")
    g = e.groupby("mi")
    rx, rz = g[name].rank(), g["fwd"].rank()
    x, z = rx - rx.groupby(e["mi"]).transform("mean"), rz - rz.groupby(e["mi"]).transform("mean")
    a = pd.DataFrame({"xy": x * z, "xx": x * x, "zz": z * z, "mi": e["mi"]})
    sm = a.groupby("mi").agg(xy=("xy", "sum"), xx=("xx", "sum"), zz=("zz", "sum"), n=("xy", "size"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = (sm["xy"] / np.sqrt(sm["xx"] * sm["zz"])).where(sm["n"] >= 3)
    ic = ic.set_axis(ic.index + 1)
    return ic[ic.index < cut]


# ---------------------------------------------------------------------------------------------------------------- checks
def shortint_checks(si: pd.DataFrame, stats: dict, signals: pd.DataFrame) -> list:
    """Printable CHECKS lines (before any result). Asserts zero duplicate (gvkey, iid, datadate). signals = build_shortint_signals output."""
    n_dup = duplicate_count(si)
    s = prepare_si(si)
    lines = [f"records: {len(s):,}; records per year: " + "  ".join(f"{int(y)}={n:,}" for y, n in s.groupby(s["datadate"].dt.year).size().items())]
    d = s.sort_values(["gvkey", "iid", "datadate"])
    gap = d.groupby(["gvkey", "iid"])["datadate"].diff().dt.days
    med = gap.groupby(d["datadate"].dt.year).median()
    lines.append("cadence: median days between records per (gvkey, iid) by year: " + "  ".join(f"{int(y)}={v:.0f}" for y, v in med.dropna().items()))
    lines.append(f"duplicate (gvkey, iid, datadate) records: {n_dup}")
    assert n_dup == 0, f"short interest has {n_dup} duplicate (gvkey, iid, datadate) records"
    lines.append(f"[PASS] zero duplicate (gvkey, iid, datadate) records")
    r, l = stats["records"], stats["linked"]
    lines.append(f"iid->liid security link hit rate: {l:,} of {r:,} records linked ({(l / r if r else float('nan')):.1%}); dropped without a link "
                 f"{stats['dropped_no_link']:,}; (permno, datadate) surplus dropped {stats['dup_permno_date_dropped']:,}")
    e = signals[signals["elig"]]
    yr = e["mi"] // 12
    cov = e["sir"].notna().groupby(yr).mean()
    lines.append("share of eligible names with a SIR by year: " + "  ".join(f"{int(y)}={v:.0%}" for y, v in cov.items()))
    sir, dtc = signals["sir"].dropna(), signals["dtc"].dropna()
    if len(sir):
        lines.append(f"SIR (sanity only): median {sir.median():.4f}, p99 {sir.quantile(.99):.4f}, share > 1 {(sir > 1).mean():.2%}; n={len(sir):,}")
    if len(dtc):
        lines.append(f"DTC (days, sanity only): median {dtc.median():.2f}, p99 {dtc.quantile(.99):.2f}; n={len(dtc):,}")
    lines.append(f"share of formation rows vetoed (top SIR decile among eligible names with a SIR): {signals['vetoed'].mean():.2%} of all rows, "
                 f"{(signals.loc[signals['elig'], 'vetoed'].mean() if len(e) else float('nan')):.2%} of eligible rows")
    if "shortintadj" in si.columns:
        a = pd.to_numeric(si["shortintadj"], errors="coerce")
        b = pd.to_numeric(si["shortint"], errors="coerce")
        lines.append(f"median shortint {b.median():,.0f} vs median shortintadj {a.median():,.0f} (shortintadj is never a signal input)")
    return lines
