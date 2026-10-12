"""Study C (pre-registration v9, commit e239a07): six alternative-data signals, all point in time by construction.

Signal       source                    available from                                 stale after
arc10k       arc (10-K only)           date_filed (first month-end on/after)          15 months
rf_len       as_firm_risks             filing_date (first month-end on/after)         15 months
blocks       blocksv                   June 30 of year+1                              18 months
cbi_neg      classified_boards         datadate + 6 months                            18 months
culture      corp_culture              June 30 of year+1                              18 months
litig        firmyear_level            June 30 of year+1                              18 months

Staleness is counted from the availability date. `attach_pit` takes DAYS: 15 months = 457 days and 18 months = 549 days
(15 * 30.5 and 18 * 30.5 rounded down; tolerance inclusive, so a record exactly that old is still used). A value is used at month-end `m` only if avail <= m, so a filing on
2012-03-15 is invisible at 2012-02-29 and visible at 2012-03-31. A June 30 that falls on a weekend attaches one month later than
the calendar date (CRSP month-ends are the last trading day); this is conservative and identical to Study B.
Rows whose signal value is missing are dropped BEFORE attachment, because merge_asof would otherwise let a missing latest
record hide an older valid one.
"""
import numpy as np
import pandas as pd

from qap.fundamentals import attach_pit_gvkey
from qap.innovation import june30_after
from qap.panel import attach_gvkey
from qap.pit import annual_available_date

TIE_EPS = 1e-9


def tie_key(permno) -> np.ndarray:
    """Fixed pseudo-random key in [0, 1) per permno (Knuth multiplicative hash): breaks exact ties the same way every month."""
    p = np.asarray(permno, dtype="uint64")
    return ((p * np.uint64(2654435761)) % np.uint64(4294967296)).astype("float64") / 4294967296.0


NAMES = ["arc10k", "rf_len", "blocks", "cbi_neg", "culture", "litig"]
STALE_15M = 457
STALE_18M = 549
BLOCKS_MIN_ROWS = 1000                 # a year with fewer raw blocksv rows is treated as not populated (no zero fill)
CULTURE_SCORES = ["s_integrity", "s_teamwork", "s_innovation", "s_respect", "s_quality"]


# ---------------------------------------------------------------------------------------------------------------- keys
def cik10(s: pd.Series) -> pd.Series:
    """Zero-padded 10-character cik string from VARCHAR '0000000020', DOUBLE 1750.0, Decimal/object or nullable input; missing -> None."""
    n = pd.to_numeric(pd.Series(s).astype(object).where(pd.Series(s).notna(), np.nan), errors="coerce")
    ok = n.notna().to_numpy() & (n.astype("float64").to_numpy() > 0)
    out = pd.Series([None] * len(n), index=n.index, dtype=object)
    if ok.any():
        out.iloc[np.flatnonzero(ok)] = n[ok].astype("float64").round().astype("int64").astype(str).str.zfill(10).to_numpy(dtype=object)
    return out


def gv6(s: pd.Series) -> pd.Series:
    """gvkey as a zero-padded 6-character string (same construction as the CCM links, so merge keys share a dtype)."""
    return s.astype(str).str.strip().str.zfill(6)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(pd.Series(s).astype(object).where(pd.Series(s).notna(), np.nan), errors="coerce").astype("float64")


def cik_gvkey_map(funda: pd.DataFrame) -> tuple:
    """Unique cik -> gvkey map from the `cik` column of Compustat annual.

    Distinct (cik, gvkey) pairs; a cik that appears with more than one gvkey is dropped. Returns (map[cik, gvkey], stats).
    """
    if "cik" not in funda.columns:
        raise KeyError("compustat_annual.parquet has no `cik` column (it is in COMPUSTAT_ANNUAL_COLS; re-extract it)")
    p = pd.DataFrame({"cik": cik10(funda["cik"]), "gvkey": gv6(funda["gvkey"])}).dropna(subset=["cik"]).drop_duplicates()
    n_gv = p.groupby("cik")["gvkey"].transform("nunique")
    m = p[n_gv == 1].reset_index(drop=True)
    return m, {"ciks": int(p["cik"].nunique()), "dropped_multi_gvkey": int((p.groupby("cik")["gvkey"].nunique() > 1).sum()),
               "gvkeys": int(m["gvkey"].nunique())}


def _to_gvkey(df: pd.DataFrame, cik_col: str, cmap: pd.DataFrame) -> pd.DataFrame:
    """Replace `cik_col` by gvkey through the unique map (unmapped ciks drop out)."""
    d = df.copy()
    d["cik"] = cik10(d[cik_col])
    return d.dropna(subset=["cik"]).merge(cmap, on="cik", how="inner")


# --------------------------------------------------------------------------------------------------------- PIT tables
def _finalize(t: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop missing values/availability, make (gvkey, avail) unique deterministically (last datadate, then largest value)."""
    t = t.dropna(subset=["gvkey", "avail", col]).copy()
    t["gvkey"] = t["gvkey"].astype(str)                                       # same dtype as attach_gvkey's gvkey
    t["avail"] = pd.to_datetime(t["avail"]).astype("datetime64[ns]")          # merge_asof needs identical datetime units
    t["datadate"] = pd.to_datetime(t["datadate"]).astype("datetime64[ns]")
    t = t[np.isfinite(t[col].astype("float64"))]
    t = t.sort_values(["gvkey", "avail", "datadate", col], kind="mergesort")
    t = t.drop_duplicates(["gvkey", "avail"], keep="last")
    return t[["gvkey", "avail", "datadate", col]].sort_values("avail", kind="mergesort").reset_index(drop=True)


def arc_table(arc: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """-arc of each 10-K (forms starting '10-K', amendments '/A' excluded), available at date_filed."""
    a = arc.copy()
    f = a["form"].astype("string").str.strip().str.upper()
    a = a[(f.str.startswith("10-K") & ~f.str.endswith("/A")).fillna(False).to_numpy(dtype=bool)]
    a = _to_gvkey(a, "cik", cmap)
    a["avail"] = pd.to_datetime(a["date_filed"], errors="coerce")
    a["datadate"] = pd.to_datetime(a["fiscal_period_end"], errors="coerce").fillna(a["avail"])
    a["arc10k"] = -_num(a["arc"])
    return _finalize(a, "arc10k")


def risk_table(risk: pd.DataFrame, cmap: pd.DataFrame) -> pd.DataFrame:
    """-log(1 + Item 1A word count) at filing_date; a zero count is a failed parse, treated as missing (the filing is dropped)."""
    r = _to_gvkey(risk, "cik", cmap)
    wc = _num(r["item_1a_word_count"])
    r["avail"] = pd.to_datetime(r["filing_date"], errors="coerce")
    r["datadate"] = r["avail"]
    r["rf_len"] = -np.log1p(wc.where(wc > 0))
    return _finalize(r, "rf_len")


def blocks_table(blocks: pd.DataFrame, cmap: pd.DataFrame, min_rows: int = BLOCKS_MIN_ROWS) -> pd.DataFrame:
    """Total blockholder `position` per (gvkey, year), available June 30 of year+1.

    A gvkey in the cik map with no row in a populated year (>= min_rows raw rows in that year) gets 0; in other years it stays missing.
    """
    b = blocks.copy()
    b["year"] = _num(b["year"])
    b["position"] = _num(b["position"])
    pop = b.dropna(subset=["year"]).groupby(b["year"].dropna().astype("int64")).size()
    years = [int(y) for y in pop.index[pop >= min_rows]]
    g = _to_gvkey(b.dropna(subset=["year", "position"]), "company_cik", cmap)
    g["year"] = g["year"].astype("int64")
    s = g.groupby(["gvkey", "year"], as_index=False)["position"].sum()
    grid = pd.MultiIndex.from_product([sorted(cmap["gvkey"].unique()), years], names=["gvkey", "year"]).to_frame(index=False)
    s = grid.merge(s, on=["gvkey", "year"], how="outer")                  # outer keeps real rows that fall in unpopulated years
    s["position"] = s["position"].where(~s["year"].isin(years), s["position"].fillna(0.0))     # zero fill: populated years only
    s["avail"] = june30_after(s["year"])
    s["datadate"] = s["avail"]
    return _finalize(s.rename(columns={"position": "blocks"}), "blocks")


def cbi_table(boards: pd.DataFrame) -> pd.DataFrame:
    """-cbi, available datadate + 6 months."""
    c = boards.copy()
    c["gvkey"] = gv6(c["gvkey"])
    c["datadate"] = pd.to_datetime(c["datadate"], errors="coerce")
    c = c.dropna(subset=["datadate"])
    c["avail"] = annual_available_date(c["datadate"], months=6)
    c["cbi_neg"] = -_num(c["cbi"])
    return _finalize(c, "cbi_neg")


def culture_table(culture: pd.DataFrame) -> pd.DataFrame:
    """Mean of the five scores z-scored within calendar year (across all gvkey-years), available June 30 of year+1.

    Duplicates across `source` are averaged per (gvkey, year) first; the mean needs all five z-scores; a year with std 0 or
    fewer than 2 rows gives missing z-scores. z uses the sample std (ddof=1).
    """
    c = culture.copy()
    c["gvkey"] = gv6(c["gvkey"])
    c["year"] = _num(c["year"])
    for s in CULTURE_SCORES:
        c[s] = _num(c[s])
    c = c.dropna(subset=["year"])
    c["year"] = c["year"].astype("int64")
    m = c.groupby(["gvkey", "year"], as_index=False)[CULTURE_SCORES].mean()
    z = pd.DataFrame(index=m.index)
    for s in CULTURE_SCORES:
        grp = m.groupby("year")[s]
        sd = grp.transform("std")
        z[s] = (m[s] - grp.transform("mean")) / sd.where(sd > 0)
    m["culture"] = z.mean(axis=1).where(z.notna().all(axis=1))
    m["avail"] = june30_after(m["year"])
    m["datadate"] = m["avail"]
    return _finalize(m, "culture")


def litig_table(lit: pd.DataFrame) -> pd.DataFrame:
    """-log(1 + case_f), available June 30 of year+1 (no row = missing, never zero)."""
    l = lit.copy()
    l["gvkey"] = gv6(l["gvkey"])
    l["year"] = _num(l["year"])
    l = l.dropna(subset=["year"])
    l["year"] = l["year"].astype("int64")
    l["avail"] = june30_after(l["year"])
    l["datadate"] = l["avail"]
    cf = _num(l["case_f"])
    l["litig"] = -np.log1p(cf.where(cf >= 0))
    return _finalize(l, "litig")


def altdata_tables(funda, arc, risk, blocks, boards, culture, lit) -> tuple:
    """({signal: PIT table}, cik map stats) for the six signals."""
    cmap, stats = cik_gvkey_map(funda)
    return {"arc10k": (arc_table(arc, cmap), STALE_15M), "rf_len": (risk_table(risk, cmap), STALE_15M),
            "blocks": (blocks_table(blocks, cmap), STALE_18M), "cbi_neg": (cbi_table(boards), STALE_18M),
            "culture": (culture_table(culture), STALE_18M), "litig": (litig_table(lit), STALE_18M)}, stats


def build_altdata_signals(keys: pd.DataFrame, funda, links, arc, risk, blocks, boards, culture, lit) -> tuple:
    """Eligible (permno, month) rows with the six signals; keys needs permno, mi, mthcaldt. Returns (frame, cik map stats)."""
    tables, stats = altdata_tables(funda, arc, risk, blocks, boards, culture, lit)
    k = keys.copy()
    k["permno"] = k["permno"].astype("int64")
    linked = attach_gvkey(k[["permno", "mi", "mthcaldt"]], links, "mthcaldt")
    linked["mthcaldt"] = pd.to_datetime(linked["mthcaldt"]).astype("datetime64[ns]")
    for col, (table, stale) in tables.items():
        a = attach_pit_gvkey(linked, table, stale, cols=[col])
        k = k.merge(a[["permno", "mi", col]], on=["permno", "mi"], how="left")
    return k, stats


def break_ties(signals: pd.DataFrame) -> pd.DataFrame:
    """v9 amendment: add TIE_EPS x a fixed per-permno key to each signal so exact ties are ordered the same way every month.

    Binary or heavily tied signals (cbi_neg, litig) would otherwise leave the top quintile empty when the tied group exceeds 40% of the ranked names.
    Distinct values keep their order (gaps are many orders of magnitude larger than TIE_EPS).
    """
    out = signals.copy()
    key = tie_key(out["permno"])
    for col in NAMES:
        out[col] = out[col] + TIE_EPS * key
    return out


# ---------------------------------------------------------------------------------------------------------------- checks
def altdata_checks(signals: pd.DataFrame, raw: dict, funda: pd.DataFrame) -> list:
    """Printable CHECKS lines: row counts per dataset, cik->gvkey hit rate, signal coverage by decade.

    raw: {'arc_data', 'risk_text', 'blockholders', 'class_boards', 'corp_culture', 'fed_litigation'} -> DataFrame.
    Hit rate = share of the dataset's distinct ciks that map to exactly one gvkey (also shown: no match, ambiguous-and-dropped).
    """
    lines = ["row counts: " + ", ".join(f"{n} {len(d):,}" for n, d in raw.items())]
    p = pd.DataFrame({"cik": cik10(funda["cik"]), "gvkey": gv6(funda["gvkey"])}).dropna(subset=["cik"]).drop_duplicates()
    ng = p.groupby("cik")["gvkey"].nunique()
    lines.append(f"Compustat cik column: {len(ng):,} distinct ciks, {int((ng > 1).sum()):,} map to more than one gvkey (dropped), "
                 f"{int((ng == 1).sum()):,} unique")
    for name, col in (("arc_data", "cik"), ("risk_text", "cik"), ("blockholders", "company_cik")):
        c = cik10(raw[name][col]).dropna().unique()
        n = pd.Series(c).map(ng)
        lines.append(f"cik->gvkey hit rate {name}: {len(c):,} distinct ciks; unique {(n == 1).mean():.1%}, ambiguous {(n > 1).mean():.1%}, "
                     f"not in Compustat {n.isna().mean():.1%}")
    b = raw["blockholders"]
    by = _num(b["year"]).dropna().astype("int64").value_counts()
    pop = sorted(int(y) for y in by.index[by >= BLOCKS_MIN_ROWS])
    lines.append(f"blocksv years with >= {BLOCKS_MIN_ROWS} rows (zero fill allowed): {pop[0] if pop else None}..{pop[-1] if pop else None} "
                 f"({len(pop)} years); sparse years (no zero fill): {sorted(int(y) for y in by.index[by < BLOCKS_MIN_ROWS])}")
    dec = (signals["mi"] // 12 // 10) * 10
    lines.append("signal availability (share of eligible rows with a value) by decade:")
    for n in NAMES:
        lines.append(f"  {n:8s} " + "  ".join(f"{int(d)}s={signals.loc[dec == d, n].notna().mean():.0%}" for d in sorted(dec.unique())))
    lines.append("top-quintile feasibility, information only (L leg is empty in a month when the group tied at the maximum holds >= 40% of ranked rows), pre-2020 months:")
    for n in NAMES:
        v = signals.loc[signals[n].notna() & (signals["mi"] < 2020 * 12), ["mi", n]]
        if v.empty:
            lines.append(f"  {n:8s} no pre-2020 values")
            continue
        g = v.groupby("mi")[n]
        tied = (v[n] == g.transform("max")).groupby(v["mi"]).mean()
        empty = int((np.ceil(v.groupby("mi")[n].rank(pct=True) * 5).groupby(v["mi"]).max() < 5).sum())
        lines.append(f"  {n:8s} median share tied at max {tied.median():.0%}; months with an empty L leg {empty} of {tied.size}")
    return lines
