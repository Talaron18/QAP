"""Study D (pre-registration v11): expectation change from earnings announcements, without analyst data.

Signals (high = good, long the top quintile): `ear`, `sue`, `rsue`, `ear_sue`. All four are known from the close of trading day +1 of
the announcement and are used from the first month-end on or after the calendar date of day +1.

Definitions (exact)
-------------------
quarters   Compustat quarterly rows, `gvkey` zero-padded, duplicates on (gvkey, fyearq, fqtr) keep the latest datadate.
           Quarter index qidx = fyearq * 4 + fqtr - 1, so "same fiscal quarter of the previous year" is qidx - 4.
change     x_q - x_{q-4} where q-4 must exist as a row (same gvkey, fqtr, fyearq - 1) with a non-missing value.
SUE/RSUE   change_q / s, where s is the sample standard deviation (ddof = 1) of the last 8 seasonal changes INCLUDING the current one
           (qidx - 7 .. qidx; a quarter without a change is missing) and at least 6 of the 8 must be non-missing. s must be > 1e-10
           (a constant history gives a missing signal, never inf). SUE uses ibq, RSUE uses saleq. Changes and standard deviations are
           computed on ALL de-duplicated quarters, rows without `rdq` are dropped only afterwards when events are formed (a quarter without
           a report date still serves as the q-4 comparison and as one of the 8 changes; nothing in it was unknown at the later date).
event      one quarter with a non-missing `rdq`. Day 0 = the first trading day on or after rdq on the daily panel's own calendar
           (sorted union of all dates in the CRSP daily file). Events whose rdq is outside the calendar span get no EAR and an
           availability date of the second weekday on/after rdq (SUE/RSUE only).
EAR        sum over calendar offsets -1, 0, +1 (trading days) of (dlyret - (mktrf + rf)), daily FF decimals. All three stock returns and all
           three FF values must exist, else missing. Day 0 comes from the union calendar, so a stock that did not trade on a calendar day
           has a missing window.
link       gvkey -> permno through CCM links (filter_ccm_links output) in force on rdq. Every valid permno of the gvkey receives the event
           (share classes); a permno with two events on the same rdq from different gvkeys keeps linkprim P, then latest linkdt.
carry      at each (permno, month) use the LATEST event whose month index of day +1 is <= the month (that includes day +1 on the last
           trading day of the month: known from its close). The latest event is used even if its own value is missing (no fall-back to an
           older quarter). If month-end minus rdq is more than 130 days (exactly 130 still counts) all signals of that row are missing.
ear_sue    mean of the within-month percentile ranks (among the eligible rows that have the value) of ear and sue; needs both.
ties       TIE_EPS x altdata.tie_key(permno) is added to every signal (after ear_sue is formed from the untied values).
"""
import glob
import pathlib

import numpy as np
import pandas as pd

from qap import config
from qap.altdata import TIE_EPS, gv6, tie_key
from qap.panel import to_mi

NAMES = ["ear", "sue", "rsue", "ear_sue"]
MAX_AGE_DAYS = 130
MIN_CHANGES, WINDOW_Q = 6, 8
SD_FLOOR = 1e-10
FIRST_YEAR, LAST_YEAR = 1989, 2025
_KEY = 1 << 20                                  # calendar index < 2**20 (a 37-year calendar has about 9,300 days)
_QKEY = 1_000_000
BUCKETS = {0: "m01", 2: "m2", 3: "m3p"}         # months since the announcement: 0-1, 2, 3 or more (130 days reach month 4 or 5)


# ------------------------------------------------------------------------------------------------------------------ SQL
def crsp_daily_ret_sql(year: int) -> str:
    """Daily returns of one calendar year from CRSP CIZ daily (portable SQL; null returns are not extracted)."""
    return (f"select permno, dlycaldt, dlyret from {config.CRSP_DAILY} where dlycaldt >= '{int(year)}-01-01' and "
            f"dlycaldt <= '{int(year)}-12-31' and dlyret is not null order by permno, dlycaldt")


_FF_ALIASES = {"date": ("date", "dateff"), "mktrf": ("mktrf", "mkt_rf", "mkt-rf"), "rf": ("rf",)}


def ff_daily_columns(have) -> dict:
    """Map {date, mktrf, rf} to the real column names of the daily FF table; fail clearly listing what is available."""
    low = {str(c).lower(): str(c) for c in have}
    out, bad = {}, []
    for k, aliases in _FF_ALIASES.items():
        hit = next((low[a] for a in aliases if a in low), None)
        if hit is None:
            bad.append(k)
        out[k] = hit
    if bad:
        raise KeyError(f"{config.FF_FACTORS_DAILY}: no column for {bad} (tried {[_FF_ALIASES[k] for k in bad]}); "
                       f"available columns: {sorted(str(c) for c in have)}")
    return out


def ff_daily_sql(cols: dict) -> str:
    return f"select {cols['date']} as date, {cols['mktrf']} as mktrf, {cols['rf']} as rf from {config.FF_FACTORS_DAILY} order by {cols['date']}"


# ------------------------------------------------------------------------------------------------------------ helpers
def _num(s) -> pd.Series:
    s = pd.Series(s)
    return pd.to_numeric(s.astype(object).where(s.notna(), np.nan), errors="coerce").astype("float64")


def _lookup(keys: np.ndarray, vals: np.ndarray, q: np.ndarray, fill=np.nan) -> np.ndarray:
    """vals[keys == q] for sorted unique `keys`; `fill` where q is absent."""
    n = len(keys)
    if n == 0:
        return np.full(len(q), fill, dtype="float64")
    pos = np.searchsorted(keys, q)
    pc = np.minimum(pos, n - 1)
    ok = (pos < n) & (keys[pc] == q)
    return np.where(ok, vals[pc], fill)


# ------------------------------------------------------------------------------------------------------ quarterly SUE
def prepare_quarters(q: pd.DataFrame) -> pd.DataFrame:
    """De-duplicated quarters (all, with or without rdq) with `qidx`, `dibq`, `dsale` (seasonal changes), `sue` and `rsue`."""
    d = pd.DataFrame({"gvkey": gv6(q["gvkey"]), "datadate": pd.to_datetime(q["datadate"], errors="coerce"),
                      "fyearq": _num(q["fyearq"]), "fqtr": _num(q["fqtr"]), "rdq": pd.to_datetime(q["rdq"], errors="coerce"),
                      "ibq": _num(q["ibq"]), "saleq": _num(q["saleq"])})
    d = d[d["datadate"].notna() & d["fyearq"].between(1900, 2200) & d["fqtr"].isin([1.0, 2.0, 3.0, 4.0]) & (d["fyearq"] == d["fyearq"].round())]
    d = d.sort_values(["gvkey", "fyearq", "fqtr", "datadate"], kind="mergesort")
    d = d.drop_duplicates(["gvkey", "fyearq", "fqtr"], keep="last").reset_index(drop=True)
    d["fyearq"], d["fqtr"] = d["fyearq"].astype("int64"), d["fqtr"].astype("int64")
    d["qidx"] = d["fyearq"] * 4 + d["fqtr"] - 1
    code = pd.factorize(d["gvkey"], sort=True)[0].astype("int64")
    key = code * _QKEY + d["qidx"].to_numpy()                     # strictly increasing: sorted by gvkey, then (fyearq, fqtr)
    for src, chg, out in (("ibq", "dibq", "sue"), ("saleq", "dsale", "rsue")):
        x = d[src].to_numpy("float64")
        c = x - _lookup(key, x, key - 4)
        d[chg] = c
        m = np.column_stack([_lookup(key, c, key - j) for j in range(WINDOW_Q)])          # changes at qidx, qidx-1, ..., qidx-7
        ok = np.isfinite(m)
        cnt = ok.sum(axis=1)
        mean = np.where(ok, m, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        var = np.where(ok, (m - mean[:, None]) ** 2, 0.0).sum(axis=1) / np.maximum(cnt - 1, 1)
        sd = np.sqrt(var)
        good = (cnt >= MIN_CHANGES) & np.isfinite(c) & (sd > SD_FLOOR)
        d[out] = np.where(good, c / np.where(sd > 0, sd, 1.0), np.nan)
    return d


def quarterly_events(q: pd.DataFrame) -> pd.DataFrame:
    """Events = prepared quarters with a non-missing rdq (see the module docstring)."""
    p = prepare_quarters(q)
    return p[p["rdq"].notna()].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------- daily panel
def ff_decimals(ff: pd.DataFrame) -> pd.DataFrame:
    """date, mktrf, rf in decimals (a whole-series maximum above 1.5 means percent units, as qap.replicate.ff_to_mi); one row per date."""
    f = pd.DataFrame({"date": pd.to_datetime(ff["date"], errors="coerce"), "mktrf": _num(ff["mktrf"]), "rf": _num(ff["rf"])})
    f = f.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if f[["mktrf", "rf"]].abs().max().max() > 1.5:
        f[["mktrf", "rf"]] = f[["mktrf", "rf"]] / 100.0
    return f


def prepare_daily(daily: pd.DataFrame, ff: pd.DataFrame) -> dict:
    """Lookup structures: calendar `cal` (sorted union of daily dates, datetime64[D]), sorted int64 `keys` = permno * 2**20 + calendar index,
    returns `ret` aligned with keys, and `mkt` = mktrf + rf on the calendar (NaN where FF is missing)."""
    days = pd.to_datetime(daily["dlycaldt"]).to_numpy().astype("datetime64[D]")
    cal = np.unique(days)
    if len(cal) >= _KEY:
        raise ValueError("calendar longer than 2**20 days")
    ret = daily["dlyret"].to_numpy(dtype="float32" if str(daily["dlyret"].dtype).lower() == "float32" else "float64", na_value=np.nan)
    keys = daily["permno"].to_numpy(dtype="int64") * _KEY + np.searchsorted(cal, days)
    if len(keys) > 1 and not (keys[1:] > keys[:-1]).all():
        order = np.argsort(keys, kind="stable")
        keys, ret = keys[order], ret[order]
        first = np.r_[True, keys[1:] != keys[:-1]]                # a duplicated (permno, date): keep the first
        keys, ret = keys[first], ret[first]
    f = ff_decimals(ff)
    fd = f["date"].to_numpy().astype("datetime64[D]")
    pos = np.searchsorted(cal, fd)
    pc = np.minimum(pos, max(len(cal) - 1, 0))
    hit = (pos < len(cal)) & (cal[pc] == fd) if len(cal) else np.zeros(len(fd), bool)
    mkt = np.full(len(cal), np.nan)
    mkt[pos[hit]] = (f["mktrf"] + f["rf"]).to_numpy()[hit]
    return {"cal": cal, "keys": keys, "ret": ret, "mkt": mkt}


def announcement_days(rdq, cal: np.ndarray) -> tuple:
    """(day0 calendar index or -1, availability date = calendar date of day +1, NaT if rdq is missing).

    Inside the calendar day 0 is the first calendar day >= rdq; outside it (or at its last day) the index is -1 and day +1 is
    approximated by the second weekday on/after rdq.
    """
    r = pd.to_datetime(pd.Series(rdq)).to_numpy().astype("datetime64[D]")
    n = len(cal)
    has = ~np.isnat(r)
    i0 = np.searchsorted(cal, r, side="left") if n else np.zeros(len(r), dtype="int64")
    inside = has & (r >= cal[0]) & (i0 + 1 < n) if n else np.zeros(len(r), bool)
    day1 = np.full(len(r), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    if n:
        day1[inside] = cal[np.minimum(i0 + 1, n - 1)][inside]
    out = has & ~inside
    if out.any():
        d0 = np.busday_offset(r[out], 0, roll="forward")
        day1[out] = np.busday_offset(d0, 1)
    return np.where(inside, i0, -1).astype("int64"), day1


def window_excess(permno: np.ndarray, day0: np.ndarray, prep: dict, offsets=(-1, 0, 1)) -> np.ndarray:
    """Sum over `offsets` of (stock return - (mktrf + rf)) around calendar index `day0`; NaN if any day is absent or missing."""
    cal, keys, ret, mkt = prep["cal"], prep["keys"], prep["ret"], prep["mkt"]
    p = np.asarray(permno, dtype="int64")
    tot = np.zeros(len(p))
    for o in offsets:
        j = day0 + o
        valid = (day0 >= 0) & (j >= 0) & (j < len(cal))
        jj = np.clip(j, 0, max(len(cal) - 1, 0))
        r = _lookup(keys, ret, p * _KEY + jj).astype("float64")
        tot += np.where(valid, r - mkt[jj], np.nan)
    return tot


# ---------------------------------------------------------------------------------------------------------- linking
def _link_merge(ev: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Every (event, permno) pair with a CCM link in force on rdq, before any tie resolution. `ev_id` = row of `ev`."""
    l = links[["gvkey", "permno", "linkprim", "linkdt", "linkenddt"]].copy()
    l["gvkey"] = gv6(l["gvkey"])
    l["prim_rank"] = (l["linkprim"] != "P").astype(int)
    e = ev.reset_index(drop=True).copy()
    e["ev_id"] = np.arange(len(e))
    m = e.merge(l, on="gvkey")
    return m[(m["linkdt"] <= m["rdq"]) & (m["rdq"] <= m["linkenddt"])]


def _resolve_links(m: pd.DataFrame) -> pd.DataFrame:
    """One event per (permno, rdq): linkprim P first, then latest linkdt, then the later fiscal quarter (a late filer can report two
    quarters on one date), then gvkey."""
    by, asc = ["permno", "rdq", "prim_rank", "linkdt"], [True, True, True, False]
    if "qidx" in m:
        by, asc = by + ["qidx"], asc + [False]
    m = m.sort_values(by + ["gvkey"], ascending=asc + [True], kind="mergesort").drop_duplicates(["permno", "rdq"], keep="first")
    m["permno"] = m["permno"].astype("int64")
    return m.drop(columns=["linkprim", "linkdt", "linkenddt", "prim_rank"]).reset_index(drop=True)


def link_events(ev: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Add `permno` by date-bounded CCM links in force on rdq (events without a link drop out). `ev_id` = row of `ev`."""
    return _resolve_links(_link_merge(ev, links))


def build_events(q: pd.DataFrame, links: pd.DataFrame, daily: pd.DataFrame, ff: pd.DataFrame) -> tuple:
    """(events, flags). events: one row per (permno, rdq) with ear, sue, rsue, rdq, day0 index, `avail` (calendar date of day +1) and
    `avail_mi`. flags: one row per event BEFORE linking (rdq, linked, complete_window, sue_ok, rsue_ok) for the checks."""
    ev = quarterly_events(q)
    prep = prepare_daily(daily, ff)
    day0, day1 = announcement_days(ev["rdq"], prep["cal"])
    ev = ev.assign(day0=day0, avail=pd.to_datetime(day1).astype("datetime64[ns]"))
    merged = _link_merge(ev, links)
    lk = _resolve_links(merged)
    lk["ear"] = window_excess(lk["permno"].to_numpy(), lk["day0"].to_numpy(), prep)
    lk = lk[lk["avail"].notna()].copy()
    lk["avail_mi"] = to_mi(lk["avail"]).to_numpy()
    flags = pd.DataFrame({"rdq": ev["rdq"], "sue_ok": ev["sue"].notna(), "rsue_ok": ev["rsue"].notna()})
    ids = lk["ev_id"].to_numpy()
    flags["linked"] = np.isin(np.arange(len(ev)), merged["ev_id"].to_numpy())             # before tie resolution
    flags["complete_window"] = np.isin(np.arange(len(ev)), ids[lk["ear"].notna().to_numpy()])
    cols = ["permno", "gvkey", "rdq", "datadate", "qidx", "day0", "avail", "avail_mi", "ear", "sue", "rsue"]
    return lk[cols].sort_values(["permno", "avail_mi", "avail", "qidx"], kind="mergesort").reset_index(drop=True), flags


# --------------------------------------------------------------------------------------------------------- signals
def build_earnings_signals(keys: pd.DataFrame, events: pd.DataFrame, max_age_days: int = MAX_AGE_DAYS) -> pd.DataFrame:
    """Eligible (permno, month) rows (keys needs permno, mi, mthcaldt) with ear, sue, rsue, ear_sue, `msa` (months since rdq) and `msa_bucket`.

    No tie-break here (see `break_ties`)."""
    k = keys.copy()
    k["permno"] = k["permno"].astype("int64")
    left = pd.DataFrame({"permno": k["permno"].to_numpy(), "mi": k["mi"].to_numpy("int64"), "_row": np.arange(len(k))}).sort_values("mi", kind="mergesort")
    e = events.copy()
    e["permno"] = e["permno"].astype("int64")
    # one event per (permno, availability month): the latest day +1, then the latest fiscal quarter
    e = e.sort_values(["permno", "avail_mi", "avail", "qidx"], kind="mergesort").drop_duplicates(["permno", "avail_mi"], keep="last")
    right = e[["permno", "avail_mi", "rdq", "ear", "sue", "rsue"]].astype({"avail_mi": "int64"}).sort_values("avail_mi", kind="mergesort")
    a = pd.merge_asof(left, right, left_on="mi", right_on="avail_mi", by="permno", direction="backward").sort_values("_row")
    rdq = pd.to_datetime(a["rdq"]).astype("datetime64[ns]").reset_index(drop=True)
    age = (pd.to_datetime(k["mthcaldt"]).astype("datetime64[ns]").reset_index(drop=True) - rdq).dt.days
    stale = (age > max_age_days).to_numpy() | rdq.isna().to_numpy()
    for c in ("ear", "sue", "rsue"):
        k[c] = np.where(stale, np.nan, a[c].to_numpy("float64"))
    msa = k["mi"].to_numpy("int64") - to_mi(rdq.fillna(pd.Timestamp("1970-01-01"))).to_numpy()
    k["msa"] = np.where(stale, np.nan, msa)
    k["msa_bucket"] = np.where(stale, np.nan, np.where(msa <= 1, 0, np.where(msa == 2, 2, 3)))
    r_e, r_s = k.groupby("mi")["ear"].rank(pct=True), k.groupby("mi")["sue"].rank(pct=True)
    k["ear_sue"] = ((r_e + r_s) / 2.0).where(r_e.notna() & r_s.notna())
    return k.reset_index(drop=True)


def break_ties(signals: pd.DataFrame) -> pd.DataFrame:
    """Add TIE_EPS x the fixed per-permno key to each Study D signal (the v9 rule); distinct values keep their order."""
    out = signals.copy()
    key = tie_key(out["permno"])
    for c in NAMES:
        out[c] = out[c] + TIE_EPS * key
    return out


# ----------------------------------------------------------------------------------------------------- information
def ic_by_bucket(sig: pd.DataFrame, panel: pd.DataFrame, name: str, cut: int) -> pd.DataFrame:
    """Information only: Spearman rank IC of `name` vs next-month return per month (index f+1 < cut) and months-since-announcement
    bucket: columns ic_m01 (0-1 months), ic_m2, ic_m3p. Same eligibility as the quintile code (ex-microcaps, signal and ME present)."""
    from qap.evaluate import _with_next_return, exclude_microcaps
    s = sig.loc[sig["mi"] < cut, ["permno", "mi", "me", "me_co", "primaryexch", "msa_bucket", name]].copy()
    p = panel.loc[panel["mi"] < cut, ["permno", "mi", "mthret_adj"]].copy()
    e = exclude_microcaps(s).dropna(subset=[name, "me", "msa_bucket"])
    e = _with_next_return(e[e["me"] > 0], p).dropna(subset=["fwd"])
    cols = [f"ic_{v}" for v in BUCKETS.values()]
    if e.empty:
        return pd.DataFrame(columns=cols, dtype="float64")
    keys = [e["mi"], e["msa_bucket"]]
    rx, rz = e.groupby(["mi", "msa_bucket"])[name].rank(), e.groupby(["mi", "msa_bucket"])["fwd"].rank()
    x, z = rx - rx.groupby(keys).transform("mean"), rz - rz.groupby(keys).transform("mean")
    a = pd.DataFrame({"xy": x * z, "xx": x * x, "zz": z * z, "mi": e["mi"], "b": e["msa_bucket"]})
    sm = a.groupby(["mi", "b"]).agg(xy=("xy", "sum"), xx=("xx", "sum"), zz=("zz", "sum"), n=("xy", "size"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = (sm["xy"] / np.sqrt(sm["xx"] * sm["zz"])).where(sm["n"] >= 3).unstack("b")
    out = pd.DataFrame({f"ic_{v}": ic[b] if b in ic else np.nan for b, v in BUCKETS.items()}, index=ic.index)
    out = out.set_axis(out.index + 1)
    return out[out.index < cut]


# ------------------------------------------------------------------------------------------------------------ loading
def load_daily(directory) -> tuple:
    """Concatenate data/raw/crsp_daily_ret/{year}.parquet -> (DataFrame[permno int32, dlycaldt, dlyret float32], {year: rows})."""
    files = sorted(glob.glob(str(pathlib.Path(directory) / "[0-9]*.parquet")))
    parts, rows = [], {}
    for f in files:
        d = pd.read_parquet(f, columns=["permno", "dlycaldt", "dlyret"])
        rows[int(pathlib.Path(f).stem)] = len(d)
        d["dlycaldt"] = pd.to_datetime(d["dlycaldt"])
        parts.append(d)
    if not parts:
        return pd.DataFrame({"permno": pd.Series(dtype="int32"), "dlycaldt": pd.Series(dtype="datetime64[ns]"),
                             "dlyret": pd.Series(dtype="float32")}), rows
    return pd.concat(parts, ignore_index=True), rows


# -------------------------------------------------------------------------------------------------------------- checks
def earnings_checks(quarters: pd.DataFrame, flags: pd.DataFrame, signals: pd.DataFrame, daily_rows: dict, ff_rows: int) -> list:
    """Printable CHECKS lines. quarters = prepare_quarters output (all quarters), flags from build_events, signals before tie-break."""
    lines = []
    tot = sum(daily_rows.values())
    yrs = sorted(daily_rows)
    lines.append(f"daily returns file: {tot:,} rows in {len(yrs)} yearly files ({yrs[0]}..{yrs[-1]}); min/max per year "
                 f"{min(daily_rows.values()):,} / {max(daily_rows.values()):,}" if yrs else "daily returns file: NO ROWS")
    lines.append(f"daily FF file: {ff_rows:,} rows")
    dec = lambda s: (pd.to_datetime(s).dt.year // 10 * 10)
    qd = dec(quarters["datadate"])
    cov = quarters["rdq"].notna().groupby(qd).agg(["mean", "size"])
    lines.append(f"quarters: {len(quarters):,}; share with rdq (kept as events) {quarters['rdq'].notna().mean():.1%}; by decade of datadate: "
                 + "  ".join(f"{int(d)}s={r['mean']:.0%} (n={int(r['size']):,})" for d, r in cov.iterrows()))
    fd = dec(flags["rdq"])
    ev = flags.groupby(fd).agg(n=("linked", "size"), linked=("linked", "mean"), sue=("sue_ok", "mean"), rsue=("rsue_ok", "mean"))
    lk = flags[flags["linked"]]
    cw = lk["complete_window"].groupby(dec(lk["rdq"])).mean()
    lines.append(f"events (rdq present): {len(flags):,}; by decade of rdq: " + "  ".join(f"{int(d)}s={int(r['n']):,}" for d, r in ev.iterrows()))
    lines.append(f"merge hit rate to permno (events with >= 1 CCM link at rdq): {flags['linked'].mean():.1%}; by decade: "
                 + "  ".join(f"{int(d)}s={r['linked']:.0%}" for d, r in ev.iterrows()))
    lines.append(f"complete 3-day window (among linked events): {lk['complete_window'].mean():.1%}; by decade: "
                 + "  ".join(f"{int(d)}s={v:.0%}" for d, v in cw.items()))
    lines.append(f"SUE available (>= {MIN_CHANGES} of {WINDOW_Q} changes): {flags['sue_ok'].mean():.1%} of events; RSUE {flags['rsue_ok'].mean():.1%}; by decade: "
                 + "  ".join(f"{int(d)}s={r['sue']:.0%}/{r['rsue']:.0%}" for d, r in ev.iterrows()))
    sd = (signals["mi"] // 12 // 10) * 10
    lines.append("signal availability (share of eligible rows with a value) by decade:")
    for n in NAMES:
        lines.append(f"  {n:8s} " + "  ".join(f"{int(d)}s={signals.loc[sd == d, n].notna().mean():.0%}" for d in sorted(sd.unique())))
    return lines
