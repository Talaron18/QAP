"""International replication of the JKP findings (pre-registration v10). DESCRIPTIVE: calibrates priors, is not a tradability test.

Nothing here logs a trial or applies a promotion rule. Conventions (all fixed by the pre-registration unless noted):
  * a country-month needs >= MIN_STOCKS eligible stocks (me > 0 and a next-month return) to enter; ineligible rows are dropped
    BEFORE ranking, so every percentile is taken over the eligible stocks of one country-month (key `cm` = country x month);
  * the 15 Study A signals plus `gp_at` (percentile within country-month) are ranked within country-month; regions pool countries;
  * all series are indexed by the RETURN month (formation month index + 1), the convention of `diagnostics.quintile_table`; periods
    and the H1-H3 windows are therefore dates of the realised return;
  * a country-month enters a given signal's portfolio only if >= MIN_RANKED stocks carry that signal (interpretation, not pre-registered).
"""
import glob
import pathlib

import numpy as np
import pandas as pd

from qap import config
from qap.jkp import add_composites, theme_scores
from qap.jkp_themes import ALL_CHARACTERISTICS, THEMES
from qap.panel import to_mi
from qap.parallel import context, pmap, set_context

MIN_STOCKS = 100
MIN_RANKED = 20
COST_MULT = 1.5
SPREAD_CAP = 0.05
FIRST_YEAR = 1990
REGION_ORDER = ["US", *config.REGIONS]
COUNTRY_REGION = {"USA": "US", **{c: r for r, cs in config.REGIONS.items() for c in cs}}
COUNTRY_CODE = {c: i for i, c in enumerate(["USA", *config.DEV_EX_US])}
SIGNALS = [*THEMES, "jkp_all13", "jkp_slow10", "gp_at"]
COMPOSITES = ("jkp_all13", "jkp_slow10")
PERIODS = {"dev": (1990, 2014), "val": (2015, 2019), "later": (2020, 2025)}
METRICS = ("ic", "q5q1", "q5u", "q1u", "net")
BASE_COLS = ["id", "excntry", "region", "mi", "cm", "ret", "me", "hs"]
MEMBER_COLS = list(dict.fromkeys(["bidaskhl_21d", "gp_at", *ALL_CHARACTERISTICS]))
BANNER = "DESCRIPTIVE: cross-market calibration, not a tradability test; no trials logged"


# ---------------------------------------------------------------- loading and scoring

def _f64(s) -> np.ndarray:
    """Numeric column (numpy or pandas nullable dtype) -> float64 ndarray with NaN for missing."""
    return pd.to_numeric(pd.Series(s)).to_numpy(dtype="float64", na_value=np.nan)


def standardise(raw: pd.DataFrame, us: bool = False) -> pd.DataFrame:
    """Raw parquet frame -> id, excntry, mi, size_grp, ret, me and the present member columns, explicit numpy dtypes."""
    ids = pd.to_numeric(raw["permno" if us else "id"])
    if ids.isna().any():
        raise ValueError("missing security ids")
    d = pd.DataFrame({"id": ids.to_numpy(dtype="int64"),
                      "excntry": np.full(len(raw), "USA", dtype=object) if us else raw["excntry"].astype(str).to_numpy(dtype=object),
                      "mi": to_mi(raw["eom"]).to_numpy(dtype="int64"),
                      "size_grp": raw["size_grp"].astype(str).to_numpy(dtype=object),
                      "ret": _f64(raw["ret_exc_lead1m"]), "me": _f64(raw["me"])})
    for c in MEMBER_COLS:
        if c in raw.columns:
            d[c] = _f64(raw[c])
    return d


def score_frame(d: pd.DataFrame, min_stocks: int = MIN_STOCKS, themes: dict = THEMES) -> tuple:
    """Standardised stock-months -> (slim frame with the 16 signals, info dict). Applies the country-month >= min_stocks rule."""
    unknown = sorted(set(d["excntry"]) - set(COUNTRY_REGION))
    if unknown:
        raise ValueError(f"countries outside the pre-registered set: {unknown}")
    if d.duplicated(["id", "mi"]).any():
        raise ValueError("duplicate (id, month) rows")
    d = d.reset_index(drop=True)
    region = d["excntry"].map(COUNTRY_REGION)
    cm = d["excntry"].map(COUNTRY_CODE).to_numpy(dtype="int64") * 1_000_000 + d["mi"].to_numpy(dtype="int64")
    elig = ((d["me"] > 0) & d["ret"].notna()).to_numpy()
    key = pd.Series(cm)
    n_elig = pd.Series(elig.astype("int64")).groupby(key).transform("sum").to_numpy()
    tab = pd.DataFrame({"cm": cm, "excntry": d["excntry"], "region": region, "mi": d["mi"]}).drop_duplicates("cm")
    tab = tab.merge(pd.DataFrame({"cm": cm, "rows": 1, "n": elig.astype("int64")}).groupby("cm").sum().reset_index(), on="cm")
    tab["kept"] = tab["n"] >= min_stocks
    info = {"cm": tab,
            "size": pd.DataFrame({"region": region, "size_grp": d["size_grp"]}).value_counts().rename("rows").reset_index()}
    keep = elig & (n_elig >= min_stocks)
    d, cm, region = d[keep].reset_index(drop=True), cm[keep], region[keep].reset_index(drop=True)
    if d.empty:
        info["slow"] = pd.DataFrame({"region": [], "rows": [], "ok": []})
        return pd.DataFrame(columns=BASE_COLS + SIGNALS), info
    d["cm"] = cm
    sc = add_composites(theme_scores(d, themes, group="cm"), d["cm"])
    if "gp_at" in d:
        sc["gp_at"] = d.groupby("cm")["gp_at"].rank(pct=True) - 0.5
    else:
        sc["gp_at"] = np.nan
    if "bidaskhl_21d" in d:
        hs = (d["bidaskhl_21d"] / 2.0).clip(0.0, SPREAD_CAP)
        hs = hs.fillna(hs.groupby(d["cm"]).transform("median")).fillna(0.0)
    else:
        hs = pd.Series(0.0, index=d.index)
    slim = pd.DataFrame({"id": d["id"].to_numpy(), "excntry": d["excntry"].to_numpy(dtype=object),
                         "region": pd.Categorical(region, categories=REGION_ORDER),
                         "mi": d["mi"].to_numpy(), "cm": d["cm"].to_numpy(), "ret": d["ret"].to_numpy(),
                         "me": d["me"].to_numpy(), "hs": hs.to_numpy(dtype="float64")})
    for s in SIGNALS:
        slim[s] = sc[s].to_numpy(dtype="float64", na_value=np.nan)
    slow = pd.DataFrame({"region": region, "ok": slim["jkp_slow10"].notna().to_numpy()}).groupby("region")["ok"].agg(["size", "sum"])
    info["slow"] = slow.rename(columns={"size": "rows", "sum": "ok"}).reset_index()
    return slim, info


def _load_one(args) -> tuple:
    path, us, min_stocks = args
    return score_frame(standardise(pd.read_parquet(path), us=us), min_stocks=min_stocks)


def _year_files(directory: str, first_year: int) -> list:
    out = []
    for f in sorted(glob.glob(str(pathlib.Path(directory) / "[0-9]*.parquet"))):
        if int(pathlib.Path(f).stem) >= first_year:
            out.append(f)
    return out


def load_all(intl_dir: str, us_dir: str, first_year: int = FIRST_YEAR, min_stocks: int = MIN_STOCKS, jobs: int = 1) -> tuple:
    """Score every year file of jkp_intl and jkp_us (same code path) -> (slim frame sorted by region/cm/id, merged info)."""
    items = [(f, False, min_stocks) for f in _year_files(intl_dir, first_year)] + [(f, True, min_stocks) for f in _year_files(us_dir, first_year)]
    if not items:
        raise ValueError("no year files found")
    res = pmap(_load_one, items, jobs=jobs)
    slim = pd.concat([r[0] for r in res], ignore_index=True)
    info = {"cm": pd.concat([r[1]["cm"] for r in res], ignore_index=True),
            "size": pd.concat([r[1]["size"] for r in res]).groupby(["region", "size_grp"], observed=True)["rows"].sum().reset_index(),
            "slow": pd.concat([r[1]["slow"] for r in res]).groupby("region", observed=True)[["rows", "ok"]].sum().reset_index()}
    slim = slim.sort_values(["cm", "id"], kind="mergesort").reset_index(drop=True)
    return slim, info


# ---------------------------------------------------------------- per (region, signal) series

def nw_t(x, lags: int = 6) -> float:
    """t-statistic of the mean with Newey-West (Bartlett) standard error (same formula as holdout_ls.nw_t)."""
    x = np.asarray(pd.Series(x).dropna(), dtype="float64")
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    S = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        S += 2.0 * (1.0 - lag / (lags + 1.0)) * float(e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(S / n)) if S > 0 else float("nan")


_COLS = ["ic", "q5", "q1", "u", "q5q1", "q5u", "q1u", "to", "cost", "net", "n"]


def signal_series(d: pd.DataFrame, sig: str, min_ranked: int = MIN_RANKED, mult: float = COST_MULT) -> pd.DataFrame:
    """Monthly series of one signal in one region (d = the region's rows), indexed by RETURN month (formation + 1).

    One eligible set per signal-month (signal, me > 0 and ret all present, country-month with >= min_ranked names) feeds the IC, the
    within-country percentile quintiles, Q5, Q1 and the universe U, so Q5-Q1 = (Q5-U) - (Q1-U) holds exactly.
    """
    s = d[sig].to_numpy(dtype="float64")
    m = np.isfinite(s) & (d["me"].to_numpy(dtype="float64") > 0) & np.isfinite(d["ret"].to_numpy(dtype="float64"))
    e = d.loc[m, ["id", "cm", "mi", "ret", "me", "hs"]].reset_index(drop=True)
    e["s"] = s[m]
    e = e[e.groupby("cm")["s"].transform("size") >= min_ranked].reset_index(drop=True)
    if e.empty:
        return pd.DataFrame(columns=_COLS, dtype="float64")
    e["id"] = e["id"].astype("int64")
    g = e.groupby("cm")
    # country-month Spearman IC (average ranks), then n-weighted average over the countries of each month
    rs, rr = g["s"].rank(), g["ret"].rank()
    x, y = rs - rs.groupby(e["cm"]).transform("mean"), rr - rr.groupby(e["cm"]).transform("mean")
    t = pd.DataFrame({"cm": e["cm"], "xy": x * y, "xx": x * x, "yy": y * y, "mi": e["mi"]}).groupby("cm").agg(
        xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"), mi=("mi", "first"), n=("xy", "size"))
    den = np.sqrt(t["xx"] * t["yy"])
    ic_cm = (t["xy"] / den.where(den > 0)).where(t["n"] >= 3)
    ok = ic_cm.notna()
    ic = (ic_cm[ok] * t["n"][ok]).groupby(t["mi"][ok]).sum() / t["n"][ok].groupby(t["mi"][ok]).sum()
    # quintiles from the within-country percentile; pooled value-weighted legs
    pct = g["s"].rank(pct=True)
    e["q"] = np.clip(np.ceil(pct * 5), 1, 5).astype("int8")
    e["wr"] = e["me"] * e["ret"]
    tot = e.groupby("mi")[["wr", "me"]].sum()
    u = tot["wr"] / tot["me"]
    gq = e.groupby(["mi", "q"])[["wr", "me"]].sum()
    r = (gq["wr"] / gq["me"]).unstack("q").reindex(index=u.index, columns=[1, 2, 3, 4, 5])
    out = pd.DataFrame({"ic": ic.reindex(u.index), "q5": r[5], "q1": r[1], "u": u})
    out["q5q1"], out["q5u"], out["q1u"] = out["q5"] - out["q1"], out["q5"] - out["u"], out["q1"] - out["u"]
    # one-way turnover of the Q5 weight vector between consecutive formation months (weights by id, drift ignored)
    t5 = e.loc[e["q"] == 5, ["mi", "id", "me", "hs"]].copy()
    t5["w"] = t5["me"] / t5.groupby("mi")["me"].transform("sum")
    cur = t5[["mi", "id", "w", "hs"]]
    prev = cur.assign(mi=cur["mi"] + 1).rename(columns={"w": "wp", "hs": "hsp"})
    mm = cur.merge(prev, on=["mi", "id"], how="outer")
    months = pd.Index(cur["mi"].unique())
    mm = mm[mm["mi"].isin(months) & (mm["mi"] - 1).isin(months)].copy()
    mm["dw"] = (mm["w"].fillna(0.0) - mm["wp"].fillna(0.0)).abs()
    mm["cdw"] = mm["dw"] * mm["hs"].fillna(mm["hsp"]).fillna(0.0)       # leavers cost at their previous-month spread
    agg = mm.groupby("mi")[["dw", "cdw"]].sum()
    out["to"] = (0.5 * agg["dw"]).reindex(out.index)
    out["cost"] = (mult * agg["cdw"]).reindex(out.index)
    out["net"] = out["q5u"] - out["cost"]
    out["n"] = e.groupby("mi").size().reindex(out.index).astype("float64")
    out.index = out.index + 1
    out.index.name = "ret_mi"
    return out[_COLS].astype("float64")


def _job(item):
    region, sig = item
    return signal_series(context()["by_region"][region], sig)


def run_series(slim: pd.DataFrame, jobs: int = 1, say=print) -> dict:
    """{(region, signal): series frame} for every region with data x the 16 signals (pmap, fork-shared frame)."""
    by_region = {r: slim[slim["region"] == r].reset_index(drop=True) for r in REGION_ORDER}
    by_region = {r: v for r, v in by_region.items() if len(v)}
    set_context(by_region=by_region)
    items = [(r, s) for r in by_region for s in SIGNALS]
    res = pmap(_job, items, jobs=jobs, say=say)
    set_context()
    return dict(zip(items, res))


# ---------------------------------------------------------------- period summaries

def period_slice(s: pd.Series, y0: int, y1: int) -> pd.Series:
    return s[(s.index >= y0 * 12) & (s.index <= y1 * 12 + 11)]


def summarize(df: pd.DataFrame) -> dict:
    """{period: {metric: (mean, NW6 t, months), 'to': mean turnover, 'short': -mean(Q1-U)/mean(Q5-Q1) or NaN}}."""
    out = {}
    for p, (y0, y1) in PERIODS.items():
        o = {}
        for mt in METRICS:
            v = period_slice(df[mt], y0, y1).dropna() if mt in df else pd.Series(dtype="float64")
            o[mt] = (float(v.mean()) if len(v) else float("nan"), nw_t(v), int(len(v)))
        o["to"] = float(period_slice(df["to"], y0, y1).mean()) if len(df) else float("nan")
        o["short"] = short_share(period_slice(df["q1u"], y0, y1), period_slice(df["q5q1"], y0, y1))
        out[p] = o
    return out


def short_share(q1u: pd.Series, q5q1: pd.Series) -> float:
    """-(mean Q1-U) / (mean Q5-Q1) over the months where both exist (ratio of pooled means); NaN unless the denominator is > 0."""
    both = pd.concat([q1u, q5q1], axis=1).dropna()
    if both.empty:
        return float("nan")
    den = float(both.iloc[:, 1].mean())
    return float(-both.iloc[:, 0].mean() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------- the three labelled questions (pure functions of the numbers)

def _fin(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def h1_ratio(ic: pd.Series) -> float:
    """Mean IC over 2015-2025 (validation + later) / mean IC over dev; NaN when the dev mean is not positive."""
    dev = period_slice(ic, *PERIODS["dev"]).dropna()
    late = period_slice(ic, PERIODS["val"][0], PERIODS["later"][1]).dropna()
    if not len(dev) or not len(late) or not dev.mean() > 0:
        return float("nan")
    return float(late.mean() / dev.mean())


def h1_label(ratios: dict) -> tuple:
    """ratios {region: (ratio jkp_all13, ratio jkp_slow10)} for the 3 non-US regions -> (label, n_decay, n_persist).

    A region counts for GLOBAL DECAY when both composite ratios are <= 0.5, for PERSISTS when both are >= 0.75; NaN ratios count for neither.
    """
    decay = sum(1 for r in ratios.values() if all(_fin(x) and x <= 0.5 for x in r))
    persist = sum(1 for r in ratios.values() if all(_fin(x) and x >= 0.75 for x in r))
    if decay >= 2:
        return "GLOBAL DECAY", decay, persist
    if persist >= 2:
        return "PERSISTS OUTSIDE THE US", decay, persist
    return "MIXED", decay, persist


def h2_label(shares: dict) -> tuple:
    """shares {region: (share jkp_all13, share jkp_slow10)} -> (label, n_regions_both_ge_0.6, n_regions_per_composite)."""
    both = sum(1 for r in shares.values() if all(_fin(x) and x >= 0.6 for x in r))
    per = tuple(sum(1 for r in shares.values() if _fin(r[i]) and r[i] >= 0.6) for i in (0, 1))
    return ("SHORT-LEG" if both >= 2 else "NOT SHORT-LEG"), both, per


def h3_region(m2122: float, other: float) -> str:
    """'yes' if the 2021-22 mean strictly exceeds 3x the other-months mean; 'n/a' for a missing or non-positive denominator."""
    if not _fin(m2122) or not _fin(other) or not other > 0:
        return "n/a"
    return "yes" if m2122 > 3.0 * other else "no"


def h3_label(means: dict) -> tuple:
    """means {region: (mean 2021-22, mean other months since 2015-01)} -> (label, n_yes, {region: verdict})."""
    v = {r: h3_region(*m) for r, m in means.items()}
    n = sum(1 for x in v.values() if x == "yes")
    return ("GLOBAL EPISODE" if n >= 2 else "US-SPECIFIC"), n, v


def h3_means(q5q1: pd.Series) -> tuple:
    s = q5q1.dropna()
    s = s[s.index >= 2015 * 12]
    ep = s[(s.index >= 2021 * 12) & (s.index <= 2022 * 12 + 11)]
    ot = s.drop(ep.index)
    return (float(ep.mean()) if len(ep) else float("nan"), float(ot.mean()) if len(ot) else float("nan"))


def questions(series: dict) -> tuple:
    """Evaluate H1, H2, H3 from the series dict -> (log lines with every number feeding each label, {'H1','H2','H3'} labels)."""
    regs = [r for r in REGION_ORDER if r != "US"]
    f = lambda x, d=3: "n/a" if not _fin(x) else f"{x:.{d}f}"
    get = lambda r, sg: series.get((r, sg), pd.DataFrame(columns=_COLS))
    L = ["== THE THREE LABELLED QUESTIONS (pre-registered; composites only; return-month dates; regions = the 3 non-US ones) =="]
    # H1
    L.append("H1 Is the decay global?  ratio = mean IC(2015-01..2025-12) / mean IC(dev 1990-2014); n/a if the dev mean IC <= 0")
    L.append("   rule: GLOBAL DECAY if both composite ratios <= 0.5 in >= 2 of 3 regions; PERSISTS OUTSIDE THE US if both >= 0.75 in >= 2 of 3; else MIXED")
    ratios = {}
    for r in ["US", *regs]:
        rr = tuple(h1_ratio(get(r, c)["ic"]) for c in COMPOSITES)
        if r != "US":
            ratios[r] = rr
        L.append(f"   {r:18s} " + "  ".join(f"{c} ratio {f(x, 2)}" for c, x in zip(COMPOSITES, rr)) + ("   (reference, not counted)" if r == "US" else ""))
    h1, nd, npz = h1_label(ratios)
    L.append(f"   regions with both ratios <= 0.5: {nd} of 3; with both >= 0.75: {npz} of 3  ->  H1 = {h1}")
    # H2
    L.append("H2 Is the edge in the short leg?  share = -mean(Q1-U) / mean(Q5-Q1) over 2015-01..2025-12 (ratio of pooled means; n/a if Q5-Q1 mean <= 0)")
    L.append("   rule: SHORT-LEG if both composite shares >= 0.6 in >= 2 of 3 regions (US reference in the pre-registration: 0.84 / 0.91)")
    shares = {}
    for r in ["US", *regs]:
        vals = []
        for c in COMPOSITES:
            d = get(r, c)
            lo, hi = PERIODS["val"][0], PERIODS["later"][1]
            vals.append(short_share(period_slice(d["q1u"], lo, hi), period_slice(d["q5q1"], lo, hi)))
        if r != "US":
            shares[r] = tuple(vals)
        L.append(f"   {r:18s} " + "  ".join(f"{c} share {f(x, 2)}" for c, x in zip(COMPOSITES, vals)) + ("   (reference, not counted)" if r == "US" else ""))
    h2, nb, per = h2_label(shares)
    L.append(f"   regions with both shares >= 0.6: {nb} of 3 (per composite: all13 {per[0]}, slow10 {per[1]})  ->  H2 = {h2}")
    # H3
    L.append("H3 Was 2021-2022 global?  jkp_slow10 gross VW Q5-Q1: mean over return months 2021-01..2022-12 vs mean over all other months since 2015-01 (%/mo)")
    L.append("   rule: GLOBAL EPISODE if the 2021-22 mean is strictly > 3x the other-months mean in >= 2 of 3 regions (non-positive other-months mean -> n/a, not global); else US-SPECIFIC")
    means = {}
    for r in ["US", *regs]:
        m = h3_means(get(r, "jkp_slow10")["q5q1"])
        if r != "US":
            means[r] = m
        ratio = m[0] / m[1] if _fin(m[0]) and _fin(m[1]) and m[1] > 0 else float("nan")
        L.append(f"   {r:18s} 2021-22 {f(m[0] * 100 if _fin(m[0]) else m[0], 2)}  other {f(m[1] * 100 if _fin(m[1]) else m[1], 2)}  ratio {f(ratio, 2)}  verdict {h3_region(*m)}"
                 + ("   (reference, not counted)" if r == "US" else ""))
    h3, ny, v = h3_label(means)
    L.append(f"   regions with 2021-22 > 3x other months: {ny} of 3  ->  H3 = {h3}")
    L.append(f"LABELS: H1 = {h1} | H2 = {h2} | H3 = {h3}")
    return L, {"H1": h1, "H2": h2, "H3": h3}


# ---------------------------------------------------------------- text formatting

def _t(x) -> str:
    return "   . " if not _fin(x) else f"{x:5.1f}"


def format_tables(series: dict) -> list:
    """Per region and period: IC, Q5-Q1, Q5-U, Q1-U (%/month, NW6 t in brackets), short-leg share, one-way Q5 turnover, proxy net."""
    L = []
    for r in REGION_ORDER:
        if not any((r, s) in series for s in SIGNALS):
            continue
        sm = {s: summarize(series[(r, s)]) for s in SIGNALS}
        for p, (y0, y1) in PERIODS.items():
            months = max(sm[s][p]["q5q1"][2] for s in SIGNALS)
            L.append(f"== {r} | {p} {y0}-{y1} (return months, up to {months}) | returns in %/month (NW6 t), IC in correlation units ==")
            L.append(f"{'signal':20s} {'IC':>8s} {'t':>5s} | {'Q5-Q1':>7s} {'t':>5s} | {'Q5-U':>7s} {'t':>5s} | {'Q1-U':>7s} {'t':>5s} | {'short':>5s} {'TO%':>5s} | {'net':>7s} {'t':>5s}")
            for s in SIGNALS:
                o = sm[s][p]
                pc = lambda x: "    nan" if not _fin(x) else f"{x * 100:7.2f}"
                ic = "     nan" if not _fin(o["ic"][0]) else f"{o['ic'][0]:8.4f}"
                sh = "  n/a" if not _fin(o["short"]) else f"{o['short']:5.2f}"
                to = "  nan" if not _fin(o["to"]) else f"{o['to'] * 100:5.1f}"
                L.append(f"{s:20s} {ic} {_t(o['ic'][1])} | {pc(o['q5q1'][0])} {_t(o['q5q1'][1])} | {pc(o['q5u'][0])} {_t(o['q5u'][1])} | "
                         f"{pc(o['q1u'][0])} {_t(o['q1u'][1])} | {sh} {to} | {pc(o['net'][0])} {_t(o['net'][1])}")
            L.append("")
    return L


def format_checks(info: dict, missing: dict) -> list:
    """CHECKS block: rows per region x decade, countries kept / dropped by the >= 100 rule, slow10 coverage, size_grp mix, missing columns."""
    cm = info["cm"].copy()
    cm["decade"] = (cm["mi"] // 12 // 10) * 10
    L = ["== CHECKS (printed before any result) ==", f"country-month rule: >= {MIN_STOCKS} eligible stocks (me > 0 and a next-month return); "
         f"a country-month enters a signal only with >= {MIN_RANKED} ranked names"]
    dec = sorted(cm["decade"].unique())
    L.append("raw screened rows per region and decade (before the rule) | rows after the rule:")
    for r in REGION_ORDER:
        c = cm[cm["region"] == r]
        if c.empty:
            continue
        a = c.groupby("decade")["rows"].sum()
        b = c[c["kept"]].groupby("decade")["n"].sum()
        L.append(f"  {r:18s} " + "  ".join(f"{int(d)}s {int(a.get(d, 0)):,}/{int(b.get(d, 0)):,}" for d in dec))
    L.append("countries: months kept / dropped by the rule, first-last kept month")
    for ctry in ["USA", *config.DEV_EX_US]:
        c = cm[cm["excntry"] == ctry]
        if c.empty:
            L.append(f"  {ctry}: NO DATA")
            continue
        k, dr = c[c["kept"]], c[~c["kept"]]
        span = "none kept (COUNTRY DROPPED)" if k.empty else f"{_ym(k['mi'].min())}..{_ym(k['mi'].max())}"
        drop = "" if dr.empty else f"; dropped {_ym(dr['mi'].min())}..{_ym(dr['mi'].max())}"
        L.append(f"  {ctry} ({COUNTRY_REGION[ctry]}): kept {len(k)} / dropped {len(dr)} months; kept {span}{drop}")
    L.append("share of kept rows with >= 8 of 10 slow themes available (jkp_slow10 defined):")
    for _, row in info["slow"].iterrows():
        L.append(f"  {str(row['region']):18s} {row['ok'] / max(row['rows'], 1):.1%} of {int(row['rows']):,}")
    L.append("size_grp mix of the raw screened rows:")
    for r in REGION_ORDER:
        z = info["size"][info["size"]["region"] == r]
        if len(z):
            L.append(f"  {r:18s} " + "  ".join(f"{g} {n / z['rows'].sum():.1%}" for g, n in zip(z["size_grp"], z["rows"])))
    for k, v in missing.items():
        L.append(f"missing theme-member columns ({k}): {len(v)} {v}")
    return L


def _ym(mi) -> str:
    return f"{int(mi) // 12}-{int(mi) % 12 + 1:02d}"
