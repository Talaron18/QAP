"""Forward-only machine-learning cross-sectional check (ROADMAP 'Pre-registration v8'): features, walk-forward model, evaluation, ledger.

Everything is built from Compustat only (comp.secm monthly rows through `qap.paper.build_pseudo_panel`, annual fundamentals through
`qap.fundamentals.prepare_annual`), so the identical code produces training data and forward data.

Universe: common stock, company ME >= NYSE 20th percentile of company ME in the month (`qap.evaluate.exclude_microcaps`), security ME > 0.
Annual items enter 4 months after the fiscal year end and are ignored once 548 days stale (as `qap.fundamentals.attach_pit_gvkey`).
Price-based features use only returns/ME through the formation month-end f; the label is the NEXT month's return (`fwd`, decimal).
Any feature that cannot be built is NaN (never imputed); ranking happens first and NaN becomes 0 afterwards (see `rank_transform`).

The 20 features (A = annual item of the latest PIT record, ME = company ME `me_co` in $ millions as the annual items, at_avg = (at + at_lag)/2
with both > 0 and `at_lag` only if the previous record is 300-430 days earlier; `*_lag` have the same guard):
  logme   log(me_co)                                         bm      be / me_co (be: Fama-French book equity, > 0)
  ep      ib / me_co                                         cfp     (oancf, else ib + dp) / me_co
  gpa     (revt - cogs) / at                                 roa     ib / at
  ag      at / at_lag - 1                                    capex   capx / at
  acc     Sloan accruals / at_avg: if oancf present (ib - oancf); else the balance-sheet version
          [(act - che) - (act - che)_lag] - [(lct - dlc) - (lct - dlc)_lag] - dp (taxes payable is not in the extract, so the usual
          'minus change in TP' is omitted; dp must be present). No lag -> NaN.
  lev     (dltt + dlc) / at (a missing one of the two counts 0, both missing -> NaN)      cash    che / at
  sg      sale / sale_lag - 1 (sale_lag > 0)                 rd      xrd / at, 0 if xrd missing but sale present
  issu    12-month change of log cshom, log cshom_f - log cshom_{f-12}, with detected split months removed: a month with
          |dlog cshom| > 0.10 whose dlog(ME) is within 25% of |dlog cshom| of log(1+ret) (price moved the opposite way, ME did not)
          contributes 0 (cshom is not split-adjusted; a genuine offering moves ME, so it is kept)
  mom     compound return over months f-11..f-1 (skips f; at least 10 of the 11 present; log1p clipped at -0.9999, as qap.replicate)
  rev     MINUS the return of month f (so a high value = recent loser = expected to rebound; sign +1)
  vol     sample std of the 12 monthly returns f-11..f (at least 10 present)
  beta    OLS slope of the stock's monthly return on the VW eligible-universe return over f-35..f (at least 24 pairs). The market
          return of month t weights the names eligible at t-1 by their ME at t-1 (names with a missing return are left out).
  turn    sale / at                                          opprof  (revt - cogs - xsga) / be (all four present, be > 0)
Windows are calendar windows on a full (permno, month) grid: a missing month is a missing observation, never a skipped row.
"""
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

from qap.evaluate import exclude_microcaps
from qap.fundamentals import attach_pit_gvkey, prepare_annual
from qap.paper import _iid_ord, build_pseudo_panel, complete_months
from qap.panel import to_mi

FEATURES = ["logme", "bm", "ep", "cfp", "gpa", "roa", "ag", "capex", "acc", "lev", "cash", "sg", "rd", "issu", "mom", "rev", "vol", "beta",
            "turn", "opprof"]
# literature signs for the equal-weight comparator (fixed in code before any run)
FEATURE_SIGNS = {"logme": -1, "bm": 1, "ep": 1, "cfp": 1, "gpa": 1, "roa": 1, "ag": -1, "capex": -1, "acc": -1, "lev": -1, "cash": 1,
                 "sg": -1, "rd": 1, "issu": -1, "mom": 1, "rev": 1, "vol": -1, "beta": -1, "turn": 1, "opprof": 1}
assert list(FEATURE_SIGNS) == FEATURES and len(FEATURES) == 20
SIGNALS = ["score_ml", "score_comp", "gpa"]
MIN_FEATURES_COVERAGE = 15
COST_ONE_WAY = 0.0010
TRAIN_START_MI = 1990 * 12
TRAIN_LAG = 2                 # last training formation month = January - 2 = previous November (label month = December)
MIN_TRAIN_ROWS = 1000
DECISION_N, DECISION_EDGE, DECISION_T = 36, 0.0015, 2.0
WALKFORWARD_LAST_MI = 2025 * 12 + 10      # information-only walk-forward stops at formation 2025-11 (return 2025-12): later months are forward-only
PSEUDO_FIRST, PSEUDO_LAST = 2025 * 12 + 11, 2026 * 12 + 7      # formation months 2025-12 .. 2026-08 are pseudo-forward

_ANNUAL_RAW = ["at", "seq", "ceq", "pstk", "pstkrv", "pstkl", "txditc", "lt", "sale", "revt", "cogs", "xsga", "xrd", "ib", "oancf", "dp",
               "capx", "dltt", "dlc", "che", "act", "lct"]
_ANNUAL_PIT = ["be", "at", "gpa", "roa", "ag", "capex", "acc", "lev", "cash", "sg", "rd", "turn", "opprof", "ib", "cfo"]
_TEST_MODEL_OVERRIDES: dict = {}       # tests only (e.g. a smaller max_iter); never set outside tests


def f64(s) -> np.ndarray:
    """Explicit float64 array (WRDS parquet gives nullable Int64/Float64, whose NA np.asarray cannot interpret)."""
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def ym(mi: int) -> str:
    return f"{int(mi) // 12}-{int(mi) % 12 + 1:02d}"


def parse_ym(s: str) -> int:
    y, m = s.split("-")
    return int(y) * 12 + int(m) - 1


def pseudo_permno(gvkey, iid) -> np.ndarray:
    """The pseudo permno of `qap.paper.build_pseudo_panel` from (gvkey, iid)."""
    g = pd.Series(gvkey).astype(str).str.strip().str.zfill(6).astype("int64").to_numpy()
    return g * 100 + _iid_ord(pd.Series(iid)).to_numpy()


# ---------------------------------------------------------------------------------------------------------------- annual features
def prepare_ml_annual(funda: pd.DataFrame, lag_months: int = 4) -> pd.DataFrame:
    """`prepare_annual` plus the fiscal-year ratios above (all float64; absent raw columns are all-NaN)."""
    f = funda.copy()
    for c in _ANNUAL_RAW:
        f[c] = f64(f[c]) if c in f.columns else np.nan
    a = prepare_annual(f, lag_months)
    g = a.groupby("gvkey")
    gap = (a["datadate"] - g["datadate"].shift(1)).dt.days
    ok = gap.between(300, 430)
    lag = {c: g[c].shift(1).where(ok) for c in ("sale", "act", "che", "lct", "dlc")}
    at = a["at"].where(a["at"] > 0)
    at_avg = ((a["at"] + a["at_lag"]) / 2).where((a["at"] > 0) & (a["at_lag"] > 0))
    a["gpa"] = a["gp"] / at
    a["roa"] = a["ib"] / at
    a["ag"] = a["at"] / a["at_lag"].where(a["at_lag"] > 0) - 1.0
    a["capex"] = a["capx"] / at
    cf_acc = (a["ib"] - a["oancf"]) / at_avg
    d_ca = (a["act"] - a["che"]) - (lag["act"] - lag["che"])
    d_cl = (a["lct"] - a["dlc"]) - (lag["lct"] - lag["dlc"])
    a["acc"] = cf_acc.where(a["oancf"].notna(), (d_ca - d_cl - a["dp"]) / at_avg)
    debt = a["dltt"].fillna(0.0) + a["dlc"].fillna(0.0)
    a["lev"] = (debt / at).where(a["dltt"].notna() | a["dlc"].notna())
    a["cash"] = a["che"] / at
    a["sg"] = a["sale"] / lag["sale"].where(lag["sale"] > 0) - 1.0
    a["rd"] = (a["xrd"] / at).where(a["xrd"].notna(), (0.0 / at).where(a["sale"].notna()))
    a["turn"] = a["sale"] / at
    a["opprof"] = (a["revt"] - a["cogs"] - a["xsga"]) / a["be"]
    a["cfo"] = a["oancf"].fillna(a["ib"] + a["dp"])
    return a


# ----------------------------------------------------------------------------------------------------------------- price features
def _rs(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing window-w sums along axis 0 (windows at the start are truncated; callers enforce counts)."""
    c = np.cumsum(a, axis=0)
    o = c.copy()
    o[w:] -= c[:-w]
    return o


def _shift1(x: np.ndarray) -> np.ndarray:
    o = np.full_like(x, np.nan)
    o[1:] = x[:-1]
    return o


def market_return(p: pd.DataFrame, elig: pd.DataFrame) -> pd.Series:
    """VW return of month t of the names eligible at t-1, weighted by their ME at t-1; indexed by t."""
    e = elig[["permno", "mi", "me"]].copy()
    e["mi"] = e["mi"] + 1
    j = e.merge(p[["permno", "mi", "mthret_adj"]], on=["permno", "mi"]).dropna(subset=["mthret_adj"])
    j["wr"] = j["me"] * j["mthret_adj"]
    g = j.groupby("mi")[["wr", "me"]].sum()
    return (g["wr"] / g["me"]).rename("mkt")


def price_features(p: pd.DataFrame, mkt: pd.Series, chunk: int = 4000) -> pd.DataFrame:
    """mom, rev, vol, beta, issu for every (permno, mi) row of `p`, from calendar-window sums on a (month x permno) grid."""
    mi_arr = p["mi"].to_numpy(dtype="int64")
    mi0, nm = int(mi_arr.min()), int(mi_arr.max() - mi_arr.min() + 1)
    codes, uniq = pd.factorize(p["permno"].to_numpy(dtype="int64"), sort=True)
    row = mi_arr - mi0
    ret_all = f64(p["mthret_adj"])
    me_all = f64(p["me"])
    sh_all = f64(p["cshom"]) if "cshom" in p.columns else np.full(len(p), np.nan)
    marr = np.full(nm, np.nan)
    marr[(mkt.index.to_numpy(dtype="int64") - mi0)] = mkt.to_numpy(dtype="float64")
    out = {k: np.full(len(p), np.nan) for k in ("mom", "rev", "vol", "beta", "issu")}
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(0, len(uniq) + chunk, chunk))
    for b in range(len(bounds) - 1):
        idx = order[bounds[b]:bounds[b + 1]]
        if len(idx) == 0:
            continue
        c0 = b * chunk
        col, rw = codes[idx] - c0, row[idx]
        nc = int(col.max()) + 1
        R = np.full((nm, nc), np.nan)
        R[rw, col] = ret_all[idx]
        ME = np.full((nm, nc), np.nan)
        ME[rw, col] = me_all[idx]
        SH = np.full((nm, nc), np.nan)
        SH[rw, col] = sh_all[idx]
        v = np.isfinite(R)
        r0 = np.where(v, R, 0.0)
        lr = np.where(v, np.log1p(np.clip(r0, -0.9999, None)), 0.0)
        # momentum f-11..f-1, at least 10 of 11
        n11 = _shift1(_rs(v.astype("float64"), 11))
        mom = np.where(n11 >= 10, np.expm1(_shift1(_rs(lr, 11))), np.nan)
        # volatility over f-11..f, at least 10 of 12
        n12, s1, s2 = _rs(v.astype("float64"), 12), _rs(r0, 12), _rs(r0 * r0, 12)
        with np.errstate(invalid="ignore", divide="ignore"):
            vol = np.sqrt(np.clip((s2 - s1 * s1 / n12) / (n12 - 1), 0.0, None))
        vol = np.where(n12 >= 10, vol, np.nan)
        # beta over f-35..f, at least 24 pairs
        vm = v & np.isfinite(marr)[:, None]
        x, m = np.where(vm, R, 0.0), np.where(vm, marr[:, None], 0.0)
        n36 = _rs(vm.astype("float64"), 36)
        sx, sm, sxm, smm = _rs(x, 36), _rs(m, 36), _rs(x * m, 36), _rs(m * m, 36)
        with np.errstate(invalid="ignore", divide="ignore"):
            den = smm - sm * sm / n36
            beta = (sxm - sx * sm / n36) / den
        beta = np.where((n36 >= 24) & (den > 1e-12), beta, np.nan)
        # net issuance: 12-month change of log shares minus detected split jumps
        with np.errstate(invalid="ignore", divide="ignore"):
            lcs = np.where(SH > 0, np.log(SH), np.nan)
            lme = np.where(ME > 0, np.log(ME), np.nan)
        dsh, dme = lcs - _shift1(lcs), lme - _shift1(lme)
        split = (np.abs(dsh) > 0.10) & (np.abs(dme - lr * v) < 0.25 * np.abs(dsh))
        adj = _rs(np.where(split, dsh, 0.0), 12)
        issu = lcs - np.vstack([np.full((12, nc), np.nan), lcs[:-12]])[:nm] - adj
        for k, arr in (("mom", mom), ("vol", vol), ("beta", beta), ("issu", issu)):
            out[k][idx] = arr[rw, col]
        out["rev"][idx] = -R[rw, col]
    return pd.DataFrame({"permno": p["permno"].to_numpy(dtype="int64"), "mi": mi_arr, **out})


def with_shares(pseudo: pd.DataFrame, secm: pd.DataFrame) -> pd.DataFrame:
    """Add the raw `cshom` of comp.secm to a `build_pseudo_panel` frame (which keeps the secm index and row order) for the `issu` feature."""
    assert pseudo.index.equals(secm.index), "pseudo panel and secm rows are not aligned"
    return pseudo.assign(cshom=f64(secm["cshom"]))


# ------------------------------------------------------------------------------------------------------------------ build_features
def build_features(pseudo: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Eligible-universe rows keyed (permno, gvkey, iid, mi) with the 20 FEATURES, me, me_co, primaryexch and `fwd` (next-month return).

    pseudo: `build_pseudo_panel` output, optionally with a `cshom` column (needed for `issu`); annual: `prepare_ml_annual` output.
    `fwd` is NaN where the next month has no return (always for the last month). Rows lacking any feature keep NaN there.
    """
    p = pseudo.copy()
    p["permno"] = p["permno"].astype("int64")
    p["mi"] = p["mi"].astype("int64")
    for c in ("me", "me_co", "mthret_adj"):
        p[c] = f64(p[c])
    form = p[p["eq"].astype(bool) & (p["me_co"] > 0) & (p["me"] > 0)]
    elig = exclude_microcaps(form)
    mkt = market_return(p, elig)
    pf = price_features(p, mkt)
    keys = elig[["permno", "gvkey", "iid", "mi", "mthcaldt", "me", "me_co", "primaryexch"]]
    k = attach_pit_gvkey(keys, annual, cols=_ANNUAL_PIT)
    k = k.merge(pf, on=["permno", "mi"], how="left")
    me_co = k["me_co"].astype("float64")
    k["logme"] = np.log(me_co)
    k["bm"] = k["be"] / me_co
    k["ep"] = k["ib"] / me_co
    k["cfp"] = k["cfo"] / me_co
    lab = p[["permno", "mi", "mthret_adj"]].assign(mi=lambda d: d["mi"] - 1).rename(columns={"mthret_adj": "fwd"})
    k = k.merge(lab, on=["permno", "mi"], how="left")
    cols = ["permno", "gvkey", "iid", "mi", *FEATURES, "me", "me_co", "primaryexch", "fwd"]
    k = k[cols].sort_values(["mi", "permno"]).reset_index(drop=True)
    k[FEATURES + ["me", "me_co", "fwd"]] = k[FEATURES + ["me", "me_co", "fwd"]].astype("float64")
    return k


def calibrate_on_raw(raw: pd.DataFrame, funda: pd.DataFrame) -> tuple:
    """`qap.paper.calibrate_secm_scale` on an already-built scale-1 pseudo panel (same median ratio, same power-of-ten rule and asserts)."""
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


def build_from_raw(secm: pd.DataFrame, funda: pd.DataFrame, max_mi: int = None) -> dict:
    """Raw comp.secm + comp.funda frames -> {pseudo, features, scale, ratio, last_mi, partial_mi}.

    The ME scale is calibrated exactly as `qap.paper.calibrate_secm_scale`. A partial latest month (fewer than half the usual rows with ME)
    is dropped before anything is built, so it can neither move the size cut nor become a label. `max_mi` hard-caps the panel.
    """
    raw = with_shares(build_pseudo_panel(secm, 1.0), secm)           # built once (the group-wise company-ME sum is slow on 1985- history)
    scale, ratio = calibrate_on_raw(raw, funda)
    pseudo = raw.assign(me=raw["me"] / scale, me_co=raw["me_co"] / scale)
    last = max(complete_months(pseudo, "me"))
    partial = int(pseudo["mi"].max()) if int(pseudo["mi"].max()) != last else None
    pseudo = pseudo[pseudo["mi"] <= last]
    if max_mi is not None:                                            # months after max_mi are removed before anything is built
        pseudo = pseudo[pseudo["mi"] <= max_mi]
        last = min(last, int(max_mi))
    return {"pseudo": pseudo, "features": build_features(pseudo, prepare_ml_annual(funda)), "scale": scale, "ratio": ratio, "last_mi": int(last),
            "partial_mi": partial}


def walkforward_inputs(b: dict, first_year: int = None, train_start_mi: int = None) -> dict:
    """Inputs of the information-only walk-forward from a `build_from_raw(..., max_mi=WALKFORWARD_LAST_MI + 1)` result.

    Formation months stop at WALKFORWARD_LAST_MI (2025-11), the returns table at the month after it, and only the January models up to that
    year are fitted, so the forward-only months (formation 2025-12 on, incl. the pseudo-forward ones) are never scored, evaluated or trained on.
    train_start / first_year default to the data-driven values of `choose_windows`; explicit infeasible ones raise `InfeasibleWindow`.
    """
    cap = min(int(b["last_mi"]), WALKFORWARD_LAST_MI)
    F = b["features"]
    F = F[F["mi"] <= cap]
    w = choose_windows(F, cap, train_start_mi, first_year)
    return {"features": F, "cap": cap, "years": list(range(w["first_year"], cap // 12 + 1)), "ret": return_table(b["pseudo"], cap + 1),
            "train_start": w["train_start"], "first_year": w["first_year"], "windows": w}


MIN_TRAIN_MONTHS = 36


class InfeasibleWindow(ValueError):
    """The requested (or any) training window has too little usable history; the message lists the feasible values."""


def usable_counts(F: pd.DataFrame) -> pd.Series:
    """Rows per formation month usable for training: forward return present and >= MIN_FEATURES_COVERAGE of the 20 features present."""
    ok = F["fwd"].notna() & (F[FEATURES].notna().sum(axis=1) >= MIN_FEATURES_COVERAGE)
    return F[ok].groupby("mi").size()


def train_stats(counts: pd.Series, train_start_mi: int, year: int, lag: int = TRAIN_LAG) -> tuple:
    """(usable rows, formation months with usable rows) in the training window of the January-`year` model."""
    lo, hi = train_window(year, train_start_mi, lag)
    c = counts[(counts.index >= lo) & (counts.index <= hi)]
    return int(c.sum()), int((c > 0).sum())


def choose_windows(F: pd.DataFrame, last_mi: int, train_start_mi: int = None, first_year: int = None) -> dict:
    """Data-driven training start and first scored year (explicit values are kept but must be feasible, else `InfeasibleWindow`).

    train_start (default): the first formation month >= 1990-01 from which the following 12 months hold >= MIN_TRAIN_ROWS usable rows
    (usable: forward return present and >= 15 of 20 features), so sparse early months are not trained on.
    first_year (default): the first year whose January model has >= MIN_TRAIN_ROWS usable training rows over >= MIN_TRAIN_MONTHS
    formation months (window train_start .. previous November). Returns train_start, first_year and the feasible values.
    """
    c = usable_counts(F)
    if c.empty:
        raise InfeasibleWindow("no formation month has a forward return and >= 15 of 20 features: nothing to train on")
    full = c.reindex(range(int(c.index.min()), int(c.index.max()) + 1), fill_value=0)
    span = full.rolling(12).sum().shift(-11)                              # rows in months m .. m+11
    cand = span[(span >= MIN_TRAIN_ROWS) & (span.index >= TRAIN_START_MI)]
    if cand.empty:
        raise InfeasibleWindow(f"no 12-month span from {ym(TRAIN_START_MI)} holds {MIN_TRAIN_ROWS} usable rows (rows per year: {dict(full.groupby(full.index // 12).sum())})")
    auto_ts = int(cand.index[0])
    ts = auto_ts if train_start_mi is None else int(train_start_mi)
    feas = next((y for y in range(ts // 12, last_mi // 12 + 1)
                 if (lambda r: r[0] >= MIN_TRAIN_ROWS and r[1] >= MIN_TRAIN_MONTHS)(train_stats(c, ts, y))), None)
    hint = f"first feasible values: train_start {ym(auto_ts)}, first_year {feas} (usable data begins {ym(int(c.index.min()))})"
    if feas is None:
        raise InfeasibleWindow(f"train_start {ym(ts)} leaves no January model with >= {MIN_TRAIN_ROWS} rows and >= {MIN_TRAIN_MONTHS} months before "
                               f"{ym(last_mi)}; {hint}")
    fy = feas if first_year is None else int(first_year)
    if fy < feas:
        rows, months = train_stats(c, ts, fy)
        raise InfeasibleWindow(f"first_year {fy} with train_start {ym(ts)} has only {rows} usable training rows over {months} months "
                               f"(needs >= {MIN_TRAIN_ROWS} and >= {MIN_TRAIN_MONTHS}); {hint}")
    return {"train_start": ts, "first_year": fy, "auto_train_start": auto_ts, "feasible_first_year": feas}


def raw_coverage_lines(secm: pd.DataFrame) -> list:
    """Per calendar year of datadate of the raw comp.secm rows: rows and shares with trt1m, prccm, cshom present and exchg == 11."""
    y = pd.to_datetime(secm["datadate"]).dt.year.to_numpy()
    d = pd.DataFrame({"year": y, "trt1m": f64(secm["trt1m"]) == f64(secm["trt1m"]), "prccm": f64(secm["prccm"]) == f64(secm["prccm"]),
                      "cshom": f64(secm["cshom"]) == f64(secm["cshom"]), "nyse": f64(secm["exchg"]) == 11})
    g = d.groupby("year")
    t = pd.concat([g.size().rename("rows"), g.mean()], axis=1)
    lines = ["RAW comp.secm by calendar year of datadate (share of rows with ...)", f"{'year':<6}{'rows':>9}{'trt1m':>8}{'prccm':>8}{'cshom':>8}{'exchg=11':>10}"]
    lines += [f"{int(yr):<6}{int(r['rows']):>9,}{r['trt1m']:>8.0%}{r['prccm']:>8.0%}{r['cshom']:>8.0%}{r['nyse']:>10.0%}" for yr, r in t.iterrows()]
    return lines


def panel_coverage_lines(F: pd.DataFrame, pseudo: pd.DataFrame = None) -> list:
    """Per formation year of the built panel: formation rows (ME > 0, if `pseudo` given), eligible rows, with a forward return, with >= 15 of 20
    features, and the shares with beta / mom / vol among eligible rows."""
    yr = F["mi"] // 12
    d = pd.DataFrame({"year": yr, "fwd": F["fwd"].notna(), "f15": F[FEATURES].notna().sum(axis=1) >= MIN_FEATURES_COVERAGE,
                      "beta": F["beta"].notna(), "mom": F["mom"].notna(), "vol": F["vol"].notna()})
    g = d.groupby("year")
    t = pd.concat([g.size().rename("elig"), g.sum()], axis=1)
    form = None
    if pseudo is not None:
        pf = pseudo[pseudo["eq"].astype(bool) & (f64(pseudo["me_co"]) > 0) & (f64(pseudo["me"]) > 0)]
        form = pf.groupby(pf["mi"] // 12).size()
    lines = ["BUILT panel by formation year (eligible = common stock, ME >= NYSE 20th percentile)",
             f"{'year':<6}{'formation':>10}{'eligible':>10}{'w/ fwd ret':>11}{'>=15 feat':>10}{'beta':>7}{'mom':>7}{'vol':>7}"]
    for yv, r in t.iterrows():
        n = r["elig"]
        lines.append(f"{int(yv):<6}{(f'{int(form.get(yv, 0)):,}' if form is not None else '-'):>10}{int(n):>10,}{int(r['fwd']):>11,}{int(r['f15']):>10,}"
                     f"{r['beta'] / n:>7.0%}{r['mom'] / n:>7.0%}{r['vol'] / n:>7.0%}")
    return lines


def coverage_text(secm: pd.DataFrame, F: pd.DataFrame, pseudo: pd.DataFrame = None) -> str:
    return "\n".join(raw_coverage_lines(secm) + [""] + panel_coverage_lines(F, pseudo))


def return_table(pseudo: pd.DataFrame, max_mi: int = None) -> pd.DataFrame:
    """(permno, mi, ret) of every row with a return (unfiltered by the formation universe, so labels carry no survivorship filter)."""
    r = pseudo[["permno", "mi", "mthret_adj"]].rename(columns={"mthret_adj": "ret"}).dropna(subset=["ret"])
    return r if max_mi is None else r[r["mi"] <= max_mi]


def coverage(features: pd.DataFrame, mi: int) -> tuple:
    """(share of rows with each feature present, share of rows with >= MIN_FEATURES_COVERAGE features) in one month."""
    d = features[features["mi"] == mi]
    pres = d[FEATURES].notna()
    return pres.mean(), float((pres.sum(axis=1) >= MIN_FEATURES_COVERAGE).mean()) if len(d) else float("nan")


# ------------------------------------------------------------------------------------------------------------------ rank / model
def rank_transform(df: pd.DataFrame, cols, by: str = "mi", fill_missing: bool = True) -> pd.DataFrame:
    """Within-`by` percentile rank mapped to [-0.5, 0.5]: (average rank - 1) / (n - 1) - 0.5 over the n present values (n == 1 -> 0).

    Ranking uses only present values; NaN is set to 0 (the neutral rank) AFTER ranking when fill_missing, else stays NaN.
    Returns a new frame with the ranked `cols` (same index); the input is untouched.
    """
    out = {}
    for c in cols:
        x = pd.Series(f64(df[c]), index=df.index)
        gx = x.groupby(df[by])
        r, n = gx.rank(method="average"), gx.transform("count")
        v = ((r - 1.0) / (n - 1.0).where(n > 1) - 0.5).where(n > 1, 0.0).where(x.notna())
        out[c] = v.fillna(0.0) if fill_missing else v
    return pd.DataFrame(out, index=df.index)


def make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    spec = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=500, l2_regularization=1.0, random_state=0, early_stopping=False)
    spec.update(_TEST_MODEL_OVERRIDES)
    return HistGradientBoostingRegressor(**spec)


def rank_features(features: pd.DataFrame) -> pd.DataFrame:
    """Model frame: ids, me, the 20 rank-transformed features, the within-month rank of `fwd` as `y`, `score_comp`, and `gpa_rank` (NaN kept)."""
    f = features.sort_values(["mi", "permno"]).reset_index(drop=True)
    R = f[["permno", "gvkey", "iid", "mi", "me"]].copy()
    R[FEATURES] = rank_transform(f, FEATURES)
    R["y"] = rank_transform(f, ["fwd"], fill_missing=False)["fwd"]
    signs = np.array([FEATURE_SIGNS[c] for c in FEATURES], dtype="float64")
    R["score_comp"] = (R[FEATURES].to_numpy(dtype="float64") * signs).mean(axis=1)
    R["gpa_rank"] = rank_transform(f, ["gpa"], fill_missing=False)["gpa"]
    return R


def model_year(mi: int) -> int:
    """The calendar year whose January retrain scores formation month `mi` (the plain protocol reading)."""
    return int(mi) // 12


def train_window(year: int, train_start_mi: int = TRAIN_START_MI, lag: int = TRAIN_LAG) -> tuple:
    """(first, last) formation month of the training set of the January-`year` model; lag=2 -> last = previous November."""
    return train_start_mi, year * 12 - lag


def assert_no_leakage(train_mi_max: int, pred_mi_min: int) -> None:
    """Every training label month (formation + 1) must lie strictly before the first prediction formation month."""
    assert train_mi_max + 1 < pred_mi_min, (
        f"leakage: last training formation {ym(train_mi_max)} has label month {ym(train_mi_max + 1)}, not before first prediction month {ym(pred_mi_min)}")


def fit_predict_year(R: pd.DataFrame, year: int, last_mi: int, train_start_mi: int = TRAIN_START_MI, lag: int = TRAIN_LAG,
                     pred_months=None) -> pd.DataFrame:
    """Train the January-`year` model and score its formation months (January..December, capped at last_mi, optionally only `pred_months`).

    Returns permno, gvkey, iid, mi, me, score_ml, score_comp, gpa (gpa = within-month rank in [-0.5, 0.5], NaN if gpa is missing).
    """
    lo, hi = train_window(year, train_start_mi, lag)
    tr = R[(R["mi"] >= lo) & (R["mi"] <= hi) & R["y"].notna()].sort_values(["mi", "permno"])
    pm = [m for m in range(year * 12, min(year * 12 + 11, last_mi) + 1) if pred_months is None or m in set(pred_months)]
    pr = R[R["mi"].isin(pm)].sort_values(["mi", "permno"])
    if len(pr) == 0:
        return pd.DataFrame(columns=["permno", "gvkey", "iid", "mi", "me", "score_ml", "score_comp", "gpa"])
    assert len(tr) >= MIN_TRAIN_ROWS, f"only {len(tr)} training rows for the {year} model"
    assert_no_leakage(int(tr["mi"].max()), int(pr["mi"].min()))
    model = make_model().fit(tr[FEATURES].to_numpy(dtype="float64"), tr["y"].to_numpy(dtype="float64"))
    out = pr[["permno", "gvkey", "iid", "mi", "me"]].copy()
    out["score_ml"] = model.predict(pr[FEATURES].to_numpy(dtype="float64"))
    out["score_comp"] = pr["score_comp"].to_numpy()
    out["gpa"] = pr["gpa_rank"].to_numpy()
    return out.reset_index(drop=True)


def walk_forward(features: pd.DataFrame, first_pred_year: int, last_mi: int, train_start_mi: int = TRAIN_START_MI,
                 lag: int = TRAIN_LAG) -> pd.DataFrame:
    """Expanding-window walk-forward: retrain each January on formation months through the previous November, score that year's months."""
    R = rank_features(features)
    parts = [fit_predict_year(R, y, last_mi, train_start_mi, lag) for y in range(first_pred_year, last_mi // 12 + 1)]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------- evaluation
def tstat(x) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def paired_t(a, b) -> tuple:
    """(mean of a - b, paired t, n) over the months present in both (aligned on the index)."""
    d = (pd.Series(a) - pd.Series(b)).dropna()
    return (float(d.mean()) if len(d) else float("nan")), tstat(d), int(len(d))


def monthly_rank_ic(sc: pd.DataFrame, ret: pd.DataFrame, col: str) -> pd.Series:
    """Spearman IC per formation month between `col` and the NEXT month's return; NaN-safe (needs >= 3 names and non-constant ranks)."""
    j = sc[["permno", "mi", col]].dropna()
    r = ret[["permno", "mi", "ret"]].assign(mi=lambda d: d["mi"] - 1)
    j = j.merge(r, on=["permno", "mi"]).dropna(subset=["ret"])
    if j.empty:
        return pd.Series(dtype="float64")
    a, b = j.groupby("mi")[col].rank(), j.groupby("mi")["ret"].rank()
    a, b = a - a.groupby(j["mi"]).transform("mean"), b - b.groupby(j["mi"]).transform("mean")
    num, da, db = (a * b).groupby(j["mi"]).sum(), (a * a).groupby(j["mi"]).sum(), (b * b).groupby(j["mi"]).sum()
    n = j.groupby("mi").size()
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / np.sqrt(da * db)
    return ic.where((n >= 3) & (da > 1e-12) & (db > 1e-12)).rename(col)


def top_quintile_weights(sc: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    """VW (by ME) weights (permno, mi, w) of the top quintile of `col` among names with the score and ME > 0."""
    s = sc[["permno", "mi", "me", col]].copy()
    s = s[s[col].notna() & (s["me"] > 0)]
    pct = s.groupby("mi")[col].rank(pct=True)
    s = s[np.minimum(np.ceil(pct * q), q) == q]
    s["w"] = s["me"] / s.groupby("mi")["me"].transform("sum")
    return s[["permno", "mi", "w"]]


def universe_weights(sc: pd.DataFrame) -> pd.DataFrame:
    """VW weights of ALL eligible names (ME > 0): the common benchmark for every signal."""
    s = sc.loc[sc["me"] > 0, ["permno", "mi", "me"]].copy()
    s["w"] = s["me"] / s.groupby("mi")["me"].transform("sum")
    return s[["permno", "mi", "w"]]


def gross_return(W: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Return earned in month f+1 by weights formed at f (missing returns count 0), indexed by the FORMATION month f."""
    r = ret[["permno", "mi", "ret"]].assign(mi=lambda d: d["mi"] - 1)
    m = W.merge(r, on=["permno", "mi"], how="left")
    m["wr"] = m["w"] * m["ret"].fillna(0.0)
    return m.groupby("mi")["wr"].sum()


def one_way_turnover(W: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """One-way turnover at each formation month f: 0.5 * sum |w_f - w_{f-1} drifted by month-f returns|, indexed by f.

    The first month, and any month whose predecessor is absent, counts as a full build (turnover 1.0, i.e. 10 bp at the pre-registered cost).
    """
    months = np.sort(W["mi"].unique())
    prev = W.rename(columns={"w": "w_prev"}).assign(mi=lambda d: d["mi"] + 1)
    prev = prev.merge(ret[["permno", "mi", "ret"]], on=["permno", "mi"], how="left")
    prev["g"] = prev["w_prev"] * (1.0 + prev["ret"].fillna(0.0))
    prev["w_pre"] = prev["g"] / prev.groupby("mi")["g"].transform("sum")
    m = W.merge(prev[["permno", "mi", "w_pre"]], on=["permno", "mi"], how="outer")
    m[["w", "w_pre"]] = m[["w", "w_pre"]].fillna(0.0)
    m = m[m["mi"].isin(months)]
    t = m.assign(dw=(m["w"] - m["w_pre"]).abs()).groupby("mi")["dw"].sum() / 2.0
    first = [mi for mi in months if (mi - 1) not in set(months)]
    t.loc[first] = 1.0
    return t


def evaluate_scores(sc: pd.DataFrame, ret: pd.DataFrame, signals=SIGNALS, cost: float = COST_ONE_WAY) -> pd.DataFrame:
    """Per formation month: bench (VW universe gross), and for each signal s: ic_s, top_s, gross_s (top minus universe), turn_s, net_s.

    sc: permno, mi, me and the signal columns, covering ALL eligible names (the benchmark is common to all signals). ret: permno, mi, ret
    (decimal, month of the return). Only formation months whose next month has returns in `ret` are evaluated.
    net = gross - one-way turnover * cost (cost on the long book only; the benchmark is costless).
    """
    have = set(ret["mi"].unique())
    sc = sc[(sc["mi"] + 1).isin(have)]
    out = {"bench": gross_return(universe_weights(sc), ret)}
    for s in signals:
        W = top_quintile_weights(sc, s)
        top = gross_return(W, ret)
        turn = one_way_turnover(W, ret)
        out[f"ic_{s}"] = monthly_rank_ic(sc, ret, s)
        out[f"top_{s}"] = top
        out[f"gross_{s}"] = top - out["bench"]
        out[f"turn_{s}"] = turn
        out[f"net_{s}"] = out[f"gross_{s}"] - turn * cost
    return pd.DataFrame(out).sort_index()


def summarize(m: pd.DataFrame, signals=SIGNALS) -> dict:
    """Mean and t of IC, gross and net long-minus-universe per signal, mean turnover, and paired net differences ML-comp, ML-gpa."""
    res = {"n": int(len(m))}
    for s in signals:
        res[s] = {"ic": float(m[f"ic_{s}"].mean()), "ic_t": tstat(m[f"ic_{s}"]), "gross": float(m[f"gross_{s}"].mean()), "gross_t": tstat(m[f"gross_{s}"]),
                  "net": float(m[f"net_{s}"].mean()), "net_t": tstat(m[f"net_{s}"]), "turn": float(m[f"turn_{s}"].mean())}
    if "score_ml" in signals:
        for other, name in (("score_comp", "ml_minus_comp"), ("gpa", "ml_minus_gpa")):
            if other in signals:
                mean, t, n = paired_t(m["net_score_ml"], m[f"net_{other}"])
                res[name] = {"mean": mean, "t": t, "n": n}
    return res


def decision_label(m: pd.DataFrame) -> str:
    """Pre-registered DECISION on the FIRST 36 settled forward months (rows of `m`, in time order).

    'ML ADDS VALUE' if mean(net_ml - net_comp) >= 0.15 %/mo with paired t >= 2 and mean net_ml > mean net_gpa (plain reading of 'exceeds
    gpa's'); else 'NO EVIDENCE'; fewer than 36 months -> 'NOT YET (n<36)'. Boundaries are inclusive (tolerance 1e-12).
    """
    d = m.sort_index().dropna(subset=["net_score_ml", "net_score_comp", "net_gpa"]).iloc[:DECISION_N]
    if len(d) < DECISION_N:
        return "NOT YET (n<36)"
    mean, t, _ = paired_t(d["net_score_ml"], d["net_score_comp"])
    if mean >= DECISION_EDGE - 1e-12 and t >= DECISION_T - 1e-12 and d["net_score_ml"].mean() > d["net_gpa"].mean():
        return "ML ADDS VALUE"
    return "NO EVIDENCE"


# ------------------------------------------------------------------------------------------------------------------------- ledger
FORM_COLS = ["gvkey", "iid", "mi", "me", "score_ml", "score_comp", "gpa"]


def model_id(train_end_mi: int, train_start_mi: int = None) -> str:
    import sklearn
    lo = f"{ym(train_start_mi)}<=" if train_start_mi is not None else ""
    return f"hgb(d3,lr0.05,it200,leaf500,l2=1,rs0;sklearn {sklearn.__version__});train {lo}formation<={ym(train_end_mi)}"


def make_form(scores: pd.DataFrame, mi: int) -> pd.DataFrame:
    f = scores[scores["mi"] == mi][FORM_COLS].copy()
    f["mi"] = f["mi"].astype("int64")
    for c in ("me", "score_ml", "score_comp", "gpa"):
        f[c] = f64(f[c])
    return f.sort_values(["gvkey", "iid"]).reset_index(drop=True)


def score_sha(form: pd.DataFrame) -> str:
    """Order-independent sha256 of a formation file (ME to 6, scores to 10 decimals; NaN gpa prints as 'nan')."""
    f = form.sort_values(["gvkey", "iid"])
    fmt = lambda x, d: f"{float(np.round(x, d)) + 0.0:.{d}f}"
    s = "\n".join(f"{g}|{i}|{int(mi)}|{fmt(me, 6)}|{fmt(a, 10)}|{fmt(b, 10)}|{fmt(c, 10)}"
                  for g, i, mi, me, a, b, c in zip(f["gvkey"], f["iid"], f["mi"], f["me"], f["score_ml"], f["score_comp"], f["gpa"]))
    return hashlib.sha256(s.encode()).hexdigest()


def manifest_line(mi: int, sha: str, commit: str, formed_on: str, mid: str, n: int = 0) -> str:
    return json.dumps({"mi": int(mi), "ym": ym(mi), "sha256": sha, "spec_commit": commit, "formed_on": formed_on, "model": mid, "n": int(n)})


def manifest_path(root=".") -> pathlib.Path:
    p = pathlib.Path(root, "paper_ledger", "ml_manifest.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def form_path(mi: int, root=".") -> pathlib.Path:
    return pathlib.Path(root, "data", "ml", "forms", f"{mi // 12}{mi % 12 + 1:02d}.parquet")


def read_manifest(root=".") -> list:
    p = manifest_path(root)
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def freeze_month(form: pd.DataFrame, mi: int, commit: str, formed_on: str, mid: str, root=".") -> str:
    """Write the frozen file and append the manifest line. Refuses (FileExistsError) if the file or a manifest entry already exists."""
    f = form_path(mi, root)
    if f.exists() or any(m["mi"] == int(mi) for m in read_manifest(root)):
        raise FileExistsError(f"formation {ym(mi)} is already frozen")
    f.parent.mkdir(parents=True, exist_ok=True)
    sha = score_sha(form)
    form.to_parquet(f, index=False)
    with open(manifest_path(root), "a") as fh:
        fh.write(manifest_line(mi, sha, commit, formed_on, mid, len(form)) + "\n")
    return sha


def load_verified(root=".") -> list:
    """[(manifest entry, form)] with every file's sha checked against its manifest line (AssertionError if a file was modified)."""
    out = []
    for m in read_manifest(root):
        form = pd.read_parquet(form_path(m["mi"], root))
        assert score_sha(form) == m["sha256"], f"ledger file for {m['ym']} was modified after it was written"
        out.append((m, form))
    return out


def form_group(mi: int, formed_on: str) -> str:
    """pseudo-forward (formation 2025-12..2026-08) / prospective (formed within 7 days of the target month's start) / retrospective."""
    import datetime as dt
    if PSEUDO_FIRST <= mi <= PSEUDO_LAST:
        return "pseudo-forward"
    tgt = dt.date((mi + 1) // 12, (mi + 1) % 12 + 1, 1)
    return "prospective" if dt.date.fromisoformat(formed_on) <= tgt + dt.timedelta(days=7) else "retrospective"
