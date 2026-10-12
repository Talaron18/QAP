"""Study G (pre-registration v14, Stage B at the INDEX level): does the model forecast carry information beyond VIX, and is there a
variance-risk-premium signal? Holdout sealed and gated on the v13 holdout.

    python scripts/wrds_extract.py cboe_vix [sp500_daily]      # once (cboe.cboe; sp500_daily is an optional robustness input)
    python scripts/extract_earnings.py                         # daily returns / FF (once; Study D / F)
    python scripts/run_vrp.py [--open-holdout]

Writes data/study_g_report.txt. Logs exactly ONE trial (T3, "studyG:vrp_payoff"; idempotent). Licenses no trade of any kind: the unconditional
variance risk premium is compensation for crash risk, not an edge. Without --open-holdout no forecast dated 2020 or later is computed and the
report contains no 2020+ number. --open-holdout REFUSES unless the v13 holdout has been opened (data/holdout_v13.done); it consults the holdout
only if T1 and T3 both passed in development and validation, once (data/holdout_v14.done, written after the holdout statistics are computed).
"""
import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from qap import volfc as V
from qap import vrp as VR
from qap.study import require_inputs
from qap.trials import trial_count

RAW = pathlib.Path("data/raw")
WINDOWS = {"dev": V.DEV, "val": V.VAL}


def main(open_holdout: bool) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, open_holdout)
    except V.HoldoutRefused:
        raise                                       # a refusal must not overwrite an earlier report
    except BaseException:
        _write_report(report)
        raise
    _write_report(report)


def _write_report(report) -> None:
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/study_g_report.txt").write_text("\n".join(report) + "\n")


# ----------------------------------------------------------------------------------------------------------- helpers
def _n(x, d=3) -> str:
    return VR.fmt_num(x, d)


def _yn(b) -> str:
    return "Y" if b else "N"


def _t1_line(label, f) -> str:
    return (f"    {label:4s} n {f['n']:3d}  a {_n(f['a'])}  b (log IV2) {_n(f['b'])} (t {_n(f['t_b'], 2)})  c (log F) {_n(f['c'])} (t {_n(f['t_c'], 2)})  "
            f"R2 {_n(f['r2'])}  | single-regressor R2: IV2 only {_n(f['r2_iv'])}, F only {_n(f['r2_f'])}")


def _t3_line(label, f) -> str:
    return f"    {label:4s} n {f['n']:3d}  intercept {_n(f['a'], 4)}  slope {_n(f['slope'], 4)} (t {_n(f['t'], 2)})  R2 {_n(f['r2'], 4)}"


def _series_on_calendar(dates, values, cal) -> np.ndarray:
    """Values on the calendar (NaN where the date is absent), exact date match."""
    dd = pd.to_datetime(pd.Series(dates), errors="coerce")
    ok = dd.notna().to_numpy()
    d64 = dd[ok].to_numpy().astype("datetime64[D]")
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy("float64")[ok]
    pos = np.searchsorted(cal, d64)
    pc = np.minimum(pos, max(len(cal) - 1, 0))
    hit = (pos < len(cal)) & (cal[pc] == d64) if len(cal) else np.zeros(len(d64), bool)
    out = np.full(len(cal), np.nan)
    out[pos[hit]] = v[hit]
    return out


def _decide_t1_t3(sample, model=VR.PRIMARY_MODEL):
    dev, val = VR.in_window(sample, V.DEV), VR.in_window(sample, V.VAL)
    t1d, t1v = VR.t1_fit(dev, model), VR.t1_fit(val, model)
    t3d, t3v = VR.t3_fit(dev), VR.t3_fit(val)
    return (t1d, t1v, VR.t1_rule(t1d, t1v)), (t3d, t3v, VR.t3_rule(t3d, t3v)), dev, val


# ----------------------------------------------------------------------------------------------------------------- run
def _run(log, open_holdout) -> None:
    if open_holdout:
        VR.require_v13_marker()                       # the new gate: refuses before anything is read or computed
    hint = ("Run: python scripts/wrds_extract.py cboe_vix  and  python scripts/extract_earnings.py")
    require_inputs([RAW / "cboe_vix.parquet", RAW / "ff_daily.parquet", RAW / "crsp_daily_ret"], hint)
    cal, daily_rows = VR.load_calendar(RAW / "crsp_daily_ret")
    last_year = int(str(cal[-1])[:4])
    cut_year = min(last_year, int(VR.PRE_END[:4]))
    ff = pd.read_parquet(RAW / "ff_daily.parquet")
    mkt = V.market_series(ff, cal)
    vix_raw = pd.read_parquet(RAW / "cboe_vix.parquet")
    vx = VR.prepare_vix(vix_raw, cal, before=VR.SEAL)
    dm = VR.market_rows(mkt, cal, cut_year)                       # sealed: forecast dates <= 2019-12-31 only
    dm = dm[dm["date"] <= np.datetime64(VR.PRE_END)].reset_index(drop=True)
    smp = VR.select_samples(dm, cal, vx["vix"])
    prim, daily = smp["primary"], smp["daily"]
    pre = cal < np.datetime64(VR.SEAL)
    span = pre & (cal >= np.datetime64(f"{V.FIRST_FC_YEAR}-01-01"))

    log("== CHECKS (printed before any result) ==")
    if len(prim) == 0:
        log("  [FAIL] no usable primary-sample observations")
        raise AssertionError("Study G: empty primary sample")
    gj = np.log(prim[f"F_{VR.PRIMARY_MODEL}"].to_numpy())
    sanity = {"median_vix": vx["stats"]["median_vix"], "mean_iv2": float(prim["iv2"].mean()), "mean_rv": float(prim["rv"].mean()),
              "corr_log": float(np.corrcoef(np.log(prim["iv2"].to_numpy()), gj)[0, 1]) if len(prim) > 2 else float("nan")}
    scount = {k: (len(VR.in_window(prim, w)), len(VR.in_window(daily, w))) for k, w in WINDOWS.items()}
    for line in VR.vrp_checks(vx["stats"], int(pre.sum()), (int(np.isfinite(vx["vix"][span]).sum()), int(span.sum())), smp["counts"], scount,
                              sanity, (int(np.isfinite(mkt[pre]).sum()), int(pre.sum())), daily_rows):
        log(line)
    ok, lines = VR.repro_check(dm, cal)
    log("Study F reproduction check (v14 pipeline through the unchanged volfc engine on the CRSP-union calendar; mean QLIKE, market series; "
        f"dev {V.DEV[0][:7]}..{V.DEV[1][:7]}, validation {V.VAL[0][:7]}..{V.VAL[1][:7]}):")
    for line in lines:
        log(line)
    if not ok:
        log("  ABORT: the v14 pipeline does not reproduce the v13 market table (calendar or index misalignment, or a changed engine)")
        raise AssertionError("Study G: reproduction check failed")

    (t1d, t1v, t1r), (t3d, t3v, t3r), dev, val = _decide_t1_t3(prim)
    t1_m2 = (VR.t1_fit(dev, VR.SECOND_MODEL), VR.t1_fit(val, VR.SECOND_MODEL))
    t1_m2r = VR.t1_rule(*t1_m2)

    log("\n== RESULTS (Study G, pre-registration v14): the FF market series, Study F engine, primary sample = non-overlapping 21-day windows ==")
    log("It licenses no trade of any kind (no tradable instrument, cost or margin data; nothing here licenses VIX futures, VIX ETNs or options). The unconditional "
        "variance risk premium is compensation for crash risk, not an edge. An index-level 'VIX already encompasses the forecast' is only a weak prior for single stocks.")
    log(f"\n-- T1 (information content; not a trial): log RV_win = a + b log IV2_win + c log F_win(model) + e; Newey-West {VR.NW_PRIMARY} lags; validation re-estimated on its own window --")
    log(f"  M3 gjr_t (PRIMARY):")
    log(_t1_line("dev", t1d))
    log(_t1_line("val", t1v))
    log(f"  M2 garch (beside it):")
    log(_t1_line("dev", t1_m2[0]))
    log(_t1_line("val", t1_m2[1]))
    log(f"  rule 'THE FORECAST ADDS INFORMATION BEYOND VIX' (dev c >= {VR.C_MIN:.2f} AND t(c) >= {VR.T_MIN:.1f} AND validation c > 0):")
    log(f"    M3 gjr_t: dev c>=0.10 {_yn(t1d['c'] >= VR.C_MIN)}  dev t(c)>=2.0 {_yn(t1d['t_c'] >= VR.T_MIN)}  val c>0 {_yn(t1v['c'] > 0)}  => "
        f"{'PASS' if t1r['passed'] else 'FAIL'}")
    log(f"    M2 garch (reported beside; the rule is not relaxed): => {'PASS' if t1m2_pass(t1_m2r) else 'FAIL'}")

    log(f"\n-- T3 (economic payoff; ONE trial): Y = (RV_win - IV2_win) / IV2_win on D = log F_win(M3) - log IV2_win; Newey-West {VR.NW_PRIMARY} lags --")
    log(_t3_line("dev", t3d))
    log(_t3_line("val", t3v))
    log(f"  rule 'A VARIANCE-RISK-PREMIUM SIGNAL EXISTS' (dev slope t >= {VR.T3_T_MIN:.1f} AND validation slope > 0): dev t>=2.0 {_yn(t3d['t'] >= VR.T3_T_MIN)}  "
        f"val slope>0 {_yn(t3v['slope'] > 0)}  => {'PASS' if t3r['passed'] else 'FAIL'}")
    wrote = VR.log_trial_once(VR.TRIAL_NAME, VR.trial_spec(), {
        "dev_slope": t3d["slope"], "dev_t": t3d["t"], "dev_n": t3d["n"], "val_slope": t3v["slope"], "val_t": t3v["t"], "val_n": t3v["n"],
        "t1_passed": t1r["passed"], "t3_passed": t3r["passed"], "dev_c": t1d["c"], "dev_t_c": t1d["t_c"], "val_c": t1v["c"]})
    log(f"  T3 is one trial ({VR.TRIAL_NAME}): {'logged' if wrote else 'already in the registry, not logged again'}")

    dec = VR.decision(t1r, t3r)
    log("\n== DECISION (the case for buying stock-level option data is made ONLY if BOTH T1 and T3 pass in development and validation) ==")
    log(f"  T1 (M3) {'PASS' if t1r['passed'] else 'FAIL'}   T3 {'PASS' if t3r['passed'] else 'FAIL'}")
    hold = None
    if open_holdout and dec["dev_val"]:
        V.assert_holdout_unopened(VR.MARKER_V14, "v14 (Study G)")
        hold = _holdout(ff, cal, last_year)
        dec = VR.decision(t1r, t3r, True, {"c": hold["t1"]["c"], "slope": hold["t3"]["slope"]})
        V.mark_holdout_opened(f"v14 holdout opened {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {dec['status']}", VR.MARKER_V14)
    log(f"  {dec['status']}")
    if open_holdout and not dec["dev_val"]:
        log("  --open-holdout given, but the holdout is NOT opened: it opens only after BOTH T1 and T3 pass in development and validation")
    if hold is not None:
        log("  HOLDOUT OPENED (once): " + f"{VR.HOLD[0][:7]}..{VR.HOLD[1][:7]} forecast dates; the case stands only if c > 0 (T1) and the T3 slope > 0")
        log(f"    T1 M3: n {hold['t1']['n']}  c {_n(hold['t1']['c'])} (t {_n(hold['t1']['t_c'], 2)})  b {_n(hold['t1']['b'])} (t {_n(hold['t1']['t_b'], 2)})  c>0 {_yn(hold['t1']['c'] > 0)}")
        log(f"    T3:    n {hold['t3']['n']}  slope {_n(hold['t3']['slope'], 4)} (t {_n(hold['t3']['t'], 2)})  slope>0 {_yn(hold['t3']['slope'] > 0)}")
        log(f"  {'CASE STANDS' if dec['stands'] else 'FAILED IN THE HOLDOUT: recorded as passed dev/validation, failed in the holdout'}")

    _info(log, ff, cal, prim, daily, dev, val, mkt, cut_year)
    log(f"\nunique trials logged so far: {trial_count()}")


def t1m2_pass(rule) -> bool:
    return bool(rule["passed"])


def _holdout(ff, cal, last_year) -> dict:
    """The single consultation of the 2020+ window: engine over all years, VIX unsealed, primary sample in the holdout window."""
    mkt = V.market_series(ff, cal)
    vx = VR.prepare_vix(pd.read_parquet(RAW / "cboe_vix.parquet"), cal, before=None)
    df = VR.market_rows(mkt, cal, last_year)
    s = VR.select_samples(df, cal, vx["vix"])["primary"]
    h = VR.in_window(s, VR.HOLD)
    return {"t1": VR.t1_fit(h, VR.PRIMARY_MODEL), "t3": VR.t3_fit(h)}


# --------------------------------------------------------------------------------------------------- information only
def _info(log, ff, cal, prim, daily, dev, val, mkt, cut_year) -> None:
    log("\n== INFORMATION ONLY (no decision weight; development and validation) ==")
    M3, M2 = VR.PRIMARY_MODEL, VR.SECOND_MODEL
    wins = {"dev": dev, "val": val}

    log(f"\n-- T2 unconditional premium: mean IV2_win - RV_win (positive = implied above realised), Newey-West {VR.NW_PRIMARY} --")
    for k, s in wins.items():
        t = VR.t2_stats(s)
        log(f"  {k:4s} n {t['n']:3d}  mean IV2_win {t['mean_iv2']:.5f}  mean RV_win {t['mean_rv']:.5f}  difference {t['mean_var']:+.5f} (t {_n(t['t_var'], 2)})  "
            f"| vol points: mean VIX {t['mean_vix']:.2f} vs realised {t['mean_rv_vol']:.2f}, difference {t['mean_vol_pts']:+.2f} (t {_n(t['t_vol_pts'], 2)})  "
            f"| share of windows with IV2_win > RV_win {t['share_pos']:.1%}")
    log("  This premium is the price of crash insurance, not an opportunity: a short-variance position loses heavily in a crash (SVXY lost about 90% in February 2018).")

    log(f"\n-- T4 market return predictability: next-21-day FF excess return (sum of mktrf over t+1..t+21) on each regressor, Newey-West {VR.NW_PRIMARY}; low power, no weight --")
    mktrf = V.market_series(ff.assign(rf=0.0), cal)
    r21 = V.targets(mktrf)[3]
    for k, s in wins.items():
        r = r21[s["tidx"].to_numpy("int64")]
        a, b = VR.t4_fit(s, r, "D"), VR.t4_fit(s, r, "iv_minus_f")
        log(f"  {k:4s} n {a['n']:3d}  on D = log F_win - log IV2_win: slope {_n(a['slope'], 4)} (t {_n(a['t'], 2)})  | on IV2_win - F_win: slope {_n(b['slope'], 4)} (t {_n(b['t'], 2)})")

    log("\n-- QLIKE of the constant-scaled implied variance k x IV2_win (k = mean RV_win / mean IV2_win on development ONLY, frozen) against M3 and M2; "
        f"mean QLIKE difference model - scaled IV (negative = the model is better), DM t with Newey-West {VR.NW_PRIMARY} --")
    k = VR.scaled_iv_k(dev)
    log(f"  k = {k:.4f}")
    for kk, s in wins.items():
        q = VR.qlike_vs_scaled_iv(s, k)
        log(f"  {kk:4s} " + "  ".join(f"{m} {_n(q[m][0], 4)} (t {_n(q[m][1], 2)}, n {q[m][2]})" for m in (M3, M2)) + f"  | mean QLIKE of scaled IV {_n(q['scaled_iv_mean'], 4)}")

    log(f"\n-- daily-overlap versions (all forecast dates, Newey-West {VR.NW_DAILY} lags; overlapping windows, information only) --")
    ddev, dval = VR.in_window(daily, V.DEV), VR.in_window(daily, V.VAL)
    for k_, s in (("dev", ddev), ("val", dval)):
        parts = []
        for m in (M3, M2):
            f = VR.t1_fit(s, m, VR.NW_DAILY)
            parts.append(f"T1 {m}: n {f['n']:,} b {_n(f['b'])} (t {_n(f['t_b'], 2)}) c {_n(f['c'])} (t {_n(f['t_c'], 2)})")
        t3 = VR.t3_fit(s, VR.NW_DAILY)
        log(f"  {k_:4s} " + "  |  ".join(parts) + f"  |  T3 slope {_n(t3['slope'], 4)} (t {_n(t3['t'], 2)})")

    log("\n-- T3 with development-frozen tercile breakpoints of D (mean Y by tercile of D; tercile 1 = lowest D) --")
    br = VR.tercile_breaks(VR.signal_d(dev, M3))
    log(f"  breakpoints (development 1/3 and 2/3 quantiles of D): {_n(br[0], 4)}, {_n(br[1], 4)}")
    for kk, s in wins.items():
        tb = VR.tercile_table(s, br)
        log(f"  {kk:4s} " + "  ".join(f"tercile {r['tercile'] + 1}: mean Y {_n(r['mean'], 4)} (t {_n(r['t'], 2)}, n {r['n']})" for r in tb))

    log("\n-- T3 level version: D = F_win(M3) - IV2_win --")
    for kk, s in wins.items():
        log(_t3_line(kk, VR.t3_fit(s, level=True)))

    log(f"\n-- T3 rerun in development excluding {VR.crash_note()} --")
    keep = ~VR.touches(dev, cal, *VR.EX_CRASH)
    ex = VR.t3_fit(dev[keep].reset_index(drop=True))
    log(f"  excluded {int((~keep).sum())} of {len(dev)} development windows")
    log(_t3_line("dev", ex))

    log("\n-- optional robustness: S&P 500 daily returns (crsp.dsp500) through the same engine, T1 for M3; information only --")
    sp_file = RAW / "sp500_daily.parquet"
    if not sp_file.exists():
        log("  skipped: data/raw/sp500_daily.parquet is absent (optional input)")
    else:
        try:
            sp = pd.read_parquet(sp_file)
            ret = _series_on_calendar(sp["caldt"], sp["sprtrn"], cal)
            vx = VR.prepare_vix(pd.read_parquet(RAW / "cboe_vix.parquet"), cal, before=VR.SEAL)
            d2 = VR.market_rows(ret, cal, cut_year)
            d2 = d2[d2["date"] <= np.datetime64(VR.PRE_END)].reset_index(drop=True)
            s2 = VR.select_samples(d2, cal, vx["vix"])["primary"]
            a, b = VR.in_window(s2, V.DEV), VR.in_window(s2, V.VAL)
            fa, fb = VR.t1_fit(a, M3), VR.t1_fit(b, M3)
            r = VR.t1_rule(fa, fb)
            log(_t1_line("dev", fa))
            log(_t1_line("val", fb))
            log(f"  T1 rule on the S&P 500 series: {'PASS' if r['passed'] else 'FAIL'}")
        except (KeyError, ValueError) as e:
            log(f"  skipped: the S&P 500 file could not be used ({type(e).__name__}: {e})")

    log("\n-- by year: mean implied versus realised variance (primary sample; vol points = 100 sqrt(window variance x 365/30)) --")
    log(f"  {'year':6s} {'n':>3s} {'mean VIX':>9s} {'realised':>9s} {'IV2-RV':>10s}")
    yrs = prim["date"].dt.year.to_numpy()
    for y in range(V.FIRST_FC_YEAR, int(VR.PRE_END[:4]) + 1):
        s = prim[yrs == y]
        if len(s) == 0:
            continue
        t = VR.t2_stats(s)
        log(f"  {y:<6d} {len(s):3d} {t['mean_vix']:9.2f} {t['mean_rv_vol']:9.2f} {t['mean_var']:+10.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true",
                    help="open the sealed 2020+ window once (only after the v13 holdout has been opened, and only if T1 and T3 both passed dev/validation)")
    a = ap.parse_args()
    main(a.open_holdout)
