"""Study F (pre-registration v13, Stage A): out-of-sample volatility forecasting, GJR-GARCH-t versus simple benchmarks. Holdout sealed.

    python scripts/extract_earnings.py                                            # daily returns / FF (once)
    python scripts/wrds_extract.py crsp_monthly crsp_delistings crsp_names        # once
    python scripts/run_volfc.py [--open-holdout] [--jobs N] [--assets N] [--save-losses]

Writes data/study_f_report.txt (and, with --save-losses, the git-ignored data/study_f_losses.parquet). No trial is logged: this script never
touches the trials registry. Without --open-holdout no forecast dated 2020 or later is computed for the primary or market series; with it
the holdout is opened once (one-shot marker data/holdout_v13.done, written after its statistics are computed; a second opening is refused), and only if all four comparisons pass the dev / validation rule are its numbers shown.
"""
import argparse
import pathlib
import time

import numpy as np
import pandas as pd

from qap import volfc as V
from qap.parallel import pmap, set_context
from qap.panel import build_monthly_panel
from qap.study import require_inputs

RAW = pathlib.Path("data/raw")
PREREG = "v13"
SEAL = "2020-01-01"                     # first sealed forecast date
PRE_END = "2019-12-31"
# The ticker block is pre-registered over the tickers' FULL history and "the last 3 years of data" (inside 2023-2025), information only, with
# no decision weight. False (default) = keep the seal: the block is capped at 2019-12-31 (so "the last 3 years" are the 3 years to the end of
# 2019) until --open-holdout, when it runs to the end of the data. True = always print it as pre-registered (holdout-era numbers on sealed names).
TICKER_BLOCK_IGNORES_SEAL = False


def main(open_holdout: bool, jobs: int = 1, assets: int = V.TOP_N, save_losses: bool = False) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, open_holdout, jobs, assets, save_losses)
    except V.HoldoutRefused:
        raise                                       # a refusal must not overwrite the report of the opening that already happened
    except BaseException:
        _write_report(report)
        raise
    _write_report(report)


def _write_report(report) -> None:
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/study_f_report.txt").write_text("\n".join(report) + "\n")


# ----------------------------------------------------------------------------------------------------------- helpers
def _n(x, d=4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _pct(x, d=2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _merge(results, cal) -> tuple:
    """(rows DataFrame with date / year, fits DataFrame, blocks DataFrame[aid-free: refit, ok]) from run_asset outputs in input order."""
    rows = [r["rows"] for r in results]
    df = pd.DataFrame({c: np.concatenate([r[c] for r in rows]) for c in rows[0]}) if rows else pd.DataFrame({c: [] for c in V._empty_rows()})
    df["date"] = cal[df["tidx"].to_numpy("int64")].astype("datetime64[ns]")
    df["year"] = df["date"].dt.year.astype("int64")
    fits = pd.DataFrame([f for r in results for f in r["fits"]], columns=V.FIT_COLS)
    blk = np.concatenate([r["blocks"] for r in results]) if results else np.empty((0, 2), "int64")
    return df, fits, pd.DataFrame(blk, columns=["refit", "ok"])


def _dl(df, h, kind="qlike", masks=None):
    m = (masks or V.valid_masks(df))[h]
    return V.date_losses(df, m, h, kind)


def _window_means(dl, cal, window) -> pd.Series:
    return V.window_slice(dl, cal, window).mean()


def _models_table(log, dls: dict, cal, windows: dict, kind="QLIKE") -> None:
    fmt = (lambda x: _n(x)) if kind == "QLIKE" else (lambda x: "n/a" if not np.isfinite(x) else f"{x:.3e}")
    for h in V.HORIZONS:
        log(f"  mean {kind} at h={h} (mean over forecast dates of the cross-asset mean):")
        log(f"    {'model':11s}" + "".join(f" {k:>10s}" for k in windows))
        means = {k: _window_means(dls[h], cal, w) for k, w in windows.items()}
        for m in V.MODELS:
            log(f"    {m:11s}" + "".join(f" {fmt(means[k][m]):>10s}" for k in windows))


def _pair_line(dl, cal, a, b, h, windows) -> str:
    parts = []
    for k, w in windows.items():
        m, t, nd = V.compare(dl, cal, a, b, h, w)
        parts.append(f"{k} {_n(m)} (t {_n(t, 2)}, {nd} dates)")
    return "  ".join(parts)


def _info_block(log, df, label) -> None:
    """Information only (no decision weight) on the rows of `df`."""
    masks = V.valid_masks(df)
    m1 = masks[1] & np.isfinite(df["r1"].to_numpy())
    log(f"  [{label}] rows with r_(t+1): {int(m1.sum()):,}")
    if not m1.any():
        return
    ls = V.row_logscore(df, m1)
    log("    predictive log score of r_(t+1) (mean; Normal, standardised-t with fitted nu for gjr_t / gjr_t_wins; fallbacks Normal): "
        + "  ".join(f"{m} {_n(float(np.mean(ls[m])))}" for m in V.MODELS))
    for a in (0.01, 0.05):
        v = V.var_violations(df, m1, a)
        n = int(m1.sum())
        log(f"    1-day {int(a * 100)}% VaR violation rate (nominal {a:.0%}; Kupiec p OPTIMISTIC: cross-sectional dependence): "
            + "  ".join(f"{m} {_pct(float(v[m].mean()), 3)} (p {_n(V.kupiec_pvalue(int(v[m].sum()), n, a), 3)})" for m in V.MODELS))
    for q, a in (("01", 0.01), ("05", 0.05)):
        parts = []
        for m in ("garch", "gjr_t", "gjr_t_wins"):
            x = df[f"mc{q}_{m}"].to_numpy()
            x = x[np.isfinite(x)]
            parts.append(f"{m} {_pct(float(x.mean()) if len(x) else float('nan'), 2)} (n {len(x)})")
        log(f"    21-day {int(a * 100)}% VaR coverage by Monte Carlo ({V.MC_PATHS:,} paths, refit dates only; violation rate vs nominal {a:.0%}): " + "  ".join(parts))


def _fit_block(log, fits, label) -> None:
    log(f"  [{label}] fitted parameters over converged fits (median) and shares:")
    for m in ("garch", "gjr_t", "gjr_t_wins"):
        s = V.fit_summary(fits, m)
        nu = "" if m == "garch" else f" nu {_n(s['nu'], 2)} share of nu at its bounds {_pct(s['nu_at_bound'])}"
        gm = "" if m == "garch" else f" gamma {_n(s['gamma'])}"
        log(f"    {m:11s} fits {s['fits']:,} ok {_pct(s['ok'], 1)} failed {_pct(s['failed'], 1)} breach {_pct(s['breach'], 1)} | alpha {_n(s['alpha'])}{gm} "
            f"beta {_n(s['beta'])} persistence {_n(s['p'], 4)}{nu}")


def _by_year(log, dls: dict, cal, years) -> None:
    for h in V.HORIZONS:
        log(f"  mean QLIKE by calendar year at h={h}:")
        log(f"    {'year':6s}" + "".join(f" {m:>10s}" for m in V.MODELS))
        for y in years:
            mm = _window_means(dls[h], cal, (f"{y}-01-01", f"{y}-12-31"))
            log(f"    {y:<6d}" + "".join(f" {_n(mm[m]):>10s}" for m in V.MODELS))


def _stress(log, dls: dict, cal, windows: dict) -> None:
    for name, w in windows.items():
        for h in V.HORIZONS:
            mm = _window_means(dls[h], cal, w)
            log(f"  stress {name} h={h}: " + "  ".join(f"{m} {_n(mm[m])}" for m in V.MODELS))


def _ticker_lines(log, tmap, summaries) -> None:
    for t in V.TICKERS:
        st, val = tmap[t]
        if st == "ok":
            s = summaries.get(t)
            if s is None:
                log(f"  {t:5s} permno {val}: insufficient history (< {V.MIN_TICKER_DAYS} days), not forecast")
            elif s["rows"] == 0:
                log(f"  {t:5s} permno {val}: no forecast rows (history rule not met)")
            else:
                log(f"  {t:5s} permno {val}: {s['rows']:,} forecast dates; mean QLIKE h=21: " + "  ".join(f"{m} {_n(s['qlike21'][m])}" for m in V.MODELS))
                log(f"        last 3 years ({s['recent_rows']} dates): mean ratio forecast/realised 21-day variance: "
                    + "  ".join(f"{m} {_n(s['ratio_mean'][m], 3)}" for m in V.MODELS))
                log("        ratio of means: " + "  ".join(f"{m} {_n(s['ratio_of_means'][m], 3)}" for m in V.MODELS))
                d = s["drag"]
                log(f"        implied annualised 2x daily-reset drag ((L^2-L) sigma^2/2 = sigma^2/yr): garch {_pct(d['garch'], 1)}  gjr_t {_pct(d['gjr_t'], 1)}  "
                    f"realised {_pct(d['realised'], 1)}")
        elif st == "ambiguous":
            log(f"  {t:5s} REFUSED: ambiguous, name records in force map to permnos {val}")
        else:
            log(f"  {t:5s} REFUSED: missing, no CRSP name record in force on the last daily date")
    for t in V.NOT_CRSP:
        log(f"  {t:5s} not a CRSP security")


# ----------------------------------------------------------------------------------------------------------------- run
def _run(log, open_holdout, jobs, assets=V.TOP_N, save_losses=False) -> None:
    hint = ("Run: python scripts/extract_earnings.py  and  python scripts/wrds_extract.py crsp_monthly crsp_delistings crsp_names")
    require_inputs([RAW / "crsp_daily_ret", RAW / "ff_daily.parquet", RAW / "crsp_monthly.parquet", RAW / "crsp_delistings.parquet",
                    RAW / "crsp_names.parquet"], hint)
    panel = build_monthly_panel(pd.read_parquet(RAW / "crsp_monthly.parquet"), pd.read_parquet(RAW / "crsp_delistings.parquet"))
    last = V.last_daily_date(RAW / "crsp_daily_ret")
    last_year = int(str(last)[:4])
    cut_year = last_year if open_holdout else min(last_year, 2019)
    universe = V.top_universe(panel, range(V.FIRST_FC_YEAR, cut_year + 1), assets)
    tmap = V.map_tickers(pd.read_parquet(RAW / "crsp_names.parquet"), V.TICKERS, last)
    tick_permnos = sorted({v for st, v in tmap.values() if st == "ok"})
    sy = V.stock_years(universe)
    need = sorted(set(sy) | set(tick_permnos))
    sub, daily_rows, cal = V.load_daily_subset(RAW / "crsp_daily_ret", need)
    ncal = len(cal)
    R = V.return_matrix(sub, cal, need)
    del sub
    mkt = V.market_series(pd.read_parquet(RAW / "ff_daily.parquet"), cal)
    row_of = {p: i for i, p in enumerate(need)}
    set_context(R=R, mkt=mkt)

    cut_idx = int(np.searchsorted(cal, np.datetime64(SEAL), side="left")) - 1             # last calendar index before the seal
    stock_jobs = [(p, row_of[p], V.year_ranges(cal, ys), True) for p, ys in sy.items()]
    mkt_ranges = V.year_ranges(cal, range(V.FIRST_FC_YEAR, cut_year + 1))
    tick_end = ncal - 1 if (TICKER_BLOCK_IGNORES_SEAL or open_holdout) else cut_idx
    tick_jobs, tick_hist = [], {}
    for t in V.TICKERS:
        st, p = tmap[t]
        if st == "ok":
            tick_hist[t] = int(np.isfinite(R[row_of[p]][:tick_end + 1]).sum())
            if tick_hist[t] >= V.MIN_TICKER_DAYS:
                tick_jobs.append((p, row_of[p], [(0, tick_end)], False))
    all_jobs = stock_jobs + [(V.MKT_ID, -1, mkt_ranges, True)] + tick_jobs
    res = pmap(V.asset_worker, all_jobs, jobs, say=log)
    n_s = len(stock_jobs)
    df_all, fits_all, blk_all = _merge(res[:n_s], cal)
    df_m, _, _ = _merge([res[n_s]], cal)
    tick_sum = {}
    for t in V.TICKERS:
        st, p = tmap[t]
        j = [i for i, jb in enumerate(tick_jobs) if jb[0] == p]
        if st == "ok" and j:
            tick_sum[t] = V.ticker_summary(res[n_s + 1 + j[0]]["rows"], cal[:tick_end + 1])
    # everything below the CHECKS / dev / validation part uses PRE rows only, so the report is identical with and without the flag until the holdout
    pre_end = np.datetime64(PRE_END)
    df = df_all[df_all["date"] <= pre_end].reset_index(drop=True)
    dm = df_m[df_m["date"] <= pre_end].reset_index(drop=True)
    fits = fits_all[cal[fits_all["refit"].to_numpy("int64")] <= pre_end]
    blk = blk_all[cal[blk_all["refit"].to_numpy("int64")] <= pre_end]

    log("== CHECKS (printed before any result) ==")
    verify = V.verify_recursion(mkt[np.isfinite(mkt)][:3000])
    tcheck = V.t_density_check()
    if not V.t_density_ok(tcheck):
        raise AssertionError(f"standardised-t density check failed: {tcheck}")
    masks = V.valid_masks(df)
    uni_pre = {y: p for y, p in universe.items() if y <= int(PRE_END[:4])}
    notes = [f"market series: {len(dm):,} forecast rows ({int(np.isfinite(dm['y21']).sum()):,} with a complete 21-day window)",
             f"forecast dates start in {V.FIRST_FC_YEAR} (earlier dates fall in no evaluation window); holdout rows (>= {SEAL[:4]}) are "
             + ("computed (--open-holdout) but reported only if the licence passes dev / validation" if open_holdout else "NOT computed (sealed)")]
    for line in V.volfc_checks(daily_rows, ncal, (int(np.isfinite(mkt).sum()), ncal), uni_pre, df, fits, (len(blk), int(blk["ok"].sum())), verify, tcheck, masks, notes):
        log(line)
    ok = len(df) > 0 and masks[1].any() and masks[21].any()
    log(f"  [{'PASS' if ok else 'FAIL'}] forecasts produced for the primary universe")
    if not ok:
        raise AssertionError("Study F: no usable forecasts were produced")
    log("No trials are logged by this study; the trial count stays at 79.")

    windows = {"dev": V.DEV, "val": V.VAL}
    dls = {(h, k): _dl(df, h, k, masks) for h in V.HORIZONS for k in ("qlike", "mse")}
    q = {h: dls[(h, "qlike")] for h in V.HORIZONS}
    mse = {h: dls[(h, "mse")] for h in V.HORIZONS}
    log("\n== RESULTS (Study F, pre-registration v13): primary universe, rolling top-100 US common stocks; holdout sealed unless opened ==")
    log(f"fixed comparison set: mean QLIKE difference (model - benchmark; negative = {V.COMPARISONS[0][0]} better), Diebold-Mariano t with Newey-West lags "
        f"{V.NW_LAGS[1]} (h=1) / {V.NW_LAGS[21]} (h=21); dev {V.DEV[0][:7]}..{V.DEV[1][:7]}, validation {V.VAL[0][:7]}..{V.VAL[1][:7]}")
    comps = {}
    for a, b in V.COMPARISONS:
        for h in V.HORIZONS:
            d = V.compare(q[h], cal, a, b, h, V.DEV)
            v = V.compare(q[h], cal, a, b, h, V.VAL)
            comps[(a, b, h)] = {"dev": d[:2], "val": v[:2]}
            c = comps[(a, b, h)]
            log(f"  {a} vs {b} h={h:<2d}: dev {_n(d[0])} (t {_n(d[1], 2)}, {d[2]} dates)  val {_n(v[0])} (t {_n(v[1], 2)}, {v[2]} dates)  | "
                f"dev<=-0.01 {'Y' if d[0] <= V.MATERIALITY else 'N'}  dev t<=-2.0 {'Y' if d[1] <= V.DM_T else 'N'}  val<0 {'Y' if v[0] < 0 else 'N'}")
    pre_verdict = V.licence_verdict(comps)
    hold_state = None
    hold_means = None
    if open_holdout and pre_verdict["dev_val"]:
        V.assert_holdout_unopened()                 # one shot: a second opening is refused (SystemExit with the marker contents)
        mh_all = V.valid_masks(df_all)
        hold_all = df_all[df_all["date"] >= np.datetime64(SEAL)]
        dh = {h: V.date_losses(df_all, mh_all[h], h, "qlike") for h in V.HORIZONS}
        hold_means = {(a, b, h): V.compare(dh[h], cal, a, b, h, V.HOLD)[0] for a, b in V.COMPARISONS for h in V.HORIZONS}
        hold_state = V.licence_verdict(comps, hold_means, True)
        # one-shot marker (v14 gate): written AFTER the holdout statistics are computed and BEFORE they are printed
        V.mark_holdout_opened(f"v13 holdout opened {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {hold_state['status']}")
    verdict = hold_state or pre_verdict
    log("\n== LICENCE VERDICT (M3 gjr_t as the position-sizing volatility model; ALL FOUR: dev diff <= -0.01 AND dev t <= -2.0 AND validation diff < 0) ==")
    log(f"  {verdict['status']}")
    if open_holdout and not pre_verdict["dev_val"]:
        log("  --open-holdout given, but the holdout is NOT opened: it opens only after all four dev / validation conditions pass")
    if hold_state:
        log("  HOLDOUT OPENED (once): mean QLIKE difference (model - benchmark), forecast dates " + f"{V.HOLD[0][:7]}..{V.HOLD[1][:7]}")
        for (a, b, h), mval in hold_means.items():
            m, t, nd = V.compare(dh[h], cal, a, b, h, V.HOLD)
            log(f"    {a} vs {b} h={h:<2d}: {_n(m)} (t {_n(t, 2)}, {nd} dates) {'< 0' if m < 0 else '>= 0'}")
        log(f"  licence {'STANDS' if hold_state['holdout_all_negative'] else 'FAILS: recorded as licensed in dev/validation, failed in the holdout'}")

    log("\n== DEFAULT SIZING MODEL (lowest dev mean QLIKE at h=21 among hist, ewma, garch, gjr_t, har; M4 is an ablation) ==")
    dev21 = _window_means(q[21], cal, V.DEV)
    for m in V.SIZING:
        log(f"  {m:11s} {_n(dev21[m])}")
    default = V.default_sizing_model({m: float(dev21[m]) for m in V.SIZING})
    log(f"  default (dev ranking): {default}")
    if verdict["status"].startswith("LICENSED ("):
        log("  M3 gjr_t is LICENSED, so it is the sizing model")
    elif verdict["dev_val"] and not verdict["opened"]:
        log(f"  licence pending the one-time holdout; until then the dev-ranking default is: {default}")
    else:
        log(f"  M3 is not (finally) licensed, so the sizing model is: {default}")

    log("\n== DESCRIPTIVE TABLES (information only) ==")
    log("all models, primary universe:")
    _models_table(log, q, cal, windows)
    log("MSE of variance (information only, mean of per-date cross-asset means):")
    _models_table(log, mse, cal, windows, kind="MSE")
    log("winsorisation ablation, M4 gjr_t_wins minus M3 gjr_t (QLIKE; negative = winsorised better):")
    for h in V.HORIZONS:
        log(f"  h={h:<2d}: " + _pair_line(q[h], cal, "gjr_t_wins", "gjr_t", h, windows))
    log("market series (mktrf + rf), own table, information only:")
    mmask = V.valid_masks(dm)
    mq = {h: V.date_losses(dm, mmask[h], h, "qlike") for h in V.HORIZONS}
    if len(dm) and mmask[1].any() and mmask[21].any():
        _models_table(log, mq, cal, windows)
        for a, b in V.COMPARISONS:
            for h in V.HORIZONS:
                log(f"  {a} minus {b} h={h:<2d}: " + _pair_line(mq[h], cal, a, b, h, windows))
    else:
        log("  no usable market-series forecasts")

    pre_info = {"dev+val": df}
    log("\n== INFORMATION ONLY (no decision weight; dev + validation rows) ==")
    for lab, d in pre_info.items():
        _info_block(log, d, lab)
    _fit_block(log, fits, "dev+val fits")
    fb = {m: _pct(float(df[f"fb_{m}"].mean())) for m in ("garch", "gjr_t", "gjr_t_wins")}
    log("  fallback share of forecast rows: " + "  ".join(f"{m} {fb[m]}" for m in fb))
    log("QLIKE by calendar year and stress windows:")
    _by_year(log, q, cal, range(V.FIRST_FC_YEAR, int(PRE_END[:4]) + 1))
    _stress(log, q, cal, {"2008-09": ("2008-01-01", "2009-12-31")})
    if hold_state:
        log("\n== HOLDOUT (opened; information only) ==")
        mh_all = V.valid_masks(df_all)
        hq = {h: V.date_losses(df_all, mh_all[h], h, "qlike") for h in V.HORIZONS}
        _models_table(log, hq, cal, {"holdout": V.HOLD})
        for h in V.HORIZONS:
            log(f"  winsorisation ablation h={h:<2d}: " + _pair_line(hq[h], cal, "gjr_t_wins", "gjr_t", h, {"holdout": V.HOLD}))
        hold_df = df_all[df_all["date"] >= np.datetime64(SEAL)]
        _info_block(log, hold_df, "holdout")
        _fit_block(log, fits_all[cal[fits_all["refit"].to_numpy("int64")] > pre_end], "holdout fits")
        _by_year(log, hq, cal, range(int(SEAL[:4]), cut_year + 1))
        _stress(log, hq, cal, {"2020": ("2020-01-01", "2020-12-31")})
        dm_h = V.valid_masks(df_m)
        mqh = {h: V.date_losses(df_m, dm_h[h], h, "qlike") for h in V.HORIZONS}
        log("market series, holdout:")
        _models_table(log, mqh, cal, {"holdout": V.HOLD})

    log("\n== INFORMATION ONLY: the user's tickers (full available history, not restricted to the top-100) ==")
    end_txt = str(cal[tick_end])
    if tick_end < ncal - 1:
        log(f"  note: sealed - this block ends on {end_txt} (the pre-registered 'last 3 years of data' lie inside the sealed 2020-2025 window; "
            "run with --open-holdout to extend it to the end of the data)")
    log(f"  name record in force on {last} (last daily date); history = non-missing daily returns up to {end_txt}")
    _ticker_lines(log, tmap, tick_sum)

    if save_losses:
        out = df_all if (hold_state and hold_state["opened"]) else df
        out = out.copy()
        mk = V.valid_masks(out)
        for h in V.HORIZONS:
            proxy = out["y1" if h == 1 else "y21"].to_numpy()
            for m in V.MODELS:
                out[f"qlike{h}_{m}"] = np.where(mk[h], V.qlike(out[f"f{h}_{m}"].to_numpy(), proxy), np.nan)
        pathlib.Path("data").mkdir(exist_ok=True)
        out.to_parquet("data/study_f_losses.parquet", index=False)
        log(f"\nsaved {len(out):,} forecast rows with losses to data/study_f_losses.parquet")
    log("\nNo trials are logged; the trial count stays at 79.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true", help="open the sealed 2020-2025 forecast dates once (only after a dev/validation licence)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes over assets (fork)")
    ap.add_argument("--assets", type=int, default=V.TOP_N, help="size of the rolling top-N universe (pre-registered: 100)")
    ap.add_argument("--save-losses", action="store_true", help="write data/study_f_losses.parquet (git-ignored)")
    a = ap.parse_args()
    main(a.open_holdout, a.jobs, a.assets, a.save_losses)
