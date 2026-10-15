"""Study H (pre-registration v15): the mechanical skeleton of the user's band-switch and momentum rules on simulated 2x daily-reset products.

    python scripts/extract_earnings.py                                            # daily returns / FF (once)
    python scripts/wrds_extract.py crsp_monthly crsp_delistings                   # once
    python scripts/extract_secd.py                                                # Compustat daily prices of the user's names and ETFs (replication check)
    python scripts/run_bandsim.py [--open-holdout] [--jobs N] [--skip-replication] [--placebo-draws N] [--assets N]

Writes data/study_h_report.txt. Logs exactly TWO trials (R1 band switch, R2 momentum; idempotent; a partial run -- --assets or a non-default
--placebo-draws -- logs none). Licenses no trade: a pass licenses at most a forward paper test. Without --open-holdout nothing dated 2020 or
later is read or computed and the report contains no 2020+ number. --open-holdout opens the holdout ONCE (marker data/holdout_v15.done, written
after its statistics are computed; a second opening is refused) and only for the rules that passed development and validation. The user's own
names (secd data, which lie in the sealed window by nature) are printed ONLY with --open-holdout (the v13 ticker-block precedent); the replication
check uses the same secd data but prints relative years of common data, never calendar dates.
"""
import argparse
import dataclasses
import pathlib
import time

import numpy as np
import pandas as pd

from qap import bandsim as B
from qap import volfc as V
from qap import vrp as VR
from qap.panel import build_monthly_panel
from qap.parallel import pmap, set_context
from qap.study import require_inputs
from qap.trials import trial_count

RAW = pathlib.Path("data/raw")
SECD = RAW / "secd_tickers.parquet"
SEAL_END_YEAR = 2019
RULES = ("R1", "R2")


def main(open_holdout: bool, jobs: int = 1, assets=None, draws: int = B.DRAWS, skip_replication: bool = False) -> None:
    report = []

    def log(m=""):
        print(m, flush=True)
        report.append(str(m))

    try:
        _run(log, open_holdout, jobs, assets, draws, skip_replication)
    except V.HoldoutRefused:
        raise                                       # a refusal must not overwrite the report of the opening that already happened
    except BaseException:
        _write_report(report)
        raise
    _write_report(report)


def _write_report(report) -> None:
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/study_h_report.txt").write_text("\n".join(report) + "\n")


# ----------------------------------------------------------------------------------------------------------- helpers
def _n(x, d=3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _pct(x, d=1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _i(x) -> str:
    return f"{int(x):,}"                              # thousands separators: no bare 4-digit counts that look like years


def _st(st) -> str:
    """mean daily excess in bp (annualised x252 in %), NW t, number of dates."""
    return f"{_n(1e4 * st['mean'], 2):>7s} bp/day ({_n(100 * ANN_ * st['mean'], 1):>6s}%/yr)  t {_n(st['t'], 2):>6s}  dates {_i(st['n'])}"


ANN_ = B.ANN


def specs_all(draws: int) -> list:
    P = B.primary
    S = B.Spec
    r = dataclasses.replace
    specs = [S("R1", params=P("R1"), placebo=True, groups=True), S("R2", params=P("R2"), placebo=True, groups=True)]
    for rule in RULES:
        p = P(rule)
        specs += [S(f"{rule}_fee0", params=r(p, fee=0.0)), S(f"{rule}_fee3", params=r(p, fee=0.03)), S(f"{rule}_cost0", params=r(p, cost=0.0)),
                  S(f"{rule}_cost25", params=r(p, cost=0.0025)), S(f"{rule}_lag0", params=r(p, lag=0))]
    p = P("R1")
    specs += [S("R1_d03", params=r(p, d=0.03)), S("R1_d10", params=r(p, d=0.10)), S("R1_N63", params=r(p, n=63))]
    specs += [S("R3", params=B.r3_params()),
              S("BH_und", "bh_u", B.Params(rule="BH", fee=0.0)), S("BH_long2", "bh_l", B.Params(rule="BH")), S("BH_short2", "bh_s", B.Params(rule="BH")),
              S("R1", params=P("R1"), sets=("top100",)), S("R2", params=P("R2"), sets=("top100",))]
    return specs


def _sleeves_for(universe: dict, q: tuple, row_of: dict) -> dict:
    out = {}
    for y, u in universe.items():
        main = []
        for p, v in zip(u["permno"], u["vol"]):
            terc = 0 if v < q[0] else (1 if v < q[1] else 2)
            main.append((int(p), row_of[int(p)], terc))
        out[y] = {"main": main, "top100": [(int(p), row_of[int(p)], 0) for p in u["top100"] if int(p) in row_of]}
    return out


def _run_years(log, years, universe, q, row_of, R, rf, cal, specs, draws, jobs):
    ranges = {y: V.year_ranges(cal, [y])[0] for y in years if V.year_ranges(cal, [y])}
    years = [y for y in years if y in ranges]
    set_context(R=R, rf=rf, draws=draws, ranges=ranges, sleeves=_sleeves_for({y: universe[y] for y in years}, q, row_of), specs=specs)
    res = pmap(B.year_worker, years, jobs, say=log)
    return B.combine(res, len(cal), draws)


# ----------------------------------------------------------------------------------------------------------------- run
def _run(log, open_holdout=False, jobs=1, assets=None, draws=B.DRAWS, skip_replication=False) -> None:
    if open_holdout:
        V.assert_holdout_unopened(B.MARKER, B.STUDY)             # one shot: refused before anything is read or computed
    hint = ("Run: python scripts/extract_earnings.py  and  python scripts/wrds_extract.py crsp_monthly crsp_delistings")
    require_inputs([RAW / "crsp_daily_ret", RAW / "ff_daily.parquet", RAW / "crsp_monthly.parquet", RAW / "crsp_delistings.parquet"], hint)
    partial = assets is not None or int(draws) != B.DRAWS
    ff = pd.read_parquet(RAW / "ff_daily.parquet")

    log("== CHECKS (printed before any result) ==")
    frame = None
    if SECD.exists():
        from qap import prices as PR
        frame, _ = PR.prepare_prices(pd.read_parquet(SECD))
        rep = B.replication_check(frame, ff)
        _print_replication(log, rep, open_holdout)
        if not rep["ok"]:
            log("  ABORT: the simulated products do not replicate the actual ETFs (wrong mapping or data): " + "; ".join(rep["reasons"]))
            raise AssertionError("Study H: replication check failed")
    elif skip_replication:
        log("replication check SKIPPED (no secd file)")
    else:
        log("replication check cannot run: data/raw/secd_tickers.parquet is absent")
        raise SystemExit("Missing data/raw/secd_tickers.parquet for the replication check. Run: python scripts/extract_secd.py "
                         "(or pass --skip-replication to continue without it)")

    panel = build_monthly_panel(pd.read_parquet(RAW / "crsp_monthly.parquet"), pd.read_parquet(RAW / "crsp_delistings.parquet"))
    last_year = max(int(f.stem) for f in (RAW / "crsp_daily_ret").glob("[0-9]*.parquet"))      # from the file names: a sealed run opens nothing later
    hold_end = int(B.HOLD[1][:4])
    pre_years = list(range(B.FIRST_YEAR, min(last_year, SEAL_END_YEAR) + 1))
    hold_years = list(range(SEAL_END_YEAR + 1, min(last_year, hold_end) + 1)) if open_holdout else []
    cand = sorted({int(p) for y in pre_years + hold_years for p in B.ranked_candidates(panel, y, B.TOP_N)})
    R, keep, cal, daily_rows = B.load_daily(RAW / "crsp_daily_ret", cand, None if open_holdout else SEAL_END_YEAR)
    row_of = {int(p): i for i, p in enumerate(keep)}
    rf = B.rf_series(ff, cal)
    pre = cal <= np.datetime64(f"{SEAL_END_YEAR}-12-31")
    span = pre & (cal >= np.datetime64(f"{B.FIRST_YEAR}-01-01"))
    log(f"calendar: {_i(len(cal))} trading days; CRSP daily files: {len(daily_rows)}; candidate permnos loaded: {_i(len(keep))}")
    log(f"rf coverage (FF daily rf on the calendar, evaluation span): {_i(int(np.isfinite(rf[span]).sum()))} of {_i(int(span.sum()))} days "
        f"({_pct(float(np.isfinite(rf[span]).mean()), 2)}); a missing rf is treated as 0 and counted per sleeve-day below")

    universe = B.select_universe(panel, pre_years, cal, R, row_of, assets=assets)
    q = B.terciles(universe)
    log(f"\nuniverse per year (top {B.TOP_N} companies by prior-December company ME -> trailing-{B.VOL_DAYS}d annualised vol >= {B.MIN_VOL:.0%} at the "
        f"last December trading day with >= {B.MIN_OBS} returns; one sleeve per stock-year; no survivor filter):")
    log(f"  {'year':6s} {'candidates':>10s} {'>= obs':>7s} {'sleeves':>8s}")
    for y, u in universe.items():
        log(f"  {y:<6d} {u['cands']:10d} {u['obs_ok']:7d} {len(u['permno']):8d}")
    allv = np.concatenate([u["vol"] for u in universe.values()]) if universe else np.empty(0)
    if len(allv) == 0:
        raise AssertionError("Study H: the universe is empty")
    pc = np.percentile(allv, [10, 25, 50, 75, 90])
    log("volatility distribution of the universe (pooled sleeve-years, trailing annualised vol at December): "
        + "  ".join(f"p{p} {_pct(v, 0)}" for p, v in zip((10, 25, 50, 75, 90), pc)) + f"   | tercile cut-offs {_pct(q[0], 0)} / {_pct(q[1], 0)}")
    if frame is not None:
        from qap import prices as PR
        parts = []
        for nm in B.USER_NAMES:
            v, n_, lab = PR.best_vol(PR.return_series(frame, nm))
            parts.append(f"{nm} {_pct(v, 0)}" if np.isfinite(v) else f"{nm} n/a")
        log("the user's names, EWMA(0.94) annualised vol from secd (levels only, no dates): " + "  ".join(parts))
    else:
        log("the user's names, EWMA vol from secd: n/a (no secd file)")

    specs = specs_all(draws)
    t0 = time.time()
    agg = _run_years(log, pre_years, universe, q, row_of, R, rf, cal, specs, draws, jobs)
    log(f"engine: {len(pre_years)} calendar years x {len(specs)} configurations, {int(draws)} placebo draws, {_i(time.time() - t0)} s")
    _print_engine_checks(log, agg, universe, pre_years)

    res = {}
    for rule in RULES:
        a = agg[("main", rule)]
        dev, val = B.window_stat(a["S"], a["C"], cal, B.DEV), B.window_stat(a["S"], a["C"], cal, B.VAL)
        pdev = B.placebo_stats(a["SP"], a["C"], cal, B.DEV)
        res[rule] = dict(dev=dev, val=val, pdev=pdev, plc=B.placebo_summary(pdev))
        res[rule]["verdict"] = B.licence_verdict(dev, val, res[rule]["plc"])

    log("\n== RESULTS (Study H, pre-registration v15): mean daily cross-sectional net excess return over rf of the sleeves, Newey-West "
        f"{B.NW_LAGS} lags; dev {B.DEV[0][:7]}..{B.DEV[1][:7]}, validation {B.VAL[0][:7]}..{B.VAL[1][:7]} ==")
    log("annualised = x252 (arithmetic); statistic for the placebo = the dev mean daily excess; costs: 10 bp per side, fee 1.5%/yr, lag 1, stop -20%")
    for rule in RULES:
        r = res[rule]
        log(f"\n-- {rule} {'band switch (N=21, d=0.059)' if rule == 'R1' else 'momentum (two up closes; +15% / -20%)'} --")
        log(f"  dev  {_st(r['dev'])}")
        log(f"  val  {_st(r['val'])}")
        p = r["plc"]
        share = float((r["pdev"] < r["dev"]["mean"]).mean()) if len(r["pdev"]) else float("nan")
        log(f"  placebo ({p['n']} draws of the circular shift, same fractions long/short/flat, same persistence, same number of transitions and costs "
            f"BY CONSTRUCTION): dev 95th pct {_n(1e4 * p['p95'], 2)} bp/day, median {_n(1e4 * p['median'], 2)} bp/day; the rule's dev mean exceeds "
            f"{_pct(share, 0)} of the draws")

    log("\n== LICENCE (per rule; nothing is promoted for use; a pass licenses at most a forward paper test) ==")
    log("  rule: dev mean > 0 AND NW t >= 2.0 AND dev > placebo 95th percentile AND validation mean > 0; then the holdout opens once (--open-holdout) and "
        "the licence stands only if its mean > 0 and above the placebo median")
    for rule in RULES:
        c = res[rule]["verdict"]["conditions"]
        log(f"  {rule}: " + "  ".join(f"{k} {'Y' if v else 'N'}" for k, v in c.items()) + f"  => {res[rule]['verdict']['status']}")
    passing = [r for r in RULES if res[r]["verdict"]["dev_val"]]

    # trials: exactly two, only for a full (non-partial) run; idempotent
    if partial:
        log("\npartial run (--assets or --placebo-draws differs from the pre-registration): NO trials logged")
    else:
        for rule in RULES:
            r = res[rule]
            wrote = VR.log_trial_once(f"studyH:{rule}_{'band' if rule == 'R1' else 'momentum'}", B.trial_spec(rule), {
                "dev_mean": r["dev"]["mean"], "dev_t": r["dev"]["t"], "val_mean": r["val"]["mean"], "val_t": r["val"]["t"],
                "placebo_p95": r["plc"]["p95"], "placebo_median": r["plc"]["median"], "dev_val_passed": r["verdict"]["dev_val"]})
            log(f"  {rule} is one trial: {'logged' if wrote else 'already in the registry, not logged again'}")

    hold = None
    if open_holdout and passing:
        hold = _holdout(log, passing, res, panel, cal, R, rf, row_of, draws, jobs, hold_years, assets)
    elif open_holdout:
        log("\n--open-holdout given, but the holdout is NOT opened: it opens only for a rule that passed development and validation")
    if hold is not None:
        _print_holdout(log, hold)

    _info(log, agg, cal, rf, universe, pre_years, res)
    if open_holdout:
        _names_block(log, frame, ff)
    else:
        log("\n-- the user's own names on secd data (IN-SAMPLE): WITHHELD. The secd history lies in the sealed window by nature, so, as with the "
            "v13 ticker block, it prints only with --open-holdout --")
    log(f"\nunique trials logged so far: {trial_count()}")


# --------------------------------------------------------------------------------------------------------------- checks
def _print_replication(log, rep, show_years) -> None:
    log("replication check (simulated 2x product = L r + (1-L) rf - 1.5%/252 against the actual ETF on common days; labels are the 1st, 2nd, ... "
        "calendar year of common data):")
    log(f"  {'pair':14s} {'L':>3s} {'days':>5s} {'slope':>7s} {'TE ann':>7s}  cumulative log gap actual - simulated per year of common data")
    for r in rep["rows"]:
        if not r["used"]:
            log(f"  {r['etf'] + '/' + r['und']:14s} {r['L']:+3.0f} {_i(r['n']):>5s}  skipped: fewer than {B.MIN_PAIR_DAYS} common days")
            continue
        gaps = "  ".join(f"y{y['rank']}({y['days']}d) {y['gap']:+.3f}" + (f"[{y['year']}]" if show_years else "") for y in r["years"])
        log(f"  {r['etf'] + '/' + r['und']:14s} {r['L']:+3.0f} {_i(r['n']):>5s} {r['slope']:7.3f} {_pct(r['te_ann'], 1):>7s}  {gaps}"
            + ("" if r["slope_ok"] else "  <-- SLOPE OUT OF RANGE") + (f"  (rf missing on {r['rf_missing']} days -> 0)" if r["rf_missing"] else ""))
    log(f"  median absolute gap per year over {rep['cells']} pair-year cells: {_n(rep['median_abs_gap'], 3)} (abort above {B.GAP_TOL}); "
        f"slope must lie within {B.SLOPE_TOL:.0%} of L  => {'PASS' if rep['ok'] else 'FAIL'}")


def _print_engine_checks(log, agg, universe, years) -> None:
    a1, a2 = agg[("main", "R1")], agg[("main", "R2")]
    log(f"\nsleeve-years: {_i(a1['sleeves'])}; sleeve-days {_i(a1['sleeve_days'])}; sleeve-years shorter than {2 * B.MIN_SHIFT} days "
        f"(delisted / ended early; their placebo shift is drawn from [n/2, n - n/2]): {_i(a1['short'])}")
    log(f"missing underlying returns inside sleeves (treated as a flat day): {_i(a1['miss_r'])}; sleeve-days with rf missing (treated as 0): {_i(a1['miss_rf'])}")
    log("transitions (position changes, a flip counts once) per sleeve-year, and share of sleeve-days held flat / long / short:")
    for rule, a in (("R1", a1), ("R2", a2)):
        d = a["days"] / max(int(a["days"].sum()), 1)
        log(f"  {rule}: {_n(a['trans'] / max(a['sleeves'], 1), 2)} transitions per sleeve-year; flat {_pct(d[0])}  long {_pct(d[1])}  short {_pct(d[2])}")
    log(f"  {'year':6s} {'sleeves':>8s} {'R1 trans/sl':>12s} {'R2 trans/sl':>12s}")
    for y in years:
        t1, n1 = a1["by_year"].get(y, (0, 0))
        t2, n2 = a2["by_year"].get(y, (0, 0))
        log(f"  {y:<6d} {n1:8d} {_n(t1 / max(n1, 1), 2):>12s} {_n(t2 / max(n2, 1), 2):>12s}")


# -------------------------------------------------------------------------------------------------------------- holdout
def _holdout(log, passing, res, panel, cal, R, rf, row_of, draws, jobs, hold_years, assets=None) -> dict:
    """The single consultation of the sealed window, for the rules that passed dev / validation. The marker is written AFTER the statistics are
    computed and BEFORE anything is printed."""
    universe = B.select_universe(panel, hold_years, cal, R, row_of, assets=assets)
    specs = [B.Spec(r, params=B.primary(r), placebo=True) for r in passing]
    agg = _run_years(log, hold_years, universe, (0.0, 0.0), row_of, R, rf, cal, specs, draws, jobs)
    out = {}
    for rule in passing:
        a = agg[("main", rule)]
        h = B.window_stat(a["S"], a["C"], cal, B.HOLD)
        ph = B.placebo_summary(B.placebo_stats(a["SP"], a["C"], cal, B.HOLD))
        v = B.licence_verdict(res[rule]["dev"], res[rule]["val"], res[rule]["plc"], h, ph, True)
        out[rule] = dict(stat=h, plc=ph, verdict=v)
    V.mark_holdout_opened(f"v15 holdout opened {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: "
                          + "; ".join(f"{r}: {out[r]['verdict']['status']}" for r in out), B.MARKER)
    return out


def _print_holdout(log, hold) -> None:
    log(f"\n== HOLDOUT OPENED (once): {B.HOLD[0][:7]}..{B.HOLD[1][:7]}; the licence stands only if the mean is > 0 and above the placebo median ==")
    for rule, h in hold.items():
        log(f"  {rule}: holdout {_st(h['stat'])}; placebo median {_n(1e4 * h['plc']['median'], 2)} bp/day (95th pct {_n(1e4 * h['plc']['p95'], 2)})  "
            f"=> {h['verdict']['status']}")


# ------------------------------------------------------------------------------------------------------ information only
def _trade_block(log, name, a, cost) -> None:
    tr = np.array(a["trades"], dtype="float64").reshape(-1, 4)
    if len(tr) == 0:
        log(f"  {name}: no trades")
        return

    def line(label, t):
        if len(t) == 0:
            return f"    {label}: no trades"
        r = t[:, 3]
        w, l = r[r > 0], r[r <= 0]
        return (f"    {label}: trades {_i(len(t))}  win rate {_pct(float((r > 0).mean()))}  mean win {_pct(float(w.mean()) if len(w) else float('nan'), 2)}  "
                f"mean loss {_pct(float(l.mean()) if len(l) else float('nan'), 2)}  expectancy {_pct(float(r.mean()), 2)}  "
                f"stop frequency {_pct(float((t[:, 2] == B.REASONS.index('stop')).mean()))}  mean holding days {_n(float(t[:, 1].mean()), 1)}")
    log(f"  {name}: transitions per sleeve-year {_n(a['trans'] / max(a['sleeves'], 1), 2)}")
    log(line("all trades", tr))
    if name == "R1":
        log(line("long-2x legs", tr[tr[:, 0] > 0]))
        log(line("inverse-2x legs", tr[tr[:, 0] < 0]))


def _info(log, agg, cal, rf, universe, years, res) -> None:
    log("\n== INFORMATION ONLY (no decision weight; development and validation years) ==")
    M = agg[("main", "R1")]
    dev_m = B.window_mask(cal, B.DEV)
    log("\n-- per-trade statistics (trade return = instrument value ratio less one entry and one exit side) --")
    for rule in RULES:
        _trade_block(log, rule, agg[("main", rule)], B.COST)
    log("\n-- sleeve-year maximum drawdown (of the sleeve's net equity incl. rf while flat) --")
    log("  NOTE: a one-day pause after a stop is a TIMEOUT, not a loss limit: the account stop (-15% of deposit) cannot be tested by single-name sleeves.")
    for rule in RULES:
        dd = np.array([x[1] for x in agg[("main", rule)]["dd"]], dtype="float64")
        if len(dd):
            log(f"  {rule}: median {_pct(float(np.median(dd)), 1)}  90th-percentile loss {_pct(float(np.percentile(dd, 10)), 1)}  "
                f"share of sleeve-years beyond -15% {_pct(float((dd < -0.15).mean()), 1)}  (n {_i(len(dd))})")

    log("\n-- benchmarks (mean daily excess over rf; dev | validation) --")
    r_dev = float(np.nanmean(rf[dev_m]))
    log(f"  cash (rf): excess 0 by construction (mean rf {_pct(ANN_ * r_dev, 2)}/yr in dev)")
    rows = [("R1 (rule)", ("main", "R1")), ("R2 (rule)", ("main", "R2")), ("buy-and-hold underlying", ("main", "BH_und")),
            ("buy-and-hold 2x long", ("main", "BH_long2")), ("buy-and-hold 2x inverse", ("main", "BH_short2"))]
    for lab, k in rows:
        a = agg[k]
        log(f"  {lab:26s} dev {_st(B.window_stat(a['S'], a['C'], cal, B.DEV))}  | val {_st(B.window_stat(a['S'], a['C'], cal, B.VAL))}")

    log(f"\n-- by trailing-volatility tercile (pooled cut-offs; tercile 1 = lowest vol; dev | validation) --")
    for rule in RULES:
        a = agg[("main", rule)]
        for k in range(3):
            S, C = a["G"][k]
            log(f"  {rule} tercile {k + 1}: dev {_st(B.window_stat(S, C, cal, B.DEV))}  | val {_st(B.window_stat(S, C, cal, B.VAL))}")

    log("\n-- by calendar year (mean daily excess, bp/day and annualised; sleeves active) --")
    log(f"  {'year':6s} {'R1 bp/day':>10s} {'R1 %/yr':>9s} {'R2 bp/day':>10s} {'R2 %/yr':>9s} {'sleeves':>8s}")
    yr = cal.astype("datetime64[Y]").astype(int) + 1970
    for y in years:
        m = yr == y
        cells = []
        for rule in RULES:
            a = agg[("main", rule)]
            d = B.daily_excess(a["S"], a["C"])[m & (a["C"] > 0)]
            mu = float(d.mean()) if len(d) else float("nan")
            cells += [_n(1e4 * mu, 2), _n(100 * ANN_ * mu, 1)]
        log(f"  {y:<6d} {cells[0]:>10s} {cells[1]:>9s} {cells[2]:>10s} {cells[3]:>9s} {agg[('main', 'R1')]['by_year'].get(y, (0, 0))[1]:8d}")

    log("\n-- sensitivities (information only; dev | validation) --")
    for name in ("R1_fee0", "R1_fee3", "R1_cost0", "R1_cost25", "R1_lag0", "R1_d03", "R1_d10", "R1_N63",
                 "R2_fee0", "R2_fee3", "R2_cost0", "R2_cost25", "R2_lag0"):
        a = agg[("main", name)]
        log(f"  {name:10s} dev {_st(B.window_stat(a['S'], a['C'], cal, B.DEV))}  | val {_st(B.window_stat(a['S'], a['C'], cal, B.VAL))}")

    log("\n-- R3 (INTU-style asymmetric band on the stock itself: L=1, buy below 0.967 M, take profit at 1.133 M, stop -20%) --")
    a = agg[("main", "R3")]
    log(f"  R3         dev {_st(B.window_stat(a['S'], a['C'], cal, B.DEV))}  | val {_st(B.window_stat(a['S'], a['C'], cal, B.VAL))}")
    _trade_block(log, "R3", a, B.COST)

    log(f"\n-- top-{B.TOP_N_INFO} universe (the v13 universe, no volatility filter; same rules; dev | validation) --")
    for rule in RULES:
        a = agg[("top100", rule)]
        log(f"  {rule} top-{B.TOP_N_INFO}: dev {_st(B.window_stat(a['S'], a['C'], cal, B.DEV))}  | val {_st(B.window_stat(a['S'], a['C'], cal, B.VAL))}  "
            f"(sleeve-years {_i(a['sleeves'])})")


def _names_block(log, frame, ff) -> None:
    log("\n== INFORMATION ONLY: the user's own names on secd data -- IN-SAMPLE (the rules were designed on exactly these names and this period) ==")
    if frame is None:
        log("  n/a: no secd file")
        return
    from qap import prices as PR
    log("  one sleeve per name over its whole available history (closed at its last close), same parameters; sleeve total return = compounded net "
        "return incl. rf while flat; B&H = the underlying's total return over the same days")
    for rule in ("R1", "R2", "R3"):
        log(f"\n  {rule}:")
        log(f"    {'name':6s} {'days':>6s} {'trades':>7s} {'win rate':>9s} {'stops':>6s} {'sleeve':>10s} {'B&H underlying':>15s}")
        for nm in B.USER_NAMES:
            r = PR.return_series(frame, nm)
            if len(r) == 0:
                log(f"    {nm:6s} not in the secd file")
                continue
            rfv = B.rf_series(ff, r.index.to_numpy())
            out = B.name_sleeve(r.to_numpy(), rfv, B.primary(rule))
            if out is None:
                log(f"    {nm:6s} {len(r):6d}  history too short for the anchor")
                continue
            log(f"    {nm:6s} {len(r):6d} {out['trades']:7d} {_pct(out['win'], 0):>9s} {out['stops']:6d} {_pct(out['total'], 1):>10s} {_pct(out['bh'], 1):>15s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-holdout", action="store_true", help="open the sealed 2020-2025 window once (only for a rule that passed dev and validation)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes over calendar years (fork)")
    ap.add_argument("--skip-replication", action="store_true", help="continue when data/raw/secd_tickers.parquet is absent (the check is otherwise mandatory)")
    ap.add_argument("--placebo-draws", type=int, default=B.DRAWS, help="placebo draws per sleeve-year (pre-registered: 200)")
    ap.add_argument("--assets", type=int, default=None, help="keep only the first N sleeves of each year (quick runs; logs no trials)")
    a = ap.parse_args()
    main(a.open_holdout, a.jobs, a.assets, a.placebo_draws, a.skip_replication)
