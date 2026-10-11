"""Shared helpers for pre-registered signal studies: costed net long-minus-universe series and the promotion table."""
import numpy as np
import pandas as pd

from qap.candidates import DEV, VAL, promotion_stats
from qap.portfolio import costed_books

SEALED_CUT = 2020 * 12


def net_lu_series(sig: pd.DataFrame, panel: pd.DataFrame, inp: pd.DataFrame, aum_t, name: str, mult: float = 1.5,
                  y: float = 1.0, cut: int = SEALED_CUT) -> pd.Series:
    """Monthly net (gross - mult * cost) long-minus-universe return of the top quintile of `name`, months before `cut`."""
    r = costed_books(sig, panel, inp, name, aum_t, y, 0.0)["L_minus_U"]
    idx = r["gross"].index.intersection(r["cost"].index)
    if cut is not None:
        idx = idx[idx < cut]
    return (r["gross"][idx] - mult * r["cost"][idx]).rename(name)


def split_means(s: pd.Series) -> dict:
    """Mean (% per month) and t-stat on dev and validation windows."""
    out = {}
    for label, (y0, y1) in (("dev", DEV), ("val", VAL)):
        v = s[(s.index >= y0 * 12) & (s.index <= y1 * 12 + 11)].dropna()
        out[label] = (round(float(v.mean()) * 100, 3) if len(v) else None,
                      round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 2) if len(v) > 2 else None)
    return out


def promotion_table(series: dict, incumbent: str = "gpa") -> dict:
    """promotion_stats for every non-incumbent series against the incumbent's net series."""
    return {n: promotion_stats(s, series[incumbent]) for n, s in series.items() if n != incumbent}


def load_context(raw_dir: str = "data/raw"):
    """Panel (ME in $ millions), links, funda, liquidity inputs and market-scaled AUM series, as in the earlier studies."""
    import pathlib

    from qap.costs import liquidity_inputs
    from qap.links import filter_ccm_links
    from qap.panel import build_monthly_panel, calibrate_me_scale
    from qap.portfolio import market_scaled_aum
    raw = pathlib.Path(raw_dir)
    monthly = pd.read_parquet(raw / "crsp_monthly.parquet")
    delists = pd.read_parquet(raw / "crsp_delistings.parquet")
    links = filter_ccm_links(pd.read_parquet(raw / "ccm_links.parquet"))
    funda = pd.read_parquet(raw / "compustat_annual.parquet")
    agg = pd.read_parquet(raw / "crsp_daily_agg.parquet")
    panel = build_monthly_panel(monthly, delists, me_scale=1.0)
    scale, _ = calibrate_me_scale(panel, funda, links)
    panel["me"] = panel["mthcap"].astype(float) / scale
    panel["me_co"] = panel.groupby(["permco", "mi"])["me"].transform("sum")
    return panel, links, funda, liquidity_inputs(panel, agg), market_scaled_aum(panel, 1e7)


def eligible_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """Formation universe: common-equity mask, positive company ME, ex NYSE-20th-percentile microcaps."""
    from qap.evaluate import exclude_microcaps
    k = panel.loc[panel["eq"] & (panel["me_co"] > 0), ["permno", "permco", "mi", "mthcaldt", "me", "me_co", "primaryexch"]]
    return exclude_microcaps(k).reset_index(drop=True)


def _series_worker(n):
    from qap.parallel import context
    c = context()
    return net_lu_series(c["sig"], c["panel"], c["inp"], c["aum_t"], n, c["mult"], c["y"])


def run_promotion_report(log, sig: pd.DataFrame, panel: pd.DataFrame, inp: pd.DataFrame, aum_t, gpa_series: pd.Series,
                         names, study: str, prereg: str, aum: float = 1e7, mult: float = 1.5, y: float = 1.0, jobs: int = 1) -> tuple:
    """Standalone (information only) lines, then the pre-registered PASS/FAIL promotion rule for each signal in `names`."""
    from qap.evaluate import evaluate_signal
    from qap.trials import log_trial
    series = {"gpa": gpa_series}
    log(f"\n== RESULTS (Study {study}): net long-minus-universe, ${aum:,.0f} (2019 market-size), {mult}x costs, Y={y}; holdout sealed ==")
    log("standalone, information only: net L-U % per month (t) on dev / validation | gross rank IC (t) on dev / validation")
    from qap.parallel import pmap, set_context
    set_context(sig=sig, panel=panel, inp=inp, aum_t=aum_t, mult=mult, y=y)
    series.update(zip(names, pmap(_series_worker, names, jobs, say=log)))      # independent per signal; logging stays in this process
    for n in names:
        ev = evaluate_signal(sig, panel, n)["by_period"]
        m = split_means(series[n])
        log(f"  {n:22s} dev {m['dev'][0]} ({m['dev'][1]})  val {m['val'][0]} ({m['val'][1]})  | IC dev {ev['dev1990_2014'].get('ic')} "
            f"({ev['dev1990_2014'].get('ic_t')})  val {ev['valid2015_2019'].get('ic')} ({ev['valid2015_2019'].get('ic_t')})")
    m = split_means(gpa_series)
    log(f"  {'gpa (incumbent)':22s} dev {m['dev'][0]} ({m['dev'][1]})  val {m['val'][0]} ({m['val'][1]})")
    log("\n== PROMOTION RULE (alpha vs gpa: dev alpha t >= 2.0 and validation alpha > 0) ==")
    promo = promotion_table(series)
    promoted = []
    for n in names:
        st = promo[n]
        log(f"  [{'PASS' if st['PROMOTE'] else 'FAIL'}] {n}: {st}")
        log_trial(f"study{study}:{n}", {"aum": aum, "mult": mult, "y": y, "vs": "gpa", "weights": "VW", "prereg": prereg}, st)
        if st["PROMOTE"]:
            promoted.append(n)
    log(f"\nPROMOTED (provisional; sealed holdout opens once for these): " + (", ".join(promoted) if promoted else "NONE"))
    return promo, promoted


def holdout_alpha(rc: pd.Series, rg: pd.Series) -> dict:
    """Once-only holdout look: alpha on 2020+ using the DEV beta of rc on rg (the pre-registered construction)."""
    from qap.candidates import _ols_alpha, _window
    j = pd.concat([rc.rename("c"), rg.rename("g")], axis=1).dropna()
    dev = _window(j, *DEV)
    _, b, _ = _ols_alpha(dev["c"].to_numpy(), dev["g"].to_numpy())
    h = j[j.index >= SEALED_CUT]
    e = h["c"] - b * h["g"]
    n = len(e)
    return {"months": int(n), "beta_dev": round(float(b), 2), "alpha_pct_mo": round(float(e.mean()) * 100, 3),
            "t": round(float(e.mean() / (e.std(ddof=1) / np.sqrt(n))), 2) if n > 2 else None,
            "standalone_pct_mo": round(float(h["c"].mean()) * 100, 3)}


def require_inputs(paths, hint: str) -> None:
    """Fail fast with a clear message when raw inputs are missing (before any heavy compute)."""
    import glob
    import pathlib
    bad = [str(p) for p in paths if not (glob.glob(str(pathlib.Path(p) / "[0-9]*.parquet")) if pathlib.Path(p).suffix == "" else pathlib.Path(p).exists())]
    if bad:
        raise SystemExit(f"Missing raw inputs: {bad}\n{hint}")
