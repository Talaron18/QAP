"""Transaction-cost model: half-spread + square-root impact (+ commission), from CRSP daily liquidity inputs.

Per-trade cost as a fraction of the traded value, for a trade of weight change |dw| on a book of `aum` dollars:
    cost = |dw| * (0.5 * spread + commission) + y * |dw| * sigma_d * sqrt(|dw| * aum / adv)
i.e. the impact part scales with |dw|**1.5 (report section 1.6 and 6.3). Everything is multiplied by
`mult` for cost-stress tests (the report's deployment rule is stated at 1.5x). Borrow fees and taxes
are not modelled. Pre-2004 Nasdaq volume is double counted in CRSP, which understates impact there.
"""
import numpy as np
import pandas as pd

CS_K = 3.0 - 2.0 * np.sqrt(2.0)
MIN_SPREAD, MAX_SPREAD = 1e-4, 0.5


def corwin_schultz(h: float, l: float, h0: float, l0: float) -> float:
    """Corwin-Schultz (2012) two-day spread estimate for one day pair, negatives floored at 0."""
    beta = np.log(h / l) ** 2 + np.log(h0 / l0) ** 2
    gamma = np.log(max(h, h0) / min(l, l0)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / CS_K - np.sqrt(gamma / CS_K)
    return max(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0.0)


def abdi_ranaldo_spread(x_mean):
    """Abdi-Ranaldo (2017) spread from the monthly mean of (c - eta_t)(c - eta_t+1): sqrt(max(4*mean, 0))."""
    return np.sqrt(np.maximum(4.0 * np.asarray(x_mean, dtype=float), 0.0))


def trade_cost(dw, spread, sigma_d, adv, aum: float, y: float = 1.0, commission: float = 0.0, mult: float = 1.0):
    """Cost (fraction of book capital) of trades of absolute weight change `dw`. Vectorised."""
    dw = np.abs(np.asarray(dw, dtype=float))
    fixed = dw * (0.5 * np.asarray(spread, dtype=float) + commission)
    impact = y * dw * np.asarray(sigma_d, dtype=float) * np.sqrt(dw * aum / np.asarray(adv, dtype=float))
    return mult * (fixed + impact)


def yyyymm_to_mi(yyyymm: pd.Series) -> pd.Series:
    y = yyyymm.astype("int64")
    return (y // 100) * 12 + (y % 100) - 1


def liquidity_inputs(panel: pd.DataFrame, agg: pd.DataFrame, min_days: int = 10) -> pd.DataFrame:
    """Per (permno, month) spread, daily volatility and dollar ADV, with explicit fallbacks.

    spread: mean quoted spread if >= min_days quotes, else Corwin-Schultz if >= min_days pairs, else the
    median spread of the same month x size decile ('fallback'). Missing sigma_d and adv use the same
    month x decile medians. Spreads are clipped to [1 bp, 50%]. ADV is |price| * volume computed by us
    (adv_b); adv_a (dlyprcvol) is kept for the unit check.
    """
    a = agg.copy()
    a["mi"] = yyyymm_to_mi(a["yyyymm"])
    a["ar_spread"] = abdi_ranaldo_spread(a["ar_x_mean"].where(a["n_ar"] >= min_days))
    base = panel[["permno", "mi", "me"]].merge(a.drop(columns="yyyymm"), on=["permno", "mi"], how="left")
    base["decile"] = np.ceil(base.groupby("mi")["me"].rank(pct=True) * 10).clip(1, 10)
    q_ok = base["n_q"].fillna(0) >= min_days
    c_ok = base["n_cs"].fillna(0) >= min_days
    base["spread"] = np.where(q_ok, base["qspread"], np.where(c_ok, base["cs_spread"], np.nan))
    base["spread_src"] = np.where(q_ok, "quoted", np.where(c_ok, "cs", "fallback"))
    base["sigma_d"] = base["sigma_d"].where(base["n_ret"].fillna(0) >= 5)
    base["adv"] = base["adv_b"].where(base["adv_b"] > 0)
    for col in ("spread", "sigma_d", "adv"):
        med = base.groupby(["mi", "decile"])[col].transform("median")
        base[col] = base[col].fillna(med)
        base[col] = base[col].fillna(base.groupby("mi")[col].transform("median"))
        base[col] = base[col].fillna(base[col].median())
    base["spread"] = base["spread"].clip(MIN_SPREAD, MAX_SPREAD)
    return base


def sanity_report(inp: pd.DataFrame) -> tuple:
    """Shape checks on the liquidity inputs. Returns (report lines, {check name: bool})."""
    lines, checks = [], {}
    yr = inp["mi"] // 12
    inp = inp.assign(decade=(yr // 10) * 10, year=yr)
    both = inp[inp["adv_a"].notna() & (inp["adv_b"] > 0)]
    ratio = (both["adv_a"] / both["adv_b"])
    share = float(ratio.between(0.9, 1.1).mean())
    lines.append(f"dollar volume: median adv_a/adv_b = {ratio.median():.3f}; share within 10%: {share:.1%}")
    checks["dollar volume columns agree (>=90% within 10%)"] = share >= 0.9

    inp["size_q"] = np.ceil(inp.groupby("mi")["me"].rank(pct=True) * 5).clip(1, 5)
    ov = inp[inp["qspread"].notna() & inp["cs_spread"].notna() & (inp["n_q"] >= 10) & (inp["n_cs"] >= 10)]
    lines.append("quoted vs Corwin-Schultz where both exist (decade x size quintile): corr / median ratio cs/quoted")
    for (dec, sq), g in ov.groupby(["decade", "size_q"]):
        if len(g) > 500:
            lines.append(f"  {int(dec)}s Q{int(sq)}: n={len(g):,} corr={g['qspread'].corr(g['cs_spread']):.2f} "
                         f"ratio={(g['cs_spread'] / g['qspread']).median():.2f}")

    med = inp.groupby(["decade", "size_q"])["spread"].median().unstack("size_q")
    lines.append("median spread (bp) by decade x size quintile (Q1 smallest):")
    mono = True
    for dec, row in med.iterrows():
        v = row.dropna()
        lines.append(f"  {int(dec)}s: " + "  ".join(f"Q{int(k)}={x * 1e4:.0f}" for k, x in v.items()))
        mono &= bool((v.diff().dropna() <= 1e-12).all())
    checks["median spread falls with size in every decade"] = mono

    pre = inp[(inp["year"] >= 1996) & (inp["year"] <= 2000)]["spread"].median()
    post = inp[(inp["year"] >= 2003) & (inp["year"] <= 2007)]["spread"].median()
    lines.append(f"decimalization: median spread 1996-2000 = {pre * 1e4:.0f} bp, 2003-2007 = {post * 1e4:.0f} bp")
    checks["post-2001 median spread below pre-2001"] = bool(post < pre)

    cov = inp.groupby("decade")["spread_src"].value_counts(normalize=True).unstack().fillna(0)
    lines.append("spread source coverage by decade: " + "; ".join(
        f"{int(d)}s " + "/".join(f"{c}={r.get(c, 0):.0%}" for c in ("quoted", "cs", "fallback")) for d, r in cov.iterrows()))
    return lines, checks
