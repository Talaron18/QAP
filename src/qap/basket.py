"""Short-basket helper (borrow-cost check): the bottom-quintile value-weighted basket for one signal and month, and the
weight-averaged borrow fee from a user-filled sheet. INFORMATION ONLY: no returns are used, nothing is logged as a trial."""
import numpy as np
import pandas as pd

from qap.portfolio import quintile_legs

# break-even annual borrow cost (%/yr) = net long-short %/mo x 12 from ROADMAP 'break-even borrow cost' (dev, validation)
BREAK_EVEN = {"jkp_slow10": (6.5, 4.6), "jkp_all13": (2.7, 2.5), "gpa": (2.7, 4.4), "Quality": (3.6, 7.1), "Profitability": (4.2, 5.5)}
NO = {"n", "no", "0", "false", "f", "na", "n/a", "x"}


def short_basket(sig: pd.DataFrame, name: str, mi: int) -> pd.DataFrame:
    """Bottom-quintile VW basket at month `mi` (same eligibility and ranking as portfolio.quintile_legs), largest weight first."""
    s = sig.loc[sig["mi"] == mi]
    if s.empty:
        raise ValueError(f"no rows for month index {mi}")
    w = quintile_legs(s, name)["S"]
    w = w[w["mi"] == mi][["permno", "w"]]
    cols = [c for c in ("permno", "me", "me_co", "primaryexch", name) if c in s.columns]
    out = w.merge(s[cols].drop_duplicates("permno").astype({"permno": "int64"}), on="permno", how="left")
    out = out.rename(columns={name: "score"}).sort_values("w", ascending=False).reset_index(drop=True)
    out["cum_w"] = out["w"].cumsum()
    return out


def attach_names(basket: pd.DataFrame, names: pd.DataFrame, mi: int) -> pd.DataFrame:
    """Add ticker and issuer name from the CRSP names history (the row active at month-end `mi`; else the latest before it)."""
    if names is None or "permno" not in names or "ticker" not in names:
        return basket.assign(ticker="", issuer="")
    n = names.copy()
    n["permno"] = n["permno"].astype("int64")
    end = pd.Timestamp(year=mi // 12, month=mi % 12 + 1, day=1) + pd.offsets.MonthEnd(0)
    n = n[pd.to_datetime(n["namedt"]) <= end] if "namedt" in n else n
    n = n.sort_values("namedt") if "namedt" in n else n
    last = n.drop_duplicates("permno", keep="last")
    iss = "issuernm" if "issuernm" in last else None
    last = last.assign(issuer=last[iss] if iss else "")[["permno", "ticker", "issuer"]]
    return basket.merge(last, on="permno", how="left").fillna({"ticker": "", "issuer": ""})


def blended_fee(sheet: pd.DataFrame, fee_col: str = "borrow_fee_pct", avail_col: str = "available",
                fill_low: float = 0.5, fill_high: float = 25.0) -> dict:
    """Weight-averaged borrow fee (%/yr) from a filled sheet with columns w, fee_col, avail_col.

    Names marked unavailable (N/no/0/false) cannot be shorted: their weight is reported and excluded (the rest renormalised).
    Names with no fee entered are bracketed: low = they cost fill_low, high = they cost fill_high.
    """
    d = sheet.copy()
    w = pd.to_numeric(d["w"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    fee = pd.to_numeric(d[fee_col], errors="coerce").to_numpy(dtype="float64") if fee_col in d else np.full(len(d), np.nan)
    av = d[avail_col].astype(str).str.strip().str.lower() if avail_col in d else pd.Series([""] * len(d))
    unavail = av.isin(NO).to_numpy()
    known = np.isfinite(fee) & ~unavail
    missing = ~np.isfinite(fee) & ~unavail
    tot = float(w.sum())
    tradable = float(w[~unavail].sum())
    out = {"total_w": tot, "unavailable_w": float(w[unavail].sum()), "unavailable_share": float(w[unavail].sum() / tot) if tot else np.nan,
           "known_fee_w": float(w[known].sum()), "missing_fee_w": float(w[missing].sum()),
           "known_share_of_tradable": float(w[known].sum() / tradable) if tradable else np.nan}
    if tradable <= 0:
        return {**out, "fee_known_avg": np.nan, "fee_low": np.nan, "fee_high": np.nan}
    wk = float(w[known].sum())
    out["fee_known_avg"] = float((w[known] * fee[known]).sum() / wk) if wk > 0 else np.nan
    base = float((w[known] * fee[known]).sum())
    out["fee_low"] = (base + fill_low * float(w[missing].sum())) / tradable
    out["fee_high"] = (base + fill_high * float(w[missing].sum())) / tradable
    return out


def verdict_lines(name: str, stats: dict, hurdle_extra: float = 1.0) -> list:
    """Compare the blended fee (plus an assumed rebate shortfall `hurdle_extra` %/yr) with the break-even table."""
    be = BREAK_EVEN.get(name)
    L = [f"  weight-averaged fee, names with a fee: {stats['fee_known_avg']:.2f}%/yr; bracket for the rest: {stats['fee_low']:.2f}-{stats['fee_high']:.2f}%/yr"]
    L.append(f"  unavailable to short: {stats['unavailable_share']:.1%} of weight; fee known for {stats['known_share_of_tradable']:.1%} of tradable weight")
    if be:
        for label, b in zip(("dev", "validation"), be):
            lo, hi = stats["fee_low"] + hurdle_extra, stats["fee_high"] + hurdle_extra
            tag = "CLEARS" if hi < b else ("FAILS" if lo >= b else "UNDETERMINED (depends on names with no fee)")
            L.append(f"  vs {label} break-even {b:.1f}%/yr (hurdle = fee + {hurdle_extra:.1f}% rebate shortfall = {lo:.2f}-{hi:.2f}%): {tag}")
    return L


# ---- ASSUMED fees (not data): a common-sense schedule by market-cap tier, used only when real borrow fees are not available ----
TIER_EDGES = (500.0, 2000.0, 10000.0)                      # $ millions; tiers: <500, 500-2000, 2000-10000, >=10000
GC_FEE = (3.0, 1.5, 0.6, 0.3)                              # general-collateral borrow fee, %/yr, by tier (smallest first)
# (share of tier weight that is hard-to-borrow, per tier smallest first; hard-to-borrow fee %/yr) per scenario
SCENARIOS = {"optimistic": ((0.05, 0.02, 0.0, 0.0), 10.0), "base": ((0.12, 0.05, 0.01, 0.0), 15.0), "pessimistic": ((0.25, 0.10, 0.03, 0.01), 25.0)}


def assumed_fee(me, scenario: str) -> np.ndarray:
    """Expected borrow fee (%/yr) for each market cap (in $ millions): general-collateral fee, blended with the hard-to-borrow share."""
    share, htb = SCENARIOS[scenario]
    tier = np.digitize(np.asarray(me, dtype="float64"), TIER_EDGES)             # 0 = smallest
    gc, s = np.asarray(GC_FEE)[tier], np.asarray(share)[tier]
    return gc * (1.0 - s) + htb * s


def assumed_summary(sheet: pd.DataFrame, name: str, rebate_shortfall: float = 1.0) -> list:
    """Weight-averaged ASSUMED all-in fee by scenario vs the break-even table, plus the hard-to-borrow weight share that would break even."""
    w = pd.to_numeric(sheet["w"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    me = pd.to_numeric(sheet["me"], errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(me) & (w > 0)
    w, me = w[ok] / w[ok].sum(), me[ok]
    tier_w = np.bincount(np.digitize(me, TIER_EDGES), weights=w, minlength=4)
    L = ["ASSUMED fees (a common-sense schedule, NOT measured data); weight by market-cap tier <500M / 500M-2B / 2-10B / >=10B: "
         + " / ".join(f"{x:.0%}" for x in tier_w)]
    be = BREAK_EVEN.get(name)
    gc_avg = float((w * np.asarray(GC_FEE)[np.digitize(me, TIER_EDGES)]).sum())
    for sc in SCENARIOS:
        fee = float((w * assumed_fee(me, sc)).sum())
        hurdle = fee + rebate_shortfall
        line = f"  {sc:12s} weighted fee {fee:5.2f}%/yr, hurdle with {rebate_shortfall:.1f}% rebate shortfall {hurdle:5.2f}%/yr"
        if be:
            line += " | " + ", ".join(f"{lab} BE {b:.1f}: {'clears' if hurdle < b else 'FAILS'}" for lab, b in zip(("dev", "val"), be))
        L.append(line)
    if be:
        htb = {sc: SCENARIOS[sc][1] for sc in SCENARIOS}["base"]
        for lab, b in zip(("dev", "val"), be):
            h = (b - rebate_shortfall - gc_avg) / (htb - gc_avg)
            L.append(f"  break-even {lab}: tolerates at most {max(h, 0.0):.0%} of basket weight at {htb:.0f}%/yr hard-to-borrow (the rest at the general-collateral schedule, avg {gc_avg:.2f}%/yr)")
    return L
