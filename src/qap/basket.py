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
