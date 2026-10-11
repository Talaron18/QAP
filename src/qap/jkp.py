"""Study A: JKP theme scores built from the Global Factor Data characteristics (pre-registration v3)."""
import glob
import math
import pathlib

import numpy as np
import pandas as pd

from qap.jkp_themes import THEMES
from qap.panel import to_mi

SLOW_EXCLUDE = ("Size", "Short-Term Reversal", "Seasonality")


def theme_scores(df: pd.DataFrame, themes: dict = THEMES, min_frac: float = 0.5, min_members: int = 3) -> pd.DataFrame:
    """Per stock-month theme score: mean over available members of sign * (within-month percentile rank - 0.5).

    df needs `mi` and the member columns (missing member columns are skipped but the availability threshold still uses the
    theme's full member count, so a missing column can only make the score rarer, never silently easier).
    A score is missing unless at least max(min_members, ceil(min_frac * n_members)) members are available.
    """
    out = {}
    for theme, members in themes.items():
        present = [(c, s) for c, s in members if c in df.columns]
        need = max(min_members, math.ceil(min_frac * len(members)))
        if len(present) < need:
            out[theme] = pd.Series(np.nan, index=df.index)
            continue
        ranks = pd.concat([(df.groupby("mi")[c].rank(pct=True) - 0.5) * s for c, s in present], axis=1)
        out[theme] = ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= need)
    return pd.DataFrame(out, index=df.index)


def add_composites(scores: pd.DataFrame, mi: pd.Series, themes=None) -> pd.DataFrame:
    """jkp_all13 (needs >= 10 themes) and jkp_slow10 (without Size, Short-Term Reversal, Seasonality; needs >= 8)."""
    themes = list(themes or scores.columns)
    r = scores[themes].groupby(mi).rank(pct=True) - 0.5
    out = scores.copy()
    out["jkp_all13"] = r.mean(axis=1).where(r.notna().sum(axis=1) >= 10)
    slow = [t for t in themes if t not in SLOW_EXCLUDE]
    out["jkp_slow10"] = r[slow].mean(axis=1).where(r[slow].notna().sum(axis=1) >= 8)
    return out


def build_theme_signals(eligible: pd.DataFrame, jkp_dir: str) -> tuple:
    """Merge JKP year files onto the eligible universe and compute theme scores and composites.

    eligible: permno, mi plus whatever the caller wants carried. Returns (signals frame, check frame with the raw columns
    gp_at, be_me, ret_exc_lead1m used by the pre-registered cross-checks).
    """
    el = eligible.copy()
    el["permno"] = el["permno"].astype("int64")
    parts, checks = [], []
    for f in sorted(glob.glob(str(pathlib.Path(jkp_dir) / "[0-9]*.parquet"))):
        j = pd.read_parquet(f)
        j["permno"] = pd.to_numeric(j["permno"]).astype("int64")
        j["mi"] = to_mi(j["eom"]).values
        j = el.merge(j.drop(columns=["me"], errors="ignore"), on=["permno", "mi"], how="inner")
        if j.empty:
            continue
        sc = add_composites(theme_scores(j), j["mi"])
        base = j[[c for c in eligible.columns if c in j.columns]].copy()
        parts.append(pd.concat([base.reset_index(drop=True), sc.reset_index(drop=True)], axis=1))
        keep = [c for c in ("permno", "mi", "gp_at", "be_me", "ret_exc_lead1m", "size_grp") if c in j.columns]
        checks.append(j[keep].reset_index(drop=True))
    return pd.concat(parts, ignore_index=True), pd.concat(checks, ignore_index=True)


def rank_corr_by_month(a: pd.DataFrame, x: str, y: str, min_names: int = 50) -> pd.Series:
    """Per-month Spearman correlation of two columns (rows with both present)."""
    d = a.dropna(subset=[x, y])
    g = d.groupby("mi")
    rx, ry = g[x].rank(), g[y].rank()
    return pd.DataFrame({"mi": d["mi"], "rx": rx, "ry": ry}).groupby("mi").apply(
        lambda t: t["rx"].corr(t["ry"]) if len(t) >= min_names else np.nan, include_groups=False).dropna()
