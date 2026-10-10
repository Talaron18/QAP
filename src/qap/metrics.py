"""Signal and backtest evaluation metrics."""
import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def rank_ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Per-date Spearman correlation of signal and forward return (dates x assets frames).

    Ranks are taken over the assets present in BOTH frames on each date.
    """
    mask = signal.notna() & fwd_ret.notna()
    return signal.where(mask).rank(axis=1).corrwith(fwd_ret.where(mask).rank(axis=1), axis=1)


def ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Per-date Pearson correlation of signal and forward return."""
    return signal.corrwith(fwd_ret, axis=1)


def icir(ic_series: pd.Series) -> float:
    s = ic_series.dropna()
    return float(s.mean() / s.std(ddof=1))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """SR_0: expected maximum per-period Sharpe over n independent null trials."""
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
                                         + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe(sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                    sr0: float = 0.0) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): P(true SR > sr0).

    sr and sr0 must be PER-PERIOD Sharpe ratios (same frequency as the n_obs returns),
    never annualized. kurt is raw (non-excess) kurtosis; Gaussian = 3.
    """
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    return float(norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / denom))
