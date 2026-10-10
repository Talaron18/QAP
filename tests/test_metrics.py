import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm
from qap.metrics import deflated_sharpe, expected_max_sharpe, icir, rank_ic


def test_dsr_gaussian_denominator_reduces_to_sqrt_1_plus_sr2_over_2():
    sr, T = 0.1, 250
    expected = norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 + sr ** 2 / 2))
    assert deflated_sharpe(sr, T, skew=0.0, kurt=3.0) == pytest.approx(expected)


def test_dsr_penalised_by_selection_and_neg_skew():
    base = deflated_sharpe(0.1, 250)
    assert deflated_sharpe(0.1, 250, sr0=expected_max_sharpe(100, 0.01)) < base
    assert deflated_sharpe(0.1, 250, skew=-1.0) < base


def test_dsr_zero_sr_is_half():
    assert deflated_sharpe(0.0, 250) == pytest.approx(0.5)


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(10, 0.01) < expected_max_sharpe(1000, 0.01)
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_rank_ic_perfect_and_inverse():
    sig = pd.DataFrame(np.arange(12.0).reshape(3, 4))
    assert np.allclose(rank_ic(sig, sig ** 3), 1.0)
    assert np.allclose(rank_ic(sig, -sig), -1.0)


def test_icir():
    s = pd.Series([0.02, 0.04, 0.03, 0.05])
    assert icir(s) == pytest.approx(s.mean() / s.std(ddof=1))


def test_rank_ic_nan_uses_common_set():
    sig = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]])
    fwd = pd.DataFrame([[10.0, np.nan, 30.0, 20.0]])
    # common set {0,2,3}: signal ranks 1,2,3 vs fwd ranks 1,3,2 -> Spearman 0.5
    assert rank_ic(sig, fwd).iloc[0] == pytest.approx(0.5)
