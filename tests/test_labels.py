import pytest
import numpy as np
import pandas as pd
from qap.labels import forward_return_label


def test_label_uses_only_t_plus_1_and_t_plus_2():
    close = pd.DataFrame({"A": [100.0, 101.0, 103.0, 90.0, 80.0]})
    y = forward_return_label(close)
    assert y["A"].iloc[0] == 103.0 / 101.0 - 1  # not 101/100: t+1 -> t+2
    assert y["A"].iloc[2] == 80.0 / 90.0 - 1
    assert y["A"].iloc[3:].isna().all()


def test_label_invariant_to_close_at_t_and_after_t_plus_2():
    base = pd.DataFrame({"A": [100.0, 101.0, 103.0, 90.0, 80.0]})
    alt = base.copy()
    alt.loc[0, "A"] = 1.0   # price at t
    alt.loc[3:, "A"] = 7.0  # prices after t+2
    assert forward_return_label(base)["A"].iloc[0] == forward_return_label(alt)["A"].iloc[0]


def test_horizon():
    close = pd.DataFrame({"A": [1.0, 2.0, 4.0, 8.0, 16.0]})
    y = forward_return_label(close, horizon=2)
    assert y["A"].iloc[0] == 8.0 / 2.0 - 1


def test_return_label_matches_price_label_when_no_delisting():
    from qap.labels import forward_return_from_returns
    close = pd.DataFrame({"A": [100.0, 101.0, 103.0, 90.0, 80.0]})
    ret = close.pct_change()
    a = forward_return_from_returns(ret)
    b = forward_return_label(close)
    assert np.allclose(a["A"].dropna(), b["A"].dropna())


def test_delisting_loss_stays_in_label_not_nan():
    from qap.labels import forward_return_from_returns
    ret = pd.DataFrame({
        "A": [0.01, 0.01, 0.01, -0.30, np.nan, np.nan],   # delists after period 3 with -30%
        "B": [0.01] * 6,                                  # survives to the end
    })
    y1 = forward_return_from_returns(ret, horizon=1)      # label at t = return of period t+2
    assert y1["A"].iloc[1] == pytest.approx(-0.30)
    y3 = forward_return_from_returns(ret, horizon=3)      # periods t+2 .. t+4
    assert y3["A"].iloc[0] == pytest.approx(1.01 * 0.70 * 1.0 - 1.0)
    assert y3["A"].iloc[1] == pytest.approx(0.70 - 1.0)
    assert np.isnan(y3["B"].iloc[4])                      # survivor: window runs past the sample


def test_leading_nans_are_not_filled():
    from qap.labels import forward_return_from_returns
    ret = pd.DataFrame({"A": [np.nan, np.nan, np.nan, 0.01, 0.01, 0.01]})
    y = forward_return_from_returns(ret)
    assert np.isnan(y["A"].iloc[0])  # not listed yet at t+2
    assert y["A"].iloc[1] == pytest.approx(0.01)


def test_cv_purge_matches_label_reach():
    from qap.cv import purged_kfold
    from qap.labels import forward_return_from_returns, label_reach
    rng = np.random.default_rng(0)
    ret = pd.DataFrame({"A": rng.normal(0, 0.01, 200)})
    h, lag, emb = 3, 1, 2
    y = forward_return_from_returns(ret, horizon=h, lag=lag)
    for train, test in purged_kfold(len(ret), 5, label_reach(h, lag), emb):
        perturbed = ret.copy()
        perturbed.iloc[test, 0] += 0.5                    # change every return inside the test fold
        y2 = forward_return_from_returns(perturbed, horizon=h, lag=lag)
        both = y.iloc[train, 0].dropna().index.intersection(y2.iloc[train, 0].dropna().index)
        assert np.allclose(y.loc[both, "A"], y2.loc[both, "A"])  # no training label saw test returns
