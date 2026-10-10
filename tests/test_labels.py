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
