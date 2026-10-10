"""Forward-return labels with an explicit one-day trade lag."""
import pandas as pd


def forward_return_label(close: pd.DataFrame, horizon: int = 1, lag: int = 1) -> pd.DataFrame:
    """Return earned from t+lag to t+lag+horizon for a signal formed at close t.

    close: dates x assets, adjusted prices. With the defaults the label at t is
    close[t+2] / close[t+1] - 1, matching Qlib's Ref($close,-2)/Ref($close,-1)-1:
    the signal is computed at close t, traded at t+1, and earns t+1 -> t+2.
    """
    return close.shift(-(lag + horizon)) / close.shift(-lag) - 1.0
