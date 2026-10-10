"""Forward-return labels with an explicit one-day trade lag."""
import pandas as pd


def label_reach(horizon: int = 1, lag: int = 1) -> int:
    """How many periods past t the label at t reaches. Pass this as `horizon` to the CV splitters."""
    return lag + horizon


def forward_return_from_returns(ret: pd.DataFrame, horizon: int = 1, lag: int = 1,
                                exit_as_zero: bool = True) -> pd.DataFrame:
    """Compounded return over periods t+lag+1 .. t+lag+horizon from a dates x assets return panel.

    Use CRSP CIZ returns (mthret/dlyret), which already include delisting payoffs; do not add
    a legacy DLRET on top. With the defaults the label at t is the return of period t+2: the
    signal is formed at close t, traded at t+1, and earns t+1 -> t+2.

    exit_as_zero: assets whose last observed return is before the panel's final date have
    exited (delisted); returns after exit count as 0 so a delisting loss stays in the label
    instead of turning it into NaN. Leading NaNs (not yet listed) stay NaN.
    """
    r = ret.copy()
    if exit_as_zero:
        last = r.apply(pd.Series.last_valid_index)
        for col in r.columns:
            if pd.notna(last[col]) and last[col] != r.index[-1]:
                tail = r.index > last[col]
                r.loc[tail, col] = 0.0
    acc = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    for k in range(lag + 1, lag + horizon + 1):
        acc = acc * (1.0 + r).shift(-k)
    return acc - 1.0


def forward_return_label(close: pd.DataFrame, horizon: int = 1, lag: int = 1) -> pd.DataFrame:
    """Price-ratio label: close[t+lag+horizon] / close[t+lag] - 1.

    Only valid for total-return-adjusted prices (dividends reinvested). Delisted names become
    NaN, which is survivorship bias; prefer `forward_return_from_returns` for research.
    Matches Qlib's Ref($close,-2)/Ref($close,-1)-1 with the defaults.
    """
    return close.shift(-(lag + horizon)) / close.shift(-lag) - 1.0
