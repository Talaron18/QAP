"""Forward-return labels with an explicit one-day trade lag."""
import numpy as np
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
    r = ret
    if exit_as_zero:
        # vectorised: mask cells after each column's last valid row, for columns that exited early
        valid = r.notna().to_numpy()
        n = len(r)
        last_pos = np.where(valid.any(axis=0), n - 1 - valid[::-1].argmax(axis=0), -1)
        exited = (last_pos >= 0) & (last_pos < n - 1)
        after = np.arange(n)[:, None] > last_pos[None, :]
        r = r.mask(after & exited[None, :], 0.0)
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
