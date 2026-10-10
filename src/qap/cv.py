"""Purged and embargoed walk-forward splits (Lopez de Prado 2018)."""
from typing import Iterator, Tuple

import numpy as np


def purged_kfold(n: int, n_splits: int, horizon: int, embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) over n time-ordered observations.

    The label of observation i spans [i, i + horizon]. `horizon` must be the label's full
    reach, i.e. `qap.labels.label_reach(horizon, lag)` (lag + horizon), not the bare horizon:
    the default 1-period label with a 1-day lag reaches t+2. Training observations whose
    label window overlaps the test fold are purged, and `embargo` observations
    after the test fold are dropped to limit leakage from serial correlation.
    """
    if n_splits < 2 or n_splits > n:
        raise ValueError("n_splits must be in [2, n]")
    idx = np.arange(n)
    for fold in np.array_split(idx, n_splits):
        start, end = fold[0], fold[-1]
        # train obs i overlaps the test fold if i + horizon >= start (and i <= end)
        before = idx[idx < start - horizon]
        after = idx[idx > end + embargo]
        yield np.concatenate([before, after]), fold


def walk_forward(n: int, n_splits: int, horizon: int, embargo: int = 0, min_train: int = 1) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward: train only on data strictly before each test fold."""
    for train, test in purged_kfold(n, n_splits, horizon, embargo):
        train = train[train < test[0]]
        if len(train) >= min_train:
            yield train, test
