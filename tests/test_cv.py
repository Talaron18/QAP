import numpy as np
import pytest
from qap.cv import purged_kfold, walk_forward


@pytest.mark.parametrize("h,emb", [(1, 0), (5, 0), (5, 3), (20, 10)])
def test_no_train_label_overlaps_test_or_embargo(h, emb):
    n = 300
    for train, test in purged_kfold(n, 5, h, emb):
        assert len(np.intersect1d(train, test)) == 0
        start, end = test[0], test[-1]
        # label window [i, i+h] must not touch the test fold
        before = train[train < start]
        assert (before + h < start).all()
        after = train[train > end]
        assert (after > end + emb).all()


def test_folds_cover_all_observations_once():
    tests = np.concatenate([t for _, t in purged_kfold(100, 4, 3, 2)])
    assert sorted(tests) == list(range(100))


def test_walk_forward_trains_only_on_past():
    for train, test in walk_forward(200, 5, horizon=5, embargo=2):
        assert train.max() < test.min()
        assert train.max() + 5 < test.min()


def test_bad_splits():
    with pytest.raises(ValueError):
        list(purged_kfold(10, 1, 1))
