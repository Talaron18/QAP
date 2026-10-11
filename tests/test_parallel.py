from qap.parallel import context, pmap, set_context


def _scale(x):
    return x * context()["k"]


def test_pmap_matches_serial_and_keeps_order():
    set_context(k=3)
    items = list(range(25))
    assert pmap(_scale, items, jobs=4) == pmap(_scale, items, jobs=1) == [3 * i for i in items]
