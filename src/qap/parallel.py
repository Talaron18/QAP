"""Process-parallel map over independent jobs (one signal / one month block each), with serial fallback.

Workers are forked, so large read-only inputs (panel, liquidity inputs) are shared copy-on-write instead of pickled; set them
with `set_context` BEFORE calling `pmap`. On platforms without fork (Windows), or with jobs <= 1, it runs serially and says so.
BLAS/OpenMP threads are pinned to 1 per worker. Results come back in input order, so callers can log deterministically.
"""
import multiprocessing as mp
import os

_CTX: dict = {}


def set_context(**kw) -> None:
    _CTX.clear()
    _CTX.update(kw)


def context() -> dict:
    return _CTX


def _init_worker() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def _call(args):
    fn, item = args
    return fn(item)


def pmap(fn, items, jobs: int = 1, say=print) -> list:
    """[fn(x) for x in items], in order. fn must be a module-level function (picklable); read big inputs via `context()`."""
    items = list(items)
    jobs = max(1, min(int(jobs), len(items)))
    if jobs == 1:
        return [fn(x) for x in items]
    if "fork" not in mp.get_all_start_methods():
        say("[parallel] fork is unavailable on this platform; running serially")
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(jobs, initializer=_init_worker) as pool:
        return pool.map(_call, [(fn, x) for x in items], chunksize=1)
