import numpy as np
import pandas as pd
import pytest

from qap.panel import build_monthly_panel, to_mi
from qap.tilt import (active_stats, benchmark_weights, cap_weights, eligible, factor_regression, holding_stats,
                      tilt_weights)

from test_pipeline import make_world


def test_cap_weights_water_filling():
    me = np.array([100.0, 50.0, 10.0, 10.0, 10.0, 10.0, 5.0, 5.0])
    w = cap_weights(me, 0.30)
    assert w.sum() == pytest.approx(1.0) and w.max() <= 0.30 + 1e-9
    assert w[0] == pytest.approx(0.30) and w[1] < 0.30 + 1e-9
    assert cap_weights(me, None).sum() == pytest.approx(1.0)
    # redistribution is pro rata among uncapped names: ratios preserved
    unc = [3, 4, 5]
    assert w[unc[0]] / w[unc[1]] == pytest.approx(1.0)
    assert w[6] / w[3] == pytest.approx(0.5)


def test_cap_infeasible_falls_back_to_equal_weight():
    w = cap_weights(np.array([10.0, 20.0, 30.0]), 0.10)           # 3 names cannot respect a 10% cap
    assert w == pytest.approx(np.full(3, 1 / 3))


def _sig(n=100, months=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(months):
        mi = 2000 * 12 + k
        for p in range(1, n + 1):
            rows.append(dict(permno=p, mi=mi, me=100.0 + p, me_co=1e4 + p, primaryexch="N", gpa=float(p), siccd=1000 + (p % 3) * 100))
    return pd.DataFrame(rows)


def test_entry_and_buffer_logic_by_hand():
    sig = _sig(n=100, months=2)           # the 20th-percentile NYSE ME cut leaves permnos 21..100 (80 names); gpa = permno
    m2 = sig["mi"] == 2000 * 12 + 1
    # month 2 edits: 90 (held) slides to the buffer zone, 95 (held) collapses, 70 (not held) jumps to the top, 83 (not held) to the buffer zone
    for p, v in {90: 80.5, 95: 40.5, 70: 99.5, 83: 79.5}.items():
        sig.loc[m2 & (sig.permno == p), "gpa"] = v
    pct = eligible(sig, "gpa").set_index(["mi", "permno"])["pct"]
    pct2 = lambda p: pct[(2000 * 12 + 1, p)]
    assert 0.70 < pct2(90) < 0.80 and 0.70 < pct2(83) < 0.80 and pct2(95) < 0.70 and pct2(70) > 0.80   # preconditions
    w0 = tilt_weights(sig, entry=0.8, exit_=0.8, cap=None)
    wb = tilt_weights(sig, entry=0.8, exit_=0.7, cap=None)
    held = lambda w, k: set(w[w.mi == 2000 * 12 + k].permno)
    assert 90 in held(wb, 0) and 95 in held(wb, 0)                           # both held in month 1 (rank > 0.8)
    assert 90 not in held(w0, 1) and 90 in held(wb, 1)                       # buffer keeps the slider
    assert 95 not in held(wb, 1)                                             # below exit -> sold
    assert 83 not in held(wb, 1) and 83 not in held(w0, 1)                   # buffer zone but not previously held -> not bought
    assert 70 in held(wb, 1) and 70 in held(w0, 1)                           # new top-ranked name enters
    assert wb.groupby("mi")["w"].sum().round(9).eq(1.0).all()


def test_buffer_reduces_turnover_on_noisy_signal():
    from qap.portfolio import leg_trades
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    rows, rets = [], []
    for k in range(40):
        mi = 2000 * 12 + k
        g = base + rng.normal(0, 0.6, 200)                                   # persistent signal + noise
        for p in range(200):
            rows.append(dict(permno=p, mi=mi, me=100.0 + p, me_co=1e4 + p, primaryexch="N", gpa=g[p]))
            rets.append(dict(permno=p, mi=mi, mthret_adj=0.0))
    sig, panel = pd.DataFrame(rows), pd.DataFrame(rets)
    to = lambda W: leg_trades(W, panel).groupby("mi")["dw"].sum().mean() / 2
    assert to(tilt_weights(sig, entry=0.8, exit_=0.7, cap=None)) < to(tilt_weights(sig, entry=0.8, exit_=0.8, cap=None))


def test_holding_stats_and_active_share():
    sig = _sig(n=100, months=1)
    W = tilt_weights(sig, entry=0.8, exit_=0.8, cap=0.10)
    B = benchmark_weights(sig)
    hs = holding_stats(W, B, sig[["permno", "mi", "siccd"]])
    r = hs.iloc[0]
    assert r["holdings"] == int((eligible(sig, "gpa")["pct"] > 0.8).sum()) and r["max_w"] <= 0.10 + 1e-9 and r["top10"] <= 1.0
    assert 0.5 < r["active_share"] < 1.0                                    # top quintile vs whole universe
    assert 0 < r["max_abs_active_sic2"] < 1.0


def test_active_stats_and_ff_regression_recover_known_loadings():
    rng = np.random.default_rng(2)
    idx = np.arange(1990 * 12, 2020 * 12)
    ff = pd.DataFrame(rng.normal(0, 0.04, (len(idx), 4)), index=idx, columns=["mktrf", "smb", "hml", "umd"])
    act = 0.002 + 0.3 * ff["mktrf"] - 0.2 * ff["hml"] + rng.normal(0, 0.005, len(idx))
    reg = factor_regression(act, ff)
    assert reg["alpha_pct_mo"] == pytest.approx(0.2, abs=0.08) and reg["alpha_t"] > 4
    assert reg["mktrf"].startswith("+0.3") and reg["hml"].startswith("-0.2")
    st = active_stats(pd.Series(act.to_numpy(), index=idx))
    assert set(st) == {"pre1990", "dev1990_2014", "valid2015_2019", "pre2020", "post1990"}
    assert st["dev1990_2014"]["IR_ann"] > 0 and st["dev1990_2014"]["TE_ann_pct"] > 0
    assert "holdout2020+" in active_stats(pd.Series(act.to_numpy(), index=idx), include_holdout=True)


def test_tilt_accepts_pandas_nullable_dtypes_like_wrds_parquet():
    sig = _sig(n=100, months=3)
    sig["permno"] = sig["permno"].astype("Int64")
    for c in ("me", "me_co", "gpa"):
        sig[c] = sig[c].astype("Float64")
    ref = tilt_weights(_sig(n=100, months=3), entry=0.8, exit_=0.7, cap=0.03)
    out = tilt_weights(sig, entry=0.8, exit_=0.7, cap=0.03)
    assert len(out) == len(ref) and out["w"].to_numpy(dtype="float64") == pytest.approx(ref["w"].to_numpy())
    holding_stats(out, benchmark_weights(sig))
