import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

import qap.holdout_ls as H
from qap.basket import assumed_fee
from qap.candidates import VAL
from qap.diagnostics import diagnose_signal
from qap.holdout_ls import (EXPECTED, assert_unopened, borrow_series, decide, ls_series, mark_opened, nw_t, rehearsal_gate, split_mean,
                            summarize)
from qap.study import SEALED_CUT

from test_costs import _agg_for
from test_pipeline import make_world

F = 100


def _sig(me_override=None, months=(F, F + 1)):
    """16 names, permno 1 an NYSE microcap, 2..6 NYSE; me = 10 p (me_override replaces); signal = permno => bottom quintile = 2, 3, 4."""
    p = np.arange(1, 17)
    me = 10.0 * p
    me[0] = 1.0
    for k, v in (me_override or {}).items():
        me[k - 1] = v
    s = pd.DataFrame({"permno": p, "mi": F, "me": me, "me_co": me, "primaryexch": np.where(p <= 6, "N", "Q"), "sig": p.astype(float)})
    return pd.concat([s.assign(mi=m) for m in months], ignore_index=True)


def _panel_inp(months=(F, F + 1, F + 2)):
    p = np.arange(1, 17)
    panel = pd.concat([pd.DataFrame({"permno": p, "mi": m, "mthret_adj": 0.01 * p * (m - F + 1)}) for m in months], ignore_index=True)
    inp = pd.concat([pd.DataFrame({"permno": p, "mi": m, "spread": 0.002, "sigma_d": 0.02, "adv": 1e7}) for m in months], ignore_index=True)
    return panel, inp


# ---------- borrow and net ----------
def test_borrow_exact_by_hand_and_net_identity():
    sig = _sig({4: 600.0})          # bottom quintile: permnos 2, 3, 4 with me 20, 30, 600 (tiers 0, 0, 1)
    w = np.array([20.0, 30.0, 600.0]) / 650.0
    fee = np.array([3 * 0.88 + 15 * 0.12, 3 * 0.88 + 15 * 0.12, 1.5 * 0.95 + 15 * 0.05])
    assert fee == pytest.approx(assumed_fee([20, 30, 600], "base"))
    b = borrow_series(sig, "sig", "base", 1.0)
    assert list(b.index) == [F + 1, F + 2] and b.iloc[0] == pytest.approx(((w * fee).sum() / 100 + 0.01) / 12)
    assert b.iloc[0] > 0 and borrow_series(sig, "sig", "optimistic", 0.0).iloc[0] < b.iloc[0] < borrow_series(sig, "sig", "pessimistic", 1.0).iloc[0]
    assert borrow_series(sig, "sig", "base", 0.0).iloc[0] == pytest.approx((w * fee).sum() / 100 / 12)
    panel, inp = _panel_inp()
    s = ls_series(sig, panel, inp, 1e6, "sig")
    assert list(s.index) == [F + 2]                                                    # first formation month builds, F+1 trades
    r = s.iloc[0]
    me4 = np.array([140.0, 150.0, 160.0])
    gL = (me4 * 0.01 * np.array([14, 15, 16]) * 3).sum() / me4.sum()
    gS = (np.array([20.0, 30.0, 600.0]) * 0.01 * np.array([2, 3, 4]) * 3).sum() / 650.0
    assert r["gross_LS"] == pytest.approx(gL - gS) and r["cost_L"] > 0 and r["cost_S"] > 0
    assert r["borrow"] == pytest.approx(b.loc[F + 2])
    assert r["net_LS_preborrow"] == pytest.approx(r["gross_LS"] - r["cost_L"] - r["cost_S"])
    assert r["net_LS"] == pytest.approx(r["net_LS_preborrow"] - r["borrow"])
    assert r["short_UmS"] == pytest.approx(r["gross_LS"] - r["long_LU"] - r["cost_L"] - r["cost_S"] - r["borrow"])      # = gross U-S - cost_S - borrow
    assert r["turnover_L"] >= 0 and r["turnover_S"] >= 0


def test_nullable_dtypes_give_identical_series():
    sig = _sig({4: 600.0})
    panel, inp = _panel_inp()
    nul = lambda d: d.astype({c: ("Int64" if d[c].dtype.kind in "iu" else "Float64") for c in d if d[c].dtype.kind in "iuf"})
    s2 = nul(sig).assign(primaryexch=sig["primaryexch"].astype("string"))
    pd.testing.assert_frame_equal(ls_series(s2, nul(panel), nul(inp), 1e6, "sig"), ls_series(sig, panel, inp, 1e6, "sig"))


# ---------- statistics ----------
def _nw_direct(x, L):
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    g = lambda k: sum(e[i] * e[i - k] for i in range(k, n)) / n
    S = g(0) + 2 * sum((1 - k / (L + 1)) * g(k) for k in range(1, L + 1))
    return x.mean() / np.sqrt(S / n)


def test_nw_t_matches_direct_computation():
    rng = np.random.default_rng(2)
    e = rng.normal(size=60)
    x = 0.01 + 0.02 * (e + 0.5 * np.r_[0, e[:-1]])
    assert nw_t(x, 6) == pytest.approx(_nw_direct(x, 6)) and nw_t(x, 0) == pytest.approx(_nw_direct(x, 0))
    assert nw_t(pd.Series(x), 3) == pytest.approx(_nw_direct(x, 3)) and np.isnan(nw_t([1.0, 2.0]))
    assert abs(nw_t(x, 6)) != abs(nw_t(x, 0))


def test_summarize_hand_checked():
    net = pd.Series([0.10, -0.05, 0.02, 0.03], index=[2020 * 12 + 10, 2020 * 12 + 11, 2021 * 12, 2021 * 12 + 1])
    s = summarize(net)
    assert s["n"] == 4 and s["mean_pct"] == pytest.approx(2.5) and s["t"] == pytest.approx(0.025 / (net.std(ddof=1) / 2))
    assert s["cum_return"] == pytest.approx(1.1 * 0.95 * 1.02 * 1.03 - 1)
    assert s["by_year"][2020] == pytest.approx(1.1 * 0.95 - 1) and s["by_year"][2021] == pytest.approx(1.02 * 1.03 - 1)
    assert s["max_drawdown"] == pytest.approx(0.95 - 1) and s["half1_pct"] == pytest.approx(2.5) and s["half2_pct"] == pytest.approx(2.5)
    assert summarize(pd.Series(dtype=float))["n"] == 0
    rng = np.random.default_rng(1)
    idx = np.arange(2020 * 12, 2020 * 12 + 60)
    ff = pd.DataFrame({"mktrf": rng.normal(0, .04, 60), "smb": rng.normal(0, .02, 60), "hml": rng.normal(0, .02, 60),
                       "umd": rng.normal(0, .03, 60), "rf": 0.001}, index=idx)
    y = pd.Series(0.004 + 0.5 * ff["mktrf"] + rng.normal(0, 1e-5, 60), index=idx)
    f = summarize(y, ff)
    assert f["capm"]["alpha_pct"] == pytest.approx(0.4, abs=0.01) and f["capm"]["b_mktrf"] == pytest.approx(0.5, abs=0.01)
    assert f["ff4"]["b_smb"] == pytest.approx(0, abs=0.01) and "alpha_t" in f["ff4"]


@pytest.mark.parametrize("mean,t,label", [(0.20, 1.0, "SUPPORTS FURTHER WORK"), (0.20, 0.99, "INCONCLUSIVE"), (0.199, 5.0, "INCONCLUSIVE"),
                                          (0.0, 3.0, "REFUTED"), (-0.1, -2.0, "REFUTED"), (0.5, 2.5, "SUPPORTS FURTHER WORK"),
                                          (0.01, 0.1, "INCONCLUSIVE"), (float("nan"), 1.0, "INCONCLUSIVE")])
def test_decide_boundaries(mean, t, label):
    assert decide(mean, t) == label


# ---------- gate and marker ----------
def _gate_series(dev, val):
    mk = lambda y0, y1, g, n: pd.DataFrame({"gross_LS": g / 100, "net_LS_preborrow": n / 100}, index=np.arange(y0 * 12, y1 * 12 + 12))
    return pd.concat([mk(1990, 2014, *dev), mk(2015, 2019, *val)])


def test_rehearsal_gate_pass_and_fail():
    ok, msgs = rehearsal_gate(_gate_series((0.899, 0.545), (0.450, 0.386)), EXPECTED)
    assert ok and len(msgs) == 4 and all("ok" in m for m in msgs)
    ok, msgs = rehearsal_gate(_gate_series((0.899, 0.545), (0.450, 0.40)), EXPECTED)
    assert not ok and sum("MISMATCH" in m for m in msgs) == 1 and "val net_LS_preborrow" in [m for m in msgs if "MISMATCH" in m][0]
    assert rehearsal_gate(_gate_series((0.905, 0.545), (0.450, 0.386)), EXPECTED, tol=0.01)[0]
    assert not rehearsal_gate(_gate_series((0.915, 0.545), (0.450, 0.386)), EXPECTED, tol=0.01)[0]
    assert not rehearsal_gate(_gate_series((0.899, 0.545), (0.450, 0.386)).iloc[:12], EXPECTED)[0]     # missing windows fail


def test_marker_refusal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert_unopened()
    mark_opened("v6 opened now")
    assert pathlib.Path(H.MARKER).exists()
    with pytest.raises(SystemExit) as e:
        assert_unopened()
    assert "ALREADY RUN" in str(e.value) and "v6 opened now" in str(e.value)


# ---------- end to end ----------
def _world(poison=False):
    m, d, funda, links = make_world(n=100, years=(2018, 2025))
    from qap.costs import liquidity_inputs
    from qap.panel import build_monthly_panel
    from qap.portfolio import market_scaled_aum
    p = build_monthly_panel(m, d)
    if poison:
        p.loc[p["mi"] >= SEALED_CUT, "mthret_adj"] = 0.5 * p.loc[p["mi"] >= SEALED_CUT, "permno"]
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    sig = p.loc[p["eq"] & (p["me_co"] > 0), ["permno", "permco", "mi", "me", "me_co", "primaryexch"]].reset_index(drop=True)
    sig["jkp_slow10"] = np.random.default_rng(4).normal(size=len(sig))
    return p, sig, liquidity_inputs(p, _agg_for(p)), market_scaled_aum(p, 1e7)


def test_end_to_end_matches_diagnostic_net_ls():
    p, sig, inp, aum = _world()
    s = ls_series(sig, p, inp, aum, "jkp_slow10")
    assert s.index.max() >= SEALED_CUT + 12 and (s["borrow"] > 0).all() and s["net_LS"].notna().all()
    r = diagnose_signal(sig, p, inp, aum, "jkp_slow10")
    pre = s[s.index < SEALED_CUT]
    for k in ("net_LS", "gross_LS", "cost_S"):
        ref = r["costs"][k]["val"][0] if k != "net_LS" else r["costs"]["net_LS"]["val"][0]
        mine = {"net_LS": split_mean(pre["net_LS_preborrow"], VAL), "gross_LS": split_mean(pre["gross_LS"], VAL), "cost_S": split_mean(pre["cost_S"], VAL)}[k]
        assert mine == pytest.approx(ref, abs=1e-10)
    assert (pre["cost_L"] + pre["cost_S"] > 0).all()
    assert np.allclose(s["net_LS"], s["net_LS_preborrow"] - s["borrow"], atol=1e-15, rtol=0)
    nul = lambda d: d.astype({c: ("Int64" if d[c].dtype.kind in "iu" else "Float64") for c in d if d[c].dtype.kind in "iuf"})
    s2 = ls_series(nul(sig), nul(p[["permno", "mi", "mthret_adj"]]), nul(inp), aum, "jkp_slow10")
    pd.testing.assert_frame_equal(s2, s)


def test_series_before_cut_ignores_poisoned_2020plus_returns():
    p, sig, inp, aum = _world()
    pp, _, inp2, aum2 = _world(poison=True)
    a, b = ls_series(sig, p, inp, aum, "jkp_slow10"), ls_series(sig, pp, inp2, aum2, "jkp_slow10")
    # gross return of month f+1 uses returns of f+1; only the formation at 2019-12 -> realised 2020-01 may touch 2020 data
    assert b[b.index >= SEALED_CUT]["gross_LS"].abs().max() > 1
    pa, pb = a[a.index < SEALED_CUT], b[b.index < SEALED_CUT]
    pd.testing.assert_frame_equal(pa, pb, atol=1e-10, rtol=0)
    assert summarize(pa["net_LS"]) == summarize(pb["net_LS"])


def _load_script():
    spec = importlib.util.spec_from_file_location("run_holdout_ls", pathlib.Path(__file__).parent.parent / "scripts" / "run_holdout_ls.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch(mod, monkeypatch, poison, gate_ok=True):
    p, sig, inp, aum = _world(poison)
    monkeypatch.setattr(mod, "require_inputs", lambda *a, **k: None)
    monkeypatch.setattr(mod, "load_context", lambda: (p, None, None, inp, aum))
    monkeypatch.setattr(mod, "eligible_universe", lambda panel: sig)
    monkeypatch.setattr(mod, "build_theme_signals", lambda el, d: (el, None))
    monkeypatch.setattr(mod, "rehearsal_gate", lambda s, e: (gate_ok, ["  [stub] gate"]))


def test_script_rehearsal_never_exposes_2020plus(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    outs = []
    for poison in (False, True):
        _patch(mod, monkeypatch, poison)
        lines = []
        mod._run(lines.append, True)
        outs.append("\n".join(lines))
    assert outs[0] == outs[1] and "SEALED WINDOW" not in outs[0] and "DECISION" not in outs[0]
    assert not pathlib.Path(H.MARKER).exists() and not pathlib.Path("data/trials.jsonl").exists()


def test_script_gate_failure_aborts_before_marker(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, False, gate_ok=False)
    lines = []
    with pytest.raises(SystemExit) as e:
        mod._run(lines.append, False)
    assert "GATE FAILED" in str(e.value) and "NOT opened" in str(e.value)
    assert not pathlib.Path(H.MARKER).exists() and not any("SEALED WINDOW" in l for l in lines)


def test_script_one_shot_marker_trial_and_refusal(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.chdir(tmp_path)
    _patch(mod, monkeypatch, False)
    lines = []
    mod._run(lines.append, False)
    txt = "\n".join(lines)
    assert pathlib.Path(H.MARKER).exists() and "72 realised months" in txt and "DECISION" in txt and "one shot used" in txt
    assert any(x in txt for x in ("REFUTED", "INCONCLUSIVE", "SUPPORTS FURTHER WORK")) and "sensitivity (optimistic" in txt
    assert len(pathlib.Path("data/trials.jsonl").read_text().strip().splitlines()) == 1
    with pytest.raises(SystemExit) as e:
        mod._run(lambda m="": None, False)
    assert "ALREADY RUN" in str(e.value)
    assert len(pathlib.Path("data/trials.jsonl").read_text().strip().splitlines()) == 1
