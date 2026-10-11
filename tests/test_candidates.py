import numpy as np
import pandas as pd
import pytest

from qap.candidates import (add_candidate_features, build_candidate_signals, merge_total_q, net_issuance,
                            promotion_stats, split_adjustment_check, total_q_checks)
from qap.fundamentals import prepare_annual
from qap.panel import build_monthly_panel

from test_pipeline import make_world


def _annual_two_years():
    base = dict(gvkey="000001", fyear=0, fyr=12, pstk=0.0, pstkrv=np.nan, pstkl=np.nan, txditc=0.0, ceq=np.nan, lt=np.nan,
                prcc_f=10.0, csho=1.0, xint=5.0, xsga=20.0)
    y1 = dict(base, datadate="2000-12-31", at=1000.0, seq=400.0, revt=500.0, cogs=300.0, ni=40.0, ib=40.0, oancf=30.0,
              dltt=300.0, act=200.0, lct=100.0, sstk=0.0)
    y2 = dict(base, datadate="2001-12-31", at=1100.0, seq=450.0, revt=600.0, cogs=340.0, ni=66.0, ib=66.0, oancf=88.0,
              dltt=280.0, act=240.0, lct=100.0, sstk=0.0)
    return prepare_annual(pd.DataFrame([y1, y2]))


def test_op_ff_and_neg_acc_hand_computed():
    a = add_candidate_features(_annual_two_years()).set_index("fyear" if False else "datadate")
    r = a.loc["2001-12-31"]
    assert r["op_ff"] == pytest.approx((600 - 340 - 20 - 5) / 450)
    assert r["neg_acc"] == pytest.approx(-(66 - 88) / 1100)


def test_fscore_all_nine_components_true():
    a = add_candidate_features(_annual_two_years()).set_index("datadate")
    # roa 6% > 0, cfo > 0, roa up (4% -> 6%), cfo/at 8% > roa 6%, leverage down, liquidity up (2.0 -> 2.4),
    # no issuance, margin up (40% -> 43.3%), turnover up (0.5 -> 0.545): 9/9
    assert a.loc["2001-12-31", "fscore"] == 9
    assert np.isnan(a.loc["2000-12-31", "fscore"])           # no prior year: >= 8 components unavailable (only no_issue + 3)


def test_fscore_missing_rule_and_sstk_treatment():
    a = _annual_two_years()
    a.loc[a.datadate == "2001-12-31", "sstk"] = 5.0           # issuance -> that component 0
    a.loc[a.datadate == "2001-12-31", "dltt"] = np.nan        # one component missing: still >= 8 available
    out = add_candidate_features(a).set_index("datadate").loc["2001-12-31"]
    assert out["fscore"] == 7                                  # 9 - 1 (leverage missing, adds nothing) - 1 (issuance)
    b = a.copy()
    b.loc[b.datadate == "2001-12-31", ["dltt", "act"]] = np.nan                # two components missing -> < 8
    assert np.isnan(add_candidate_features(b).set_index("datadate").loc["2001-12-31", "fscore"])


def test_total_q_merge_and_checks():
    ann = _annual_two_years()
    tq = pd.DataFrame({"gvkey": ["1", "1"], "datadate": ["2000-12-31", "2001-12-31"], "k_int": [300.0, 330.0]})
    a, rate = merge_total_q(ann, tq)
    assert rate == 1.0 and a["k_int"].tolist() == [300.0, 330.0]      # gvkey '1' padded to '000001'
    lines, checks = total_q_checks(a, tq, rate)
    assert checks["median k_int/at in [0.05, 2] (units)"]
    assert not checks["Total Q covers validation years (>= 2019)"]     # ends in 2001
    a2 = a.assign(k_int=a["k_int"] * 1000)
    assert not total_q_checks(a2, tq, rate)[1]["median k_int/at in [0.05, 2] (units)"]


def _panel_with_shares(split: bool, fix: bool, n_names: int = 200):
    """n_names stocks, each with a 2:1 split at month 18 (raw shares double)."""
    mi = np.arange(2000 * 12, 2000 * 12 + 30)
    frames = []
    for p in range(1, n_names + 1):
        sh = np.full(len(mi), 100.0 + p)
        fac = np.ones(len(mi))
        if split:
            sh[18:] = sh[18:] * 2.0
            fac[:18] = 2.0 if fix else 1.0                     # cumfac adjusts pre-split shares up when the convention is right
        frames.append(pd.DataFrame({"permno": p, "mi": mi, "shrout": sh, "mthcumfacshr": fac}))
    return pd.concat(frames, ignore_index=True)


def test_net_issuance_ignores_properly_adjusted_splits():
    ok = _panel_with_shares(split=True, fix=True)
    msg, passed = split_adjustment_check(ok)
    assert passed and "shrout*fac" in msg and "shrout/fac" in msg
    ni = net_issuance(ok).dropna()
    assert (ni["neg_issuance"].abs() < 1e-9).all()
    broken = _panel_with_shares(split=True, fix=False)
    assert not split_adjustment_check(broken)[1]                # raw 2:1 splits spike at ln 2 = 0.69


def test_split_check_ignores_fund_share_jumps():
    p = _panel_with_shares(split=False, fix=True, n_names=50)
    p["eq"] = True
    fund = _panel_with_shares(split=True, fix=False, n_names=50).assign(permno=lambda d: d["permno"] + 1000, eq=False)
    assert split_adjustment_check(pd.concat([p, fund], ignore_index=True))[1]    # broken series are non-equity: excluded


def test_net_issuance_sign():
    p = _panel_with_shares(split=False, fix=True, n_names=1)
    p.loc[p.mi >= 2000 * 12 + 20, "shrout"] = 110.0             # roughly 10% issuance (shares 101 -> 110)
    ni = net_issuance(p).set_index("mi")["neg_issuance"]
    assert ni.loc[2000 * 12 + 25] == pytest.approx(-np.log(110.0 / 101.0))   # issuance -> negative (bad)


def test_promotion_rule_is_relative_to_gpa():
    rng = np.random.default_rng(0)
    idx = np.arange(1990 * 12, 2020 * 12)
    g = pd.Series(rng.normal(0.002, 0.03, len(idx)), index=idx)
    clone = g * 1.0 + rng.normal(0, 0.0005, len(idx))                       # essentially gpa: no alpha
    indep = pd.Series(rng.normal(0.012, 0.03, len(idx)), index=idx)         # independent with big alpha
    flip = indep.copy()
    flip[(flip.index >= 2015 * 12) & (flip.index <= 2019 * 12 + 11)] -= 0.05    # alpha disappears in validation
    s_clone, s_indep, s_flip = promotion_stats(clone, g), promotion_stats(indep, g), promotion_stats(flip, g)
    assert not s_clone["PROMOTE"] and s_clone["corr_with_gpa_dev"] > 0.99
    assert s_indep["PROMOTE"] and s_indep["dev_alpha_t"] >= 2.0
    assert s_flip["dev_alpha_t"] >= 2.0 and not s_flip["PROMOTE"]            # validation sign failure blocks promotion
    assert s_indep["ew_gpa_plus_c_dev_pct"] > s_indep["gpa_alone_dev_pct"]


def test_build_candidate_signals_runs_on_synthetic_world():
    m, d, funda, links = make_world(n=240)
    m["shrout"], m["mthcumfacshr"] = 1000.0, 1.0
    p = build_monthly_panel(m, d)
    p["me"] = p["mthcap"] / 1000.0
    p["me_co"] = p.groupby(["permco", "mi"])["me"].transform("sum")
    ann = prepare_annual(funda)
    for c in ("ib", "oancf", "dltt", "act", "lct", "sstk", "ni", "xsga", "xint"):
        ann[c] = 10.0
    ann = add_candidate_features(ann)
    tq = pd.DataFrame({"gvkey": ann["gvkey"], "datadate": ann["datadate"], "k_int": 50.0})
    ann, _ = merge_total_q(ann, tq)
    sig = build_candidate_signals(p, ann, links)
    for c in ["gpa", "bm_timely", "bm_int", "op_ff", "neg_acc", "fscore", "neg_issuance", "quality3"]:
        assert c in sig and sig[c].notna().any(), c
    assert (sig["bm_int"].dropna() > sig["bm_timely"].dropna().reindex(sig["bm_int"].dropna().index)).all()


def test_promotion_handles_empty_series():
    idx = np.arange(1990 * 12, 1990 * 12 + 5)
    out = promotion_stats(pd.Series(dtype=float), pd.Series(np.zeros(5), index=idx))
    assert out["PROMOTE"] is False and "note" in out
