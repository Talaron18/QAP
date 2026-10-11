import numpy as np
import pandas as pd
import pytest

from qap.evaluate import build_research_signals
from qap.fundamentals import prepare_annual
from qap.paper import (build_pseudo_panel, calibrate_secm_scale, form_sha, month_form, overlap_stats, research_signals,
                       settle_form, top_n_variant)
from qap.panel import build_monthly_panel
from qap.tilt import benchmark_weights, tilt_weights
from test_pipeline import make_world


def _secm_from_world(m):
    s = pd.DataFrame({"gvkey": m["permno"].map(lambda p: f"{p:06d}"), "iid": "01", "datadate": m["mthcaldt"], "prccm": 1.0,
                      "cshom": m["mthcap"], "trt1m": m["mthret"] * 100.0, "exchg": np.where(m["permno"] % 2 == 0, 11, 14), "tpci": "0"})
    s["gvkey"] = s["gvkey"].astype("string"); s["cshom"] = s["cshom"].astype("Float64")
    s["trt1m"] = s["trt1m"].astype("Float64"); s["exchg"] = s["exchg"].astype("Int64")
    return s


def test_compustat_path_reproduces_crsp_path_on_a_clean_world():
    m, d, funda, links = make_world(n=120, years=(1994, 1999), seed=3)
    panel = build_monthly_panel(m, d, me_scale=1000.0)
    crsp_sig = build_research_signals(panel, prepare_annual(funda), links)
    secm = _secm_from_world(m)
    # the world makes permnos 1 and 2 one company; in secm they are two gvkeys, so use the ME-consistent subset (permno >= 3)
    secm = secm[secm["gvkey"].astype(int) >= 3]
    scale, _ = calibrate_secm_scale(secm, funda[funda["gvkey"].astype(int) >= 3])
    assert scale == 1000.0
    pseudo = build_pseudo_panel(secm, scale)
    sig = research_signals(pseudo, prepare_annual(funda))
    a = crsp_sig[crsp_sig["permno"] >= 3].assign(gvkey=lambda x: x["gvkey"]).set_index(["gvkey", "mi"])["gpa"].sort_index()
    b = sig.set_index(["gvkey", "mi"])["gpa"].sort_index()
    pd.testing.assert_series_equal(a.dropna(), b.dropna(), check_names=False, rtol=1e-12)
    # tilt weights agree by gvkey for the last month (permno-level in CRSP, gvkey-level in the Compustat path)
    mi = int(sig["mi"].max())
    Wc = tilt_weights(crsp_sig[crsp_sig["permno"] >= 3], "gpa")
    Wp = tilt_weights(sig, "gpa")
    wc = Wc[Wc["mi"] == mi].assign(gvkey=lambda x: x["permno"].map(lambda p: f"{p:06d}")).set_index("gvkey")["w"]
    wp = Wp[Wp["mi"] == mi].merge(sig[["permno", "mi", "gvkey"]], on=["permno", "mi"]).set_index("gvkey")["w"]
    pd.testing.assert_series_equal(wc.sort_index(), wp.sort_index(), check_names=False, rtol=1e-12)


def test_top_n_variant_is_normalised_capped_subset():
    w = pd.DataFrame({"permno": range(200), "w": np.r_[0.03, np.full(199, 0.97 / 199)]})
    t = top_n_variant(w, n=50, cap=0.03)
    assert len(t) == 50 and t["w"].sum() == pytest.approx(1.0) and t["w"].max() <= 0.03 + 1e-12


def test_form_sha_order_independent_and_settlement_by_hand():
    form = pd.DataFrame({"kind": ["spec", "spec", "bench", "bench", "top50"], "gvkey": ["000001", "000002", "000001", "000003", "000001"],
                         "iid": "01", "w": [0.6, 0.4, 0.5, 0.5, 1.0], "mi": 24000})
    assert form_sha(form) == form_sha(form.iloc[::-1])
    ret = pd.DataFrame({"gvkey": ["000001", "000002", "000003"], "iid": "01", "mi": 24001, "ret": [0.10, np.nan, -0.02]})
    r = settle_form(form, ret)
    assert r["spec"] == pytest.approx(0.06) and r["spec_unmatched_w"] == pytest.approx(0.4)     # missing return counts 0, weight reported
    assert r["bench"] == pytest.approx(0.04) and r["active_spec"] == pytest.approx(0.02) and r["active_top50"] == pytest.approx(0.06)


def test_month_form_and_overlap():
    m, d, funda, links = make_world(n=120, years=(1994, 1999), seed=4)
    secm = _secm_from_world(m)
    pseudo = build_pseudo_panel(secm, 1000.0)
    sig = research_signals(pseudo, prepare_annual(funda))
    W, B = tilt_weights(sig, "gpa"), benchmark_weights(sig, "gpa")
    mi = int(sig["mi"].max())
    f = month_form(W, B, sig, mi)
    assert set(f["kind"]) == {"spec", "bench", "top50"}
    for k in ("spec", "bench", "top50"):
        assert f.loc[f["kind"] == k, "w"].sum() == pytest.approx(1.0)
    g = f[f["kind"] == "spec"][["gvkey", "mi", "w"]]
    assert overlap_stats(g, g, mi) == pytest.approx(1.0)
