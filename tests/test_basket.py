import numpy as np
import pandas as pd
import pytest

from qap.basket import blended_fee, short_basket, attach_names, verdict_lines


def _sig(n=200, mi=24000, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"permno": np.arange(1, n + 1), "permco": np.arange(1, n + 1), "mi": mi,
                      "me": np.exp(r.normal(7, 1, n)), "primaryexch": np.where(np.arange(n) % 2 == 0, "N", "Q"), "s": r.normal(size=n)})
    d["me_co"] = d["me"]
    return d


def test_bottom_quintile_is_lowest_scores_vw_and_sorted():
    d = _sig()
    b = short_basket(d, "s", 24000)
    e = d[d["me_co"] >= d[d["primaryexch"] == "N"]["me_co"].quantile(0.2)]
    k = np.ceil(e["s"].rank(pct=True) * 5).clip(upper=5)
    exp = e[k == 1]
    assert set(b["permno"]) == set(exp["permno"])
    assert b["w"].sum() == pytest.approx(1.0) and b["w"].is_monotonic_decreasing
    assert b["cum_w"].iloc[-1] == pytest.approx(1.0)
    w = exp.set_index("permno")["me"]
    assert b.set_index("permno")["w"].to_dict() == pytest.approx((w / w.sum()).to_dict())


def test_attach_names_uses_active_row_and_handles_missing():
    d = short_basket(_sig(), "s", 24000)
    names = pd.DataFrame({"permno": [d["permno"].iloc[0]] * 2, "ticker": ["OLD", "NEW"], "issuernm": ["Old Inc", "New Inc"],
                          "namedt": pd.to_datetime(["2000-01-01", "2020-01-01"])})
    a = attach_names(d, names, 2021 * 12)
    assert a.loc[a["permno"] == d["permno"].iloc[0], "ticker"].iloc[0] == "NEW"
    assert attach_names(d, names, 2010 * 12).loc[lambda x: x["permno"] == d["permno"].iloc[0], "ticker"].iloc[0] == "OLD"       # name active in 2010 and (a["ticker"] == "").sum() == len(d) - 1
    assert (attach_names(d, None, 24000)["ticker"] == "").all()


def test_blended_fee_by_hand_unavailable_and_bracket():
    s = pd.DataFrame({"w": [0.4, 0.3, 0.2, 0.1], "borrow_fee_pct": [1.0, 10.0, np.nan, 5.0], "available": ["Y", "Y", "", "N"]})
    st = blended_fee(s)
    assert st["unavailable_share"] == pytest.approx(0.1)
    assert st["fee_known_avg"] == pytest.approx((0.4 * 1 + 0.3 * 10) / 0.7)          # unavailable row ignored even though a fee was typed
    assert st["fee_low"] == pytest.approx((0.4 * 1 + 0.3 * 10 + 0.2 * 0.5) / 0.9)
    assert st["fee_high"] == pytest.approx((0.4 * 1 + 0.3 * 10 + 0.2 * 25) / 0.9)
    assert st["known_share_of_tradable"] == pytest.approx(0.7 / 0.9)


def test_verdict_labels():
    ok = {"fee_known_avg": 1.0, "fee_low": 1.0, "fee_high": 1.2, "unavailable_share": 0.0, "known_share_of_tradable": 1.0}
    assert "CLEARS" in "\n".join(verdict_lines("jkp_slow10", ok))
    bad = dict(ok, fee_low=9.0, fee_high=9.0)
    assert "FAILS" in "\n".join(verdict_lines("jkp_slow10", bad))
    mid = dict(ok, fee_low=1.0, fee_high=20.0, known_share_of_tradable=0.5)
    assert "UNDETERMINED" in "\n".join(verdict_lines("jkp_slow10", mid))
    assert "CLEARS" not in "\n".join(verdict_lines("jkp_all13", dict(ok, fee_low=2.0, fee_high=2.0)))        # 3.0 > 2.5 break-even
