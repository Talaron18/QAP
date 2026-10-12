"""Compustat daily price helpers (qap.prices). SYNTHETIC data only."""
import math

import numpy as np
import pandas as pd
import pytest

from qap import prices as px


def mkraw(rows, tic="AAA", gvkey="1", iid="01"):
    """rows: (date, prccd[, ajexdi[, trfd]])."""
    recs = []
    for r in rows:
        recs.append(dict(gvkey=gvkey, iid=iid, tic=tic, datadate=r[0], prccd=r[1],
                         ajexdi=r[2] if len(r) > 2 else 1.0, trfd=r[3] if len(r) > 3 else 1.0))
    return pd.DataFrame(recs)


# ------------------------------------------------------------------------------------------------ chain
def test_split_adjusts_through_ajexdi_and_dividend_through_trfd():
    # 2-for-1 split on 01-06: ajexdi 2 before, 1 after (price halves). Dividend on 01-08: price -1%, trfd +1%.
    raw = mkraw([("2025-01-02", 100.0, 2.0), ("2025-01-03", 102.0, 2.0), ("2025-01-06", 51.0, 1.0),
                 ("2025-01-07", 52.0, 1.0), ("2025-01-08", 52 / 1.01, 1.0, 1.01)])
    d, diag = px.prepare_prices(raw)
    assert d["adj_close"].tolist() == pytest.approx([50.0, 51.0, 51.0, 52.0, 52 / 1.01])
    r = d["ret"].tolist()
    assert math.isnan(r[0])
    assert r[1] == pytest.approx(0.02)
    assert r[2] == pytest.approx(0.0)                      # the split is not a return
    assert r[3] == pytest.approx(52 / 51 - 1)
    assert r[4] == pytest.approx(0.0, abs=1e-12)           # price -1% (ex-dividend) with factor +1%: flat total return
    assert d["tri"].iloc[-1] == pytest.approx(52.0)
    assert diag["AAA"]["rows"] == 5 and diag["AAA"]["gaps"] == 0
    assert px.last_price(d, "aaa") == (pytest.approx(52 / 1.01), pd.Timestamp("2025-01-08"))     # unadjusted last close


def test_missing_adjustment_columns_default_to_one():
    raw = mkraw([("2025-01-02", 10.0), ("2025-01-03", 11.0)]).drop(columns=["ajexdi", "trfd"])
    d, _ = px.prepare_prices(raw)
    assert d["adj_close"].tolist() == [10.0, 11.0] and d["ret"].iloc[1] == pytest.approx(0.1)
    raw = mkraw([("2025-01-02", 10.0, np.nan, np.nan), ("2025-01-03", 11.0, 0.0, np.nan)])
    d, _ = px.prepare_prices(raw)
    assert d["ret"].iloc[1] == pytest.approx(0.1)


def test_gap_breaks_chain_and_is_counted():
    # Fri 01-03 -> Wed 01-08 is 5 calendar days (no gap); Wed 01-08 -> Thu 01-16 is 8 days (gap)
    raw = mkraw([("2025-01-03", 100.0), ("2025-01-08", 110.0), ("2025-01-16", 121.0), ("2025-01-17", 121.0)])
    d, diag = px.prepare_prices(raw)
    assert d["ret"].iloc[1] == pytest.approx(0.1)
    assert math.isnan(d["ret"].iloc[2]) and d["ret"].iloc[3] == pytest.approx(0.0)
    assert diag["AAA"]["gaps"] == 1
    assert list(px.return_series(d, "AAA").index) == [pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-17")]


def test_dedupe_and_nonpositive_dropped_and_counted():
    raw = mkraw([("2025-01-02", 10.0), ("2025-01-03", 11.0), ("2025-01-03", 12.0), ("2025-01-06", 0.0),
                 ("2025-01-07", -1.0), ("2025-01-08", np.nan), ("2025-01-09", 13.0)])
    d, diag = px.prepare_prices(raw)
    assert d["datadate"].dt.strftime("%m-%d").tolist() == ["01-02", "01-03", "01-09"]
    assert d["prccd"].tolist() == [10.0, 12.0, 13.0]       # later duplicate kept
    assert (diag["AAA"]["dups"], diag["AAA"]["nonpositive"]) == (1, 3)
    assert px.last_price(d, "ZZZ")[1] is None and math.isnan(px.last_price(d, "ZZZ")[0])


def test_one_security_per_tic_ambiguity_raises_and_label_is_latest_tic():
    a = mkraw([("2025-01-02", 10.0), ("2025-01-03", 11.0)], tic="SNDK", gvkey="1")
    b = mkraw([("2025-01-02", 50.0), ("2025-01-03", 51.0)], tic="SNDK", gvkey="2")
    with pytest.raises(ValueError, match="several securities"):
        px.prepare_prices(pd.concat([a, b]))
    # a security that was renamed is labelled by its latest tic, so its chain stays whole
    c = pd.concat([mkraw([("2025-01-02", 10.0)], tic="OLD"), mkraw([("2025-01-03", 11.0)], tic="NEW")])
    d, _ = px.prepare_prices(c)
    assert set(d["tic"]) == {"NEW"} and d["ret"].iloc[1] == pytest.approx(0.1)


def test_load_prices_roundtrip(tmp_path):
    f = tmp_path / "p.parquet"
    mkraw([("2025-01-02", 10.0), ("2025-01-03", 11.0)]).to_parquet(f)
    d, diag = px.load_prices(f)
    assert len(d) == 2 and "AAA" in diag
    with pytest.raises(ValueError):
        px.prepare_prices(pd.DataFrame({"gvkey": ["1"]}))


def test_close_on_or_before():
    d, _ = px.prepare_prices(mkraw([("2025-01-03", 10.0), ("2025-01-06", 11.0)]))
    assert px.close_on_or_before(d, "AAA", "2025-01-05") == (10.0, pd.Timestamp("2025-01-03"))
    assert px.close_on_or_before(d, "AAA", "2025-01-06") == (11.0, pd.Timestamp("2025-01-06"))
    assert px.close_on_or_before(d, "AAA", "2025-01-01")[1] is None
    assert px.close_on_or_before(d, "AAA", "2025-02-01")[1] is None     # too old


# ------------------------------------------------------------------------------------------------ volatility
def test_ewma_hand_values():
    r = [0.01, -0.02, 0.03]
    v0 = (0.01 ** 2 + 0.02 ** 2) / 2                       # seed: second moment of the first 2
    v = v0
    for x in r:
        v = 0.94 * v + 0.06 * x * x
    assert v == pytest.approx(0.0002895076)
    assert px.ewma_vol(r, seed_n=2) == pytest.approx(math.sqrt(252 * 0.0002895076))
    assert px.ewma_vol([0.01] * 60) == pytest.approx(0.01 * math.sqrt(252))     # constant |r|: EWMA stays at r^2
    assert math.isnan(px.ewma_vol([])) and math.isnan(px.ewma_vol([np.nan]))


def test_sample_vol_hand_values():
    assert px.sample_vol([0.01, 0.02, 0.03]) == pytest.approx(0.01 * math.sqrt(252))
    assert px.sample_vol([0.01, 0.02, 0.03], n=2) == pytest.approx(0.01 / math.sqrt(2) * math.sqrt(252))
    assert px.sample_vol([0.01, np.nan, 0.03]) == pytest.approx(np.std([0.01, 0.03], ddof=1) * math.sqrt(252))
    assert math.isnan(px.sample_vol([0.01]))


@pytest.mark.parametrize("n,label", [(0, "insufficient"), (19, "insufficient"), (20, "sample"), (59, "sample"), (60, "ewma94")])
def test_best_vol_labels(n, label):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, n)
    v, k, lab = px.best_vol(np.concatenate([x, [np.nan]]))   # NaNs are not observations
    assert (k, lab) == (n, label)
    if label == "insufficient":
        assert math.isnan(v)
    elif label == "sample":
        assert v == pytest.approx(px.sample_vol(x))
    else:
        assert v == pytest.approx(px.ewma_vol(x))


# ------------------------------------------------------------------------------------------------ leveraged products
def simulate(L, n=5000, sigma=0.02, seed=3, noise=0.0):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0004, sigma, n)
    e = np.maximum(L * u + rng.normal(0, noise, n), -0.95) if noise else L * u
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(e, idx), pd.Series(u, idx)


@pytest.mark.parametrize("L", [2.0, -2.0, 3.0])
def test_leveraged_check_recovers_slope_and_drag(L):
    e, u = simulate(L)
    c = px.leveraged_check(e, u, L)
    assert c["n"] == 5000 and c["slope"] == pytest.approx(L, abs=1e-9) and c["r2"] == pytest.approx(1.0)
    assert c["te_ann"] == pytest.approx(0.0, abs=1e-9)
    expected = (L * L - L) * c["sigma_ann"] ** 2 * (5000 / 252) / 2
    assert c["drag_expected"] == pytest.approx(expected)
    assert c["drag_realised"] == pytest.approx(c["drag_expected"], rel=0.03)
    assert c["drag_realised"] > 0
    assert c["start"] == e.index[0] and c["end"] == e.index[-1]


def test_leveraged_check_tracking_error_common_dates_and_short_input():
    e, u = simulate(2.0, n=3000, noise=0.002)
    c = px.leveraged_check(e.iloc[100:], u.iloc[:-100], 2.0)          # only the overlap is used
    assert c["n"] == 2800 and c["start"] == e.index[100] and c["end"] == u.index[-101]
    assert c["te_ann"] == pytest.approx(0.002 * math.sqrt(252), rel=0.1)
    assert 1.9 < c["slope"] < 2.1 and c["r2"] > 0.99
    short = px.leveraged_check(e.iloc[:2], u.iloc[:2], 2.0)
    assert short["n"] == 2 and math.isnan(short["slope"]) and math.isnan(short["drag_realised"])


def test_a_missing_trfd_is_filled_from_neighbours_not_set_to_one_which_would_fake_a_jump():
    import numpy as np
    import pandas as pd
    from qap.prices import prepare_prices
    d = pd.DataFrame({"gvkey": ["000001"] * 4, "iid": ["01"] * 4, "tic": ["ZZ"] * 4,
                      "datadate": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
                      "prccd": [10.0, 10.0, 10.0, 10.0], "ajexdi": [1.0, 1.0, 1.0, 1.0], "trfd": [1.50, np.nan, 1.50, 1.50]})
    out, _ = prepare_prices(d)
    assert out["ret"].dropna().abs().max() < 1e-12            # flat price and a constant cumulative factor: no return anywhere
    d2 = d.assign(trfd=np.nan)
    out2, _ = prepare_prices(d2)
    assert (out2["trfd"] == 1.0).all()                         # missing for the whole security -> 1
