import json
import re

import numpy as np
import pandas as pd
import pytest

import qap.mlcheck as ml
from qap.paper import build_pseudo_panel
from qap.panel import to_mi


# ------------------------------------------------------------------------------------------------------------- synthetic worlds
def make_secm_world(n=400, start="2000-01", end="2014-12", seed=0, k=0.0, nullable=True, first_fy=None):
    """Vectorised comp.secm-like + comp.funda-like world. Returns planted by the PIT gross-profit/assets z-score with slope k (monthly)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{start}-01", f"{end}-01", freq="MS") + pd.offsets.MonthEnd(0)
    nt = len(dates)
    yrs, mons = dates.year.to_numpy(), dates.month.to_numpy()
    fy_t = np.where(mons >= 4, yrs - 1, yrs - 2)                        # latest fiscal year known at month-end t (4-month lag)
    fy0 = (first_fy or int(yrs.min()) - 2)
    fys = np.arange(fy0, int(yrs.max()) + 1)
    firm = rng.normal(0.30, 0.08, n)
    gpa = firm[:, None] + rng.normal(0.0, 0.03, (n, len(fys)))           # persistent firm level + noise
    z = (gpa - gpa.mean(axis=0)) / gpa.std(axis=0)
    sig_t = np.stack([z[:, np.clip(f - fy0, 0, len(fys) - 1)] for f in fy_t], axis=1)    # (n, nt) signal known at t
    eps = rng.normal(0.0, 0.08, (n, nt))
    ret = 0.008 + eps
    ret[:, 1:] += k * sig_t[:, :-1]
    shares = np.exp(rng.normal(9.0, 1.0, n))                              # thousands
    price = np.exp(rng.normal(2.5, 0.8, n))[:, None] * np.cumprod(1 + ret, axis=1)
    me = price * shares[:, None]
    nyse = np.arange(n) % 2 == 0
    secm = pd.DataFrame({
        "gvkey": np.repeat([f"{i + 1:06d}" for i in range(n)], nt), "iid": "01", "datadate": np.tile(dates, n),
        "prccm": price.ravel(), "cshom": np.repeat(shares, nt), "trt1m": (ret * 100.0).ravel(),
        "exchg": np.repeat(np.where(nyse, 11, 14), nt), "tpci": "0"})
    rows = []
    for j, fy in enumerate(fys):
        dd = pd.Timestamp(f"{fy}-12-31")
        t = np.where(dates == dd)[0]
        cap = (me[:, t[0]] / 1000.0) if len(t) else (me[:, 0] / 1000.0)    # $ millions at the fiscal year end (or first month)
        at = cap * rng.uniform(1.0, 3.0, n)
        gp = gpa[:, j] * at
        dp = 0.03 * at
        ib = 0.05 * at + rng.normal(0, 0.01, n) * at
        oancf = np.where(rng.random(n) < 0.8, ib + dp - 0.01 * at, np.nan)
        rows.append(pd.DataFrame({
            "gvkey": [f"{i + 1:06d}" for i in range(n)], "datadate": dd, "at": at, "seq": 0.5 * at, "ceq": np.nan, "pstk": np.nan, "pstkrv": np.nan,
            "pstkl": np.nan, "txditc": 0.0, "lt": 0.5 * at, "sale": gp + 0.5 * at, "revt": gp + 0.5 * at, "cogs": 0.5 * at, "xsga": 0.1 * at,
            "xrd": np.where(rng.random(n) < 0.3, 0.02 * at, np.nan), "ib": ib, "oancf": oancf, "dp": dp, "capx": 0.05 * at, "dltt": 0.2 * at,
            "dlc": 0.05 * at, "che": 0.1 * at, "act": 0.3 * at, "lct": 0.15 * at, "csho": cap / (price[:, t[0]] if len(t) else price[:, 0]) * 1000.0 / 1000.0,
            "prcc_f": (price[:, t[0]] if len(t) else price[:, 0])}))
    funda = pd.concat(rows, ignore_index=True)
    if nullable:
        for c in ("prccm", "cshom", "trt1m"):
            secm[c] = secm[c].astype("Float64")
        secm["exchg"] = secm["exchg"].astype("Int64")
        secm["gvkey"] = secm["gvkey"].astype("string")
        for c in funda.columns:
            if c not in ("gvkey", "datadate"):
                funda[c] = funda[c].astype("Float64")
    return secm, funda


def features_of(secm, funda, scale=1000.0):
    pseudo = ml.with_shares(build_pseudo_panel(secm, scale), secm)
    return ml.build_features(pseudo, ml.prepare_ml_annual(funda))


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """HGB's OpenMP pool must not outlive this module: tests/test_parallel.py forks workers later in the same process."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def small_override():
    ml._TEST_MODEL_OVERRIDES.update(max_iter=40)
    yield
    ml._TEST_MODEL_OVERRIDES.clear()


@pytest.fixture(scope="module")
def planted(small_override):
    secm, funda = make_secm_world(n=400, seed=11, k=0.012)
    return features_of(secm, funda)


@pytest.fixture(scope="module")
def null_world(small_override):
    secm, funda = make_secm_world(n=400, seed=12, k=0.0)
    return features_of(secm, funda)


# ------------------------------------------------------------------------------------------------------------------- tiny panels
def tiny_pseudo(returns: dict, shares: dict = None, start="1998-01", exch="N", me_const=None):
    """pseudo panel for a few stocks from explicit monthly returns {gvkey: array}; price compounds returns, shares given or constant 100."""
    rows = []
    for g, r in returns.items():
        r = np.asarray(r, dtype="float64")
        dates = pd.date_range(f"{start}-01", periods=len(r), freq="MS") + pd.offsets.MonthEnd(0)
        sh = np.full(len(r), 100.0) if shares is None or g not in shares else np.asarray(shares[g], dtype="float64")
        px = 10.0 * np.cumprod(1 + r)
        # a split-style share jump keeps ME continuous: price is divided by the share ratio
        if shares is not None and g in shares:
            px = px / (sh / sh[0])
        me = px * sh if me_const is None else np.full(len(r), me_const)
        for i, d in enumerate(dates):
            rows.append(dict(gvkey=g, iid="01", datadate=d, prccm=px[i], cshom=sh[i], trt1m=r[i] * 100.0, exchg=11 if exch == "N" else 14, me=me[i]))
    d = pd.DataFrame(rows)
    secm = d[["gvkey", "iid", "datadate", "prccm", "cshom", "trt1m", "exchg"]].copy()
    pseudo = ml.with_shares(build_pseudo_panel(secm, 1.0), secm)
    if me_const is not None:
        pseudo["me"] = d["me"].to_numpy()
        pseudo["me_co"] = d["me"].to_numpy()
    return pseudo


def tiny_annual(gvkeys, rows):
    """rows: list of dicts per fiscal year (datadate + items), applied to every gvkey."""
    f = pd.DataFrame([dict(gvkey=g, **r) for g in gvkeys for r in rows])
    for c in ml._ANNUAL_RAW:
        if c not in f.columns:
            f[c] = np.nan
    f["datadate"] = pd.to_datetime(f["datadate"])
    return ml.prepare_ml_annual(f)


def at_month(df, g, ymstr):
    mi = ml.parse_ym(ymstr)
    r = df[(df["gvkey"] == g) & (df["mi"] == mi)]
    assert len(r) == 1, (g, ymstr, len(r))
    return r.iloc[0]


# ------------------------------------------------------------------------------------------------------------------ feature formulas
def test_feature_list_and_signs():
    assert len(ml.FEATURES) == 20 and len(set(ml.FEATURES)) == 20
    assert list(ml.FEATURE_SIGNS) == ml.FEATURES and set(ml.FEATURE_SIGNS.values()) == {-1, 1}
    assert ml.FEATURE_SIGNS["logme"] == -1 and ml.FEATURE_SIGNS["acc"] == -1 and ml.FEATURE_SIGNS["rev"] == 1


def test_beta_against_constructed_market_and_calendar_windows():
    rng = np.random.default_rng(1)
    n = 40
    rb, rc = rng.normal(0.01, 0.05, n), rng.normal(0.0, 0.05, n)
    m = rb + rc                                        # equal ME -> market = (rA + rB + rC)/3 with rA = 2 m  => m = rB + rC
    ra = 2.0 * m
    # market of month t uses ME(t-1); ME is held equal by me_const so the construction is exact
    p = tiny_pseudo({"000001": ra, "000002": rb, "000003": rc}, me_const=1000.0)
    f = ml.build_features(p, tiny_annual(["000001", "000002", "000003"], [dict(datadate="1997-12-31", at=100.0, seq=50.0)]))
    last = ml.parse_ym("1998-01") + n - 1
    a = f[(f["gvkey"] == "000001") & (f["mi"] == last)].iloc[0]
    assert a["beta"] == pytest.approx(2.0, abs=1e-9)
    # hand OLS for B over the 36 months ending at `last`, market from month 1 (month 0 has no market)
    mk = ((ra + rb + rc) / 3.0)
    lo = n - 36
    x, y = mk[lo:], rb[lo:]
    exp_b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    b = f[(f["gvkey"] == "000002") & (f["mi"] == last)].iloc[0]
    assert b["beta"] == pytest.approx(exp_b, rel=1e-9)
    # fewer than 24 pairs -> NaN: month index 20 has 20 months of history, 19 with a market return
    early = f[(f["gvkey"] == "000002") & (f["mi"] == ml.parse_ym("1998-01") + 20)].iloc[0]
    assert np.isnan(early["beta"])


def test_momentum_skips_formation_month_reversal_vol_and_gaps():
    rng = np.random.default_rng(2)
    n = 30
    r = {g: rng.normal(0.01, 0.06, n) for g in ("000001", "000002", "000003")}
    p = tiny_pseudo(r, me_const=1000.0)
    ann = tiny_annual(list(r), [dict(datadate="1997-12-31", at=100.0, seq=50.0)])
    f = ml.build_features(p, ann)
    base = ml.parse_ym("1998-01")
    fm = base + 20
    row = at_month(f, "000001", ml.ym(fm))
    x = r["000001"]
    assert row["mom"] == pytest.approx(np.prod(1 + x[fm - base - 11:fm - base]) - 1.0, rel=1e-12)       # months f-11 .. f-1
    assert row["rev"] == pytest.approx(-x[fm - base], rel=1e-12)
    assert row["vol"] == pytest.approx(np.std(x[fm - base - 11:fm - base + 1], ddof=1), rel=1e-9)
    # changing the formation month's own return moves rev/vol but never mom
    r2 = {g: v.copy() for g, v in r.items()}
    r2["000001"][fm - base] = 0.9
    f2 = ml.build_features(tiny_pseudo(r2, me_const=1000.0), ann)
    row2 = at_month(f2, "000001", ml.ym(fm))
    assert row2["mom"] == pytest.approx(row["mom"], rel=1e-12) and row2["rev"] == pytest.approx(-0.9)
    # label: fwd is NEXT month's return, NaN in the last month
    assert row["fwd"] == pytest.approx(x[fm - base + 1])
    assert np.isnan(at_month(f, "000001", ml.ym(base + n - 1))["fwd"])
    # one missing month in the window is tolerated (10 of 11), two are not; a gap is a calendar gap, not a skipped row
    pg = p[~((p["gvkey"] == "000001") & (p["mi"] == fm - 5))]
    row_g = at_month(ml.build_features(pg, ann), "000001", ml.ym(fm))
    idx = [i for i in range(fm - base - 11, fm - base) if i != fm - base - 5]
    assert row_g["mom"] == pytest.approx(np.prod(1 + x[idx]) - 1.0, rel=1e-12)
    pg2 = pg[~((pg["gvkey"] == "000001") & (pg["mi"] == fm - 7))]
    assert np.isnan(at_month(ml.build_features(pg2, ann), "000001", ml.ym(fm))["mom"])
    # the vol window with a gap uses the 11 present returns of the calendar window
    idxv = [i for i in range(fm - base - 11, fm - base + 1) if i != fm - base - 5]
    assert row_g["vol"] == pytest.approx(np.std(x[idxv], ddof=1), rel=1e-9)


def test_net_issuance_and_split_handling():
    n = 30
    rng = np.random.default_rng(3)
    r = {g: rng.normal(0.0, 0.03, n) for g in ("000001", "000002", "000003", "000004", "000009")}
    sh = {"000009": np.full(n, 1.0),                                          # tiny NYSE name anchoring the 20th-percentile size cut
          "000001": np.r_[np.full(15, 100.0), np.full(15, 110.0)],           # 10% offering at month 15 (price continuous -> ME jumps)
          "000002": np.r_[np.full(15, 100.0), np.full(15, 200.0)],           # 2:1 split at month 15 (price halves, ME continuous)
          "000003": np.full(n, 100.0), "000004": np.full(n, 100.0)}
    p = tiny_pseudo(r, shares=sh)
    # build the offering case with a price that does NOT halve: tiny_pseudo divides price by the share ratio, so redo 000001 by hand
    p = p.copy()
    base = ml.parse_ym("1998-01")
    m1 = p["gvkey"] == "000001"
    px = 10.0 * np.cumprod(1 + r["000001"])
    p.loc[m1, "me"] = px * sh["000001"]
    p.loc[m1, "me_co"] = px * sh["000001"]
    ann = tiny_annual(list(r), [dict(datadate="1997-12-31", at=100.0, seq=50.0)])
    f = ml.build_features(p, ann)
    fm = base + 20                                                        # window f-12 .. f contains month 15
    assert at_month(f, "000001", ml.ym(fm))["issu"] == pytest.approx(np.log(1.1), abs=1e-12)
    assert at_month(f, "000002", ml.ym(fm))["issu"] == pytest.approx(0.0, abs=1e-12)    # split removed (raw change would be log 2)
    assert at_month(f, "000003", ml.ym(fm))["issu"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(at_month(f, "000003", ml.ym(base + 10))["issu"])      # no month f-12 yet


def test_annual_features_accruals_both_versions_and_lag_guard():
    rows = [dict(datadate="1999-12-31", at=80.0, act=40.0, che=8.0, lct=26.0, dlc=6.0, dp=2.0, sale=100.0, ib=5.0, seq=40.0),
            dict(datadate="2000-12-31", at=100.0, act=50.0, che=10.0, lct=30.0, dlc=5.0, dp=1.0, sale=130.0, ib=10.0, oancf=4.0, seq=60.0,
                 dltt=20.0, revt=130.0, cogs=70.0, xsga=20.0, capx=7.0, xrd=3.0),
            dict(datadate="2001-12-31", at=120.0, act=60.0, che=12.0, lct=33.0, dlc=7.0, dp=2.0, sale=150.0, ib=12.0, seq=70.0),
            dict(datadate="2003-12-31", at=140.0, act=60.0, che=12.0, lct=33.0, dlc=7.0, dp=2.0, sale=150.0, ib=12.0, seq=70.0)]
    a = tiny_annual(["000001"], rows).set_index("datadate")
    y0, y1, y2, y3 = (a.loc[pd.Timestamp(f"{y}-12-31")] for y in (1999, 2000, 2001, 2003))
    assert y1["acc"] == pytest.approx((10.0 - 4.0) / 90.0)                                   # cash-flow version, average assets
    assert y2["acc"] == pytest.approx((((60 - 12) - (50 - 10)) - ((33 - 7) - (30 - 5)) - 2.0) / 110.0)  # balance-sheet version
    assert y2["acc"] == pytest.approx((8.0 - 1.0 - 2.0) / 110.0)
    assert np.isnan(y0["acc"]) and np.isnan(y3["acc"])                                       # no lag record (first year; 2-year gap)
    assert y1["ag"] == pytest.approx(0.25) and np.isnan(y3["ag"])
    assert y1["sg"] == pytest.approx(0.30) and y1["gpa"] == pytest.approx(0.6) and y1["roa"] == pytest.approx(0.10)
    assert y1["lev"] == pytest.approx(25.0 / 100.0) and y1["cash"] == pytest.approx(0.10) and y1["capex"] == pytest.approx(0.07)
    assert y1["rd"] == pytest.approx(0.03) and y2["rd"] == pytest.approx(0.0)               # xrd missing but sale present -> 0
    assert y1["turn"] == pytest.approx(1.3) and y1["opprof"] == pytest.approx((130 - 70 - 20) / 60.0)
    assert y1["cfo"] == pytest.approx(4.0) and y2["cfo"] == pytest.approx(14.0)             # oancf else ib + dp
    assert np.isnan(y2["opprof"])


def test_fundamentals_enter_after_four_month_lag_and_expire():
    n = 40
    rng = np.random.default_rng(4)
    r = {g: rng.normal(0.0, 0.03, n) for g in ("000001", "000002", "000003")}
    p = tiny_pseudo(r, me_const=1000.0)
    ann = tiny_annual(list(r), [dict(datadate="1998-12-31", at=100.0, seq=200.0, ib=20.0, oancf=30.0, revt=300.0, cogs=100.0)])
    f = ml.build_features(p, ann)
    assert np.isnan(at_month(f, "000001", "1999-03")["bm"])                   # available 1999-04-30
    r_apr = at_month(f, "000001", "1999-04")
    assert r_apr["bm"] == pytest.approx(200.0 / 1000.0) and r_apr["ep"] == pytest.approx(0.02) and r_apr["cfp"] == pytest.approx(0.03)
    assert r_apr["gpa"] == pytest.approx(2.0) and r_apr["logme"] == pytest.approx(np.log(1000.0))
    assert at_month(f, "000001", "2000-09")["bm"] == pytest.approx(0.2)       # still within 548 days of 1999-04-30 (2000-10-30)
    assert np.isnan(at_month(f, "000001", "2000-12")["bm"])                   # stale


def test_no_lookahead_truncating_future_changes_nothing():
    rng = np.random.default_rng(5)
    n = 50
    r = {f"{g:06d}": rng.normal(0.01, 0.07, n) for g in range(1, 7)}
    p = tiny_pseudo(r)
    ann = tiny_annual(list(r), [dict(datadate="1998-12-31", at=100.0, seq=60.0, ib=5.0, revt=90.0, cogs=40.0)])
    full = ml.build_features(p, ann)
    cut = ml.parse_ym("1998-01") + 39
    part = ml.build_features(p[p["mi"] <= cut], ann)
    a = full[full["mi"] <= cut].reset_index(drop=True)
    b = part.reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns="fwd"), b.drop(columns="fwd"))
    assert b[b["mi"] == cut]["fwd"].isna().all()


def test_nullable_dtype_inputs_match_float_inputs():
    s_null, f_null = make_secm_world(n=60, start="2000-01", end="2004-12", seed=7, nullable=True)
    s_flt, f_flt = make_secm_world(n=60, start="2000-01", end="2004-12", seed=7, nullable=False)
    a, b = features_of(s_null, f_null), features_of(s_flt, f_flt)
    assert all(a[c].dtype == np.float64 for c in ml.FEATURES) and a["permno"].dtype == np.int64
    pd.testing.assert_frame_equal(a, b)
    assert a[ml.FEATURES].notna().mean().min() > 0.3


# -------------------------------------------------------------------------------------------------------------------- rank / model
def test_rank_transform_properties():
    df = pd.DataFrame({"mi": [1, 1, 1, 1, 2, 2, 2, 3], "x": [3.0, 1.0, 2.0, np.nan, 5.0, 5.0, 9.0, 4.0]})
    r = ml.rank_transform(df, ["x"])["x"]
    assert r.tolist() == [0.5, -0.5, 0.0, 0.0, -0.25, -0.25, 0.5, 0.0]       # NaN ranked out, then 0; ties average; single obs 0
    assert r.min() >= -0.5 and r.max() <= 0.5
    keep = ml.rank_transform(df, ["x"], fill_missing=False)["x"]
    assert np.isnan(keep.iloc[3]) and keep.iloc[0] == 0.5
    assert df["x"].iloc[3] != df["x"].iloc[3] and df["x"].iloc[0] == 3.0       # input untouched
    big = pd.DataFrame({"mi": 1, "x": np.random.default_rng(0).normal(size=101)})
    rr = ml.rank_transform(big, ["x"])["x"]
    assert rr.min() == -0.5 and rr.max() == 0.5 and rr.mean() == pytest.approx(0.0, abs=1e-12)
    assert (np.argsort(big["x"].to_numpy()) == np.argsort(rr.to_numpy())).all()    # monotone
    nullable = pd.DataFrame({"mi": pd.array([1, 1, 1], dtype="Int64"), "x": pd.array([1.0, None, 3.0], dtype="Float64")})
    assert ml.rank_transform(nullable, ["x"])["x"].tolist() == [-0.5, 0.0, 0.5]


def test_make_model_spec_is_exact():
    ml._TEST_MODEL_OVERRIDES.clear()
    p = ml.make_model().get_params()
    assert (p["max_depth"], p["learning_rate"], p["max_iter"], p["min_samples_leaf"], p["l2_regularization"], p["random_state"]) == (3, 0.05, 200, 500, 1.0, 0)
    assert p["early_stopping"] is False and type(ml.make_model()).__name__ == "HistGradientBoostingRegressor"


def test_training_window_and_leakage_guard(planted):
    assert ml.train_window(2020) == (1990 * 12, 2019 * 12 + 10)               # previous November
    assert ml.model_year(2026 * 12 + 3) == 2026 and ml.model_year(2025 * 12 + 11) == 2025
    ml.assert_no_leakage(2019 * 12 + 10, 2020 * 12)                           # Nov formation -> Dec label < Jan
    with pytest.raises(AssertionError, match="leakage"):
        ml.assert_no_leakage(2019 * 12 + 11, 2020 * 12)                       # Dec formation -> Jan label


def test_walk_forward_never_trains_on_unrealised_labels(planted, monkeypatch):
    R = ml.rank_features(planted)
    seen = {}

    class Rec:
        def fit(self, X, y):
            seen["n"] = len(X)
            return self

        def predict(self, X):
            return np.zeros(len(X))
    monkeypatch.setattr(ml, "make_model", lambda: Rec())
    start = 2001 * 12
    ml.fit_predict_year(R, 2008, planted["mi"].max(), train_start_mi=start)
    expect = ((R["mi"] >= start) & (R["mi"] <= 2007 * 12 + 10) & R["y"].notna()).sum()
    with_dec = ((R["mi"] >= start) & (R["mi"] <= 2007 * 12 + 11) & R["y"].notna()).sum()
    assert seen["n"] == expect and expect < with_dec
    # a deliberately broken variant (training through December) trips the guard
    with pytest.raises(AssertionError, match="leakage"):
        ml.fit_predict_year(R, 2008, planted["mi"].max(), train_start_mi=start, lag=1)


def test_walk_forward_is_deterministic_and_order_free(planted):
    kw = dict(first_pred_year=2012, last_mi=planted["mi"].max(), train_start_mi=2001 * 12)
    a = ml.walk_forward(planted, **kw)
    b = ml.walk_forward(planted, **kw)
    pd.testing.assert_frame_equal(a, b)
    c = ml.walk_forward(planted.sample(frac=1.0, random_state=1).reset_index(drop=True), **kw)
    pd.testing.assert_frame_equal(a.sort_values(["mi", "permno"]).reset_index(drop=True), c.sort_values(["mi", "permno"]).reset_index(drop=True))
    assert set(a.columns) == {"permno", "gvkey", "iid", "mi", "me", "score_ml", "score_comp", "gpa"}
    assert a["mi"].min() == 2012 * 12 and a["mi"].max() == planted["mi"].max()
    # protocol mapping: scoring one month alone (as ml_form does) equals the walk-forward row set for that month
    R = ml.rank_features(planted)
    mi = 2013 * 12 + 4
    one = ml.fit_predict_year(R, ml.model_year(mi), planted["mi"].max(), train_start_mi=2001 * 12, pred_months=[mi])
    ref = a[a["mi"] == mi].reset_index(drop=True)
    pd.testing.assert_frame_equal(one, ref)


def test_comparator_composite_is_signed_mean_of_ranks(planted):
    R = ml.rank_features(planted)
    row = R.iloc[1234]
    expect = np.mean([row[c] * ml.FEATURE_SIGNS[c] for c in ml.FEATURES])
    assert row["score_comp"] == pytest.approx(expect)


# ------------------------------------------------------------------------------------------------------------------- evaluation
def test_evaluation_by_hand_two_months():
    # month 0: 10 names, score = i, me = i+1 ; top quintile (rank > 8 of 10) = names 8, 9
    ids = np.arange(10)
    r1 = np.array([0.0, 0.01, 0.02, 0.0, -0.01, 0.03, 0.02, 0.01, 0.10, -0.05])
    r2 = np.zeros(10)
    r2[8] = 0.5
    sc = pd.concat([pd.DataFrame({"permno": ids, "mi": 100, "me": ids + 1.0, "s": ids.astype(float)}),
                    pd.DataFrame({"permno": ids, "mi": 101, "me": ids + 1.0, "s": ids.astype(float)})])
    ret = pd.concat([pd.DataFrame({"permno": ids, "mi": 101, "ret": r1}), pd.DataFrame({"permno": ids, "mi": 102, "ret": r2})])
    m = ml.evaluate_scores(sc, ret, signals=["s"])
    w_top = np.array([9.0, 10.0]) / 19.0
    top = float(w_top @ np.array([0.10, -0.05]))
    bench = float((np.arange(1, 11) / 55.0) @ r1)
    assert m.loc[100, "top_s"] == pytest.approx(top) and m.loc[100, "bench"] == pytest.approx(bench)
    assert m.loc[100, "gross_s"] == pytest.approx(top - bench)
    assert m.loc[100, "turn_s"] == 1.0                                         # first month = full build
    assert m.loc[100, "net_s"] == pytest.approx(top - bench - 0.0010)
    # month 1: weights drift with month-101 returns before the same target weights are reapplied
    g = w_top * (1 + np.array([0.10, -0.05]))
    w_pre = g / g.sum()
    assert m.loc[101, "turn_s"] == pytest.approx(0.5 * np.abs(w_top - w_pre).sum())
    assert m.loc[100, "ic_s"] == pytest.approx(pd.Series(ids).rank().corr(pd.Series(r1).rank()))
    assert m.loc[101, "ic_s"] > 0
    # a month whose next month has no returns at all is dropped, not scored as zero
    assert set(m.index) == {100, 101}
    sc3 = pd.concat([sc, sc.assign(mi=102)])
    assert set(ml.evaluate_scores(sc3, ret, signals=["s"]).index) == {100, 101}


def test_rank_ic_is_nan_safe():
    sc = pd.DataFrame({"permno": [1, 2, 3, 4], "mi": [5, 5, 5, 6], "s": [1.0, np.nan, 3.0, 2.0]})
    ret = pd.DataFrame({"permno": [1, 2, 3, 4], "mi": [6, 6, 6, 7], "ret": [0.1, 0.2, 0.3, 0.0]})
    ic = ml.monthly_rank_ic(sc, ret, "s")
    assert np.isnan(ic.loc[5]) and np.isnan(ic.loc[6])                          # fewer than 3 names -> NaN, never an error


def test_paired_t_and_summary():
    mean, t, n = ml.paired_t(pd.Series([2.0, 4.0, 6.0]), pd.Series([1.0, 2.0, 3.0]))
    assert (mean, n) == (2.0, 3) and t == pytest.approx(2.0 / (1.0 / np.sqrt(3)))
    assert np.isnan(ml.paired_t(pd.Series([1.0, 2.0]), pd.Series([0.0, 0.0]))[1])
    idx = pd.Series([1.0, np.nan, 3.0, 4.0], index=[0, 1, 2, 3])
    assert ml.paired_t(idx, idx * 0)[2] == 3                                    # NaN months dropped pairwise


def _decision_frame(diff_mean, diff_sd, n=36, gpa_gap=0.001):
    z = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    z = z / z.std(ddof=1)                                                       # mean 0, sd 1
    comp = np.full(n, 0.002)
    mlv = comp + diff_mean + diff_sd * z
    return pd.DataFrame({"net_score_ml": mlv, "net_score_comp": comp, "net_gpa": np.full(n, mlv.mean() - gpa_gap)}, index=range(n))


def test_decision_label_boundaries():
    edge_sd = 0.0015 * 6.0 / 2.0                                                 # t = mean/(sd/6) = 2.0 exactly at mean 0.15%
    assert ml.decision_label(_decision_frame(0.0015, edge_sd)) == "ML ADDS VALUE"             # both inclusive boundaries met
    assert ml.decision_label(_decision_frame(0.00149, 0.00149 * 3.0)) == "NO EVIDENCE"        # edge just below
    assert ml.decision_label(_decision_frame(0.0016, 0.0016 * 6.0 / 1.9)) == "NO EVIDENCE"    # t = 1.9
    assert ml.decision_label(_decision_frame(0.0015, edge_sd, gpa_gap=-0.001)) == "NO EVIDENCE"   # does not beat gpa
    assert ml.decision_label(_decision_frame(0.0030, 0.001, n=35)) == "NOT YET (n<36)"
    big = _decision_frame(0.0030, 0.001, n=40)
    big.iloc[36:, big.columns.get_loc("net_score_ml")] = -1.0                      # months after the 36th must not matter
    assert ml.decision_label(big) == "ML ADDS VALUE"


# ----------------------------------------------------------------------------------------------------------------------- ledger
def _form(seed=0, n=8):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"gvkey": [f"{i:06d}" for i in range(n)], "iid": "01", "mi": 24300, "me": rng.uniform(100, 1000, n),
                         "score_ml": rng.normal(size=n), "score_comp": rng.normal(size=n), "gpa": np.r_[np.nan, rng.uniform(-.5, .5, n - 1)]})


def test_score_sha_is_order_independent_and_sensitive():
    f = _form()
    assert ml.score_sha(f) == ml.score_sha(f.iloc[::-1]) == ml.score_sha(f.sample(frac=1.0, random_state=3))
    g = f.copy()
    g.loc[3, "score_ml"] += 1e-6
    assert ml.score_sha(g) != ml.score_sha(f)
    h = f.copy()
    h.loc[2, "me"] += 1.0
    assert ml.score_sha(h) != ml.score_sha(f)
    z = f.copy()
    z.loc[1, "score_comp"] = -0.0
    z2 = f.copy()
    z2.loc[1, "score_comp"] = 0.0
    assert ml.score_sha(z) == ml.score_sha(z2)


def test_freeze_refuses_overwrite_and_detects_tampering(tmp_path):
    f = _form()
    mi = 24300
    sha = ml.freeze_month(f, mi, "abc123", "2026-10-11", ml.model_id(mi - 2), root=tmp_path)
    assert (tmp_path / "data" / "ml" / "forms" / "202501.parquet").exists() and ml.form_path(mi, tmp_path).exists()
    line = json.loads((tmp_path / "paper_ledger" / "ml_manifest.jsonl").read_text().splitlines()[0])
    assert line["sha256"] == sha and line["mi"] == mi and line["spec_commit"] == "abc123" and line["formed_on"] == "2026-10-11" and "hgb" in line["model"]
    with pytest.raises(FileExistsError):
        ml.freeze_month(_form(seed=9), mi, "def", "2026-10-12", "x", root=tmp_path)
    assert len(ml.read_manifest(tmp_path)) == 1
    (m, form), = ml.load_verified(tmp_path)
    assert ml.score_sha(form) == m["sha256"]
    bad = form.copy()
    bad.loc[0, "score_ml"] += 0.5
    bad.to_parquet(ml.form_path(mi, tmp_path), index=False)
    with pytest.raises(AssertionError, match="modified"):
        ml.load_verified(tmp_path)


def test_form_groups():
    assert ml.form_group(2025 * 12 + 11, "2026-10-11") == "pseudo-forward" and ml.form_group(2026 * 12 + 7, "2026-10-11") == "pseudo-forward"
    assert ml.form_group(2026 * 12 + 8, "2026-10-03") == "prospective"        # target Oct 2026 starts 10-01; formed within 7 days
    assert ml.form_group(2026 * 12 + 8, "2026-10-09") == "retrospective"


# --------------------------------------------------------------------------------------------------------------------- end to end
def _forward_eval(features, first_year, train_start):
    last = int(features["mi"].max())
    wf = ml.walk_forward(features, first_year, last, train_start_mi=train_start)
    ret = features[["permno", "mi", "fwd"]].dropna().assign(mi=lambda d: d["mi"] + 1).rename(columns={"fwd": "ret"})
    return ml.evaluate_scores(wf, ret), wf


def test_end_to_end_planted_signal_is_found(planted):
    m, wf = _forward_eval(planted, 2008, 2001 * 12)
    s = ml.summarize(m)
    assert s["n"] >= 70
    assert s["score_ml"]["ic"] > 0.03 and s["score_ml"]["ic_t"] > 4
    assert s["score_comp"]["ic"] > 0.0 and s["gpa"]["ic"] > 0.03          # the planted driver is gross profit / assets
    assert s["score_ml"]["gross"] > 0


def test_end_to_end_null_world_has_no_ic(null_world):
    m, wf = _forward_eval(null_world, 2008, 2001 * 12)
    ic = m["ic_score_ml"].dropna()
    se = ic.std(ddof=1) / np.sqrt(len(ic))
    assert len(ic) >= 70 and abs(ic.mean()) < 4 * se + 0.005
    assert abs(ic.mean()) < 0.03
    assert ml.summarize(m)["score_comp"]["ic"] == pytest.approx(m["ic_score_comp"].mean())


def test_calibration_and_build_from_raw_match_the_paper_harness():
    from qap.paper import calibrate_secm_scale
    secm, funda = make_secm_world(n=120, start="2000-01", end="2005-12", seed=5)
    scale, ratio = calibrate_secm_scale(secm, funda)
    raw = ml.with_shares(build_pseudo_panel(secm, 1.0), secm)
    assert ml.calibrate_on_raw(raw, funda) == (scale, ratio) and scale == 1000.0
    b = ml.build_from_raw(secm, funda)
    ref = features_of(secm, funda, scale)
    pd.testing.assert_frame_equal(b["features"], ref, rtol=1e-12)
    assert b["scale"] == 1000.0 and b["partial_mi"] is None and b["last_mi"] == ml.parse_ym("2005-12")
    # a partial latest month is dropped, not used as a label
    part = secm[~((secm["datadate"] == secm["datadate"].max()) & (secm["gvkey"].astype(str) > "000010"))]
    b2 = ml.build_from_raw(part, funda)
    assert b2["partial_mi"] == ml.parse_ym("2005-12") and b2["last_mi"] == ml.parse_ym("2005-11") and b2["features"]["mi"].max() == ml.parse_ym("2005-11")
    assert b2["features"]["fwd"].isna().groupby(b2["features"]["mi"]).all().iloc[-1]


def test_walkforward_never_touches_forward_only_months():
    secm, funda = make_secm_world(n=150, start="2018-01", end="2027-03", seed=31, k=0.01, first_fy=2016)
    b = ml.build_from_raw(secm, funda, max_mi=ml.WALKFORWARD_LAST_MI + 1)
    assert b["features"]["mi"].max() == ml.WALKFORWARD_LAST_MI + 1 and b["pseudo"]["mi"].max() == ml.WALKFORWARD_LAST_MI + 1
    w = ml.walkforward_inputs(b, 2022)
    assert w["cap"] == ml.WALKFORWARD_LAST_MI == ml.parse_ym("2025-11") and w["years"] == [2022, 2023, 2024, 2025]
    assert w["features"]["mi"].max() == w["cap"] and w["ret"]["mi"].max() == w["cap"] + 1
    ml._TEST_MODEL_OVERRIDES.update(max_iter=10, min_samples_leaf=50)
    try:
        wf = ml.walk_forward(w["features"], 2022, w["cap"], train_start_mi=2019 * 12)
    finally:
        ml._TEST_MODEL_OVERRIDES.clear()
    m = ml.evaluate_scores(wf, w["ret"])
    assert wf["mi"].max() == w["cap"] and m.index.max() == w["cap"] and m.index.max() < ml.parse_ym("2025-12")


# ------------------------------------------------------------------------------------------------- data-driven windows, coverage
def _late_history_world(n=150, start="1995-01", end="2008-12", blind_before=1998, seed=41):
    """World whose returns are missing (trt1m NaN) before `blind_before`, emulating a comp.secm pull whose usable history starts late."""
    secm, funda = make_secm_world(n=n, start=start, end=end, seed=seed, k=0.01)
    secm.loc[secm["datadate"] < pd.Timestamp(f"{blind_before}-01-01"), "trt1m"] = pd.NA
    return secm, funda


def test_coverage_text_has_expected_years_and_shares():
    secm, funda = _late_history_world()
    b = ml.build_from_raw(secm, funda)
    txt = ml.coverage_text(secm, b["features"], b["pseudo"])
    raw = {int(ln.split()[0]): ln.split() for ln in ml.raw_coverage_lines(secm)[2:]}
    assert set(raw) == set(range(1995, 2009))
    assert raw[1996][2] == "0%" and raw[1999][2] == "100%" and raw[1999][5] == "50%"          # trt1m share, exchg == 11 share
    built = {int(ln.split()[0]): ln.split() for ln in ml.panel_coverage_lines(b["features"], b["pseudo"])[2:]}
    assert min(built) == 1995 and "RAW comp.secm" in txt and "BUILT panel" in txt
    assert built[1996][3] == "0"                                                               # no forward return before 1998
    assert int(built[2000][3].replace(",", "")) > 0 and built[2000][5] != "0%"                 # beta share present once history exists


def test_windows_are_data_driven_and_explicit_infeasible_values_get_a_clear_message():
    secm, funda = _late_history_world()
    b = ml.build_from_raw(secm, funda)
    F = b["features"]
    w = ml.choose_windows(F, b["last_mi"])
    assert 1997 * 12 + 11 <= w["train_start"] <= 1998 * 12 + 6               # 1997-12 is the first month with a forward return (Jan 1998)
    rows, months = ml.train_stats(ml.usable_counts(F), w["train_start"], w["first_year"])
    assert rows >= ml.MIN_TRAIN_ROWS and months >= ml.MIN_TRAIN_MONTHS
    assert ml.train_stats(ml.usable_counts(F), w["train_start"], w["first_year"] - 1)[1] < ml.MIN_TRAIN_MONTHS     # the previous year is infeasible
    with pytest.raises(ml.InfeasibleWindow, match=r"first_year 1995.*first feasible values: train_start (1997|1998)-\d\d, first_year %d" % w["first_year"]):
        ml.choose_windows(F, b["last_mi"], first_year=1995)
    with pytest.raises(ml.InfeasibleWindow, match="first feasible"):
        ml.choose_windows(F, b["last_mi"], train_start_mi=1990 * 12, first_year=w["first_year"] - 1)
    with pytest.raises(ml.InfeasibleWindow, match="no 12-month span"):
        ml.choose_windows(F[F["mi"] < 1999 * 12 - 6], 1999 * 12 - 7)
    # explicit feasible values are kept
    w2 = ml.choose_windows(F, b["last_mi"], train_start_mi=w["train_start"], first_year=w["first_year"] + 1)
    assert w2["first_year"] == w["first_year"] + 1 and w2["train_start"] == w["train_start"]
    # the walk-forward inputs use them and fit without error
    wi = ml.walkforward_inputs(b, None)
    assert wi["years"][0] == w["first_year"] and wi["train_start"] == w["train_start"]
    ml._TEST_MODEL_OVERRIDES.update(max_iter=10)
    try:
        wf = ml.walk_forward(wi["features"], wi["first_year"], wi["cap"], train_start_mi=wi["train_start"])
    finally:
        ml._TEST_MODEL_OVERRIDES.clear()
    assert wf["mi"].min() == w["first_year"] * 12


def _run_script(tmp_path, name, *args):
    import pathlib
    import subprocess
    import sys
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / name
    return subprocess.run([sys.executable, str(script), *args], cwd=tmp_path, capture_output=True, text=True, timeout=600)


def _write_raw(tmp_path, secm, funda):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    secm.to_parquet(tmp_path / "data" / "raw" / "comp_secm_hist.parquet", index=False)
    funda.to_parquet(tmp_path / "data" / "raw" / "compustat_annual.parquet", index=False)


def test_scripts_pick_feasible_windows_or_fail_clearly(tmp_path):
    secm, funda = _late_history_world(n=120, start="1995-01", end="2008-12")
    _write_raw(tmp_path, secm, funda)
    bad = _run_script(tmp_path, "ml_walkforward.py", "--first-year", "1996")
    assert bad.returncode != 0 and "INFEASIBLE WINDOW" in (bad.stderr + bad.stdout) and "first feasible values" in (bad.stderr + bad.stdout)
    assert "Traceback" not in bad.stderr.split("INFEASIBLE")[0][-400:] or "AssertionError" not in bad.stderr
    ok = _run_script(tmp_path, "ml_walkforward.py")
    assert ok.returncode == 0, ok.stderr[-2000:]
    assert "COVERAGE DIAGNOSTIC" in ok.stdout and re.search(r"chosen: train_start (1997|1998)-", ok.stdout) and "(data-driven)" in ok.stdout
    assert (tmp_path / "data" / "ml_walkforward_report.txt").exists()
    # ml_form: data-driven start recorded in the manifest; frozen months survive a second run
    f = _run_script(tmp_path, "ml_form.py", "--from", "2008-10")
    assert f.returncode == 0, f.stderr[-2000:]
    assert "COVERAGE DIAGNOSTIC" in f.stdout and re.search(r"train_start (1997|1998)-", f.stdout)
    man = [json.loads(ln) for ln in (tmp_path / "paper_ledger" / "ml_manifest.jsonl").read_text().splitlines()]
    assert [m["ym"] for m in man] == ["2008-10", "2008-11", "2008-12"] and all(re.search(r"train (1997|1998)-", m["model"]) for m in man)


def test_ml_form_refuses_when_history_is_too_short(tmp_path):
    secm, funda = make_secm_world(n=120, start="2006-01", end="2008-12", seed=3, k=0.0)       # 3 years: no January model reaches 36 months
    _write_raw(tmp_path, secm, funda)
    r = _run_script(tmp_path, "ml_form.py", "--from", "2008-06")
    assert r.returncode != 0 and "INFEASIBLE WINDOW" in (r.stderr + r.stdout)
    assert not (tmp_path / "paper_ledger" / "ml_manifest.jsonl").exists() or (tmp_path / "paper_ledger" / "ml_manifest.jsonl").read_text() == ""
