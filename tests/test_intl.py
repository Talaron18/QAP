import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from qap import config, intl
from qap.jkp import theme_scores
from qap.jkp_themes import ALL_CHARACTERISTICS, THEMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIGN = {c: s for m in THEMES.values() for c, s in m}


# ---------------------------------------------------------------- synthetic raw files (also used by the smoke run)

def synth_year(year, countries, n, seed, beta, us=False, scale=None, nan_frac=0.05, base=0):
    """One year file in the extraction's layout (nullable dtypes). Every theme member = sign * latent + noise; ret = beta(c, t) * latent."""
    members = [c for c in ALL_CHARACTERISTICS if c != "bidaskhl_21d"]
    sg = np.array([SIGN[c] for c in members], dtype="float64")
    parts = []
    for ci, c in enumerate(countries):
        f = np.random.RandomState(seed + 31 * ci).randn(n)               # persistent per-stock latent (same every year)
        sc = (scale or {}).get(c, 1.0)
        cols = {k: [] for k in ("eom", "size_grp", "me", "ret", "hs")}
        V = []
        for m in range(1, 13):
            t = year + (m - 1) / 12
            rng = np.random.RandomState(seed + 1000 * ci + 7 + 97 * m)
            cols["eom"].append(np.full(n, pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0), dtype="datetime64[ns]"))
            cols["size_grp"].append(rng.choice(["mega", "large", "small"], n))
            cols["me"].append(np.exp(rng.randn(n) + 7.0))
            cols["ret"].append(0.01 * beta(c, t) * f + 0.08 * rng.randn(n))
            cols["hs"].append(np.abs(0.01 + 0.005 * rng.randn(n)))
            v = (sg[None, :] * f[:, None] + 0.7 * rng.randn(n, len(members))) * sc
            v[rng.rand(n, len(members)) < nan_frac] = np.nan
            V.append(v)
        V = np.vstack(V)
        data = {"id": pd.array(np.tile(base + 100_000_000 * (ci + 1) + np.arange(n), 12), dtype="Int64"),
                "eom": np.concatenate(cols["eom"]), "size_grp": np.concatenate(cols["size_grp"]),
                "me": np.concatenate(cols["me"]), "ret_exc_lead1m": np.concatenate(cols["ret"]).astype("float32"),
                "bidaskhl_21d": np.concatenate(cols["hs"]).astype("float32")}
        for j, col in enumerate(members):
            data[col] = pd.array(V[:, j].astype("float32"), dtype="Float32")
        df = pd.DataFrame(data)
        if us:
            df = df.rename(columns={"id": "permno"})
        else:
            df.insert(1, "excntry", pd.array([c] * len(df), dtype="string"))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def write_synth(root, intl_years, us_years, countries, n=110, n_us=120, beta=None, scale=None):
    beta = beta or (lambda c, t: 1.0)
    for y in intl_years:
        d = pathlib.Path(root, "jkp_intl")
        d.mkdir(parents=True, exist_ok=True)
        synth_year(y, countries, n, 11 + y, beta, scale=scale).to_parquet(d / f"{y}.parquet", index=False)
        (d / "_missing_columns.txt").write_text("fake_col_a\nfake_col_b\n")
    for y in us_years:
        d = pathlib.Path(root, "jkp_us")
        d.mkdir(parents=True, exist_ok=True)
        synth_year(y, ["USA"], n_us, 5 + y, beta, us=True).to_parquet(d / f"{y}.parquet", index=False)
        (d / "_missing_columns.txt").write_text("\n")


def _beta(c, t):
    return 0.1 if (c in config.REGIONS["Developed Europe"] and t >= 2015) else 1.0


YEARS = [2013, 2014, 2015, 2016, 2021, 2022]
CTRY = ["GBR", "DEU", "JPN", "AUS", "CAN"]


# ---------------------------------------------------------------- ranking within country-month

def test_theme_scores_group_ranks_within_group_and_default_is_unchanged():
    th = {"T": [("a", 1), ("b", 1), ("c", -1)]}
    a = np.array([1.0, 2.0, 3.0, 4.0])
    df = pd.DataFrame({"mi": 5, "cm": [1] * 4 + [2] * 4,
                       "a": np.r_[a, a * 1000.0], "b": np.r_[a, a * 1000.0], "c": np.r_[a, a * 1000.0]})
    within = theme_scores(df, th, group="cm")["T"]
    expected = (pd.Series(a).rank(pct=True) - 0.5)
    expected = ((expected * 1 + expected * 1 + expected * -1) / 3).to_numpy()
    np.testing.assert_allclose(within.to_numpy(), np.r_[expected, expected])
    pooled = theme_scores(df, th)["T"]                      # default group="mi": country 2 (large scale) dominates the ranking
    assert not np.allclose(pooled.to_numpy(), within.to_numpy())
    np.testing.assert_allclose(theme_scores(df, th, group="mi")["T"], pooled)


def _base(n, ctry="JPN", mi=24000, **kw):
    d = pd.DataFrame({"id": np.arange(n, dtype="int64"), "excntry": ctry, "mi": mi, "size_grp": "small",
                      "ret": np.linspace(-0.05, 0.05, n), "me": 10.0})
    for k, v in kw.items():
        d[k] = v
    return d


def test_score_frame_ranks_within_country_month_with_different_scales():
    n = 120
    x = np.random.RandomState(0).randn(n)
    a, b = _base(n, "JPN", gp_at=x), _base(n, "GBR", gp_at=x * 1e6 + 5e8)
    b["id"] += 1000
    slim, info = intl.score_frame(pd.concat([a, b], ignore_index=True))
    for c in ("JPN", "GBR"):
        np.testing.assert_allclose(np.sort(slim.loc[slim["excntry"] == c, "gp_at"].to_numpy()),
                                   np.sort(pd.Series(x).rank(pct=True).to_numpy() - 0.5))
    assert set(slim["region"].astype(str)) == {"Japan", "Developed Europe"}
    assert slim["gp_at"].dtype == np.float64 and slim["id"].dtype == np.int64


def test_country_month_needs_100_eligible_stocks():
    ok, edge = _base(100, "JPN"), _base(99, "GBR")
    edge["id"] += 1000
    # 100 listed names but one is ineligible (me <= 0) and one has no return -> 98 eligible -> dropped
    bad = _base(100, "AUS")
    bad["id"] += 5000
    bad.loc[0, "me"], bad.loc[1, "ret"] = 0.0, np.nan
    slim, info = intl.score_frame(pd.concat([ok, edge, bad], ignore_index=True))
    assert set(slim["excntry"]) == {"JPN"} and len(slim) == 100
    cm = info["cm"].set_index("excntry")
    assert cm.loc["JPN", "kept"] and not cm.loc["GBR", "kept"] and not cm.loc["AUS", "kept"]
    assert cm.loc["AUS", "n"] == 98 and cm.loc["AUS", "rows"] == 100
    # the same country in another month is judged on its own month
    nxt = _base(100, "GBR", mi=24001)
    nxt["id"] += 1000
    slim2, info2 = intl.score_frame(pd.concat([edge, nxt], ignore_index=True))
    assert set(slim2["mi"]) == {24001} and info2["cm"]["kept"].sum() == 1


def test_unknown_country_and_duplicates_are_rejected():
    with pytest.raises(ValueError):
        intl.score_frame(_base(5, "IND"))
    d = _base(5, "JPN")
    with pytest.raises(ValueError):
        intl.score_frame(pd.concat([d, d], ignore_index=True))


# ---------------------------------------------------------------- series: IC, legs, identity, turnover

def _sig_frame(blocks):
    """blocks: list of (cm, mi, signal, ret, me, ids, hs)."""
    rows = []
    for cm, mi, sg, rt, me, ids, hs in blocks:
        rows.append(pd.DataFrame({"id": ids, "cm": cm, "mi": mi, "s": sg, "ret": rt, "me": me, "hs": hs}))
    return pd.concat(rows, ignore_index=True)


def test_ic_is_n_weighted_over_countries_and_nan_safe():
    a = _sig_frame([(1, 100, [1, 2, 3, 4], [.1, .2, .3, .4], 1.0, [1, 2, 3, 4], 0.0)])                       # IC +1, n 4
    b = _sig_frame([(2, 100, np.arange(8.0), -np.arange(8.0), 1.0, np.arange(10, 18), 0.0)])               # IC -1, n 8
    c = _sig_frame([(3, 100, np.arange(5.0), np.ones(5), 1.0, np.arange(20, 25), 0.0)])                    # constant ret -> IC NaN
    d = pd.concat([a, b, c], ignore_index=True).rename(columns={"s": "x"})
    out = intl.signal_series(d, "x", min_ranked=3)
    assert out.index.tolist() == [101]
    assert out.loc[101, "ic"] == pytest.approx((4 * 1 + 8 * -1) / 12)
    only_nan = intl.signal_series(c.rename(columns={"s": "x"}), "x", min_ranked=3)
    assert np.isnan(only_nan.loc[101, "ic"])


def test_vw_quintile_legs_hand_checked_and_identity():
    n = 10
    me = np.arange(1.0, n + 1)
    ret = np.array([.01, .02, .03, .04, .05, .06, .07, .08, .09, .10])
    d = _sig_frame([(1, 50, np.arange(n, dtype=float), ret, me, np.arange(n), 0.0)]).rename(columns={"s": "x"})
    o = intl.signal_series(d, "x", min_ranked=3).loc[51]
    q5 = (9 * .09 + 10 * .10) / 19                    # pct 0.9, 1.0 -> q 5
    q1 = (1 * .01 + 2 * .02) / 3                      # pct 0.1, 0.2 -> q 1
    u = (me * ret).sum() / me.sum()
    assert o["q5"] == pytest.approx(q5) and o["q1"] == pytest.approx(q1) and o["u"] == pytest.approx(u)
    assert o["q5q1"] == pytest.approx(q5 - q1) and o["q5u"] == pytest.approx(q5 - u) and o["q1u"] == pytest.approx(q1 - u)
    rng = np.random.RandomState(3)
    big = pd.DataFrame({"id": np.arange(600), "cm": np.repeat([1, 2, 3], 200), "mi": np.tile(np.repeat([7, 8], 100), 3) + 1000,
                        "x": rng.randn(600), "ret": rng.randn(600) * .1, "me": np.exp(rng.randn(600)), "hs": 0.01})
    s = intl.signal_series(big, "x", min_ranked=10)
    np.testing.assert_allclose((s["q5u"] - s["q1u"]).to_numpy(), s["q5q1"].to_numpy(), atol=1e-14)


def test_turnover_fixed_portfolio_swap_gap_and_cost():
    n = 10
    def month(mi, ids):
        return _sig_frame([(1, mi, np.arange(n, dtype=float), np.full(n, .01), 1.0, ids, 0.01)])
    top = np.arange(n)
    fixed = pd.concat([month(10, top), month(11, top)]).rename(columns={"s": "x"})
    o = intl.signal_series(fixed, "x", min_ranked=3)
    assert np.isnan(o.loc[11, "to"]) and o.loc[12, "to"] == pytest.approx(0.0) and o.loc[12, "cost"] == pytest.approx(0.0)
    swap = pd.concat([month(10, top), month(11, top + 100)]).rename(columns={"s": "x"})
    o = intl.signal_series(swap, "x", min_ranked=3)
    assert o.loc[12, "to"] == pytest.approx(1.0)
    # cost = 1.5 * sum|dw| * half-spread = 1.5 * 2 * 0.01 (leavers and entrants both carry 0.01); net = Q5-U - cost
    assert o.loc[12, "cost"] == pytest.approx(0.03) and o.loc[12, "net"] == pytest.approx(o.loc[12, "q5u"] - 0.03)
    gap = pd.concat([month(10, top), month(12, top)]).rename(columns={"s": "x"})        # month 11 missing -> no turnover
    o = intl.signal_series(gap, "x", min_ranked=3)
    assert o["to"].isna().all()


# ---------------------------------------------------------------- statistics and labels

def test_nw_t_matches_holdout_ls():
    from qap.holdout_ls import nw_t
    x = np.random.RandomState(1).randn(80) + 0.2
    assert intl.nw_t(x) == pytest.approx(nw_t(x)) and np.isnan(intl.nw_t([1.0, 2.0]))


def _ic(dev, late, val=None):
    idx = list(range(1990 * 12, 2015 * 12)) + list(range(2015 * 12, 2026 * 12))
    return pd.Series([dev] * (2015 - 1990) * 12 + [late] * (2026 - 2015) * 12, index=idx, dtype="float64")


def test_h1_ratio_and_label_boundaries():
    assert intl.h1_ratio(_ic(0.04, 0.02)) == pytest.approx(0.5)
    assert np.isnan(intl.h1_ratio(_ic(0.0, 0.02))) and np.isnan(intl.h1_ratio(_ic(-0.01, 0.02)))
    D, P = (0.5, 0.5), (0.75, 0.75)
    assert intl.h1_label({"a": D, "b": D, "c": (0.9, 0.9)})[0] == "GLOBAL DECAY"                # exactly 0.5 counts
    assert intl.h1_label({"a": (0.5000001, 0.4), "b": D, "c": (0.9, 0.9)})[0] == "MIXED"
    assert intl.h1_label({"a": D, "b": (0.4, 0.51), "c": D})[0] == "GLOBAL DECAY"
    assert intl.h1_label({"a": P, "b": P, "c": (0.2, 0.2)})[0] == "PERSISTS OUTSIDE THE US"       # exactly 0.75 counts
    assert intl.h1_label({"a": (0.7499, 0.9), "b": P, "c": (0.2, 0.2)})[0] == "MIXED"
    assert intl.h1_label({"a": D, "b": P, "c": (0.6, 0.6)})[0] == "MIXED"                          # one decay, one persist
    assert intl.h1_label({"a": P, "b": P, "c": D})[0] == "PERSISTS OUTSIDE THE US"
    assert intl.h1_label({"a": (np.nan, 0.1), "b": D, "c": (0.6, 0.6)})[0] == "MIXED"             # n/a counts for neither
    assert intl.h1_label({"a": (0.8, 0.8), "b": (0.8, 0.7), "c": (0.8, 0.74)})[0] == "MIXED"   # 'both composites' per region


def test_short_share_and_h2_boundaries():
    i = pd.RangeIndex(5)
    assert intl.short_share(pd.Series([-0.6] * 5, index=i), pd.Series([1.0] * 5, index=i)) == pytest.approx(0.6)
    assert np.isnan(intl.short_share(pd.Series([-0.6] * 5, index=i), pd.Series([-1.0] * 5, index=i)))     # non-positive Q5-Q1
    assert np.isnan(intl.short_share(pd.Series([-0.6] * 5, index=i), pd.Series([0.0] * 5, index=i)))
    S = (0.6, 0.6)
    assert intl.h2_label({"a": S, "b": S, "c": (0.1, 0.1)})[0] == "SHORT-LEG"
    assert intl.h2_label({"a": S, "b": (0.5999, 0.9), "c": (0.1, 0.1)})[0] == "NOT SHORT-LEG"
    assert intl.h2_label({"a": S, "b": (0.9, np.nan), "c": (0.1, 0.1)})[0] == "NOT SHORT-LEG"
    assert intl.h2_label({"a": S, "b": S, "c": S})[1:] == (3, (3, 3))


def test_h3_boundaries_and_non_positive_denominator():
    assert intl.h3_region(3.0, 1.0) == "no" and intl.h3_region(3.0001, 1.0) == "yes"          # strictly greater than 3x
    assert intl.h3_region(5.0, 0.0) == "n/a" and intl.h3_region(5.0, -1.0) == "n/a" and intl.h3_region(np.nan, 1.0) == "n/a"
    assert intl.h3_label({"a": (4.0, 1.0), "b": (4.0, 1.0), "c": (1.0, 1.0)})[0] == "GLOBAL EPISODE"
    assert intl.h3_label({"a": (4.0, 1.0), "b": (3.0, 1.0), "c": (9.0, 1.0)})[0] == "GLOBAL EPISODE"
    assert intl.h3_label({"a": (4.0, 1.0), "b": (3.0, 1.0), "c": (9.0, -1.0)})[0] == "US-SPECIFIC"
    lab, n, v = intl.h3_label({"a": (4.0, 1.0), "b": (4.0, 0.0), "c": (4.0, -2.0)})
    assert lab == "US-SPECIFIC" and n == 1 and v["b"] == v["c"] == "n/a"
    idx = [2014 * 12 + 11] + list(range(2015 * 12, 2023 * 12))                                   # Dec 2014 is excluded (since 2015-01)
    s = pd.Series(1.0, index=idx)
    s[(s.index >= 2021 * 12) & (s.index <= 2022 * 12 + 11)] = 5.0
    s.iloc[0] = 100.0
    assert intl.h3_means(s) == (5.0, 1.0)


# ---------------------------------------------------------------- dtypes, determinism, end-to-end

def test_standardise_converts_nullable_dtypes():
    raw = synth_year(2014, ["JPN"], 10, 1, _beta, nan_frac=0.5).iloc[:10]
    d = intl.standardise(raw)
    assert d["id"].dtype == np.int64 and d["mi"].dtype == np.int64
    assert all(d[c].dtype == np.float64 for c in ["ret", "me", "gp_at", "bidaskhl_21d"])
    assert d["gp_at"].isna().any() and not any(d[c].dtype == object for c in ["ret", "me", "gp_at"])
    us = intl.standardise(synth_year(2014, ["USA"], 10, 1, _beta, us=True).iloc[:10], us=True)
    assert set(us["excntry"]) == {"USA"} and us["id"].dtype == np.int64


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    write_synth(root, YEARS, YEARS, CTRY, beta=_beta, scale={"JPN": 1e4})
    return root


def test_load_and_series_are_deterministic_across_jobs(synth_root):
    s1, i1 = intl.load_all(str(synth_root / "jkp_intl"), str(synth_root / "jkp_us"), jobs=1)
    s3, i3 = intl.load_all(str(synth_root / "jkp_intl"), str(synth_root / "jkp_us"), jobs=3)
    pd.testing.assert_frame_equal(s1, s3)
    assert set(s1["region"].astype(str)) == {"US", "Developed Europe", "Japan", "Other developed"}
    assert i1["cm"]["kept"].all() and (i1["cm"]["n"] >= 100).all()
    r1, r3 = intl.run_series(s1, jobs=1), intl.run_series(s1, jobs=3)
    assert r1.keys() == r3.keys() and len(r1) == 4 * 16
    for k in r1:
        pd.testing.assert_frame_equal(r1[k], r3[k])
    # planted signal: Europe's IC collapses after 2014, Japan's (scaled 1e4, ranked within country) does not
    eu, jp = r1[("Developed Europe", "jkp_all13")]["ic"], r1[("Japan", "jkp_all13")]["ic"]
    assert intl.h1_ratio(eu) < 0.5 and intl.h1_ratio(jp) > 0.6


def test_no_trial_logging_anywhere():
    for f in ("src/qap/intl.py", "scripts/run_intl.py"):
        text = (ROOT / f).read_text()
        assert "log_trial" not in text and "qap.trials" not in text and "trials.jsonl" not in text, f


def test_end_to_end_script_main_path(synth_root, tmp_path, monkeypatch, capsys):
    import qap.trials
    def boom(*a, **k):
        raise AssertionError("a trial was logged")
    monkeypatch.setattr(qap.trials, "log_trial", boom)
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for sub in ("jkp_intl", "jkp_us"):
        (raw / sub).symlink_to(synth_root / sub)
    spec = importlib.util.spec_from_file_location("run_intl_script", ROOT / "scripts" / "run_intl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    labels = mod.main(jobs=2)
    rep = (tmp_path / "data" / "intl_report.txt").read_text()
    assert rep.startswith("DESCRIPTIVE: cross-market calibration, not a tradability test; no trials logged")
    assert rep.index("== CHECKS") < rep.index("== Developed Europe | dev") < rep.index("THE THREE LABELLED QUESTIONS")
    assert "fake_col_a" in rep and "COUNTRY DROPPED" not in rep
    assert set(labels) == {"H1", "H2", "H3"} and f"LABELS: H1 = {labels['H1']}" in rep
    assert labels["H1"] != "GLOBAL DECAY"                            # only Europe decays (1 of 3); MIXED or PERSISTS depending on noise
    assert not (tmp_path / "data" / "trials.jsonl").exists()
    # the report is also written when the run fails (try/finally)
    (raw / "jkp_us").unlink()
    with pytest.raises(SystemExit):
        mod.main(jobs=1)
    assert (tmp_path / "data" / "intl_report.txt").exists()


def test_all_countries_dropped_returns_empty_frame():
    slim, info = intl.score_frame(_base(50, "JPN"))
    assert slim.empty and list(slim.columns) == intl.BASE_COLS + intl.SIGNALS and not info["cm"]["kept"].any()


def test_checks_separate_thin_months_from_missing_lead_and_flag_missing_spreads():
    parts = []
    for mi in (24000, 24001, 24002):                                  # thin early run: 60 rows
        d = _base(60, "NZL", mi=mi)
        parts.append(d)
    for mi in (24003, 24004):                                         # healthy months, no spread data
        parts.append(_base(120, "NZL", mi=mi))
    last = _base(120, "NZL", mi=24005)                                # 120 listed rows but no next-month return
    last["ret"] = np.nan
    parts.append(last)
    slim, info = intl.score_frame(pd.concat(parts, ignore_index=True))
    text = "\n".join(intl.format_checks(info, {"jkp_intl": []}))
    line = [l for l in text.splitlines() if l.startswith("  NZL")][0]
    assert "kept 2 months" in line and "dropped thin (< 100 listed rows) 3" in line and "no next-month return / me) 1" in line
    assert "(2000-01..2000-03)" in line
    assert "NO bidaskhl_21d at all" in text and ": 2 of 2" in text
