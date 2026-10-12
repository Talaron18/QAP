"""Trade-journal analyzer tests. All data is SYNTHETIC (made-up tickers/prices in the real formats); nothing real here."""
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from qap import journal as j

LOG = """Week 1
* 2025-09-23 00:30:05: Bought 20 shares of Acme Widgets Inc. (ACME) at $289.98 per share.
* 2025-09-23 00:32:50: Bought 400 shares of 2x Short ZZQX ETF-Foo (ZQS) at $9.10 per share.
- 2025-09-23 21:40:00: Sold 1,000 shares of Zeta Quux Corp (ZZQX) at $1,234.50 per share. fee $2.50
   *   2025-09-24 22:00: Bought 5 shares of Acme Widgets Inc. (ACME) at 290.00 per share. commission $1.25
some random chatter line
"""


def mk(rows):
    """rows: (datetime, action, ticker, qty, price[, fee[, name]]) -> frame with session dates (naive times)."""
    recs = []
    for i, r in enumerate(rows):
        recs.append(dict(ts=pd.Timestamp(r[0]), action=r[1], ticker=r[2], qty=float(r[3]), price=float(r[4]),
                         fee=float(r[5]) if len(r) > 5 else 0.0, name=r[6] if len(r) > 6 else "", src="t", line_no=i + 1))
    df = pd.DataFrame(recs).sort_values("ts", kind="stable").reset_index(drop=True)
    df["typo"] = False
    return j.assign_sessions(df, None)


# ------------------------------------------------------------------------------------------------ parsing
def test_parse_log_variants_and_ignored():
    fills, ign = j.parse_log_text(LOG)
    assert len(fills) == 4 and ign == 2  # "Week 1" header + chatter
    a, b, c, d = fills
    assert (a["ticker"], a["action"], a["qty"], a["price"], a["fee"]) == ("ACME", "buy", 20, 289.98, 0.0)
    assert a["ts"] == pd.Timestamp("2025-09-23 00:30:05") and a["name"] == "Acme Widgets Inc."
    assert b["name"] == "2x Short ZZQX ETF-Foo" and b["ticker"] == "ZQS"
    assert (c["action"], c["qty"], c["price"], c["fee"]) == ("sell", 1000, 1234.5, 2.5)
    assert d["fee"] == 1.25 and d["price"] == 290.0 and d["ts"] == pd.Timestamp("2025-09-24 22:00")


def test_parse_number_variants():
    assert j.parse_number("$1,234.50") == 1234.5
    assert j.parse_number("+20.17%") == 20.17
    assert j.parse_number("(1,110.00)") == -1110
    assert j.parse_number("-$5") == -5
    assert math.isnan(j.parse_number("abc"))


def test_leverage_and_underlying():
    assert j.parse_leverage("2x Short SNDK ETF-Tradr") == (-2.0, "SNDK")
    assert j.parse_leverage("2x Short Hynix ETF-Defiance") == (-2.0, "HYNIX")
    assert j.parse_leverage("3x Long FOOB ETF") == (3.0, "FOOB")
    assert j.parse_leverage("2X bear QQQQ Fund")[0] == -2.0
    assert j.parse_leverage("Acme Widgets Inc.") == (1.0, None)
    info = j.instrument_info({"ZQS": "2x Short ZZQX ETF", "ACME": "Acme Widgets Inc.", "FBL": "3x Long FOOB ETF"})
    assert info["ZQS"]["underlying"] == "ZZQX" and info["ZQS"]["leverage"] == -2
    assert info["ACME"]["underlying"] == "ACME" and info["ACME"]["leverage"] == 1
    info = j.instrument_info({"ZQS": "2x Short ZZQX ETF"}, meta={"ZQS": ("OTHR", -3.0)})
    assert info["ZQS"]["underlying"] == "OTHR" and info["ZQS"]["leverage"] == -3


def test_meta_csv(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("ticker,underlying,leverage\nzqs,zzqx,-2\n")
    assert j.load_meta(p) == {"ZQS": ("ZZQX", -2.0)}
    assert j.load_meta(tmp_path / "nope.csv") == {}


# ------------------------------------------------------------------------------------------------ positions
def test_average_cost_partial_exit_and_fees():
    df = mk([("2025-01-02 10:00", "buy", "AAA", 10, 10.0, 1.0), ("2025-01-03 10:00", "buy", "AAA", 10, 20.0),
             ("2025-01-06 10:00", "sell", "AAA", 5, 18.0)])
    r = j.process_fills(df, {})
    assert r["roundtrips"] == []
    p = r["positions"]["AAA"]
    assert p["qty"] == 15 and p["avg"] == pytest.approx(15.0)
    df = mk([("2025-01-02 10:00", "buy", "AAA", 10, 10.0, 1.0), ("2025-01-03 10:00", "buy", "AAA", 10, 20.0),
             ("2025-01-06 10:00", "sell", "AAA", 5, 18.0), ("2025-01-10 10:00", "sell", "AAA", 15, 12.0, 2.0)])
    r = j.process_fills(df, {"AAA": dict(leverage=2.0, underlying="AAA")})
    assert len(r["roundtrips"]) == 1 and r["positions"] == {}
    rt = r["roundtrips"][0]
    assert rt["holding_days"] == 8 and rt["avg_entry"] == 15 and rt["avg_exit"] == pytest.approx((90 + 180) / 20)
    assert rt["pnl_usd"] == pytest.approx(5 * 3 + 15 * -3 - 3.0)
    assert rt["ret_pct"] == pytest.approx(100 * (-33.0) / 300.0) and rt["leverage"] == 2
    assert j.recompute_cash(r, 1000) == pytest.approx(1000 - 300 + 90 + 180 - 3.0)


def test_reentry_gives_two_round_trips_and_open_since():
    df = mk([("2025-02-03 10:00", "buy", "AAA", 10, 10.0), ("2025-02-05 10:00", "sell", "AAA", 10, 11.0),
             ("2025-02-10 10:00", "buy", "AAA", 4, 12.0), ("2025-02-12 10:00", "sell", "AAA", 2, 13.0),
             ("2025-02-14 10:00", "buy", "AAA", 1, 12.0)])
    r = j.process_fills(df, {})
    assert [x["pnl_usd"] for x in r["roundtrips"]] == [10.0]
    p = r["positions"]["AAA"]
    assert p["qty"] == 3 and p["avg"] == pytest.approx(12.0) and p["open_since"] == date(2025, 2, 10)


def test_short_cross_through_zero_splits_fill():
    df = mk([("2025-02-03 10:00", "buy", "AAA", 10, 10.0), ("2025-02-04 10:00", "sell", "AAA", 15, 12.0, 3.0)])
    r = j.process_fills(df, {})
    assert r["roundtrips"][0]["pnl_usd"] == pytest.approx(20 - 2.0)  # fee split 10/15 to the close
    assert r["positions"]["AAA"]["qty"] == -5 and r["positions"]["AAA"]["avg"] == 12.0


def test_fifo_short_vs_long_term_and_tax():
    df = mk([("2024-01-02 10:00", "buy", "AAA", 10, 10.0), ("2025-06-02 10:00", "buy", "AAA", 10, 20.0),
             ("2025-09-01 10:00", "sell", "AAA", 15, 30.0, 3.0)])
    pcs, unm = j.fifo_pieces(df)
    assert unm == 0 and len(pcs) == 2
    lt, st = pcs
    assert (lt["term"], lt["qty"], lt["days"]) == ("LT", 10, 608)
    assert (st["term"], st["qty"], st["days"]) == ("ST", 5, 91)
    assert lt["pnl"] == pytest.approx(10 * 30 - 2.0 - 100) and st["pnl"] == pytest.approx(5 * 30 - 1.0 - 100)
    tv = j.tax_view(pcs, 0.4, 0.2)
    assert tv["tax"] == pytest.approx(0.2 * lt["pnl"] + 0.4 * st["pnl"])
    assert tv["aftertax"] == pytest.approx(tv["pretax"] - tv["tax"])


def test_tax_loss_offset_only_within_period():
    p = lambda pnl, term: dict(pnl=pnl, term=term)
    tv = j.tax_view([p(-100, "ST"), p(300, "LT")], 0.4, 0.2)
    assert tv["tax"] == pytest.approx(200 * 0.2)
    tv = j.tax_view([p(-100, "LT"), p(50, "ST")], 0.4, 0.2)
    assert tv["tax"] == 0.0 and tv["pretax"] == -50  # residual loss is not refunded
    tv = j.tax_view([p(100, "ST"), p(-30, "ST")], 0.4, 0.2)
    assert tv["tax"] == pytest.approx(28.0)


def test_wash_sale_warning_same_underlying():
    info = {"ZQS": dict(underlying="ZZQX", leverage=-2.0), "ZZQX": dict(underlying="ZZQX", leverage=1.0)}
    df = mk([("2025-03-03 10:00", "buy", "ZQS", 10, 10.0), ("2025-03-10 10:00", "sell", "ZQS", 10, 8.0),
             ("2025-03-20 10:00", "buy", "ZZQX", 1, 5.0)])
    w = j.wash_sale_warnings(df, j.fifo_pieces(df)[0], info)
    assert len(w) == 1 and "WASH SALE" in w[0]
    df = mk([("2025-03-03 10:00", "buy", "ZQS", 10, 10.0), ("2025-03-10 10:00", "sell", "ZQS", 10, 8.0),
             ("2025-05-20 10:00", "buy", "ZZQX", 1, 5.0)])
    assert j.wash_sale_warnings(df, j.fifo_pieces(df)[0], info) == []


# ------------------------------------------------------------------------------------------------ dates / time zones
def test_year_typo_detection_and_fix():
    rows = [(f"2025-09-2{d} 10:00", "buy", "AAA", 1, 1.0) for d in (2, 3, 4, 5)] + [("2026-09-22 10:00", "buy", "BBB", 1, 1.0)]
    df = mk(rows)
    ty = j.detect_year_typos(df, today=date(2025, 12, 31))
    assert len(ty) == 1 and ty[0]["ticker"] == "BBB" and ty[0]["proposed"] == pd.Timestamp("2025-09-22 10:00")
    fixed = j.apply_year_fixes(df, ty)
    assert fixed["ts"].max() == pd.Timestamp("2025-09-25 10:00") and df["ts"].max().year == 2026
    assert j.detect_year_typos(fixed, today=date(2025, 12, 31)) == []


def test_timezone_session_dates_and_dst():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-09-23 00:30", "2025-09-23 21:30", "2025-12-23 00:30", "2026-01-15 21:30"])})
    out = j.assign_sessions(df, "Asia/Shanghai")
    assert list(out["ts_et"]) == [pd.Timestamp("2025-09-22 12:30"), pd.Timestamp("2025-09-23 09:30"),
                                  pd.Timestamp("2025-12-22 11:30"), pd.Timestamp("2026-01-15 08:30")]
    assert out["sdate"].iloc[0] == date(2025, 9, 22) and out["sdate"].iloc[2] == date(2025, 12, 22)
    assert out["sdate"].iloc[1] == date(2025, 9, 23)
    assert list(j.assign_sessions(df, None)["sdate"])[0] == date(2025, 9, 23)


def test_outside_rth_hint():
    assert j.outside_rth(pd.DataFrame({"ts": pd.to_datetime(["2025-09-23 00:30", "2025-09-23 22:10"])}))
    assert not j.outside_rth(pd.DataFrame({"ts": pd.to_datetime(["2025-09-23 00:30", "2025-09-23 10:10"])}))


# ------------------------------------------------------------------------------------------------ snapshots
SNAP = """Week 2 (2025-10-03)
| Security Name | Quantity | Cost Price | Current Price | Floating P/L | P/L Ratio (%) |
|---|---|---|---|---|---|
| ACME (Acme Widgets) | 20 | 100.00 | 110.00 | 200.00 | +10.00% |
| ZQS (2x Short ZZQX) | 1,000 | 5.00 | 4.50 | (500.00) | -10.00% |
Account Summary
* Total Market Value: $6,700.00
* Remaining Cash: $ 28,000.00
* Total Assets: $34,700.00
* Total Return Rate: -0.86%
"""
SNAP_FILLS = [("2025-09-30 10:00", "buy", "ACME", 20, 100.0), ("2025-09-30 10:05", "buy", "ZQS", 1000, 5.0)]


def test_snapshot_parser():
    (s,) = j.parse_snapshots(SNAP)
    assert s["week"] == 2 and s["date"] == date(2025, 10, 3) and len(s["rows"]) == 2
    assert s["rows"][1]["ticker"] == "ZQS" and s["rows"][1]["qty"] == 1000 and s["rows"][1]["pl"] == -500
    assert s["rows"][0]["ratio"] == 10.0
    assert s["summary"] == dict(market_value=6700.0, cash=28000.0, total_assets=34700.0, return_rate=-0.86)


def test_multiple_snapshots_in_one_file():
    snaps = j.parse_snapshots(SNAP + "\nWeek 3\n" + SNAP.split("\n", 1)[1])
    assert [s["week"] for s in snaps] == [2, 3] and all(len(s["rows"]) == 2 for s in snaps)


def _recon(snap_text, fills=SNAP_FILLS, start=35000.0):
    (s,) = j.parse_snapshots(snap_text)
    df = mk(fills)
    r = j.process_fills(df, {})
    return j.reconcile(s, r["positions"], r, start)


def test_reconcile_clean_snapshot():
    out = _recon(SNAP)
    assert [o for o in out if o.startswith("MISMATCH")] == []


def test_reconcile_catches_wrong_sign_ratio_and_cost_price():
    bad = SNAP.replace("-10.00%", "+10.00%").replace("| 20 | 100.00 |", "| 20 | 100.50 |")
    out = _recon(bad)
    assert "MISMATCH ZQS P/L ratio: snapshot +10.00% vs recomputed -10.00%" in out
    assert any(o.startswith("MISMATCH ACME cost price: snapshot 100.5000 vs recomputed 100.0000") for o in out)
    # within tolerance: 0.01 cost difference is not flagged
    assert not any("cost price" in o for o in _recon(SNAP.replace("| 20 | 100.00 |", "| 20 | 100.01 |")))


def test_reconcile_cash_total_assets_and_return_rate():
    bad = SNAP.replace("28,000.00", "27,900.00")  # 100 USD lower than recomputed (after-fee slippage)
    out = _recon(bad)
    assert any("MISMATCH ACCOUNT remaining cash: snapshot 27900.00 vs recomputed 28000.00" in o for o in out)
    assert any("implied fees/slippage" in o and "-100.00" in o for o in out)
    assert any("total assets (market value + cash)" in o for o in out)
    ok = _recon(SNAP.replace("28,000.00", "27,990.00").replace("34,700.00", "34,690.00").replace("-0.86%", "-0.89%"))
    assert [o for o in ok if o.startswith("MISMATCH")] == []  # 10 USD < 25 USD cash tolerance
    out = _recon(SNAP.replace("-0.86%", "+3.00%"))
    assert any("total return rate: snapshot +3.00% vs recomputed -0.86%" in o for o in out)


def test_reconcile_missing_and_extra_positions():
    out = _recon(SNAP, fills=SNAP_FILLS + [("2025-09-30 10:10", "buy", "NEWC", 3, 10.0)])
    assert any(o.startswith("MISMATCH NEWC quantity") for o in out)


# ------------------------------------------------------------------------------------------------ drag / exposure / risk
def test_drag_and_breakeven_hand_values():
    T = 21 / 252
    assert j.drag_log(-2, 0.7, T) == pytest.approx(6 * 0.49 * T / 2) and round(100 * j.drag_log(-2, 0.7, T), 1) == 12.2
    assert round(100 * j.breakeven_move(-2, 0.7, T), 1) == -5.9
    assert j.drag_log(1, 0.7, T) == 0 and j.breakeven_move(1, 0.7, T) == 0
    # consistency: at the break-even underlying move the product's log return is exactly zero
    L, s = 3.0, 0.8
    x = math.log1p(j.breakeven_move(L, s, T))
    assert L * x - j.drag_log(L, s, T) == pytest.approx(0)


def test_exposure_aggregation_by_underlying():
    pos = [dict(ticker="ZZQX", mv=2000.0, leverage=1.0, underlying="ZZQX"),
           dict(ticker="ZQS", mv=1000.0, leverage=-2.0, underlying="ZZQX"),
           dict(ticker="ACME", mv=3000.0, leverage=1.0, underlying="ACME"),
           dict(ticker="FBL", mv=500.0, leverage=3.0, underlying="FOOB")]
    e = j.exposure_summary(pos, 10000.0)
    assert e["by_underlying"] == pytest.approx({"ZZQX": 0.0, "ACME": 0.3, "FOOB": 0.15})
    assert e["net"] == pytest.approx(0.45) and e["gross"] == pytest.approx(0.45)
    assert e["long"] == pytest.approx(0.45) and e["short"] == 0
    assert e["leveraged_share"] == pytest.approx(0.15) and e["invested"] == pytest.approx(0.65)
    assert e["largest"] == pytest.approx(0.3) and e["top2_share"] == pytest.approx(1.0)
    e2 = j.exposure_summary(pos[1:2], 10000.0)
    assert e2["short"] == pytest.approx(0.2) and e2["net"] == pytest.approx(-0.2)


def test_book_risk_two_asset_hand_case():
    e = {"A": 0.6, "B": -0.3}
    v = {"A": 0.5, "B": 0.8}
    var = 0.36 * 0.25 + 0.09 * 0.64 + 2 * 0.6 * -0.3 * 0.5 * 0.5 * 0.8
    r = j.book_risk(e, v, 0.5, 10000.0)
    assert r["annual_sd"] == pytest.approx(math.sqrt(var))
    assert r["daily_sd"] == pytest.approx(math.sqrt(var / 252)) and r["weekly_sd"] == pytest.approx(math.sqrt(var * 5 / 252))
    assert r["usd_per_day"] == pytest.approx(10000 * math.sqrt(var / 252))
    assert r["loss95_week"] == pytest.approx(1.6448536 * math.sqrt(var * 5 / 252), rel=1e-5)
    assert r["ratio_to_bench"] == pytest.approx(math.sqrt(var) / 0.16)
    one = j.book_risk({"A": 1.0}, {"A": 0.4}, 0.9, 1000.0)  # single asset: correlation irrelevant
    assert one["annual_sd"] == pytest.approx(0.4)


# ------------------------------------------------------------------------------------------------ luck vs skill
def test_sign_test_matches_scipy():
    for n in range(1, 16):
        for k in range(n + 1):
            assert j.sign_test_p(k, n) == pytest.approx(stats.binomtest(k, n, 0.5).pvalue, abs=1e-12)


def test_mean_t_and_power():
    x = [0.1, -0.05, 0.2, 0.0, 0.05]
    m = j.mean_t(x)
    assert m["t"] == pytest.approx(stats.ttest_1samp(x, 0).statistic) and m["p"] == pytest.approx(stats.ttest_1samp(x, 0).pvalue)
    assert j.weeks_needed(0.10, 0.01) == pytest.approx(400) and j.weeks_needed(0.10, 0.05) == pytest.approx(16)
    assert math.isnan(j.mean_t([0.1])["t"])


# ------------------------------------------------------------------------------------------------ CLI
CLI_LOG = """Week 1
* 2025-09-23 22:30:05: Bought 20 shares of Acme Widgets Inc. (ACME) at $100.00 per share.
* 2025-09-23 22:32:50: Bought 1,000 shares of 2x Short ZZQX ETF-Foo (ZQS) at $5.00 per share.
* 2025-09-24 23:00:00: Bought 10 shares of Quux Corp (QUUX) at $50.00 per share.
* 2025-09-26 01:00:00: Sold 10 shares of Quux Corp (QUUX) at $55.00 per share.
* 2026-09-22 22:40:00: Bought 1 shares of Acme Widgets Inc. (ACME) at $100.00 per share.
"""
CLI_SNAP = """Week 1 (2025-09-26)
| Security Name | Quantity | Cost Price | Current Price | Floating P/L | P/L Ratio (%) |
|---|---|---|---|---|---|
| ACME (Acme Widgets) | 20 | 100.00 | 110.00 | 200.00 | +10.00% |
| ZQS (2x Short ZZQX) | 1,000 | 5.00 | 4.50 | -500.00 | +10.00% |
Account Summary
Total Market Value: $6,700.00
Remaining Cash: $27,950.00
Total Assets: $34,650.00
Total Return Rate: -1.00%
"""


def test_cli_end_to_end_and_idempotent(tmp_path, capsys):
    (tmp_path / "log.txt").write_text(CLI_LOG)
    (tmp_path / "snap.txt").write_text(CLI_SNAP)
    rep, rt = tmp_path / "out" / "report.txt", tmp_path / "out" / "rt.csv"
    argv = [str(tmp_path / "*.txt"), "--log-tz", "Asia/Shanghai", "--fix-years", "--vol", "ZZQX=0.7", "--vol", "ACME=0.4",
            "--corr", "0.3", "--today", "2025-12-31", "--prices", "ACME=111", "--report", str(rep), "--roundtrips", str(rt),
            "--meta", str(tmp_path / "none.csv")]
    assert j.main(argv) == 0
    out1 = capsys.readouterr().out
    text1, csv1 = rep.read_text(), rt.read_text()
    assert out1.strip() == text1.strip()
    for must in ("1. PARSE CHECKS", "suspected year typo", "APPLIED", "MISMATCH ZQS P/L ratio: snapshot +10.00% vs recomputed -10.00%",
                 "log(ETF_T / ETF_0) = L * log(S_T / S_0) - (L^2 - L) * sigma^2 * T / 2", "3. EXPOSURE", "5. RISK SNAPSHOT",
                 "ASSUMPTION", "6. LUCK VS SKILL", "7. WARNINGS", "too small", "QUUX"):
        assert must in text1, must
    df = pd.read_csv(rt)
    assert list(df["ticker"]) == ["QUUX"] and df["pnl_usd"].iloc[0] == pytest.approx(50.0)
    assert df["holding_days"].iloc[0] == 1  # 23:00 Shanghai 9/24 = 11:00 ET 9/24 ; 01:00 on 9/26 = 13:00 ET 9/25
    assert j.main(argv) == 0
    capsys.readouterr()
    assert rep.read_text() == text1 and rt.read_text() == csv1


def test_cli_hint_without_tz_and_unfixed_years(tmp_path):
    (tmp_path / "log.txt").write_text(CLI_LOG)
    rep, rt = tmp_path / "r.txt", tmp_path / "rt.csv"
    j.main([str(tmp_path / "log.txt"), "--report", str(rep), "--roundtrips", str(rt), "--meta", str(tmp_path / "x.csv"),
            "--today", "2025-12-31"])
    t = rep.read_text()
    assert "pass --log-tz" in t and "not applied; use --fix-years" in t and "UNRELIABLE" in t
    assert "2025-09-23" in t  # still analysed with dates as written


def test_cli_fills_csv(tmp_path):
    (tmp_path / "f.csv").write_text("datetime,action,ticker,qty,price,fee,name\n"
                                    "2025-03-03 10:00:00,buy,AAA,10,10.0,0.5,2x Long ZZQX Fund\n"
                                    "2025-03-04 11:00:00,SELL,AAA,10,11.0,0.5,\n")
    rep, rt = tmp_path / "r.txt", tmp_path / "rt.csv"
    j.main(["--fills-csv", str(tmp_path / "f.csv"), "--report", str(rep), "--roundtrips", str(rt), "--meta", "none"])
    df = pd.read_csv(rt)
    assert df["pnl_usd"].iloc[0] == pytest.approx(9.0) and df["leverage"].iloc[0] == 2


# ------------------------------------------------------------------------------------------------ undated snapshots / ordering
WEEKLOG = """Week 1
* 2025-09-23 22:30:00: Bought 20 shares of Acme Widgets Inc. (ACME) at $100.00 per share.
* 2025-09-24 22:30:00: Bought 1,000 shares of 2x Short ZZQX ETF-Foo (ZQS) at $5.00 per share.
Week 2
* 2025-09-30 22:30:00: Bought 10 shares of Acme Widgets Inc. (ACME) at $110.00 per share.
"""
WEEKSNAP = """Week 1
| Security Name | Quantity | Cost Price | Current Price | Floating P/L | P/L Ratio (%) |
|---|---|---|---|---|---|
| ACME (Acme Widgets) | 20 | 100.50 | 105.00 | 100.00 | +5.00% |
| ZQS (2x Short ZZQX) | 1,000 | 5.00 | 4.50 | -500.00 | +10.00% |
Account Summary
Total Assets: $34,000.00
Week 2
| Security Name | Quantity | Cost Price | Current Price | Floating P/L | P/L Ratio (%) |
|---|---|---|---|---|---|
| ACME (Acme Widgets) | 30 | 103.33 | 105.00 | 50.00 | +1.61% |
| ZQS (2x Short ZZQX) | 1,000 | 5.00 | 4.50 | -500.00 | -10.00% |
Total Assets: $34,100.00
Week 3
Total Assets: $34,300.00
"""


def _fills_df(text):
    f, _ = j.parse_log_text(text)
    return j.assign_sessions(j.fills_frame(f), None)


def test_log_week_tags_and_cutoff_for_undated_snapshots():
    df = _fills_df(WEEKLOG)
    assert list(df["week"]) == [1, 1, 2]
    s1, s2, s3 = j.parse_snapshots(WEEKSNAP)
    assert (s1["date"], s2["date"]) == (None, None)
    sub, how = j.snapshot_cutoff(s1, df, False)
    assert len(sub) == 2 and "week tags" in how
    sub, _ = j.snapshot_cutoff(s2, df, False)
    assert len(sub) == 3


def test_quantity_match_cutoff_without_week_tags():
    df = _fills_df(WEEKLOG.replace("Week 1\n", "").replace("Week 2\n", ""))
    s1, s2, _ = j.parse_snapshots(WEEKSNAP)
    sub, how = j.snapshot_cutoff(s1, df, False)
    assert len(sub) == 2 and "quantity match" in how
    assert len(j.snapshot_cutoff(s2, df, False)[0]) == 3
    bad = dict(s1, rows=[dict(s1["rows"][0], qty=7.0)])
    assert j.snapshot_cutoff(bad, df, False)[0] is None  # skipped, not silently wrong
    assert j.snapshot_cutoff(bad, df, True)[0] is df


def test_undated_early_snapshot_errors_are_caught_in_cli(tmp_path):
    (tmp_path / "log.txt").write_text(WEEKLOG)
    (tmp_path / "snap.txt").write_text(WEEKSNAP)
    rep = tmp_path / "r.txt"
    j.main([str(tmp_path / "log.txt"), str(tmp_path / "snap.txt"), "--report", str(rep), "--roundtrips", str(tmp_path / "rt.csv"),
            "--meta", "none"])
    t = rep.read_text()
    assert "MISMATCH ZQS P/L ratio: snapshot +10.00% vs recomputed -10.00%" in t  # Week 1 (not the last snapshot)
    assert "MISMATCH ACME cost price: snapshot 100.50 " in t.replace("100.5000", "100.50 ")
    assert "not reconciled" not in t and "cutoff by log week tags (week <= 1)" in t


def test_snapshot_ordering_not_by_file_name():
    mk_s = lambda w, d=None: dict(week=w, date=d, rows=[], summary={})
    assert [x["week"] for x in j.order_snapshots([mk_s(10), mk_s(2), mk_s(3)])] == [2, 3, 10]
    d = lambda k: date(2025, 10, k)
    assert [x["week"] for x in j.order_snapshots([mk_s(1, d(9)), mk_s(2, d(2))])] == [2, 1]
    assert [x["week"] for x in j.order_snapshots([mk_s(None), mk_s(1)])] == [None, 1]  # mixed -> parse order


def test_fix_years_clears_unreliable_flag(tmp_path):
    log = ("* 2025-09-23 22:30:00: Bought 10 shares of 2x Short ZZQX ETF-Foo (ZQS) at $5.00 per share.\n"
           "* 2025-09-24 22:30:00: Bought 10 shares of Acme Widgets Inc. (ACME) at $5.00 per share.\n"
           "* 2025-09-25 22:30:00: Bought 10 shares of Acme Widgets Inc. (ACME) at $5.00 per share.\n"
           "* 2026-09-26 22:30:00: Bought 10 shares of 2x Short ZZQX ETF-Foo (ZQS) at $5.00 per share.\n")
    (tmp_path / "log.txt").write_text(log)
    texts = {}
    for fix in (False, True):
        rep = tmp_path / f"r{fix}.txt"
        j.main([str(tmp_path / "log.txt"), "--report", str(rep), "--roundtrips", str(tmp_path / "rt.csv"), "--meta", "none", "--today", "2025-12-31"]
               + (["--fix-years"] if fix else []))
        texts[fix] = rep.read_text()
    assert "open?" in texts[False] and "UNRELIABLE" in texts[False]
    assert "open?" not in texts[True] and "UNRELIABLE" not in texts[True]


# ------------------------------------------------------------------------------------------------ year-typo direction / horizons
SEVEN = [("2026-09-22 22:40", "buy", "AAA", 1, 1.0)] + [(f"2025-09-2{d} 22:40", "buy", "BBB", 1, 1.0) for d in (3, 4, 5, 6, 7, 8)]
TODAY = date(2026, 10, 12)


def test_year_proposal_uses_latest_year_not_after_today():
    df = mk(SEVEN)  # one 2026 line (listed first in the file), six 2025 lines, same week
    ty = j.detect_year_typos(df, today=TODAY)
    assert len(ty) == 6 and all(t["ticker"] == "BBB" and t["proposed"].year == 2026 for t in ty)
    assert all(t["alt"].year == 2025 and not t["future"] for t in ty)
    fixed = j.apply_year_fixes(df, ty)
    assert fixed["ts"].dt.year.eq(2026).all()
    # with an earlier "today" the 2026 line is the typo instead (old direction)
    ty = j.detect_year_typos(df, today=date(2025, 12, 31))
    assert [t["ticker"] for t in ty] == ["AAA"] and ty[0]["proposed"].year == 2025
    # --assume-year overrides; future proposals are marked
    ty = j.detect_year_typos(df, today=TODAY, assume_year=2027)
    assert len(ty) == 7 and all(t["future"] for t in ty) and ty[0]["proposed"].year == 2027


def test_cli_assume_year_implies_apply_and_prints_alternative(tmp_path):
    lines = [f"* {d} 22:30:00: Bought 1 shares of Acme Widgets Inc. (ACME) at $10.00 per share." for d in
             ["2026-09-22"] + [f"2025-09-2{k}" for k in (3, 4, 5, 6, 7, 8)]]
    (tmp_path / "log.txt").write_text("\n".join(lines) + "\n")
    rep = tmp_path / "r.txt"
    base = [str(tmp_path / "log.txt"), "--report", str(rep), "--roundtrips", str(tmp_path / "rt.csv"), "--meta", "none",
            "--today", "2026-10-12"]
    j.main(base)
    t = rep.read_text()
    assert "flagged fills: 6 of 7" in t and "corrected range would be 2026-09-22 .. 2026-09-28" in t
    assert "alternative (median-year) correction: 2025-09-23" in t and "not applied" in t
    j.main(base + ["--assume-year", "2026"])
    t = rep.read_text()
    assert "APPLIED" in t and "session dates (after correction): 2026-09-22 .. 2026-09-28" in t


def test_open_horizon_flagged_when_last_date_seen_is_flagged(tmp_path):
    # ZQS (clean 2025 dates) is open; the last date seen comes from a typo'd ACME fill dated 2026
    log = ("* 2025-09-23 22:30:00: Bought 10 shares of 2x Short ZZQX ETF-Foo (ZQS) at $5.00 per share.\n"
           "* 2025-09-24 22:30:00: Bought 10 shares of Acme Widgets Inc. (ACME) at $5.00 per share.\n"
           "* 2025-09-25 22:30:00: Bought 10 shares of Acme Widgets Inc. (ACME) at $5.00 per share.\n"
           "* 2026-09-26 22:30:00: Bought 10 shares of Acme Widgets Inc. (ACME) at $5.00 per share.\n")
    (tmp_path / "log.txt").write_text(log)
    rep = tmp_path / "r.txt"
    args = [str(tmp_path / "log.txt"), "--report", str(rep), "--roundtrips", str(tmp_path / "rt.csv"), "--meta", "none",
            "--today", "2025-12-31"]
    j.main(args)
    t = rep.read_text()
    assert "open?" in t and "dates suspect" in t and "ZQS (open?)" in t
    j.main(args + ["--fix-years"])
    t = rep.read_text()
    assert "open?" not in t and "dates suspect" not in t


def test_closed_round_trip_with_flagged_fill_gets_warning(tmp_path):
    log = ("* 2025-09-23 22:30:00: Bought 10 shares of Acme Widgets Inc. (ACME) at $5.00 per share.\n"
           "* 2025-09-24 22:30:00: Bought 10 shares of Quux Corp (QUUX) at $5.00 per share.\n"
           "* 2025-09-25 22:30:00: Bought 10 shares of Quux Corp (QUUX) at $5.00 per share.\n"
           "* 2026-09-26 22:30:00: Sold 10 shares of Acme Widgets Inc. (ACME) at $6.00 per share.\n")
    (tmp_path / "log.txt").write_text(log)
    rep = tmp_path / "r.txt"
    j.main([str(tmp_path / "log.txt"), "--report", str(rep), "--roundtrips", str(tmp_path / "rt.csv"), "--meta", "none",
            "--today", "2025-12-31"])
    t = rep.read_text()
    assert "dates suspect (flagged fill) for closed round trips: ACME" in t
