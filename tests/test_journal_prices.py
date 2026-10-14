"""Journal analyzer with --prices-file (Compustat daily prices). SYNTHETIC tickers and prices only."""
import re

import numpy as np
import pandas as pd
import pytest

from qap import journal as j
from qap import prices as px

LOG = """* 2025-09-22 10:00:00: Bought 20 shares of Acme Widgets Inc. (ACME) at $100.00 per share.
* 2025-09-22 10:05:00: Bought 1000 shares of 2x Short ZZQX ETF-Foo (ZQS) at $5.00 per share.
* 2025-09-23 10:00:00: Bought 10 shares of Quux Corp (QUUX) at $50.00 per share.
"""
SNAP = """Week 1 (2025-09-26)
| Security Name | Quantity | Cost Price | Current Price | Floating P/L | P/L Ratio (%) |
|---|---|---|---|---|---|
| ACME (Acme Widgets) | 20 | 100.00 | {acme:.2f} | 0.00 | +0.00% |
| ZQS (2x Short ZZQX) | 1,000 | 5.00 | {zqs:.2f} | 0.00 | +0.00% |
Account Summary
Total Market Value: $1.00
Remaining Cash: $1.00
Total Assets: $2.00
"""
DAYS = pd.bdate_range("2025-03-03", "2025-09-26")


def frame(tic, gvkey, ret, start_px, days=DAYS):
    p = start_px * np.cumprod(1 + np.asarray(ret))
    return pd.DataFrame(dict(gvkey=gvkey, iid="01", tic=tic, datadate=days[-len(p):], prccd=p, ajexdi=1.0, trfd=1.0))


def world(tmp_path, end=None, drop=()):
    rng = np.random.default_rng(7)
    n = len(DAYS)
    zz = rng.normal(0, 0.03, n)                       # underlying ZZQX, ~48% annual vol
    ac = rng.normal(0, 0.015, n)                      # ACME
    zqs = -2 * zz[-100:]                              # inverse 2x product: shorter history (100 days)
    parts = {"ZZQX": frame("ZZQX", "1", zz, 40.0), "ACME": frame("ACME", "2", ac, 100.0), "ZQS": frame("ZQS", "3", zqs, 5.0)}
    f = pd.concat([v for k, v in parts.items() if k not in drop])
    if end:
        f = f[f["datadate"] <= end]
    path = tmp_path / "p.parquet"
    f.to_parquet(path)
    (tmp_path / "log.txt").write_text(LOG)
    return path, parts


def run(tmp_path, *extra, snap=None, pfile=None, today="2025-12-31"):
    files = [str(tmp_path / "log.txt")]
    if snap is not None:
        (tmp_path / "snap.txt").write_text(snap)
        files.append(str(tmp_path / "snap.txt"))
    argv = files + ["--today", today, "--report", str(tmp_path / "r.txt"), "--roundtrips", str(tmp_path / "rt.csv"),
                    "--meta", str(tmp_path / "none.csv"), *extra]
    if pfile:
        argv += ["--prices-file", str(pfile)]
    assert j.main(argv) == 0
    return (tmp_path / "r.txt").read_text()


def section(text, head, nxt):
    return text.split(head)[1].split(nxt)[0]


def test_current_prices_come_from_file_when_no_override_or_snapshot(tmp_path):
    pf, parts = world(tmp_path)
    t = run(tmp_path, "--vol", "ZZQX=0.5", pfile=pf)
    acme_last = parts["ACME"]["prccd"].iloc[-1]
    assert f"ACME=price file 2025-09-26" in t and f"{acme_last:.2f}" in t
    assert "QUUX=none" in t and "EXCLUDED (no current price): QUUX" in t        # not in the file: unpriced as before
    assert "open positions (current prices from latest snapshot / --prices / price file)" in t
    # --prices beats the file
    t2 = run(tmp_path, "--prices", "ACME=123.45", pfile=pf)
    assert "ACME=--prices" in t2 and "123.45" in t2


def test_snapshot_beats_file_and_mismatch_line_over_one_percent(tmp_path):
    pf, parts = world(tmp_path)
    acme, zqs = (parts[k]["prccd"].iloc[-1] for k in ("ACME", "ZQS"))
    t = run(tmp_path, snap=SNAP.format(acme=acme * 1.05, zqs=zqs * 1.005), pfile=pf)
    assert "ACME=snapshot" in t and "ZQS=snapshot" in t
    mm = [ln for ln in t.splitlines() if "MISMATCH ACME price: snapshot" in ln]
    assert len(mm) == 1 and "+5.00%" in mm[0] and "file close" in mm[0] and "2025-09-26" in mm[0]
    assert not any("MISMATCH ZQS price" in ln for ln in t.splitlines())        # 0.5% < 1%
    assert "price MISMATCH lines" in t.split("7. WARNINGS")[1]
    # the snapshot price is the one used for the position
    assert f"{acme * 1.05:.2f}" in section(t, "open positions", "unrealised")
    ok = run(tmp_path, snap=SNAP.format(acme=acme, zqs=zqs), pfile=pf)
    assert "MISMATCH ACME price" not in ok and "MISMATCH ZQS price" not in ok


def test_stale_note(tmp_path):
    pf, _ = world(tmp_path, end="2025-09-12")
    t = run(tmp_path, pfile=pf)
    assert "STALE price file: last date 2025-09-12 is 11 calendar days before the latest fill/snapshot date 2025-09-23" in t
    assert "price file is stale" in t.split("7. WARNINGS")[1]
    pf2, _ = world(tmp_path, end="2025-09-19")                                   # 4 days before: not stale
    assert "STALE price file" not in run(tmp_path, pfile=pf2)


def test_vol_precedence_explicit_over_file_over_default(tmp_path):
    pf, parts = world(tmp_path)
    d, _ = px.load_prices(pf)
    v_file, n, lab = px.best_vol(px.return_series(d, "ZZQX").to_numpy())
    assert lab == "ewma94" and n > 100
    t = run(tmp_path, "--default-vol", "0.61", pfile=pf)
    s5 = section(t, "5. RISK SNAPSHOT", "6. LUCK")
    assert f"ZZQX={v_file:.2f}," in s5 + "," or f"ZZQX={v_file:.2f} " in s5 or f"ZZQX={v_file:.2f}\n" in s5
    assert f"ZZQX={v_file:.2f}*" not in s5                                                   # data, not flagged
    assert "ACME=" in s5 and re.search(r"ACME=\d\.\d\d(?!\*)", s5)                           # in the file as well
    assert "QUUX" not in s5                                                                  # unpriced: not in the book risk
    assert f"ZZQX: {v_file:.2f}  <- price file: ewma94, {n} returns, as of 2025-09-26" in s5
    # explicit wins
    t = run(tmp_path, "--vol", "ZZQX=0.9", pfile=pf)
    s5 = section(t, "5. RISK SNAPSHOT", "6. LUCK")
    assert "ZZQX=0.90," in s5 + "," or "ZZQX=0.90 " in s5 or "ZZQX=0.90\n" in s5
    assert "ZZQX: 0.90  <- --vol (explicit)" in s5
    # not in the file -> default, flagged
    pf2, _ = world(tmp_path, drop=("ZZQX",))
    t = run(tmp_path, "--default-vol", "0.61", pfile=pf2)
    s5 = section(t, "5. RISK SNAPSHOT", "6. LUCK")
    assert "ZZQX=0.61*" in s5 and "ZZQX: 0.61  <- --default-vol (ASSUMPTION; not in the price file)" in s5


def test_section_4_uses_file_vol_unflagged(tmp_path):
    pf, _ = world(tmp_path)
    d, _ = px.load_prices(pf)
    v = px.best_vol(px.return_series(d, "ZZQX").to_numpy())[0]
    s4 = section(run(tmp_path, pfile=pf), "4. VOLATILITY", "4b.")
    row = [ln for ln in s4.splitlines() if ln.strip().startswith("ZQS")][0]
    assert f"{v:.2f}" in row and f"{v:.2f}*" not in row
    assert "explicit --vol or realised vol from the price file" in s4


def test_section_4b_only_with_file_and_shows_slope_and_drag(tmp_path):
    pf, parts = world(tmp_path)
    t = run(tmp_path, pfile=pf)
    assert "4b. REALISED DRAG OF YOUR LEVERAGED PRODUCTS" in t and t.index("4b.") < t.index("5. RISK SNAPSHOT")
    s = section(t, "4b. REALISED", "5. RISK")
    row = [ln for ln in s.splitlines() if ln.strip().startswith("ZQS")][0].split()
    assert row[1] == "ZZQX" and row[2] == "-2" and row[3] == "99"            # 100 prices -> 99 returns
    assert row[5] == "-2.00" and row[6] == "1.000"                          # slope, R2
    d, _ = px.load_prices(pf)
    c = px.leveraged_check(px.return_series(d, "ZQS"), px.return_series(d, "ZZQX"), -2.0)
    assert row[9] == f"{100 * c['drag_realised']:+.2f}%" and row[10] == f"{100 * c['drag_expected']:+.2f}%"
    assert "check the leverage/underlying mapping" not in t
    # wrong nominal leverage via meta -> warning
    (tmp_path / "meta.csv").write_text("ticker,underlying,leverage\nZQS,ZZQX,-3\n")
    t3 = run(tmp_path, "--meta", str(tmp_path / "meta.csv"), pfile=pf)
    assert "ZQS: realised slope -2.00 on ZZQX differs from the nominal L=-3" in t3.split("7. WARNINGS")[1]
    # product or underlying missing from the file: explained, not silently skipped
    pf2, _ = world(tmp_path, drop=("ZZQX",))
    assert "ZQS (L=-2, underlying ZZQX): not checked, ZZQX not in the price file" in run(tmp_path, pfile=pf2)
    assert "no leveraged" not in section(t, "4b. REALISED", "5. RISK")


def test_without_file_nothing_changes(tmp_path):
    world(tmp_path)
    t = run(tmp_path, snap=SNAP.format(acme=105.0, zqs=4.5))
    for absent in ("1c.", "4b.", "price file", "PRICE FILE", "STALE", "--prices-file"):
        assert absent not in t, absent
    assert "all inputs below are ASSUMPTIONS)" in t and "open positions (current prices from latest snapshot / --prices):" in t
    assert "ZZQX=0.60*" in t


def _frozen():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("journal_prefile", pathlib.Path(__file__).parent / "fixtures" / "journal_prefile.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cli_scenarios(tmp_path):
    import test_journal as tj
    world(tmp_path)
    (tmp_path / "snap.txt").write_text(SNAP.format(acme=105.0, zqs=4.5))
    (tmp_path / "cli_log.txt").write_text(tj.CLI_LOG)
    (tmp_path / "cli_snap.txt").write_text(tj.CLI_SNAP)
    (tmp_path / "f.csv").write_text("datetime,action,ticker,qty,price,fee,name\n"
                                    "2025-03-03 10:00:00,buy,AAA,10,10.0,0.5,2x Long ZZQX Fund\n"
                                    "2025-03-04 11:00:00,SELL,AAA,10,11.0,0.5,\n"
                                    "2025-03-05 11:00:00,buy,AAA,5,12.0,0.5,\n")
    none = str(tmp_path / "none.csv")
    return {
        "open book with snapshot": [str(tmp_path / "log.txt"), str(tmp_path / "snap.txt"), "--today", "2025-12-31",
                                    "--vol", "ZZQX=0.7", "--meta", none],
        "open book, default vols, prices override": [str(tmp_path / "log.txt"), "--today", "2025-12-31", "--prices", "ACME=111", "--meta", none],
        "cli log + snapshot, tz, fix years": [str(tmp_path / "cli_log.txt"), str(tmp_path / "cli_snap.txt"), "--log-tz", "Asia/Shanghai",
                                              "--fix-years", "--vol", "ZZQX=0.7", "--vol", "ACME=0.4", "--corr", "0.3",
                                              "--today", "2025-12-31", "--prices", "ACME=111", "--meta", none],
        "fills csv, leveraged": ["--fills-csv", str(tmp_path / "f.csv"), "--meta", none],
    }


@pytest.mark.parametrize("name", ["open book with snapshot", "open book, default vols, prices override",
                                  "cli log + snapshot, tz, fix years", "fills csv, leveraged"])
def test_report_byte_identical_to_frozen_pre_change_analyzer(tmp_path, name):
    """Without --prices-file the report and round-trip csv equal those of the frozen copy of the analyzer from before this feature."""
    old = _frozen()
    argv = _cli_scenarios(tmp_path)[name]
    assert old.main(argv + ["--report", str(tmp_path / "old.txt"), "--roundtrips", str(tmp_path / "old.csv")]) == 0
    assert j.main(argv + ["--report", str(tmp_path / "new.txt"), "--roundtrips", str(tmp_path / "new.csv")]) == 0
    assert (tmp_path / "old.txt").read_bytes() == (tmp_path / "new.txt").read_bytes() and len((tmp_path / "new.txt").read_bytes()) > 500
    assert (tmp_path / "old.csv").read_bytes() == (tmp_path / "new.csv").read_bytes()


def test_section_4_closed_trip_underlying_gets_a_vol_source_line(tmp_path):
    pf, _ = world(tmp_path)
    (tmp_path / "log.txt").write_text(LOG + "* 2025-09-24 10:00:00: Sold 1000 shares of 2x Short ZZQX ETF-Foo (ZQS) at $6.00 per share.\n")
    t = run(tmp_path, pfile=pf)
    s4 = section(t, "4. VOLATILITY", "4b.")
    assert "closed" in s4 and "vol source: ZZQX: price file: ewma94" in s4


def test_future_dated_price_row_is_ignored_and_stale_uses_latest_real_date(tmp_path):
    pf, _ = world(tmp_path, end="2025-09-12")
    f = pd.read_parquet(pf)
    fut = f[f["tic"] == "ACME"].tail(1).assign(datadate=pd.Timestamp("2099-01-01"), prccd=999.0)
    pd.concat([f, fut]).to_parquet(pf)
    t = run(tmp_path, pfile=pf)
    assert "STALE price file: last date 2025-09-12" in t and "future-dated 1" in t and "999.00" not in t
