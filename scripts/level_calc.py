"""Decision-support calculator for price-level rules (buy / take profit / stop / account stop). NOT a predictor.

    python scripts/level_calc.py TICKER [--prices-file data/raw/secd_tickers.parquet] [--leverage 1|2|-2] [--entry P]
        [--target P_or_pct] [--stop P_or_pct] [--horizon 21] [--paths 100000] [--drift 0.0] [--seed N] [--close-only]
        [--size-frac F --account 45000 --account-stop 0.15] [--meta data/journal/meta.csv]

TICKER is a stock, or a listed leveraged ETF named in the meta file (columns ticker,underlying,leverage; the same file the journal reads, e.g.
`SNDQ,SNDK,-2` or `SNXX,SNDK,2`): then the UNDERLYING is simulated with that leverage and the ETF's own history is used only for the one-day-move
statistic and a tracking line (--leverage together with such a ticker is refused). --leverage 2 / -2 on a plain stock simulates a daily-reset
2x / 2x inverse product on it. A value ending in % is a return on the INSTRUMENT (--target 15% --stop -20%); a plain number is a price of the
UNDERLYING. Entry defaults to the last close. Model: GJR-GARCH-t (qap.volfc) when there are >= 750 returns, else flat Student-t(5) at the
EWMA(0.94) vol (stated in the report). Zero drift unless --drift (annual). Levels are limit / stop orders filled at the level, with Brownian-bridge
intraday touches between closes; the report also shows the close-only numbers; --close-only drops the bridge. Prints the report and writes
data/level_report.txt (git-ignored; change with --report). Details: src/qap/levels.py.
"""
import argparse
import pathlib
import sys

from qap import journal, levels, prices


def _glue_negative(argv):
    """argparse reads '--stop -20%' as a missing value; turn it into '--stop=-20%' (also --entry/--drift)."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in ("--target", "--stop", "--drift") and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            out.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ticker")
    ap.add_argument("--prices-file", default="data/raw/secd_tickers.parquet")
    ap.add_argument("--leverage", type=float, default=None)
    ap.add_argument("--entry", type=float)
    ap.add_argument("--target")
    ap.add_argument("--stop")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--paths", type=int, default=100000)
    ap.add_argument("--drift", type=float, default=0.0, help="annual drift of the underlying (default 0: no view on direction)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--size-frac", type=float)
    ap.add_argument("--account", type=float, default=45000.0)
    ap.add_argument("--account-stop", type=float, default=0.15)
    ap.add_argument("--meta", default="data/journal/meta.csv", help="ticker,underlying,leverage (used only if the file exists)")
    ap.add_argument("--close-only", action="store_true", help="check levels on closes only (no intraday-touch bridge)")
    ap.add_argument("--report", default="data/level_report.txt")
    a = ap.parse_args(_glue_negative(list(sys.argv[1:] if argv is None else argv)))
    t = a.ticker.upper()
    df, _ = prices.load_prices(a.prices_file)
    try:
        inst = levels.resolve_instrument(t, journal.load_meta(a.meta), df["tic"].unique(), a.leverage)
    except ValueError as e:
        raise SystemExit(str(e))
    st = inst["sim_ticker"]
    rets = prices.return_series(df, st)
    if rets.empty:
        raise SystemExit(f"{st} not in {a.prices_file} (tickers: {', '.join(sorted(df['tic'].unique()))})")
    last, day = prices.last_price(df, st)
    etf = None
    if inst["etf_ticker"]:
        etf = dict(ticker=t, underlying=st, returns=prices.return_series(df, t), underlying_returns=rets, note="")
        if etf["returns"].empty:
            raise SystemExit(f"{t} is listed in meta as {inst['leverage']:+g}x {st} but has no history in {a.prices_file}")
        etf["note"] = f"{t} last close {prices.last_price(df, t)[0]:.4f}; levels below refer to {st}'s price and to {t}'s returns"
    elif inst["note"]:
        print(inst["note"])
    try:
        res = levels.run(rets.to_numpy(), ticker=st, last_close=last, last_date=day, leverage=inst["leverage"], entry=a.entry,
                         target=levels.parse_level(a.target), stop=levels.parse_level(a.stop), horizon=a.horizon, n_paths=a.paths,
                         drift=a.drift, seed=a.seed, size_frac=a.size_frac, account=a.account, account_stop=a.account_stop,
                         close_only=a.close_only, etf=etf)
    except ValueError as e:
        raise SystemExit(str(e))
    text = levels.format_report(res)
    print(text)
    p = pathlib.Path(a.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
