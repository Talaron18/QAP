"""Decision-support calculator for price-level rules (buy / take profit / stop / account stop). NOT a predictor.

    python scripts/level_calc.py TICKER [--prices-file data/raw/secd_tickers.parquet] [--leverage 1|2|-2] [--entry P]
        [--target P_or_pct] [--stop P_or_pct] [--horizon 21] [--paths 100000] [--drift 0.0] [--seed N]
        [--size-frac F --account 45000 --account-stop 0.15]

TICKER is the UNDERLYING stock; --leverage 2 / -2 simulates a daily-reset 2x / 2x inverse product on it. A value ending in % is a return on the
INSTRUMENT (--target 15% --stop -20%); a plain number is a price of the UNDERLYING. Entry defaults to the last close. Model: GJR-GARCH-t
(qap.volfc) when the ticker has >= 750 returns, else flat Student-t(5) at the EWMA(0.94) vol (stated in the report). Zero drift unless --drift
(annual). Prints the report and writes data/level_report.txt (git-ignored; change with --report). Full description: src/qap/levels.py.
"""
import argparse
import pathlib
import sys

from qap import levels, prices


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
    ap.add_argument("--leverage", type=float, default=1.0)
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
    ap.add_argument("--report", default="data/level_report.txt")
    a = ap.parse_args(_glue_negative(list(sys.argv[1:] if argv is None else argv)))
    t = a.ticker.upper()
    df, _ = prices.load_prices(a.prices_file)
    rets = prices.return_series(df, t)
    if rets.empty:
        raise SystemExit(f"{t} not in {a.prices_file} (tickers: {', '.join(sorted(df['tic'].unique()))})")
    last, day = prices.last_price(df, t)
    try:
        res = levels.run(rets.to_numpy(), ticker=t, last_close=last, last_date=day, leverage=a.leverage, entry=a.entry,
                         target=levels.parse_level(a.target), stop=levels.parse_level(a.stop), horizon=a.horizon, n_paths=a.paths,
                         drift=a.drift, seed=a.seed, size_frac=a.size_frac, account=a.account, account_stop=a.account_stop)
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
