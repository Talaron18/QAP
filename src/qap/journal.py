"""Personal trade-journal analyzer. DESCRIPTIVE ONLY: not a pre-registered study, no trials logged, no alpha claim.

Inputs (all tolerant of markdown bullets, ``$``, thousands separators):
  (A) operation-log lines ``* 2025-09-23 00:30:05: Bought 20 shares of Intuit Inc. (INTU) at $289.98 per share.``
  (B) optional CSV fills (datetime, action, ticker, qty, price[, fee][, name])
  (C) optional holdings snapshots (markdown table + Account Summary), one or several, optional ``Week N`` headers
  (D) optional current-price overrides
Everything is pure functions on plain Python / pandas objects so it can be unit-tested with synthetic data.
Real journals live under data/journal/ (git-ignored); nothing real belongs in the repo or in tests.
"""
from __future__ import annotations

import argparse
import glob
import math
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

NY = ZoneInfo("America/New_York")
EPS = 1e-9
TOL_COST, TOL_PL, TOL_RATIO, TOL_CASH, TOL_MV = 0.015, 1.0, 0.05, 25.0, 1.0
BENCH_VOL = 0.16
Z95 = float(stats.norm.ppf(0.95))

# ----------------------------------------------------------------------------------------------- numbers / parsing
def parse_number(s) -> float:
    """'$1,234.50' -> 1234.5 ; '+20.17%' -> 20.17 ; '(1,110.00)' -> -1110 ; '-$5' -> -5 ; junk -> nan."""
    if s is None:
        return float("nan")
    t = str(s).strip().replace("−", "-")
    neg = t.startswith("(") and t.endswith(")")
    t = re.sub(r"(?i)usd|[,$%\s()+]", "", t)
    try:
        v = float(t)
    except ValueError:
        return float("nan")
    return -v if neg else v


LOG_RE = re.compile(
    r"^\s*(?:[*\-•]+\s*)?(?P<date>\d{4}-\d{2}-\d{2})[T ]+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*:?\s*"
    r"(?P<action>Bought|Sold|Buy|Sell)\s+(?P<qty>[\d,]*\.?\d+)\s+shares?\s+of\s+(?P<name>.+?)\s*"
    r"\((?P<ticker>[A-Za-z0-9.\-^]+)\)\s+at\s+\$?\s*(?P<price>[\d,]*\.?\d+)\s*(?:per\s+share)?\.?(?P<rest>.*)$",
    re.I,
)
FEE_RE = re.compile(r"(?:fee|commission)s?\s*[:=]?\s*\$?\s*([\d,]*\.?\d+)", re.I)
WEEK_RE = re.compile(r"^\W*Week\s+(\d+)\b(.*)$", re.I)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
FILL_COLS = ["ts", "action", "ticker", "qty", "price", "fee", "name", "src", "line_no", "week"]


def parse_log_text(text: str, src: str = "") -> tuple[list[dict], int]:
    """Parse operation-log text. Returns (fills, n_ignored). Blank and ``|`` table lines are not counted as ignored."""
    fills: list[dict] = []
    ignored = 0
    week = None  # current "Week N" header, tagged onto the fills below it
    for i, line in enumerate(text.splitlines(), 1):
        wm = WEEK_RE.match(line.strip().lstrip("*-\u2022 "))
        if wm:
            week = int(wm.group(1))
        if not line.strip() or line.lstrip().startswith("|") or re.search(r"(?i)account\s+summary", line) \
                or any(re.search(pat, line, re.I) for _, pat in SUMMARY_KEYS):
            continue  # snapshot content is handled by parse_snapshots
        m = LOG_RE.match(line)
        if not m:
            ignored += 1
            continue
        fm = FEE_RE.search(m["rest"] or "")
        try:
            ts = pd.Timestamp(f"{m['date']} {m['time']}")
        except ValueError:
            ignored += 1
            continue
        fills.append(dict(
            ts=ts, action="buy" if m["action"].lower() in ("bought", "buy") else "sell",
            ticker=m["ticker"].upper(), qty=parse_number(m["qty"]), price=parse_number(m["price"]),
            fee=parse_number(fm.group(1)) if fm else 0.0, name=m["name"].strip(), src=src, line_no=i, week=week))
    return fills, ignored


def read_fills_csv(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [str(c).strip().lower() for c in d.columns]
    for c in ("datetime", "action", "ticker", "qty", "price"):
        if c not in d.columns:
            raise ValueError(f"fills csv {path}: missing column '{c}'")
    out = pd.DataFrame({
        "ts": pd.to_datetime(d["datetime"]).dt.tz_localize(None),
        "action": d["action"].astype(str).str.strip().str.lower().map(lambda a: "buy" if a.startswith("b") else "sell"),
        "ticker": d["ticker"].astype(str).str.strip().str.upper(),
        "qty": d["qty"].map(parse_number), "price": d["price"].map(parse_number),
        "fee": d["fee"].map(parse_number).fillna(0.0) if "fee" in d.columns else 0.0,
        "name": d["name"].fillna("").astype(str) if "name" in d.columns else "",
        "src": str(path), "line_no": np.arange(1, len(d) + 1), "week": np.nan})
    return out[FILL_COLS]


def fills_frame(fills: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(fills, columns=FILL_COLS)
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def parse_snapshots(text: str, src: str = "") -> list[dict]:
    """Parse holdings-snapshot blocks. A new block starts at a ``Week N`` header or at a second table header."""
    snaps: list[dict] = []
    cur = None

    def new(week=None, sdate=None):
        return dict(week=week, date=sdate, rows=[], summary={}, src=src, has_table=False)

    def push():
        if cur is not None and (cur["rows"] or cur["summary"]):
            snaps.append(cur)

    for line in text.splitlines():
        s = line.strip().lstrip("*-• ").strip()
        wm = WEEK_RE.match(s)
        if wm:
            push()
            dm = DATE_RE.search(wm.group(2))
            cur = new(int(wm.group(1)), pd.Timestamp(dm.group(0)).date() if dm else None)
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            low = [c.lower() for c in cells]
            if low and low[0].startswith("security name"):
                if cur is None or cur["has_table"]:
                    prev = cur
                    push()
                    cur = new(None if prev is None else None)
                cur["has_table"] = True
                continue
            if all(re.fullmatch(r":?-{2,}:?", c) or c == "" for c in cells):
                continue
            if len(cells) >= 6:
                tm = re.match(r"\s*\$?([A-Za-z0-9.\-^]+)", cells[0])
                if tm:
                    if cur is None:
                        cur = new()
                    cur["rows"].append(dict(
                        ticker=tm.group(1).upper(), name=cells[0], qty=parse_number(cells[1]),
                        cost=parse_number(cells[2]), price=parse_number(cells[3]),
                        pl=parse_number(cells[4]), ratio=parse_number(cells[5])))
                    continue
        for key, pat in SUMMARY_KEYS:
            m = re.search(pat + r"[^\d\-+$(]*[:|\s]*\$?\s*([+\-]?\(?[\d,]*\.?\d+\)?)", s, re.I)
            if m:
                if cur is None:
                    cur = new()
                cur["summary"][key] = parse_number(m.group(1))
                break
    push()
    return snaps


SUMMARY_KEYS = [("market_value", r"Total\s+Market\s+Value"), ("cash", r"Remaining\s+Cash"),
                ("total_assets", r"Total\s+Assets"), ("return_rate", r"Total\s+Return\s+Rate")]

# ----------------------------------------------------------------------------------------------- instrument metadata
LEV_RE = re.compile(r"(\d)x\s+(Short|Long|Inverse|Bull|Bear)\b(?:\s+([A-Za-z0-9.\-]+))?", re.I)


def parse_leverage(name: str) -> tuple[float, str | None]:
    """'2x Short SNDK ETF-Tradr' -> (-2, 'SNDK'); no match -> (1.0, None)."""
    m = LEV_RE.search(name or "")
    if not m:
        return 1.0, None
    k = float(m.group(1))
    sign = 1.0 if m.group(2).lower() in ("long", "bull") else -1.0
    return sign * k, (m.group(3).upper() if m.group(3) else None)


def load_meta(path: str | Path | None) -> dict[str, tuple[str, float]]:
    if not path or not Path(path).exists():
        return {}
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    return {str(r["ticker"]).strip().upper(): (str(r["underlying"]).strip().upper(), float(r["leverage"]))
            for _, r in d.iterrows()}


def instrument_info(names: dict[str, str], meta: dict | None = None) -> dict[str, dict]:
    """ticker -> {name, leverage, underlying}. Meta override wins; plain stocks: underlying = ticker."""
    meta = meta or {}
    out = {}
    for t, nm in names.items():
        lev, und = parse_leverage(nm)
        if und is None:
            und = t
        if t in meta:
            und, lev = meta[t]
        out[t] = dict(name=nm, leverage=float(lev), underlying=und.upper())
    return out


# ----------------------------------------------------------------------------------------------- time handling
def _with_year(ts: pd.Timestamp, year: int) -> pd.Timestamp:
    try:
        return ts.replace(year=year)
    except ValueError:  # Feb 29
        return ts.replace(year=year, day=28)


def detect_year_typos(df: pd.DataFrame, max_days: int = 150, today: date | None = None,
                      assume_year: int | None = None) -> list[dict]:
    """Fills whose written date is > max_days from the median fill date.

    Proposal year = assume_year if given, else the LATEST written year not after `today` (logs are recent, so a lone
    later year is the right one and the older majority is the typo side); same month/day. `alt` is the median-year
    alternative; `future` marks proposals that would lie after `today`.
    """
    if len(df) == 0:
        return []
    today = today or date.today()
    ords = np.array([t.toordinal() for t in df["ts"]])
    med = date.fromordinal(int(np.median(ords)))
    ok = [y for y in {t.year for t in df["ts"]} if y <= today.year]
    year = assume_year if assume_year is not None else (max(ok) if ok else med.year)
    outliers = [idx for idx, ts in df["ts"].items() if abs(ts.toordinal() - med.toordinal()) > max_days]
    if not outliers:
        return []
    # If, moved into the proposal year, all fills form one tight cluster, the whole disagreeing-year group is the typo
    # (e.g. six lines written 2025 and one 2026 in the same week: the lone line is right when 2026 is the latest year).
    shifted = np.array([_with_year(t, year).toordinal() for t in df["ts"]])
    if np.abs(shifted - np.median(shifted)).max() <= max_days:
        flagged = [idx for idx, t in df["ts"].items() if t.year != year]
    else:
        flagged = [idx for idx in outliers if df.at[idx, "ts"].year != year]
    out = []
    for idx in flagged:
        ts = df.at[idx, "ts"]
        prop = _with_year(ts, year)
        out.append(dict(idx=idx, written=ts, proposed=prop, alt=_with_year(ts, med.year),
                        future=prop.date() > today, ticker=df.at[idx, "ticker"]))
    return out


def apply_year_fixes(df: pd.DataFrame, typos: list[dict]) -> pd.DataFrame:
    df = df.copy()
    for t in typos:
        df.at[t["idx"], "ts"] = t["proposed"]
    return df


def assign_sessions(df: pd.DataFrame, log_tz: str | None) -> pd.DataFrame:
    """Add ts_et (naive ET wall time) and sdate (US session date = ET calendar date). No tz: naive times as is."""
    df = df.copy()
    if len(df) == 0:
        df["ts_et"], df["sdate"] = pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=object)
        return df
    if log_tz:
        loc = df["ts"].dt.tz_localize(ZoneInfo(log_tz), ambiguous="NaT", nonexistent="shift_forward")
        et = loc.dt.tz_convert(NY).dt.tz_localize(None)
    else:
        et = df["ts"]
    df["ts_et"] = et
    df["sdate"] = et.dt.normalize().dt.date
    return df


def outside_rth(df: pd.DataFrame) -> bool:
    """True when every fill lies outside 09:30-16:00 (so the log is probably in a non-US time zone)."""
    if len(df) == 0:
        return False
    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return bool(((mins < 570) | (mins > 960)).all())


# ----------------------------------------------------------------------------------------------- positions / round trips
def process_fills(df: pd.DataFrame, info: dict) -> dict:
    """Average-cost bookkeeping per ticker. Round trip = flat -> non-flat -> flat (partial exits stay inside).

    Returns dict(positions={ticker: {qty, avg, open_since, fees}}, roundtrips=[...], buys, sells, fees).
    Fee on a fill that crosses through zero is split pro rata between the closing and the opening part.
    """
    pos: dict[str, dict] = {}
    rts: list[dict] = []
    totals = dict(buys=0.0, sells=0.0, fees=0.0)
    open_rt: dict[str, dict] = {}
    for r in df.itertuples():
        t, sgn = r.ticker, (1.0 if r.action == "buy" else -1.0)
        totals["buys" if sgn > 0 else "sells"] += r.qty * r.price
        totals["fees"] += r.fee
        p = pos.setdefault(t, dict(qty=0.0, avg=0.0, open_since=None, fees=0.0))
        rem = r.qty
        typo = bool(getattr(r, "typo", False))
        while rem > EPS:
            q = p["qty"]
            fee_part = r.fee * rem / r.qty if r.qty else 0.0
            if abs(q) < EPS or q * sgn > 0:  # open / add
                if abs(q) < EPS:
                    open_rt[t] = dict(ticker=t, entry_date=r.sdate, entry_sign=sgn, entry_qty=0.0, entry_value=0.0,
                                      exit_qty=0.0, exit_value=0.0, gross=0.0, fees=0.0, unreliable=False)
                    p["open_since"] = r.sdate
                rt = open_rt[t]
                p["avg"] = (abs(q) * p["avg"] + rem * r.price) / (abs(q) + rem)
                p["qty"] = q + sgn * rem
                rt["entry_qty"] += rem
                rt["entry_value"] += rem * r.price
                rt["fees"] += fee_part
                rt["last_date"] = r.sdate
                rt["unreliable"] |= typo
                rem = 0.0
            else:  # reduce / close (possibly cross through zero on the next loop pass)
                rt = open_rt[t]
                close = min(rem, abs(q))
                fee_c = r.fee * close / r.qty if r.qty else 0.0
                rt["exit_qty"] += close
                rt["exit_value"] += close * r.price
                rt["gross"] += (r.price - p["avg"]) * close * (1.0 if q > 0 else -1.0)
                rt["fees"] += fee_c
                rt["last_date"] = r.sdate
                rt["unreliable"] |= typo
                p["qty"] = q + sgn * close
                rem -= close
                if abs(p["qty"]) < EPS:
                    p["qty"], p["avg"], p["open_since"] = 0.0, 0.0, None
                    rts.append(_finish_rt(open_rt.pop(t), info))
    positions = {t: v for t, v in pos.items() if abs(v["qty"]) > EPS}
    for t, rt in open_rt.items():
        positions[t]["unreliable"] = rt["unreliable"]
        positions[t]["fees"] = rt["fees"]
    return dict(positions=positions, roundtrips=rts, **totals)


def _finish_rt(rt: dict, info: dict) -> dict:
    ev = rt["entry_value"]
    pnl = rt["gross"] - rt["fees"]
    inf = info.get(rt["ticker"], dict(leverage=1.0, underlying=rt["ticker"]))
    return dict(ticker=rt["ticker"], entry_date=rt["entry_date"], exit_date=rt["last_date"],
                holding_days=(rt["last_date"] - rt["entry_date"]).days,
                avg_entry=ev / rt["entry_qty"], avg_exit=rt["exit_value"] / rt["exit_qty"],
                ret_pct=100.0 * pnl / ev if ev else float("nan"), pnl_usd=pnl, fees=rt["fees"], qty=rt["entry_qty"],
                leverage=inf["leverage"], underlying=inf["underlying"],
                side="long" if rt["entry_sign"] > 0 else "short", dates_unreliable=bool(rt["unreliable"]))


def recompute_cash(totals: dict, start_capital: float) -> float:
    return start_capital - totals["buys"] + totals["sells"] - totals["fees"]


# ----------------------------------------------------------------------------------------------- FIFO tax view
def fifo_pieces(df: pd.DataFrame) -> tuple[list[dict], int]:
    """Realised FIFO lot pieces for long positions. Returns (pieces, n_unmatched_short_sell_shares)."""
    lots: dict[str, list[list]] = {}
    pieces: list[dict] = []
    unmatched = 0.0
    for r in df.itertuples():
        q = lots.setdefault(r.ticker, [])
        fee_ps = r.fee / r.qty if r.qty else 0.0
        if r.action == "buy":
            q.append([r.Index, r.sdate, r.qty, r.price + fee_ps])
            continue
        rem = r.qty
        while rem > EPS and q:
            bi, bd, bq, bc = q[0]
            m = min(rem, bq)
            pieces.append(dict(ticker=r.ticker, sell_idx=r.Index, buy_idx=bi, buy_date=bd, sell_date=r.sdate, qty=m,
                               cost=m * bc, proceeds=m * (r.price - fee_ps), days=(r.sdate - bd).days))
            pieces[-1]["pnl"] = pieces[-1]["proceeds"] - pieces[-1]["cost"]
            pieces[-1]["term"] = "ST" if pieces[-1]["days"] <= 365 else "LT"
            q[0][2] -= m
            rem -= m
            if q[0][2] <= EPS:
                q.pop(0)
        unmatched += max(rem, 0.0)
    return pieces, int(round(unmatched))


def tax_view(pieces: list[dict], tax_st: float, tax_lt: float) -> dict:
    """Net ST and LT within the report period; a net loss in one bucket offsets a net gain in the other; no carry-forward
    or refund of residual losses, no wash-sale adjustment."""
    st = sum(p["pnl"] for p in pieces if p["term"] == "ST")
    lt = sum(p["pnl"] for p in pieces if p["term"] == "LT")
    st_net, lt_net = st, lt
    if st < 0 < lt:
        off = min(-st, lt)
        st_net, lt_net = st + off, lt - off
    elif lt < 0 < st:
        off = min(-lt, st)
        st_net, lt_net = st - off, lt + off
    tax = max(st_net, 0.0) * tax_st + max(lt_net, 0.0) * tax_lt
    return dict(st=st, lt=lt, tax=tax, pretax=st + lt, aftertax=st + lt - tax)


def wash_sale_warnings(df: pd.DataFrame, pieces: list[dict], info: dict) -> list[str]:
    out = []
    und = lambda t: info.get(t, {}).get("underlying", t)
    by_sell: dict[int, list[dict]] = {}
    for p in pieces:
        by_sell.setdefault(p["sell_idx"], []).append(p)
    buys = df[df["action"] == "buy"]
    for si, ps in by_sell.items():
        if sum(p["pnl"] for p in ps) >= 0:
            continue
        t, sd = ps[0]["ticker"], ps[0]["sell_date"]
        used: dict[int, float] = {}
        for p in ps:
            used[p["buy_idx"]] = used.get(p["buy_idx"], 0.0) + p["qty"]
        hit = [(b.ticker, b.sdate) for b in buys.itertuples()
               if und(b.ticker) == und(t) and abs((b.sdate - sd).days) <= 30 and b.qty - used.get(b.Index, 0.0) > EPS]
        if hit:
            out.append(f"possible WASH SALE: loss-closing sale of {t} on {sd} has a buy within 30 days "
                       f"({', '.join(f'{a} {b}' for a, b in hit)}); loss may be disallowed (not modelled)")
    return out


# ----------------------------------------------------------------------------------------------- leverage drag
def drag_log(L: float, sigma: float, t_years: float) -> float:
    """Daily-reset volatility drag in log terms: (L^2 - L) sigma^2 T / 2 ; log ETF = L log S - drag."""
    return (L * L - L) * sigma * sigma * t_years / 2.0


def breakeven_move(L: float, sigma: float, t_years: float) -> float:
    """Underlying simple move at which the daily-reset product is flat: exp((L-1) sigma^2 T / 2) - 1."""
    return math.expm1((L - 1.0) * sigma * sigma * t_years / 2.0)


# ----------------------------------------------------------------------------------------------- exposure / risk
def exposure_summary(positions: list[dict], total_assets: float) -> dict:
    """positions: dicts with ticker, mv, leverage, underlying. Effective exposure = L x MV / total assets."""
    by_und: dict[str, float] = {}
    for p in positions:
        by_und[p["underlying"]] = by_und.get(p["underlying"], 0.0) + p["leverage"] * p["mv"] / total_assets
    eff = list(by_und.values())
    gross_sorted = sorted((abs(e) for e in eff), reverse=True)
    gross = sum(gross_sorted)
    lev_share = sum(p["mv"] for p in positions if abs(p["leverage"] - 1.0) > EPS) / total_assets
    return dict(by_underlying=by_und, net=sum(eff), gross=gross, long=sum(e for e in eff if e > 0),
                short=-sum(e for e in eff if e < 0), leveraged_share=lev_share,
                largest=gross_sorted[0] if gross_sorted else 0.0,
                top2_share=(sum(gross_sorted[:2]) / gross) if gross > 0 else 0.0,
                invested=sum(p["mv"] for p in positions) / total_assets)


def book_risk(exposures: dict[str, float], vols: dict[str, float], corr: float, total_assets: float) -> dict:
    names = sorted(exposures)
    e = np.array([exposures[n] for n in names])
    s = np.array([vols[n] for n in names])
    C = np.full((len(names), len(names)), corr)
    np.fill_diagonal(C, 1.0)
    cov = np.outer(s, s) * C
    ann = float(math.sqrt(max(e @ cov @ e, 0.0)))
    daily, weekly = ann / math.sqrt(252), ann * math.sqrt(5 / 252)
    return dict(annual_sd=ann, daily_sd=daily, weekly_sd=weekly, usd_per_day=daily * total_assets,
                loss95_week=Z95 * weekly, loss95_week_usd=Z95 * weekly * total_assets,
                bench_annual=BENCH_VOL, bench_daily=BENCH_VOL / math.sqrt(252),
                bench_weekly=BENCH_VOL * math.sqrt(5 / 252), ratio_to_bench=ann / BENCH_VOL)


# ----------------------------------------------------------------------------------------------- luck vs skill
def sign_test_p(wins: int, n: int) -> float:
    """Exact two-sided binomial p-value vs p = 0.5."""
    if n == 0:
        return float("nan")
    pm = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    return min(1.0, sum(p for p in pm if p <= pm[wins] * (1 + 1e-12)))


def mean_t(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    mean = float(x.mean()) if n else float("nan")
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), n - 1)) if not math.isnan(t) else float("nan")
    return dict(n=n, mean=mean, sd=sd, t=t, p=p)


def weeks_needed(sd: float, edge: float) -> float:
    """Weeks until t = 2 for a true weekly edge: (2 sd / e)^2."""
    return (2.0 * sd / edge) ** 2


# ----------------------------------------------------------------------------------------------- reconciliation
def reconcile(snap: dict, positions: dict, totals: dict, start_capital: float) -> list[str]:
    """Compare a snapshot with positions/cash recomputed from fills. Returns MISMATCH / NOTE lines."""
    out: list[str] = []
    mm = lambda t, f, a, b: out.append(f"MISMATCH {t} {f}: snapshot {a} vs recomputed {b}")
    rows = {r["ticker"]: r for r in snap["rows"]}
    rec_mv = 0.0
    has_rows = bool(rows)
    if not has_rows:
        out.append("NOTE summary-only snapshot (no holdings table): only account-level internal checks done")
    for t in sorted((set(rows) | set(positions)) if has_rows else ()):
        r, p = rows.get(t), positions.get(t)
        if r is None:
            mm(t, "quantity", "0 (not listed)", f"{p['qty']:g}")
            continue
        if p is None:
            mm(t, "quantity", f"{r['qty']:g}", "0 (no open position from fills)")
            continue
        q, avg, px = p["qty"], p["avg"], r["price"]
        rec_mv += q * px
        if abs(r["qty"] - q) > 1e-6:
            mm(t, "quantity", f"{r['qty']:g}", f"{q:g}")
        if abs(r["cost"] - avg) > TOL_COST:
            mm(t, "cost price", f"{r['cost']:.4f}", f"{avg:.4f}")
        pl = q * (px - avg)
        if abs(r["pl"] - pl) > TOL_PL:
            mm(t, "floating P/L", f"{r['pl']:.2f}", f"{pl:.2f}")
        ratio = 100.0 * pl / (abs(q) * avg) if avg else float("nan")
        if abs(r["ratio"] - ratio) > TOL_RATIO:
            mm(t, "P/L ratio", f"{r['ratio']:+.2f}%", f"{ratio:+.2f}%")
    s = snap["summary"]
    rec_cash = recompute_cash(totals, start_capital)
    if has_rows and "market_value" in s and abs(s["market_value"] - rec_mv) > TOL_MV:
        mm("ACCOUNT", "total market value", f"{s['market_value']:.2f}", f"{rec_mv:.2f}")
    if has_rows and "cash" in s:
        d = s["cash"] - rec_cash
        if abs(d) > TOL_CASH:
            mm("ACCOUNT", "remaining cash", f"{s['cash']:.2f}", f"{rec_cash:.2f}")
            out.append(f"NOTE unexplained cash difference {d:+.2f} USD = implied fees/slippage "
                       f"(or missing fills / deposits); recomputed = start capital - net purchases - fees")
    if "total_assets" in s:
        if "market_value" in s and "cash" in s and abs(s["total_assets"] - s["market_value"] - s["cash"]) > TOL_MV:
            mm("ACCOUNT", "total assets (market value + cash)", f"{s['total_assets']:.2f}",
               f"{s['market_value'] + s['cash']:.2f}")
        if has_rows and abs(s["total_assets"] - (rec_mv + rec_cash)) > TOL_CASH:
            mm("ACCOUNT", "total assets", f"{s['total_assets']:.2f}", f"{rec_mv + rec_cash:.2f}")
        if "return_rate" in s:
            rr = 100.0 * (s["total_assets"] / start_capital - 1.0)
            if abs(s["return_rate"] - rr) > TOL_RATIO:
                mm("ACCOUNT", "total return rate", f"{s['return_rate']:+.2f}%", f"{rr:+.2f}%")
    return out


def order_snapshots(snaps: list[dict]) -> list[dict]:
    """Chronological order: by date if all dated, else by week number if all numbered, else parse order."""
    if snaps and all(x["date"] for x in snaps):
        return sorted(snaps, key=lambda x: x["date"])
    if snaps and all(x["week"] is not None for x in snaps):
        return sorted(snaps, key=lambda x: x["week"])
    return list(snaps)


def snapshot_cutoff(snap: dict, df: pd.DataFrame, is_last: bool) -> tuple[pd.DataFrame | None, str]:
    """Fills that existed at the snapshot. Order of methods: written date; log 'Week N' tags (week <= N);
    shortest time-ordered prefix whose quantities exactly match the snapshot rows; all fills (latest snapshot only)."""
    if snap["date"]:
        return df[df["sdate"] <= snap["date"]], f"cutoff by date {snap['date']}"
    if snap["week"] is not None and "week" in df and df["week"].notna().any():
        return df[df["week"].isna() | (df["week"] <= snap["week"])], f"cutoff by log week tags (week <= {snap['week']})"
    if snap["rows"]:
        want = {r["ticker"]: r["qty"] for r in snap["rows"]}
        q: dict[str, float] = {}
        for n, r in enumerate(df.itertuples(), 1):
            q[r.ticker] = q.get(r.ticker, 0.0) + (r.qty if r.action == "buy" else -r.qty)
            got = {t: v for t, v in q.items() if abs(v) > 1e-6}
            if got.keys() == want.keys() and all(abs(got[t] - want[t]) < 1e-6 for t in want):
                return df.iloc[:n], f"cutoff inferred by quantity match after {n} fills"
    if is_last:
        return df, "cutoff: all fills (latest snapshot, undated)"
    return None, "undated, no week tags and no quantity match"


def snapshot_total_assets(snap: dict) -> float:
    s = snap["summary"]
    if "total_assets" in s:
        return s["total_assets"]
    mv = s.get("market_value", sum(r["qty"] * r["price"] for r in snap["rows"]))
    return mv + s.get("cash", float("nan"))


# ----------------------------------------------------------------------------------------------- report helpers
def _tab(rows: list[list], headers: list[str], indent: str = "  ") -> list[str]:
    cells = [headers] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    return [indent + "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r)) for r in cells]


def _pc(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.{d}f}%"


def _usd(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+,.2f}"


def _f(x: float, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


# ----------------------------------------------------------------------------------------------- orchestration
def analyze(paths: list[str], fills_csv: str | None = None, log_tz: str | None = None, fix_years: bool = False,
            vols: dict[str, float] | None = None, default_vol: float = 0.60, corr: float = 0.5,
            tax_st: float = 0.408, tax_lt: float = 0.238, start_capital: float = 35000.0,
            prices: dict[str, float] | None = None, meta_path: str | None = None, assume_year: int | None = None,
            today: date | None = None) -> tuple[str, pd.DataFrame]:
    vols = {k.upper(): float(v) for k, v in (vols or {}).items()}
    prices_in = {k.upper(): float(v) for k, v in (prices or {}).items()}
    L: list[str] = []
    warn: list[str] = []

    fills, snaps, n_ign = [], [], 0
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        f, ig = parse_log_text(text, src=Path(p).name)
        fills += f
        n_ign += ig
        snaps += parse_snapshots(text, src=Path(p).name)
    snaps = order_snapshots(snaps)
    df = fills_frame(fills)
    if fills_csv:
        c = read_fills_csv(fills_csv)
        df = pd.concat([df, c], ignore_index=True) if len(df) else c
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["typo"] = False

    # ---- 1. parse checks
    L += ["=" * 100, "TRADE JOURNAL REPORT (descriptive only: not a pre-registered study, no alpha claim)", "=" * 100, "",
          "1. PARSE CHECKS"]
    typos = detect_year_typos(df, today=today, assume_year=assume_year)
    fix_years = fix_years or assume_year is not None  # --assume-year implies applying it
    orig_df = df
    if fix_years and typos:
        df = apply_year_fixes(df, typos)  # corrected: dates are no longer flagged unreliable
    elif typos:
        df.loc[[t["idx"] for t in typos], "typo"] = True
    df = df.sort_values("ts", kind="stable").reset_index(drop=True) if fix_years and typos else df
    tz_hint = log_tz is None and outside_rth(df)
    df = assign_sessions(df, log_tz)
    if len(df) == 0:
        L += ["  no fills parsed", f"  ignored (non-fill) lines: {n_ign}"]
        return "\n".join(L) + "\n", pd.DataFrame()
    L.append(f"  fills: {len(df)}   ignored (non-fill) lines: {n_ign}   tickers: {df['ticker'].nunique()} "
             f"({', '.join(sorted(df['ticker'].unique()))})")
    L.append(f"  written date range: {orig_df['ts'].min():%Y-%m-%d %H:%M} .. {orig_df['ts'].max():%Y-%m-%d %H:%M}"
             f"   session dates{' (after correction)' if fix_years and typos else ''}: {min(df['sdate'])} .. {max(df['sdate'])}"
             f"   time zone: {log_tz + ' -> America/New_York' if log_tz else 'none given (naive)'}")
    if typos:
        corr_ts = orig_df["ts"].copy()
        for t in typos:
            corr_ts.loc[t["idx"]] = t["proposed"]
        L.append(f"  flagged fills: {len(typos)} of {len(orig_df)} ({', '.join(sorted({t['ticker'] for t in typos}))}); "
                 f"range as written {orig_df['ts'].min():%Y-%m-%d} .. {orig_df['ts'].max():%Y-%m-%d}, "
                 f"corrected range would be {corr_ts.min():%Y-%m-%d} .. {corr_ts.max():%Y-%m-%d}")
    for t in typos:
        L.append(f"  suspected year typo: {t['ticker']} written {t['written']:%Y-%m-%d %H:%M} -> proposed "
                 f"{t['proposed']:%Y-%m-%d %H:%M}{' (IN THE FUTURE vs today)' if t['future'] else ''} "
                 f"({'APPLIED' if fix_years else 'not applied; use --fix-years or --assume-year'}); "
                 f"alternative (median-year) correction: {t['alt']:%Y-%m-%d %H:%M}")
        if not fix_years:
            warn.append("dates as written include a suspected year typo: holding periods involving it are UNRELIABLE")
    if not typos:
        L.append("  suspected year typos: none")
    if tz_hint:
        L.append("  HINT: every fill lies outside 09:30-16:00, so the log is probably in a non-US time zone "
                 "(e.g. UTC+8): pass --log-tz Asia/Shanghai")
        warn.append("no --log-tz given although all fills fall outside US regular hours: session dates may be off by a day")
    dup = df.duplicated(["ts", "ticker", "qty", "price"], keep=False)
    L.append(f"  duplicates (same timestamp/ticker/qty/price): {int(dup.sum())} rows"
             + ("" if not dup.any() else " -> kept, check for double-pasted lines"))
    if dup.any():
        warn.append(f"{int(dup.sum())} possibly duplicated fill rows (kept in the analysis)")
    L.append(f"  snapshots parsed: {len(snaps)}   start capital: {start_capital:,.2f}")

    names = {}
    for r in df.itertuples():
        if r.name or r.ticker not in names:
            names[r.ticker] = r.name
    for s in snaps:
        for r in s["rows"]:
            if not names.get(r["ticker"]):
                names[r["ticker"]] = r["name"]
    info = instrument_info(names, load_meta(meta_path))
    L.append("  instruments: " + "; ".join(f"{t} L={i['leverage']:+g} und={i['underlying']}" for t, i in sorted(info.items())))

    res = process_fills(df, info)
    cash = recompute_cash(res, start_capital)
    last_date = max(df["sdate"])
    last_flagged = bool(df.loc[df["sdate"] == last_date, "typo"].any())  # 'last date seen' comes from a flagged fill

    # prices: latest snapshot rows, overridden by --prices
    prices_cur: dict[str, float] = {}
    for sn in snaps:  # later snapshots override earlier ones
        for r in sn["rows"]:
            prices_cur[r["ticker"]] = r["price"]
    prices_cur.update(prices_in)

    # ---- reconciliation
    L += ["", "1b. RECONCILIATION vs SNAPSHOTS (tolerances: cost 0.015, P/L 1.0, ratio 0.05 pct pts, cash 25 USD)"]
    if not snaps:
        L.append("  no snapshots given")
    n_mm = 0
    for i, s in enumerate(snaps):
        label = f"Week {s['week']}" if s["week"] is not None else f"snapshot {i + 1}"
        label += f" ({s['src']})" + (f" as of {s['date']}" if s["date"] else "")
        sub, how = snapshot_cutoff(s, df, i == len(snaps) - 1)
        if sub is None:
            L.append(f"  {label}: not reconciled ({how}; add a date 'Week N (YYYY-MM-DD)' or 'Week N' tags in the log)")
            continue
        r2 = process_fills(sub, info)
        out = reconcile(s, r2["positions"], r2, start_capital)
        n_mm += sum(o.startswith("MISMATCH") for o in out)
        L.append(f"  {label} [{how}]: {sum(o.startswith('MISMATCH') for o in out)} mismatches")
        L += ["    " + o for o in out]
    if n_mm:
        warn.append(f"{n_mm} snapshot/fill MISMATCH lines: the log and the broker snapshot disagree (see 1b)")

    # ---- 2. positions & round trips
    L += ["", "2. POSITIONS & ROUND TRIPS (average-cost; round trip = flat -> position -> flat; session dates, calendar days)"]
    rts = pd.DataFrame(res["roundtrips"])
    if len(rts):
        rows = [[r.ticker, r.entry_date, r.exit_date, r.holding_days + 0, f"{r.avg_entry:.2f}", f"{r.avg_exit:.2f}",
                 f"{r.ret_pct:+.2f}%", f"{r.pnl_usd:+,.2f}", f"{r.leverage:+g}", "?" if r.dates_unreliable else ""]
                for r in rts.itertuples()]
        L += _tab(rows, ["ticker", "entry", "exit", "days", "avg_in", "avg_out", "ret", "pnl_usd", "L", "date?"])
        L.append(f"  closed round trips: {len(rts)}   total net P/L {rts['pnl_usd'].sum():+,.2f} USD   ('?' = dates unreliable)")
        if rts["dates_unreliable"].any():
            warn.append("dates suspect (flagged fill) for closed round trips: " + ", ".join(rts.loc[rts["dates_unreliable"], "ticker"]))
    else:
        L.append("  no closed round trips")
    openp = res["positions"]
    open_rows, unpriced = [], []
    for t, p in sorted(openp.items()):
        px = prices_cur.get(t)
        if px is None:
            unpriced.append(t)
            continue
        open_rows.append([t, f"{p['qty']:g}", f"{p['avg']:.4f}", f"{px:.2f}", f"{p['qty'] * px:,.2f}",
                          f"{p['qty'] * (px - p['avg']):+,.2f}",
                          f"{100 * (px / p['avg'] - 1) * (1 if p['qty'] > 0 else -1):+.2f}%"])
    L.append("  open positions (current prices from latest snapshot / --prices):")
    L += _tab(open_rows, ["ticker", "qty", "avg_cost", "price", "mkt_value", "unreal_pnl", "ret"]) if open_rows else ["    none priced"]
    if unpriced:
        L.append(f"  EXCLUDED (no current price): {', '.join(unpriced)}  (give --prices TICKER=PRICE)")
        warn.append(f"no current price for {', '.join(unpriced)}: excluded from exposure/risk/unrealised P/L")
    unreal = sum(r_[0] for r_ in [(p["qty"] * (prices_cur[t] - p["avg"]),) for t, p in openp.items() if t in prices_cur])
    L.append(f"  unrealised P/L (priced positions): {unreal:+,.2f} USD   cash (recomputed): {cash:,.2f}   "
             f"fees paid: {res['fees']:,.2f}")

    pieces, unmatched = fifo_pieces(df)
    tv = tax_view(pieces, tax_st, tax_lt)
    L += ["", f"  FIFO realised-lot tax view (ST <= 365 days at {tax_st:.1%}, LT at {tax_lt:.1%}; ASSUMPTION: losses offset only "
          "against gains realised in this report period, no carry-forward/refund, no wash-sale adjustment, long lots only):"]
    if pieces:
        L += _tab([[p["ticker"], p["buy_date"], p["sell_date"], p["days"], f"{p['qty']:g}", f"{p['pnl']:+,.2f}", p["term"]]
                   for p in pieces], ["ticker", "bought", "sold", "days", "qty", "pnl", "term"], indent="    ")
        L.append(f"    short-term P/L {tv['st']:+,.2f}   long-term P/L {tv['lt']:+,.2f}   pre-tax {tv['pretax']:+,.2f}   "
                 f"estimated tax {tv['tax']:,.2f}   after-tax P/L {tv['aftertax']:+,.2f}")
    else:
        L.append("    no realised lots")
    if unmatched:
        L.append(f"    NOTE {unmatched} shares sold beyond held longs (short selling): not in the tax view")
        warn.append("sells exceed holdings (short positions): tax view ignores them")
    ws = wash_sale_warnings(df, pieces, info)
    L += ["  " + w for w in ws]
    warn += ws
    if tv["st"] > 0:
        warn.append(f"realised gains are short-term: estimated tax {tv['tax']:,.2f} USD "
                    f"({tv['tax'] / tv['st']:.0%} of the ST gain) at {tax_st:.1%}")

    # ---- 3. exposure
    L += ["", "3. EXPOSURE"]
    priced = [dict(ticker=t, mv=p["qty"] * prices_cur[t], leverage=info[t]["leverage"], underlying=info[t]["underlying"])
              for t, p in sorted(openp.items()) if t in prices_cur]
    ta = None
    rowsnaps = [x for x in snaps if x["rows"]]
    if rowsnaps:
        ta = snapshot_total_assets(rowsnaps[-1])
        if math.isnan(ta) or prices_in:
            ta = None
    if ta is None:
        ta = sum(p["mv"] for p in priced) + cash
    L.append(f"  total assets used: {ta:,.2f} ({'latest snapshot with holdings' if rowsnaps and ta == snapshot_total_assets(rowsnaps[-1]) else 'recomputed: priced market value + cash'})")
    ex = None
    if priced and ta > 0:
        ex = exposure_summary(priced, ta)
        L += _tab([[p["ticker"], f"{p['mv']:,.2f}", f"{p['mv'] / ta:.1%}", f"{p['leverage']:+g}", p["underlying"],
                    f"{p['leverage'] * p['mv'] / ta:+.1%}"] for p in priced],
                  ["ticker", "mkt_value", "weight", "L", "underlying", "eff_exposure"])
        L.append("  effective underlying exposure (L x MV / total assets):")
        L += _tab([[u, f"{e:+.1%}"] for u, e in sorted(ex["by_underlying"].items(), key=lambda kv: -abs(kv[1]))],
                  ["underlying", "exposure"], indent="    ")
        L.append(f"  invested {ex['invested']:.1%}   gross {ex['gross']:.1%}   net {ex['net']:+.1%}   long {ex['long']:.1%}   "
                 f"short {ex['short']:.1%}   leveraged-product share of account {ex['leveraged_share']:.1%}")
        L.append(f"  concentration: largest underlying exposure {ex['largest']:.1%}, top-2 share of gross {ex['top2_share']:.1%}")
        if ex["largest"] > 0.40:
            warn.append(f"concentration: largest single-underlying exposure is {ex['largest']:.0%} of the account")
        if ex["gross"] > 1.0:
            warn.append(f"gross effective exposure {ex['gross']:.0%} of total assets (> 100%: leverage)")
        if ex["leveraged_share"] > 0.25:
            warn.append(f"{ex['leveraged_share']:.0%} of the account sits in leveraged/inverse products")
        for u, grp in {u: [p for p in priced if p["underlying"] == u] for u in {p["underlying"] for p in priced}}.items():
            if any(p["leverage"] * p["mv"] > 0 for p in grp) and any(p["leverage"] * p["mv"] < 0 for p in grp):
                warn.append(f"{u}: long and short exposure to the same underlying are held at once (they net)")
    else:
        L.append("  nothing to show (no priced open positions)")

    # ---- 4. drag
    L += ["", "4. VOLATILITY DRAG & BREAK-EVEN (daily-reset leveraged/inverse products)",
          "  Formula: log(ETF_T / ETF_0) = L * log(S_T / S_0) - (L^2 - L) * sigma^2 * T / 2   (T in years)",
          "  => drag = (L^2 - L) sigma^2 T / 2 ; flat product needs log(S_T/S_0) = (L - 1) sigma^2 T / 2",
          "  Holding-period T = calendar days / 365; the generic table uses trading days / 252. Vol from --vol per underlying,"
          f" else --default-vol {default_vol:.2f} (ASSUMPTION)."]

    def vol_for(und: str, tk: str = "") -> tuple[float, bool]:
        for k in (und, tk):
            if k and k.upper() in vols:
                return vols[k.upper()], False
        return default_vol, True

    levered = []
    for r in res["roundtrips"]:
        if abs(r["leverage"] - 1) > EPS:
            levered.append((r["ticker"], r["underlying"], r["leverage"], r["holding_days"],
                            "closed" + ("?" if r["dates_unreliable"] else "")))
    for t, p in sorted(openp.items()):
        if abs(info[t]["leverage"] - 1) > EPS:
            levered.append((t, info[t]["underlying"], info[t]["leverage"], (last_date - p["open_since"]).days,
                            "open" + ("?" if p.get("unreliable") or last_flagged else "")))
    if levered:
        rows = []
        for t, u, lev, days, st in levered:
            sg, assumed = vol_for(u, t)
            T = days / 365.0
            rows.append([t, st, f"{lev:+g}", u, days, f"{sg:.2f}" + ("*" if assumed else ""),
                         f"{100 * drag_log(lev, sg, T):.2f}%", f"{100 * breakeven_move(lev, sg, T):+.2f}%"])
        L += _tab(rows, ["ticker", "state", "L", "underlying", "days", "sigma", "drag(log)", "underlying b/e move"])
        L.append("  * = ASSUMPTION (default vol). Open positions: days from first fill to the last date seen "
                 f"({last_date}). Drag is a log-return deduction on top of L x the underlying's move.")
        bad = [f"{x[0]} ({x[4]})" for x in levered if x[4].endswith("?")]
        if bad:
            warn.append("dates suspect (flagged fill is the entry/exit or the last date seen) for: " + ", ".join(bad))
        warn.append("leveraged/inverse daily-reset products held: volatility drag erodes the multiple over time")
    else:
        L.append("  no leveraged/inverse positions")
    levs = sorted({i["leverage"] for i in info.values() if abs(i["leverage"] - 1) > EPS})
    if levs:
        L.append("  drag (and underlying break-even move) by horizon, trading days / 252:")
        rows = []
        for lev in levs:
            for td in (5, 21, 63):
                rows.append([f"{lev:+g}", td] + [f"{100 * drag_log(lev, s, td / 252):.1f}% ({100 * breakeven_move(lev, s, td / 252):+.1f}%)"
                                               for s in (0.5, 0.7, 0.9)])
        L += _tab(rows, ["L", "T_days", "sigma=0.5", "sigma=0.7", "sigma=0.9"], indent="    ")

    # ---- 5. risk
    L += ["", "5. RISK SNAPSHOT (normal approximation, zero mean, constant pairwise correlation; all inputs below are ASSUMPTIONS)"]
    if ex and ex["by_underlying"]:
        uv = {}
        tick_of = {p["underlying"]: p["ticker"] for p in priced}
        for u in ex["by_underlying"]:
            uv[u] = vol_for(u, tick_of.get(u, ""))
        rk = book_risk(ex["by_underlying"], {u: v[0] for u, v in uv.items()}, corr, ta)
        L.append("  assumed annual vols: " + ", ".join(f"{u}={v:.2f}{'*' if a else ''}" for u, (v, a) in sorted(uv.items()))
                 + f"   (* = --default-vol {default_vol:.2f})")
        L.append(f"  assumed correlation between distinct underlyings: {corr:.2f}; 252-day convention; cash has zero risk")
        L.append(f"  book annual sd {rk['annual_sd']:.1%}  daily sd {rk['daily_sd']:.2%} ({rk['usd_per_day']:,.0f} USD per day at 1 sd)  "
                 f"weekly sd {rk['weekly_sd']:.2%}")
        L.append(f"  95% one-week loss (normal approximation: 1.645 x weekly sd): {rk['loss95_week']:.1%} = {rk['loss95_week_usd']:,.0f} USD")
        L.append(f"  16% annual benchmark: daily sd {rk['bench_daily']:.2%}, weekly sd {rk['bench_weekly']:.2%}; "
                 f"book risk = {rk['ratio_to_bench']:.1f}x the benchmark")
        if rk["ratio_to_bench"] > 2:
            warn.append(f"book volatility is {rk['ratio_to_bench']:.1f}x a 16% benchmark (under the stated vol assumptions)")
    else:
        L.append("  nothing to show (no priced open positions)")

    # ---- 6. luck vs skill
    L += ["", "6. LUCK VS SKILL"]
    n_tr = len(rts)
    if n_tr:
        rets = rts["ret_pct"].to_numpy() / 100.0
        wins = int((rets > 0).sum())
        nz = int((rets != 0).sum())
        mt = mean_t(rets)
        L.append(f"  (a) round trips: n={n_tr}  win rate {wins / n_tr:.0%} ({wins}/{n_tr})  mean return {_pc(mt['mean'], 2)}  "
                 f"sd {_f(100 * mt['sd'], 2)}{'' if math.isnan(mt['sd']) else '%'}  t={_f(mt['t'], 2)} (p={_f(mt['p'], 3)})  sign-test p={_f(sign_test_p(wins, nz), 3)} "
                 f"(exact two-sided binomial vs 50%, {nz} non-zero trips)")
    else:
        L.append("  (a) round trips: none closed")
    eq = [(s, snapshot_total_assets(s)) for s in snaps]
    eq = [(s, v) for s, v in eq if not math.isnan(v)]
    pr = np.array([])
    if len(eq) >= 3:
        vals = [start_capital] + [v for _, v in eq]
        pr = np.diff(vals) / np.array(vals[:-1])
        eqt = mean_t(pr)
        L.append(f"  (b) equity curve ({len(eq)} snapshots, baseline = start capital {start_capital:,.0f}): total assets "
                 + ", ".join(f"{v:,.0f}" for _, v in eq))
        L.append(f"      period returns: " + ", ".join(_pc(x, 2) for x in pr) + f"   mean {_pc(eqt['mean'], 2)}  sd {_f(100 * eqt['sd'], 2)}{'' if math.isnan(eqt['sd']) else '%'}"
                 f"  t={_f(eqt['t'], 2)} (p={_f(eqt['p'], 3)})  [periods treated as weeks]")
    else:
        L.append(f"  (b) equity curve: {len(eq)} snapshot(s), need >= 3")
    if len(pr) >= 4:
        wsd, wlab = float(np.std(pr, ddof=1)), "observed"
    else:
        wsd, wlab = 0.10, "ASSUMED 10%/week (fewer than 4 weekly points)"
    L.append(f"  (c) POWER: weekly sd {wsd:.1%} ({wlab}); weeks needed for t = 2 is (2 sd / e)^2:")
    L += _tab([[f"{e:.0%}", f"{weeks_needed(wsd, e):,.0f}", f"{weeks_needed(wsd, e) / 52:,.1f}"] for e in (0.01, 0.02, 0.05)],
              ["true edge/week", "weeks", "years"], indent="    ")
    need5 = weeks_needed(wsd, 0.05)
    if len(pr) >= 2 and np.mean(pr) > 0:
        L.append(f"    at the OBSERVED mean {_pc(float(np.mean(pr)), 2)}/week: {weeks_needed(wsd, float(np.mean(pr))):,.0f} weeks "
                 "(an observed mean is itself noisy and biased upwards by selection)")
    if n_tr < 30 or len(pr) < need5:
        L.append(f"  CONCLUSION: n is far too small to distinguish skill from luck (round trips n={n_tr}, weekly points n={len(pr)}; "
                 f"even a 5% weekly edge needs ~{need5:,.0f} weeks). Results are descriptive.")
        warn.append("sample far too small to tell skill from luck: do not extrapolate")

    # ---- 7. warnings
    L += ["", "7. WARNINGS"]
    seen = []
    for w in warn:
        if w not in seen:
            seen.append(w)
    L += [f"  - {w}" for w in seen] or ["  none"]
    return "\n".join(L) + "\n", rts


# ----------------------------------------------------------------------------------------------- CLI
def _kv(items: list[str] | None, what: str) -> dict[str, float]:
    out = {}
    for tok in items or []:
        for part in tok.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise SystemExit(f"{what}: expected TICKER=VALUE, got {part!r}")
            k, v = part.split("=", 1)
            out[k.strip().upper()] = float(v.replace("$", "").replace(",", ""))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Personal trade-journal analyzer (descriptive).")
    ap.add_argument("paths", nargs="*", help="operation-log / snapshot text files or globs, e.g. data/journal/*.txt")
    ap.add_argument("--fills-csv")
    ap.add_argument("--log-tz", help="time zone of the log timestamps, e.g. Asia/Shanghai")
    ap.add_argument("--fix-years", action="store_true")
    ap.add_argument("--assume-year", type=int, help="year to move every suspected year-typo line to (implies applying it)")
    ap.add_argument("--vol", action="append", nargs="+", metavar="UND=VOL")
    ap.add_argument("--default-vol", type=float, default=0.60)
    ap.add_argument("--corr", type=float, default=0.5)
    ap.add_argument("--tax-st", type=float, default=0.408)
    ap.add_argument("--tax-lt", type=float, default=0.238)
    ap.add_argument("--start-capital", type=float, default=35000.0)
    ap.add_argument("--prices", nargs="*", metavar="TICKER=PRICE")
    ap.add_argument("--meta", default="data/journal/meta.csv")
    ap.add_argument("--today", help="YYYY-MM-DD used as 'today' for the year-typo proposal (default: real today)")
    ap.add_argument("--report", default="data/journal_report.txt")
    ap.add_argument("--roundtrips", default="data/journal/roundtrips.csv")
    a = ap.parse_args(argv)
    files: list[str] = []
    for p in a.paths:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        files += [h for h in hits if Path(h).is_file()]
    if not files and not a.fills_csv:
        raise SystemExit("no input files found")
    vols = _kv([x for grp in (a.vol or []) for x in grp], "--vol")
    report, rts = analyze(files, a.fills_csv, a.log_tz, a.fix_years, vols, a.default_vol, a.corr, a.tax_st, a.tax_lt,
                          a.start_capital, _kv(a.prices, "--prices"), a.meta, a.assume_year,
                          date.fromisoformat(a.today) if a.today else None)
    print(report)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(report, encoding="utf-8")
    Path(a.roundtrips).parent.mkdir(parents=True, exist_ok=True)
    rts.to_csv(a.roundtrips, index=False)
    return 0
