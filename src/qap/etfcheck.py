"""Single-stock leveraged/inverse ETF check (pre-registration v5): inventory, coverage of signal quintiles, realised shortfall.

INFORMATION ONLY: nothing here logs a trial or computes a signal return. Signals enter through RANKS (who sits in a quintile)
and the ETF / underlying monthly returns are plain CRSP returns. Pure functions with explicit numpy dtypes (WRDS parquet gives
pandas nullable Int64 / Float64 / string).
"""
import math
import re

import numpy as np
import pandas as pd

from qap.costs import yyyymm_to_mi
from qap.evaluate import exclude_microcaps

ETF_EXCLUDE_WORDS = (
    "S&P", "SP500", "Nasdaq", "Russell", "Dow", "Index", "Sector", "Treasury", "Treasuries", "Gold", "Silver", "Oil", "Crude",
    "Gas", "Bitcoin", "Ether", "Ethereum", "Crypto", "Semiconductor", "Semis", "China", "Japan", "Europe", "Emerging", "Bond",
    "Volatility", "VIX", "Biotech", "Energy", "Financial", "Technology", "Retail", "Homebuilders", "Metals", "Miners", "Bank",
    "Banks", "Dividend", "Real Estate")
STOPWORDS = frozenset("""ETF ETFS ETN ETP FUND DAILY LONG SHORT BULL BEAR INVERSE INV SHARES SHARE TRUST TARGET TARGETS LEVERAGED LEVERAGE
    ULTRA ULTRAPRO ULTRASHORT PRO THE AND FOR ON OF NEW CLASS INC CORP CO LTD PLC SERIES TIMES REVERSE EXPOSURE SINGLE STOCK STOCKS
    LEVERAGES DIREXION GRANITESHARES GRANITE TRADR DEFIANCE TREX REX PROSHARES ROUNDHILL TIDAL VOLSHARES VOLATILITYSHARES KURV
    AXS INNOVATOR TUTTLE T-REX BULLISH BEARISH INVERSED PERFORMANCE DAILYS DAY
    II III IV V VI FUNDS INVESTMENT INVESTMENTS MANAGERS MANAGER OPPORTUNITIES SOLUTIONS LISTED WORLD STRATEGY STRATEGIES CAPITAL ADVISORS
    ADVISOR ASSET MANAGEMENT SERIES PORTFOLIO PORTFOLIOS GLOBAL AMERICAN UNITED HOLDINGS GROUP PARTNERS COMPANY ETFS EXCHANGE TRADED
    PRODUCTS PRODUCT COMMODITY INDEXED LP LLC""".split())
TICKER_COLS = ("ticker", "tradingsymbol", "tsymbol")
NAME_COLS = ("securitynm", "issuernm", "comnam", "securityname", "issuername", "issuer_name", "name")
FIRST_YEAR = 2022            # coverage years start here (products launched 2022+)
LAST_FULL_YEAR = 2025        # decision rule: bottom-quintile VW coverage in this year
COVERAGE_MIN = 0.30          # decision rule thresholds (pre-registration v5 section 4)
CARRY_SHARE = 0.50
GAINS = {"slow10": (0.641, 0.412), "all13": (0.480, 0.263), "Quality": (0.181, 0.309), "gpa": (0.081, 0.221)}  # (dev, val) %/mo Q1-U
SIGNAL_OF = {"slow10": "jkp_slow10", "all13": "jkp_all13"}
BUCKETS = {"all": lambda L: np.ones(len(L), bool), "L<=-2": lambda L: L <= -2, "-2<L<0": lambda L: (L > -2) & (L < 0),
           "L>0": lambda L: L > 0}

_MULT = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*[xX](?![A-Za-z])")
_DAILY = re.compile(r"\bdaily\b", re.I)
_SHORT = re.compile(r"\b(?:short|bear|inverse|inv)\b", re.I)
_EXCL = re.compile(r"(?<![A-Za-z])(" + "|".join(r"\s+".join(map(re.escape, w.split())) for w in ETF_EXCLUDE_WORDS) + r")(?![A-Za-z])", re.I)
_FAR_PAST, _FAR_FUTURE = pd.Timestamp("1800-01-01"), pd.Timestamp("2200-01-01")


def parse_etf_name(name) -> dict:
    """{'L': signed leverage or None, 'single_stock_candidate': bool, 'excluded': matched exclusion word or None}.

    Leveraged = whole word 'daily' and a multiple like 2X / 1.5x; L < 0 if the name has short|bear|inverse|inv, else > 0.
    Candidate = leveraged and no word of ETF_EXCLUDE_WORDS (case-insensitive, letters must not run on either side).
    """
    s = "" if name is None or (isinstance(name, float) and math.isnan(name)) else str(name)
    m = _MULT.search(s)
    if not _DAILY.search(s) or not m or float(m.group(1)) == 0:
        return {"L": None, "single_stock_candidate": False, "excluded": None}
    L = float(m.group(1)) * (-1.0 if _SHORT.search(s) else 1.0)
    x = _EXCL.search(s)
    return {"L": L, "single_stock_candidate": x is None, "excluded": x.group(1) if x else None}


def pick_columns(names: pd.DataFrame) -> dict:
    """Locate permno, ticker, name column(s), namedt and nameenddt in crsp.stocknames_v2; ValueError lists the columns if missing."""
    cols = list(names.columns)
    low = {c.lower(): c for c in cols}
    pre = lambda p: next((c for c in cols if c.lower().startswith(p)), None)
    out = {"permno": low.get("permno"), "ticker": next((low[c] for c in TICKER_COLS if c in low), None),
           "names": [low[c] for c in NAME_COLS if c in low], "start": low.get("namedt") or pre("namedt"),
           "end": low.get("nameenddt") or pre("nameenddt")}
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise ValueError(f"crsp_names: cannot find {missing} (ticker from {list(TICKER_COLS)}, name from {list(NAME_COLS)}, "
                         f"dates namedt / nameenddt); available columns: {cols}")
    return out


def _clean(s: pd.Series) -> np.ndarray:
    return s.astype("string").fillna("").str.strip().to_numpy(dtype=object)


def _join(parts) -> str:
    """Join the name columns, skipping empty pieces and pieces already contained (case-insensitive) in a longer kept piece."""
    kept = []
    for p in sorted({p for p in parts if p}, key=len, reverse=True):
        if not any(p.lower() in k.lower() for k in kept):
            kept.append(p)
    return " ".join(kept)


def prep_names(names: pd.DataFrame) -> pd.DataFrame:
    """permno (int64), ticker (upper, alnum only), start / end (datetime64[ns]), text (all name columns joined), fws (first word of each)."""
    c = pick_columns(names)
    pn = pd.to_numeric(names[c["permno"]].astype("float64"), errors="coerce")
    keep = pn.notna().to_numpy()
    n = names.loc[keep]
    parts = [_clean(n[k]) for k in c["names"]]
    first = lambda v: (re.findall(r"[A-Za-z0-9]+", v) or [""])[0].upper()
    tick = n[c["ticker"]].astype("string").fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return pd.DataFrame({
        "permno": pn[keep].to_numpy(dtype="float64").astype("int64"), "ticker": tick.to_numpy(dtype=object),
        "start": pd.to_datetime(n[c["start"]], errors="coerce").fillna(_FAR_PAST).astype("datetime64[ns]").to_numpy(),
        "end": pd.to_datetime(n[c["end"]], errors="coerce").fillna(_FAR_FUTURE).astype("datetime64[ns]").to_numpy(),
        "text": [_join(ps) for ps in zip(*parts)],
        "fws": [tuple(dict.fromkeys(f for f in map(first, ps) if f)) for ps in zip(*parts)]})


def _parse_rows(nm: pd.DataFrame) -> pd.DataFrame:
    """Name-history rows that parse as leveraged (cheap 'daily' prefilter, one regex parse per distinct text)."""
    d = nm[nm["text"].str.contains("daily", case=False, regex=False)].copy()
    cache = {t: parse_etf_name(t) for t in d["text"].unique()}
    d["L"] = d["text"].map(lambda t: cache[t]["L"]).astype("float64")
    d["candidate"] = d["text"].map(lambda t: cache[t]["single_stock_candidate"]).astype(bool)
    d["excluded"] = d["text"].map(lambda t: cache[t]["excluded"])
    return d[d["L"].notna()]


def month_end(mi: float) -> pd.Timestamp:
    return pd.Timestamp(year=int(mi // 12), month=int(mi % 12) + 1, day=1) + pd.offsets.MonthEnd(0)


def mi_str(mi) -> str:
    return "n/a" if mi is None or not np.isfinite(mi) else f"{int(mi) // 12}-{int(mi) % 12 + 1:02d}"


def _slim(panel: pd.DataFrame, extra=()) -> pd.DataFrame:
    p = panel[["permno", "mi", "mthret_adj", *extra]].copy()
    p["permno"], p["mi"] = p["permno"].astype("int64"), p["mi"].astype("int64")
    p["mthret_adj"] = p["mthret_adj"].to_numpy(dtype="float64", na_value=np.nan)
    return p


def first_last_months(panel: pd.DataFrame) -> pd.DataFrame:
    """Per permno (index): first_mi / last_mi = first and last month with a (delisting-adjusted) return."""
    p = _slim(panel)
    return p[p["mthret_adj"].notna()].groupby("permno")["mi"].agg(first_mi="min", last_mi="max")


def _eq_by_month(panel: pd.DataFrame) -> dict:
    """{mi: DataFrame(permno, me_co)} of rows flagged eq (the formation universe) that month."""
    k = panel[["permno", "mi", "me_co", "eq"]].copy()
    ok = k["eq"].to_numpy(dtype=bool, na_value=False)
    k = pd.DataFrame({"permno": k["permno"].astype("int64")[ok], "mi": k["mi"].astype("int64")[ok],
                      "me_co": k["me_co"].to_numpy(dtype="float64", na_value=np.nan)[ok]})
    return {int(m): g[["permno", "me_co"]] for m, g in k.groupby("mi")}


def etf_tokens(text: str) -> list:
    """All-upper-case tokens (punctuation stripped, >= 2 chars, not a multiple like 2X, not a STOPWORD), in order."""
    out = []
    for tok in str(text).split():
        s = re.sub(r"[^A-Za-z0-9]", "", tok)
        if len(s) >= 2 and s.isupper() and any(ch.isalpha() for ch in s) and not re.fullmatch(r"\d+X", s) and s not in STOPWORDS:
            out.append(s)
    return list(dict.fromkeys(out))


def etf_words(text: str) -> list:
    """Alphabetic words of >= 4 letters, upper-cased, not STOPWORDS (name fallback)."""
    return list(dict.fromkeys(w.upper() for w in re.findall(r"[A-Za-z]+", str(text)) if len(w) >= 4 and w.upper() not in STOPWORDS))


def find_underlying(text: str, date, nm: pd.DataFrame, eq_month: pd.DataFrame) -> tuple:
    """(permno, ticker, method) of the underlying common stock, method 'ticker' | 'name' | 'unmatched'.

    Pool = name-history rows active at `date` (start <= date <= end) of stocks with an eq row that month (`eq_month`: permno, me_co).
    (a) ticker equal to an upper-case token of `text`; (b) else a word of >= 4 letters equal to the first word of a pool name.
    Several hits: largest company ME, then lowest permno.
    """
    pool = nm[(nm["start"] <= date) & (nm["end"] >= date)].merge(eq_month, on="permno")
    pool = pool.sort_values(["me_co", "permno"], ascending=[False, True], na_position="last")
    toks, words = etf_tokens(text), set(etf_words(text))
    for method, hit in (("ticker", pool[pool["ticker"].isin(toks)]), ("name", pool[[bool(words & set(f)) for f in pool["fws"]]])):
        if len(hit):
            return int(hit["permno"].iloc[0]), str(hit["ticker"].iloc[0]), method
    return None, None, "unmatched"


def _pick_row(g: pd.DataFrame, date) -> pd.Series:
    """The candidate name row active at `date` (latest start), else the latest row."""
    g = g.sort_values("start")
    if date is not None:
        a = g[(g["start"] <= date) & (g["end"] >= date)]
        if len(a):
            return a.iloc[-1]
    return g.iloc[-1]


def build_etf_table(names: pd.DataFrame, panel: pd.DataFrame, monthly_first_last: pd.DataFrame = None) -> pd.DataFrame:
    """One row per leveraged single-stock candidate ETF permno.

    Columns: etf_permno, etf_ticker, name, L, underlying_permno, underlying_ticker, method, first_mi, last_mi (float64, NaN when the
    permno has no CRSP return row), variants = distinct (L, ticker tokens) among its candidate name rows (> 1: renamed, L is fixed per
    permno from the row active at the first month, so check the list). Unmatched ETFs are kept with underlying NaN.
    `monthly_first_last` = first_last_months(panel).
    """
    nm = prep_names(names)
    cand = _parse_rows(nm)
    cand = cand[cand["candidate"]]
    fl = first_last_months(panel) if monthly_first_last is None else monthly_first_last
    eq = _eq_by_month(panel)
    cols = ["etf_permno", "etf_ticker", "name", "L", "underlying_permno", "underlying_ticker", "method", "first_mi", "last_mi", "variants"]
    rows = []
    for pid, g in cand.groupby("permno"):
        f, l = (float(fl.at[pid, "first_mi"]), float(fl.at[pid, "last_mi"])) if pid in fl.index else (np.nan, np.nan)
        date = month_end(f) if np.isfinite(f) else None
        r = _pick_row(g, date)
        u = (None, None, "unmatched")
        if date is not None and int(f) in eq:
            u = find_underlying(r["text"], date, nm, eq[int(f)])
        rows.append((int(pid), r["ticker"], r["text"], float(r["L"]), np.nan if u[0] is None else float(u[0]), u[1], u[2], f, l,
                     int(len({(x.L, tuple(etf_tokens(x.text))) for x in g.itertuples()}))))
    t = pd.DataFrame(rows, columns=cols)
    t["underlying_permno"] = t["underlying_permno"].astype("float64")
    return t.sort_values(["first_mi", "etf_permno"], na_position="last").reset_index(drop=True)


def excluded_etfs(names: pd.DataFrame) -> pd.DataFrame:
    """Leveraged-looking permnos dropped by the word list: permno, ticker, name, L, excluded (the matched word)."""
    d = _parse_rows(prep_names(names))
    ok = set(d.loc[d["candidate"], "permno"])
    d = d[~d["permno"].isin(ok)].sort_values("start").groupby("permno").tail(1)
    return d[["permno", "ticker", "text", "L", "excluded"]].rename(columns={"text": "name"}).sort_values("permno").reset_index(drop=True)


def shortfall_panel(etf_table: pd.DataFrame, panel: pd.DataFrame, daily_agg: pd.DataFrame, rf: pd.Series = None) -> pd.DataFrame:
    """ETF-month shortfall = (r_etf - rf) - L * (r_u - rf), i.e. in excess-of-rf terms (rf: decimal monthly, indexed by mi; None = 0).

    Swap-based funds earn (1 - L) * rf on the cash they hold, so the raw gap r_etf - L * r_u has a large positive rf term for inverse funds;
    the excess-return form isolates fees, swap spread and path effects (ROADMAP v5 deviation, recorded before any real run).
    First and last ETF month dropped, both returns present.

    Columns: etf_permno, etf_ticker, mi, L, r_etf, r_u, shortfall, v = sigma_d^2 * n_ret (underlying, crsp_daily_agg; yyyymm -> mi),
    conv = (L^2 - L) / 2 * (r_u^2 - v).
    """
    cols = ["etf_permno", "etf_ticker", "mi", "L", "r_etf", "r_u", "shortfall", "v", "conv"]
    t = etf_table.dropna(subset=["underlying_permno", "first_mi", "last_mi"])
    if t.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    p = _slim(panel)
    p = p[p["mthret_adj"].notna()]
    t = t[["etf_permno", "etf_ticker", "L", "underlying_permno", "first_mi", "last_mi"]].astype(
        {"etf_permno": "int64", "underlying_permno": "int64", "L": "float64", "first_mi": "float64", "last_mi": "float64"})
    e = p.merge(t, left_on="permno", right_on="etf_permno").rename(columns={"mthret_adj": "r_etf"})
    e = e[(e["mi"] > e["first_mi"]) & (e["mi"] < e["last_mi"])]
    u = p.rename(columns={"permno": "underlying_permno", "mthret_adj": "r_u"})
    m = e.merge(u, on=["underlying_permno", "mi"])
    a = pd.DataFrame({"underlying_permno": daily_agg["permno"].astype("int64"), "mi": yyyymm_to_mi(daily_agg["yyyymm"]).astype("int64"),
                      "v": daily_agg["sigma_d"].to_numpy(dtype="float64", na_value=np.nan) ** 2
                      * daily_agg["n_ret"].to_numpy(dtype="float64", na_value=np.nan)})
    m = m.merge(a.drop_duplicates(["underlying_permno", "mi"]), on=["underlying_permno", "mi"], how="left")
    m["rf"] = 0.0 if rf is None else m["mi"].map(rf).astype("float64")
    m["shortfall"] = (m["r_etf"] - m["rf"]) - m["L"] * (m["r_u"] - m["rf"])
    m["conv"] = (m["L"] ** 2 - m["L"]) / 2.0 * (m["r_u"] ** 2 - m["v"])
    return m[cols].sort_values(["etf_permno", "mi"]).reset_index(drop=True)


def ols_cluster(y, x, g) -> dict:
    """y = a + b x by OLS; standard errors clustered by `g` (CR1: G/(G-1) * (n-1)/(n-2)). NaN when degenerate."""
    y, x = np.asarray(y, dtype="float64"), np.asarray(x, dtype="float64")
    n = len(y)
    out = {k: float("nan") for k in ("a", "b", "se_a", "se_b", "t_a", "t_b")}
    out.update(n=int(n), n_clusters=int(pd.Series(g).nunique()) if n else 0)
    X = np.column_stack([np.ones(n), x]) if n else np.zeros((0, 2))
    if n < 3 or np.linalg.matrix_rank(X) < 2:
        return out
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    e = y - X @ beta
    codes, G = pd.factorize(np.asarray(g))
    S = np.zeros((len(G), 2))
    np.add.at(S, codes, X * e[:, None])
    out["a"], out["b"] = float(beta[0]), float(beta[1])
    if len(G) > 1:
        V = inv @ (S.T @ S) @ inv * (len(G) / (len(G) - 1)) * ((n - 1) / (n - 2))
        out["se_a"], out["se_b"] = float(np.sqrt(V[0, 0])), float(np.sqrt(V[1, 1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            out["t_a"], out["t_b"] = float(beta[0] / out["se_a"]), float(beta[1] / out["se_b"])
    return out


def _fm(d: pd.DataFrame, col: str) -> dict:
    """Fama-MacBeth style: monthly cross-sectional mean across ETFs, then mean / std / t over months; noise = std over ETF-months."""
    m = d.groupby("mi")[col].mean()
    n = len(m)
    mean = float(m.mean()) if n else float("nan")
    sd = float(m.std(ddof=1)) if n > 1 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        t = float(mean / (sd / np.sqrt(n))) if n > 2 else float("nan")
    noise = float(d[col].std(ddof=1)) if len(d) > 1 else float("nan")
    return {"n_obs": int(len(d)), "n_etfs": int(d["etf_permno"].nunique()), "n_months": int(n), "mean": mean, "std_months": sd, "t": t,
            "noise": noise}


def shortfall_summary(sp: pd.DataFrame) -> dict:
    """{'fm': {bucket: ...}, 'fm_per_unit': {bucket: ...}, 'regression': {'all' | 'short' | 'long': ols_cluster + a_per_unit}}.

    Buckets 'all', 'L<=-2', '-2<L<0', 'L>0'; per-unit = shortfall / |L|. Regression shortfall ~ a + b * conv (month-clustered SEs):
    a = realised fee + financing carry per month, b should be near 1; a_per_unit = a / mean |L| of the sample. All plain floats.
    """
    d = sp.copy()
    for c in ("L", "shortfall", "conv"):
        d[c] = d[c].astype("float64")
    d["mi"] = d["mi"].astype("int64")
    d["su"] = d["shortfall"] / d["L"].abs()
    L = d["L"].to_numpy()
    fm, fu = {}, {}
    for k, f in BUCKETS.items():
        sub = d[f(L)] if len(d) else d
        fm[k], fu[k] = _fm(sub, "shortfall"), _fm(sub, "su")
    reg = {}
    for k, f in (("all", BUCKETS["all"]), ("short", lambda l: l < 0), ("long", lambda l: l > 0)):
        s = d[f(L)] if len(d) else d
        s = s[np.isfinite(s["conv"]) & np.isfinite(s["shortfall"])]
        r = ols_cluster(s["shortfall"], s["conv"], s["mi"])
        r["mean_abs_L"] = float(s["L"].abs().mean()) if len(s) else float("nan")
        r["a_per_unit"] = r["a"] / r["mean_abs_L"] if len(s) else float("nan")
        reg[k] = r
    return {"fm": fm, "fm_per_unit": fu, "regression": reg}


def _live_pairs(etf: pd.DataFrame, key: str) -> pd.DataFrame:
    """(key, mi) for every live month first_mi..last_mi of each ETF in `etf` (key = 'underlying_permno' or 'etf_permno')."""
    e = etf.dropna(subset=[key, "first_mi", "last_mi"])
    if e.empty:
        return pd.DataFrame({key: pd.Series(dtype="int64"), "mi": pd.Series(dtype="int64")})
    k = np.concatenate([np.full(int(l - f) + 1, int(p)) for p, f, l in zip(e[key], e["first_mi"], e["last_mi"])])
    m = np.concatenate([np.arange(int(f), int(l) + 1) for f, l in zip(e["first_mi"], e["last_mi"])])
    return pd.DataFrame({key: k.astype("int64"), "mi": m.astype("int64")})


def coverage(sig: pd.DataFrame, etf_table: pd.DataFrame, name: str, panel=None, first_year: int = FIRST_YEAR) -> dict:
    """Share of the bottom quintile covered by a live inverse product (and the top quintile by a live long product), by calendar year.

    Quintiles as `portfolio.quintile_legs` (ex-microcaps, signal and ME present, me > 0, within-month pct rank, q = ceil(5 pct)); uses
    RANKS only. A name is covered in month mi if it is the underlying of an ETF with L < 0 (bottom) / L > 0 (top) and first_mi <= mi <= last_mi.
    By year (formation months >= first_year): monthly count share and VW (me) share, averaged over the year's months; n_inv_live /
    n_long_live = all live inverse / long products (matched or not) in the year's last month. `panel` is accepted for symmetry, unused.
    """
    cols = [c for c in ("permno", "mi", "me", "me_co", "primaryexch", name) if c in sig.columns]
    s = sig[cols].copy()
    s["permno"], s["mi"] = s["permno"].astype("int64"), s["mi"].astype("int64")
    for c in ("me", "me_co", name):
        s[c] = s[c].to_numpy(dtype="float64", na_value=np.nan)
    s["primaryexch"] = s["primaryexch"].astype("string").fillna("").to_numpy(dtype=object)
    s = exclude_microcaps(s).dropna(subset=[name, "me"])
    s = s[s["me"] > 0].copy()
    s["q"] = np.minimum(np.ceil(s.groupby("mi")[name].rank(pct=True) * 5), 5).astype(int)
    inv, lng = etf_table[etf_table["L"] < 0], etf_table[etf_table["L"] > 0]
    live = {}
    for tag, e in (("bot", inv), ("top", lng)):
        u = _live_pairs(e, "underlying_permno").rename(columns={"underlying_permno": "permno"}).drop_duplicates()
        live[tag] = u.assign(cov=True)
        allp = _live_pairs(e, "etf_permno")
        live["n_" + tag] = allp.groupby("mi").size()
    out = {"name": name, "by_year": {}, "last_month": None}
    legs = {}
    for tag, q in (("bot", 1), ("top", 5)):
        t = s[s["q"] == q].merge(live[tag], on=["permno", "mi"], how="left")
        t["cov"] = t["cov"].eq(True)
        g = t.groupby("mi")
        legs[tag] = pd.DataFrame({"n": g.size(), "nc": g["cov"].sum(), "me": g["me"].sum(),
                                  "mec": t.assign(w=t["me"] * t["cov"]).groupby("mi")["w"].sum()})
    months = np.sort(s["mi"].unique())
    for y in sorted({int(m) // 12 for m in months if m // 12 >= first_year}):
        ms = [int(m) for m in months if m // 12 == y]
        row = {"n_months": len(ms), "n_inv_live": int(live["n_bot"].get(ms[-1], 0)), "n_long_live": int(live["n_top"].get(ms[-1], 0))}
        for tag in ("bot", "top"):
            L_ = legs[tag].reindex(ms)
            row[f"{tag}_n"] = float(L_["n"].mean())
            row[f"{tag}_cnt_share"] = float((L_["nc"] / L_["n"]).mean())
            row[f"{tag}_vw_share"] = float((L_["mec"] / L_["me"]).mean())
        out["by_year"][y] = row
    if len(months):
        m = int(months[-1])
        b = legs["bot"].reindex([m]).fillna(0)
        out["last_month"] = {"mi": m, "bottom_n": int(b["n"].iloc[0]), "bottom_covered_n": int(b["nc"].iloc[0])}
    return out


def _f(x, d=3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _pct(x, d=1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def decision(summary: dict, cov: dict, last_year: int = LAST_FULL_YEAR) -> dict:
    """Pre-registration section 4, mechanically: per signal, carry drag >= 50% of the validation short-leg gain, or VW coverage < 30%."""
    a = summary["regression"]["short"]["a_per_unit"]
    drag = -a * 100.0 if np.isfinite(a) else float("nan")      # %/mo lost per unit of short exposure (intercept is negative = cost)
    out = {"intercept_pct_mo": a * 100.0 if np.isfinite(a) else float("nan"), "drag_pct_mo": drag, "signals": {}}
    for k, (dev, val) in GAINS.items():
        c = cov.get(SIGNAL_OF.get(k, k))
        row = None if c is None else c["by_year"].get(last_year)
        vw = float("nan") if row is None else row["bot_vw_share"]
        carry_hit = bool(np.isfinite(drag) and drag >= CARRY_SHARE * val)
        cov_hit = bool(np.isfinite(vw) and vw < COVERAGE_MIN)
        out["signals"][k] = {"dev": dev, "val": val, "carry_hit": carry_hit, "vw_cov": vw, "cov_hit": cov_hit, "not_viable": carry_hit or cov_hit}
    out["verdict"] = "NOT VIABLE" if all(s["not_viable"] for s in out["signals"].values()) else "PILOT CONSIDERED"
    out["unevaluable"] = (["carry"] if not np.isfinite(drag) else []) + [f"coverage {last_year}: {k}" for k, s in out["signals"].items() if not np.isfinite(s["vw_cov"])]
    return out


def format_report(etf_table: pd.DataFrame, excluded: pd.DataFrame, cov: dict, summary: dict, last_year: int = LAST_FULL_YEAR) -> list:
    """Text lines: parsed ETF list, unmatched, excluded-by-word-list, coverage tables, shortfall summary, decision-rule block."""
    L = [f"== PARSED SINGLE-STOCK LEVERAGED / INVERSE ETF CANDIDATES ({len(etf_table)}) =="]
    L.append(f"{'etf':9s} {'L':>5s} {'underlying':10s} {'method':9s} {'first':8s} {'last':8s} name (full text of all name columns)")
    for r in etf_table.itertuples():
        L.append(f"{r.etf_ticker:9s} {r.L:+5.1f} {(r.underlying_ticker if isinstance(r.underlying_ticker, str) else '-'):10s} {r.method:9s} "
                 f"{mi_str(r.first_mi):8s} {mi_str(r.last_mi):8s} {r.name}" + (f"  [CHECK: {r.variants} name variants]" if r.variants > 1 else ""))
    un = etf_table[etf_table["method"] == "unmatched"]
    L += ["", f"== UNMATCHED ({len(un)}) =="] + [f"  {r.etf_ticker:9s} {r.name}  L={r.L:+.1f} first {mi_str(r.first_mi)}" for r in un.itertuples()]
    L += ["", f"== LEVERAGED-LOOKING BUT EXCLUDED BY THE WORD LIST ({len(excluded)}) =="]
    L += [f"  {r.ticker:9s} {r.name}  L={r.L:+.1f}  excluded by '{r.excluded}'" for r in excluded.itertuples()]
    L += ["", "== COVERAGE of the bottom / top quintile by a live inverse / long single-stock product (means over the year's months) =="]
    for nme, c in cov.items():
        L.append(f"  {nme}:")
        L.append(f"    {'year':>4s} {'months':>6s} {'bot n':>6s} {'bot cnt':>8s} {'bot VW':>8s} {'top cnt':>8s} {'top VW':>8s} {'inv live':>8s} {'long live':>9s}")
        for y, r in c["by_year"].items():
            L.append(f"    {y:4d} {r['n_months']:6d} {r['bot_n']:6.0f} {_pct(r['bot_cnt_share']):>8s} {_pct(r['bot_vw_share']):>8s} "
                     f"{_pct(r['top_cnt_share']):>8s} {_pct(r['top_vw_share']):>8s} {r['n_inv_live']:8d} {r['n_long_live']:9d}")
        lm = c["last_month"]
        L.append("    last month " + ("n/a" if lm is None else f"{mi_str(lm['mi'])}: bottom-quintile names covered {lm['bottom_covered_n']} of {lm['bottom_n']}"))
    L += ["", "== REALISED SHORTFALL = r_ETF - L * r_underlying (monthly, %; first and last ETF month dropped) =="]
    for title, key, sc in (("shortfall", "fm", 100.0), ("shortfall per unit exposure (/|L|)", "fm_per_unit", 100.0)):
        L.append(f"  {title}: Fama-MacBeth monthly cross-sectional mean, mean / std over months / t; noise = std over ETF-months")
        for b, r in summary[key].items():
            L.append(f"    {b:8s} ETFs {r['n_etfs']:3d} obs {r['n_obs']:4d} months {r['n_months']:3d}: mean {_f(r['mean'] * sc)} "
                     f"std {_f(r['std_months'] * sc)} t {_f(r['t'], 2)} | noise {_f(r['noise'] * sc)}")
    L.append("  convexity regression shortfall ~ a + b * (L^2 - L)/2 * (r_u^2 - v), month-clustered SEs (a in %/mo; b should be ~1)")
    for k, r in summary["regression"].items():
        L.append(f"    {k:6s} n {r['n']:4d} clusters {r['n_clusters']:3d}: a {_f(r['a'] * 100)} (se {_f(r['se_a'] * 100)}, t {_f(r['t_a'], 2)}) "
                 f"b {_f(r['b'])} (se {_f(r['se_b'])}, t {_f(r['t_b'], 2)}) | a/mean|L| {_f(r['a_per_unit'] * 100)} %/mo (mean|L| {_f(r['mean_abs_L'], 2)})")
    d = decision(summary, cov, last_year)
    L += ["", "== DECISION RULE (pre-registration v5 section 4; labels only) ==",
          f"  intercept per unit exposure, L<0 ETFs: {_f(d['intercept_pct_mo'])} %/mo (carry drag = {_f(d['drag_pct_mo'])} %/mo)",
          f"  short-leg gross gains from diagnostic (b), %/mo Q1-U (dev, val); carry rule: drag >= {CARRY_SHARE:.0%} of val; "
          f"coverage rule: {last_year} VW bottom-quintile coverage < {COVERAGE_MIN:.0%}"]
    for k, s in d["signals"].items():
        L.append(f"  {k:7s} gain dev {s['dev']:.3f} val {s['val']:.3f} | carry >= 50% of val: {'YES' if s['carry_hit'] else 'no'} | "
                 f"{last_year} VW coverage {_pct(s['vw_cov'])} < 30%: {'YES' if s['cov_hit'] else ('no' if np.isfinite(s['vw_cov']) else 'n/a (year absent)')}"
                 f" -> {'NOT VIABLE' if s['not_viable'] else 'pilot considered'}")
    L.append(f"VERDICT: {d['verdict']}  (NOT VIABLE only if the rule holds for EVERY listed signal; per-signal lines above)"
             + (f"  [INCOMPLETE, clauses not evaluable: {'; '.join(d['unevaluable'])}]" if d["unevaluable"] else ""))
    return L
