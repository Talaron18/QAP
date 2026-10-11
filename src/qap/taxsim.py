"""Pre-registration v7: mechanical after-tax benefit of tax-loss harvesting on a ~150-name direct index (NOT an alpha test).

Three arms on one rolling 10-year window (see ROADMAP.md, 'Pre-registration v7'): an ETF-like value-weighted market benchmark,
a DIY index without harvesting and a DIY index with monthly lot-level harvesting. One engine (`_Arm`) runs all three; harvesting is
a branch inside it, so with a threshold no lot can reach the harvest arm is bit-identical to the no-harvest arm.

Conventions chosen where the spec is silent (all reported):
- Tax cash runs through the account. Each calendar year's tax (dividends, realised gains net of losses, the exogenous outside LT gain,
  the USD 3,000 ordinary-income deduction) is settled at the year-end close against the NEXT month's investable cash: tax owed reduces
  it, a refund adds to it, so every arm has the same out-of-pocket outlay (start + contributions). A negative balance (e.g. with zero
  contributions) is carried at zero interest. The final year's tax is deducted from terminal value to give after-tax wealth.
- Purchases cost `cost_bp` of notional and the cost is capitalised into the lot's basis; sales cost `cost_bp` of value and reduce proceeds.
- Lots bought in the same month in the same name are merged (identical tax date). Long-term means held MORE than 12 months, with the
  initial purchase dated at month -1 (start of month 0), purchases at month ends, sales at month ends.
- Annualisation of the benefit is geometric: (W_a / W_b) ** (1 / years) - 1.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaxParams:
    st_rate: float = 0.408          # 32% federal + 3.8% NIIT + 5% state
    lt_rate: float = 0.238          # 15% + 3.8% + 5%
    div_rate: float = 0.238         # qualified dividends
    loss_cap: float = 3000.0        # net loss deductible against ordinary income per year (valued at st_rate)
    outside_lt_gains: float = 3000.0  # exogenous LT gain realised every year elsewhere (all arms)
    contrib: float = 500.0          # per month, at month end
    start_capital: float = 35000.0
    cost_bp: float = 5.0            # per side, DIY arms only
    harvest_pct: float = 0.10       # loss >= pct * basis
    harvest_min_usd: float = 50.0
    n_names: int = 150
    lt_months: int = 12             # long-term iff held MORE than this many months

    def with_(self, **kw) -> "TaxParams":
        return dataclasses.replace(self, **kw)


def is_long_term(sale_mi, buy_mi, params: TaxParams):
    return (np.asarray(sale_mi) - np.asarray(buy_mi)) > params.lt_months


def net_year(st: float, lt: float, carry_st: float, carry_lt: float, p: TaxParams) -> dict:
    """One year's capital-gain tax. st/lt are net realised (incl. outside gains in lt); carry_* are positive loss carryforwards.

    ST nets with ST, LT with LT, then cross-netting; remaining net loss is deducted up to loss_cap (ST losses first) at st_rate;
    the rest carries forward indefinitely keeping its character. Returns tax (negative = saving) and the new carryforwards.
    """
    st = st - carry_st
    lt = lt - carry_lt
    if st > 0 and lt < 0:
        x = min(st, -lt)
        st, lt = st - x, lt + x
    elif lt > 0 and st < 0:
        x = min(lt, -st)
        lt, st = lt - x, st + x
    tax = max(st, 0.0) * p.st_rate + max(lt, 0.0) * p.lt_rate
    loss_st, loss_lt = max(-st, 0.0), max(-lt, 0.0)
    use_st = min(loss_st, p.loss_cap)
    use_lt = min(loss_lt, p.loss_cap - use_st)
    tax -= (use_st + use_lt) * p.st_rate
    return {"tax": tax, "carry_st": loss_st - use_st, "carry_lt": loss_lt - use_lt, "st": st, "lt": lt}


class TaxLedger:
    """Per-calendar-year accumulator of realised ST/LT gains and dividends with loss carryforward."""

    def __init__(self, params: TaxParams):
        self.p = params
        self.carry_st = 0.0
        self.carry_lt = 0.0
        self.open: dict = {}
        self.paid: dict = {}
        self.realised_st = 0.0
        self.realised_lt = 0.0
        self.dividends = 0.0

    def _b(self, year):
        return self.open.setdefault(int(year), [0.0, 0.0, 0.0])

    def add_gain(self, year, amount: float, long_term: bool) -> None:
        self._b(year)[1 if long_term else 0] += amount
        if long_term:
            self.realised_lt += amount
        else:
            self.realised_st += amount

    def add_div(self, year, amount: float) -> None:
        self._b(year)[2] += amount
        self.dividends += amount

    def close_year(self, year) -> float:
        """Settle the year: adds the outside LT gain, taxes dividends, updates carryforwards; returns the year's tax."""
        st, lt, dv = self.open.pop(int(year), [0.0, 0.0, 0.0])
        r = net_year(st, lt + self.p.outside_lt_gains, self.carry_st, self.carry_lt, self.p)
        self.carry_st, self.carry_lt = r["carry_st"], r["carry_lt"]
        tax = r["tax"] + dv * self.p.div_rate
        self.paid[int(year)] = tax
        return tax

    def copy(self) -> "TaxLedger":
        return copy.deepcopy(self)


def waterfill(v: np.ndarray, w: np.ndarray, elig: np.ndarray, X: float) -> np.ndarray:
    """Spend X on eligible names so the most underweight (lowest value / target weight) are lifted to a common level."""
    buy = np.zeros(len(v))
    idx = np.nonzero(elig & (w > 0))[0]
    if X <= 0 or idx.size == 0:
        return buy
    r = v[idx] / w[idx]
    o = np.argsort(r, kind="stable")
    ii = idx[o]
    lam = (X + np.cumsum(v[ii])) / np.cumsum(w[ii])
    k = np.nonzero(lam > r[o])[0].max()
    sel = ii[: k + 1]
    buy[sel] = np.maximum(lam[k] * w[sel] - v[sel], 0.0)
    return buy


def exit_mask_from_returns(ret: np.ndarray) -> np.ndarray:
    """True at each column's last finite observation if it is before the final row (a delisting / last observation)."""
    fin = np.isfinite(ret)
    T, N = ret.shape
    last = T - 1 - np.argmax(fin[::-1], axis=0)
    cols = np.nonzero(fin.any(0) & (last < T - 1))[0]
    m = np.zeros((T, N), bool)
    m[last[cols], cols] = True
    return m


class _Arm:
    def __init__(self, pr, dy, exitm, w0, start_mi, p: TaxParams, harvest: bool, cost_bp: float):
        self.pr, self.dy, self.exitm, self.w0 = pr, dy, exitm, w0
        self.start_mi, self.p, self.harvest, self.bp = start_mi, p, harvest, cost_bp / 1e4

    def _sell(self, mask, t, year, led):
        """Sell the masked lots at month end; returns (proceeds, st_loss, lt_loss) and records gains in the ledger."""
        val = self.lot_val[mask]
        proceeds = val * (1.0 - self.bp)
        gain = proceeds - self.lot_basis[mask]
        lt = is_long_term(t, self.lot_mon[mask], self.p)
        led.add_gain(year, float(gain[lt].sum()), True)
        led.add_gain(year, float(gain[~lt].sum()), False)
        self.sold_notional += float(val.sum())
        self.costs += self.bp * float(val.sum())
        keep = ~mask
        self.lot_name, self.lot_basis = self.lot_name[keep], self.lot_basis[keep]
        self.lot_val, self.lot_mon = self.lot_val[keep], self.lot_mon[keep]
        loss = np.where(gain < 0, -gain, 0.0)
        return float(proceeds.sum()), float(loss[~lt].sum()), float(loss[lt].sum())

    def run(self) -> dict:
        pr, dy, exitm, w0, p, bp = self.pr, self.dy, self.exitm, self.w0, self.p, self.bp
        T, N = pr.shape
        sm = self.start_mi
        years = list(range(sm // 12, (sm + T - 1) // 12 + 1))
        yi = {y: i for i, y in enumerate(years)}
        harv_st, harv_lt, taxes = np.zeros(len(years)), np.zeros(len(years)), np.zeros(len(years))
        led = TaxLedger(p)
        alive = np.ones(N, bool)
        last_buy = np.full(N, -1, np.int64)
        sold_mon = np.full(N, -10, np.int64)
        self.sold_notional = 0.0
        self.costs = 0.0
        buy_notional = 0.0
        skipped = 0
        n_events = 0
        cash = 0.0
        wn = w0 / w0.sum()
        amt0 = wn * p.start_capital
        names0 = np.arange(N)
        self.lot_name = names0
        self.lot_basis = amt0.copy()
        self.lot_val = amt0 * (1.0 - bp)
        self.lot_mon = np.full(N, -1, np.int64)
        self.costs += bp * p.start_capital
        buy_notional += p.start_capital
        ret = np.full(T, np.nan)
        v0_sum = 0.0
        contrib_total = 0.0
        taxes_prior = 0.0
        for t in range(T):
            year = (sm + t) // 12
            V0 = float(self.lot_val.sum())
            v0_sum += V0
            if self.lot_name.size:
                D = float((self.lot_val * dy[t][self.lot_name]).sum())
                self.lot_val = self.lot_val * (1.0 + pr[t][self.lot_name])
            else:
                D = 0.0
            led.add_div(year, D)
            E = float(self.lot_val.sum()) + D
            cost_trade0 = self.costs
            pool = 0.0
            ex = exitm[t]
            if ex.any() and self.lot_name.size:
                m = ex[self.lot_name]
                if m.any():
                    pool += self._sell(m, t, year, led)[0]
            alive &= ~ex
            if self.harvest and self.lot_name.size:
                loss = self.lot_basis - self.lot_val
                q = (loss >= p.harvest_pct * self.lot_basis) & (loss >= p.harvest_min_usd)
                if q.any():
                    recent = last_buy[self.lot_name] >= t - 1
                    skipped += int((q & recent).sum())
                    h = q & ~recent
                    if h.any():
                        n_events += int(h.sum())
                        names_h = np.unique(self.lot_name[h])
                        pr_, ls, ll = self._sell(h, t, year, led)
                        pool += pr_
                        harv_st[yi[year]] += ls
                        harv_lt[yi[year]] += ll
                        sold_mon[names_h] = t
            blocked = sold_mon >= t - 1
            elig = alive & ~blocked
            buy = np.zeros(N)
            if pool > 0.0:
                if elig.any():
                    v = np.bincount(self.lot_name, self.lot_val, minlength=N)
                    wv = np.where(elig, v, 0.0)
                    s = wv.sum()
                    if s > 0:
                        buy += pool * wv / s
                    else:
                        wt = np.where(elig, wn, 0.0)
                        buy += pool * wt / wt.sum()
                else:
                    cash += pool
            cost_pool_buy = bp * float(buy.sum())
            X = p.contrib + D + cash
            cash = 0.0
            contrib_total += p.contrib
            if X > 0.0 and elig.any():
                v = np.bincount(self.lot_name, self.lot_val, minlength=N) + buy * (1.0 - bp)
                wa = np.where(alive, wn, 0.0)
                wa = wa / wa.sum()
                buy += waterfill(v, wa, elig, X)
            else:
                cash = X
            idx = np.nonzero(buy > 0.0)[0]
            if idx.size:
                self.lot_name = np.concatenate([self.lot_name, idx])
                self.lot_basis = np.concatenate([self.lot_basis, buy[idx]])
                self.lot_val = np.concatenate([self.lot_val, buy[idx] * (1.0 - bp)])
                self.lot_mon = np.concatenate([self.lot_mon, np.full(idx.size, t, np.int64)])
                last_buy[idx] = t
                bn = float(buy[idx].sum())
                buy_notional += bn
                self.costs += bp * bn
            trade_cost = (self.costs - cost_trade0) - (bp * float(buy.sum()) - cost_pool_buy)   # sells + pool re-buys only
            if V0 > 0:
                ret[t] = (E - V0 - trade_cost) / V0
            if t < T - 1 and (sm + t + 1) // 12 != year:
                tax = led.close_year(year)
                taxes[yi[year]] = tax
                taxes_prior += tax
                cash -= tax
        year = (sm + T - 1) // 12
        mv_lots = float(self.lot_val.sum())
        mv = mv_lots + cash
        unreal = float((self.lot_val - self.lot_basis).sum())
        led_h = led.copy()
        tax_hold = led_h.close_year(year)
        led_l = led.copy()
        proceeds = self.lot_val * (1.0 - bp)
        gain = proceeds - self.lot_basis
        lt = is_long_term(T - 1, self.lot_mon, p)
        led_l.add_gain(year, float(gain[lt].sum()), True)
        led_l.add_gain(year, float(gain[~lt].sum()), False)
        tax_liq = led_l.close_year(year)
        liq_cost = bp * mv_lots
        taxes[yi[year]] = tax_hold
        avg_v = v0_sum / T if T else 0.0
        return {
            "mv": mv, "mv_lots": mv_lots, "cash": cash, "unrealised": unreal, "years": np.array(years), "taxes": taxes,
            "tax_hold_final": tax_hold, "tax_liq_final": tax_liq, "liq_cost": liq_cost,
            "wealth_hold": mv - tax_hold, "wealth_liq": mv - liq_cost - tax_liq,
            "ret": ret, "harv_st": harv_st, "harv_lt": harv_lt,
            "turnover": self.sold_notional / avg_v / (T / 12.0) if avg_v > 0 else 0.0,
            "n_names": int(np.unique(self.lot_name).size), "n_lots": int(self.lot_name.size),
            "costs": self.costs, "sold_notional": self.sold_notional, "buy_notional": buy_notional,
            "skipped_recent": skipped, "harvest_lots": n_events,
            "carry_st_end": led.carry_st, "carry_lt_end": led.carry_lt,
            "realised_st": led.realised_st, "realised_lt": led.realised_lt, "dividends": led.dividends,
            "contrib_total": contrib_total, "taxes_prior": taxes_prior, "start_capital": p.start_capital,
            "lots": {"name": self.lot_name.copy(), "basis": self.lot_basis.copy(), "val": self.lot_val.copy(), "mon": self.lot_mon.copy()},
        }


def _clean(a):
    return np.where(np.isfinite(a), a, 0.0)


def simulate_window(pret: np.ndarray, dyld: np.ndarray, me0: np.ndarray, mkt_pret: np.ndarray, mkt_dy: np.ndarray,
                    params: TaxParams = TaxParams(), start_mi: int = 0, exit_mask: np.ndarray | None = None) -> dict:
    """Run the three arms on one window.

    pret, dyld: (T, N) monthly price return (mthretx) and dividend yield (mthret - mthretx); NaN = no observation.
    me0: (N,) company ME at window start (NaN / <= 0 = ineligible); the top `params.n_names` with a month-0 return are held.
    exit_mask: (T, N) True in a name's delisting / last-observation month (default: last finite observation before the window end).
    mkt_pret, mkt_dy: (T,) value-weighted total-market price return and dividend yield.
    start_mi: calendar month index (year*12 + month - 1) of month 0; fixes calendar-year tax boundaries.
    """
    pret = np.asarray(pret, float)
    dyld = np.asarray(dyld, float)
    me0 = np.asarray(me0, float)
    T = pret.shape[0]
    ok = np.isfinite(me0) & (me0 > 0) & np.isfinite(pret[0])
    cand = np.nonzero(ok)[0]
    if cand.size == 0:
        raise ValueError("no eligible names at window start")
    sel = cand[np.argsort(-me0[cand], kind="stable")][: params.n_names]
    sub = pret[:, sel]
    em = exit_mask_from_returns(sub) if exit_mask is None else np.asarray(exit_mask, bool)[:, sel]
    pr, dy = _clean(sub), _clean(dyld[:, sel])
    w0 = me0[sel]
    out = {"start_mi": int(start_mi), "n_months": int(T), "n_sel": int(sel.size), "sel": sel}
    out["etf"] = _Arm(_clean(np.asarray(mkt_pret, float))[:, None], _clean(np.asarray(mkt_dy, float))[:, None],
                      np.zeros((T, 1), bool), np.ones(1), start_mi, params, False, 0.0).run()
    out["nohv"] = _Arm(pr, dy, em, w0, start_mi, params, False, params.cost_bp).run()
    out["hv"] = _Arm(pr, dy, em, w0, start_mi, params, True, params.cost_bp).run()
    return out


# ---------------------------------------------------------------- inputs from the monthly panel

def _f(s) -> np.ndarray:
    return pd.Series(s).to_numpy(dtype="float64", na_value=np.nan)


def build_inputs(panel: pd.DataFrame) -> dict:
    """Wide (month x permno) arrays from `qap.panel.build_monthly_panel` output (nullable dtypes are converted explicitly).

    Columns are permnos that are ever in the equity universe. pr = mthret_adj - dy (price part incl. the delisting return),
    dy = clip(mthret - mthretx, 0) (0 if mthretx is unavailable). The market series is VW by LAGGED permno-level ME over names
    with eq True at t-1 (month 0 of the panel, which has no lag, uses its own month's ME and eq).
    """
    eq_all = panel["eq"].fillna(False).to_numpy(bool)
    perm = np.sort(panel.loc[eq_all, "permno"].unique().astype("int64"))
    d = panel[panel["permno"].isin(perm)]
    mi = d["mi"].to_numpy("int64")
    first, last = int(panel["mi"].min()), int(panel["mi"].max())
    T, P = last - first + 1, len(perm)
    ti = mi - first
    pi = np.searchsorted(perm, d["permno"].to_numpy("int64"))
    tot = _f(d["mthret_adj"])
    if "mthretx" in d.columns:
        dyv = np.clip(_f(d["mthret"]) - _f(d["mthretx"]), 0.0, None)
        dyv = np.where(np.isfinite(dyv), dyv, 0.0)
    else:
        dyv = np.zeros(len(d))
    prv = np.clip(tot - dyv, -1.0, None)
    pr = np.full((T, P), np.nan)
    dy = np.zeros((T, P))
    me = np.full((T, P), np.nan)
    me_co = np.full((T, P), np.nan)
    eq = np.zeros((T, P), bool)
    permco = np.full((T, P), -1, np.int32)
    pr[ti, pi], dy[ti, pi] = prv, dyv
    me[ti, pi], me_co[ti, pi] = _f(d["me"]), _f(d["me_co"])
    eq[ti, pi] = d["eq"].fillna(False).to_numpy(bool)
    permco[ti, pi] = d["permco"].fillna(-1).to_numpy("int32")
    wgt = np.where(eq & np.isfinite(me) & (me > 0), me, 0.0)
    wl = np.vstack([wgt[:1], wgt[:-1]])                       # lagged weights (month 0: own month)
    valid = np.isfinite(pr)
    ws = np.where(valid, wl, 0.0)
    den = ws.sum(1)
    den = np.where(den > 0, den, np.nan)
    mkt_pr = (ws * np.where(valid, pr, 0.0)).sum(1) / den
    mkt_dy = (ws * dy).sum(1) / den
    return {"first_mi": first, "months": np.arange(first, last + 1), "permnos": perm, "pr": pr, "dy": dy, "me": me,
            "me_co": me_co, "eq": eq, "permco": permco, "mkt_pr": mkt_pr, "mkt_dy": mkt_dy}


def window_me0(inp: dict, start_mi: int) -> np.ndarray:
    """Company ME at the window start for the eligible universe, one permno per permco (the largest class carries me_co).

    Selection row is the month BEFORE the window if present, else month 0. Eligible: eq True, me_co > 0, finite month-0 return.
    """
    i0 = start_mi - inp["first_mi"]
    r = i0 - 1 if i0 >= 1 else i0
    eqr, mec, me, pc = inp["eq"][r], inp["me_co"][r], inp["me"][r], inp["permco"][r]
    ok = eqr & np.isfinite(mec) & (mec > 0) & np.isfinite(me) & np.isfinite(inp["pr"][i0])
    idx = np.nonzero(ok)[0]
    out = np.full(len(eqr), np.nan)
    if idx.size == 0:
        return out
    o = idx[np.lexsort((idx, -me[idx], pc[idx]))]              # by permco, then largest class first, then index
    first = np.r_[True, pc[o][1:] != pc[o][:-1]]
    keep = o[first]
    out[keep] = mec[keep]
    return out


def _window_worker(start_year):
    from qap.parallel import context
    c = context()
    inp, p, years = c["inp"], c["params"], c["years"]
    mi0 = start_year * 12
    i0 = mi0 - inp["first_mi"]
    T = years * 12
    if i0 < 0 or i0 + T > len(inp["months"]):
        return None
    sl = slice(i0, i0 + T)
    r = simulate_window(inp["pr"][sl], inp["dy"][sl], window_me0(inp, mi0), inp["mkt_pr"][sl], inp["mkt_dy"][sl], p, start_mi=mi0)
    r["start_year"] = int(start_year)
    r["years"] = int(years)
    r.pop("sel")
    return r


def run_windows(inp: dict, start_year: int, end_year: int, params: TaxParams, jobs: int = 1, years: int = 10, say=print) -> list:
    from qap.parallel import pmap, set_context
    set_context(inp=inp, params=params, years=years)
    res = pmap(_window_worker, range(start_year, end_year + 1), jobs, say=say)
    return [r for r in res if r is not None]


# ---------------------------------------------------------------- summary and decision

def annualised(w_a: float, w_b: float, years: float) -> float:
    """Geometric annualised % difference (w_a / w_b) ** (1 / years) - 1."""
    if not (w_a > 0 and w_b > 0):
        return float("nan")
    return ((w_a / w_b) ** (1.0 / years) - 1.0) * 100.0


def tracking_error(r_a: np.ndarray, r_b: np.ndarray) -> float:
    d = np.asarray(r_a, float) - np.asarray(r_b, float)
    d = d[np.isfinite(d)]
    return float(np.std(d, ddof=1) * np.sqrt(12.0) * 100.0) if d.size > 2 else float("nan")


def _dist(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan"), "p10": float("nan"), "share_pos": float("nan")}
    return {"n": int(x.size), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "p10": float(np.percentile(x, 10)), "share_pos": float((x > 0).mean())}


def decision_label(median_benefit: float, share_pos: float, te: float, liq_median: float,
                   min_benefit: float = 0.5, min_share: float = 0.75, max_te: float = 2.0) -> str:
    """Fixed rule: median hold-indefinitely benefit >= 0.5 %/yr AND share of positive windows >= 0.75 AND harvest TE <= 2.0 %/yr."""
    ok = bool(median_benefit >= min_benefit and share_pos >= min_share and te <= max_te)
    label = "WORTH PURSUING" if ok else "NOT WORTH THE COMPLEXITY"
    if liq_median <= 0:
        label += " (deferral only: liquidate-at-end median benefit <= 0)"
    return label


def summarize_windows(results: list) -> dict:
    rs = [r for r in results if r is not None]
    yrs = np.array([r["years"] for r in rs], float)
    sy = np.array([r["start_year"] for r in rs])
    g = lambda arm, k: np.array([r[arm][k] for r in rs], float)
    prim = np.array([annualised(r["hv"]["wealth_hold"], r["nohv"]["wealth_hold"], r["years"]) for r in rs])
    liq = np.array([annualised(r["hv"]["wealth_liq"], r["nohv"]["wealth_liq"], r["years"]) for r in rs])
    hv_etf = np.array([annualised(r["hv"]["wealth_hold"], r["etf"]["wealth_hold"], r["years"]) for r in rs])
    hv_etf_liq = np.array([annualised(r["hv"]["wealth_liq"], r["etf"]["wealth_liq"], r["years"]) for r in rs])
    nh_etf = np.array([annualised(r["nohv"]["wealth_hold"], r["etf"]["wealth_hold"], r["years"]) for r in rs])
    te_hv = np.array([tracking_error(r["hv"]["ret"], r["etf"]["ret"]) for r in rs])
    te_nh = np.array([tracking_error(r["nohv"]["ret"], r["etf"]["ret"]) for r in rs])
    dec = {d: _dist(prim[(sy // 10 * 10) == d]) for d in sorted(set((sy // 10 * 10).tolist()))}
    p = _dist(prim)
    te_med = float(np.nanmedian(te_hv)) if len(te_hv) else float("nan")
    liq_med = float(np.nanmedian(liq)) if len(liq) else float("nan")
    skipped = g("hv", "skipped_recent")
    hl = g("hv", "harvest_lots")
    return {
        "n_windows": len(rs), "primary": p, "primary_values": prim, "start_years": sy, "liq": _dist(liq), "hv_vs_etf": _dist(hv_etf),
        "hv_vs_etf_liq": _dist(hv_etf_liq), "nohv_vs_etf": _dist(nh_etf),
        "te_hv": {"median": te_med, "max": float(np.nanmax(te_hv)) if len(te_hv) else float("nan")},
        "te_nohv": {"median": float(np.nanmedian(te_nh)) if len(te_nh) else float("nan"),
                    "max": float(np.nanmax(te_nh)) if len(te_nh) else float("nan")},
        "by_decade": dec,
        "harvest_st_per_year": float(np.nanmedian([r["hv"]["harv_st"].mean() for r in rs])) if rs else float("nan"),
        "harvest_lt_per_year": float(np.nanmedian([r["hv"]["harv_lt"].mean() for r in rs])) if rs else float("nan"),
        "carry_end_median": float(np.nanmedian(g("hv", "carry_st_end") + g("hv", "carry_lt_end"))) if rs else float("nan"),
        "turnover": {a: float(np.nanmedian(g(a, "turnover"))) for a in ("nohv", "hv")} if rs else {},
        "n_names_end": {a: float(np.nanmedian(g(a, "n_names"))) for a in ("etf", "nohv", "hv")} if rs else {},
        "n_lots_end": {a: float(np.nanmedian(g(a, "n_lots"))) for a in ("etf", "nohv", "hv")} if rs else {},
        "wealth_median": {a: {"hold": float(np.nanmedian(g(a, "wealth_hold"))), "liq": float(np.nanmedian(g(a, "wealth_liq")))}
                          for a in ("etf", "nohv", "hv")} if rs else {},
        "skipped_recent_median": float(np.nanmedian(skipped)) if rs else float("nan"),
        "harvest_lots_median": float(np.nanmedian(hl)) if rs else float("nan"),
        "decision": decision_label(p["median"], p["share_pos"], te_med, liq_med),
    }


def format_summary(s: dict, title: str = "") -> list:
    f = lambda d: (f"median {d['median']:+.3f}  IQR [{d['q25']:+.3f}, {d['q75']:+.3f}]  p10 {d['p10']:+.3f}  "
                   f"share>0 {d['share_pos']:.2f}  (n={d['n']})")
    L = [title] if title else []
    L += [f"  PRIMARY harvest - no-harvest, hold-indefinitely, %/yr of wealth : {f(s['primary'])}",
          f"  harvest - no-harvest, LIQUIDATE at end, %/yr                    : {f(s['liq'])}",
          f"  harvest vs ETF benchmark, hold, %/yr                            : {f(s['hv_vs_etf'])}",
          f"  harvest vs ETF benchmark, liquidate, %/yr                       : {f(s['hv_vs_etf_liq'])}",
          f"  no-harvest vs ETF benchmark, hold, %/yr                         : {f(s['nohv_vs_etf'])}",
          f"  tracking error vs ETF, %/yr: harvest median {s['te_hv']['median']:.2f} (max {s['te_hv']['max']:.2f}); "
          f"no-harvest median {s['te_nohv']['median']:.2f} (max {s['te_nohv']['max']:.2f})",
          f"  median harvested loss per year (USD): ST {s['harvest_st_per_year']:,.0f}  LT {s['harvest_lt_per_year']:,.0f}; "
          f"end carryforward {s['carry_end_median']:,.0f}; harvest lots sold {s['harvest_lots_median']:.0f}; "
          f"lots blocked by the recent-purchase rule {s['skipped_recent_median']:.0f}",
          f"  turnover (sold/avg value/yr): no-harvest {s['turnover'].get('nohv', float('nan')):.3f}, harvest {s['turnover'].get('hv', float('nan')):.3f}; "
          f"end names {s['n_names_end']}; end lots {s['n_lots_end']}",
          f"  median terminal after-tax wealth (hold / liquidate), USD: "
          + "; ".join(f"{a} {w['hold']:,.0f} / {w['liq']:,.0f}" for a, w in s["wealth_median"].items()),
          f"  DECISION: {s['decision']}"]
    return L
