# Research map: economic mechanisms, what was tested, what is left (2026-10-12)

Companion to [REGISTRY.md](REGISTRY.md) (per-study verdicts) and [research_directions.md](research_directions.md) (data inventory).
Status codes: **TESTED** (pre-registered, judged), **INSIDE JKP** (covered by the 153 JKP characteristics / 13 themes of Study A, not separately),
**PENDING** (frozen forward test), **UNTESTED-DATA?** (data believed accessible, access not yet verified by a probe), **UNTESTED-DATA OK** (data in hand), **BLOCKED** (data confirmed absent).
Prior = my rough probability that a clean, pre-registered test would produce a signal that survives costs in a liquid long-only book (not an estimate of statistical significance).

## 1. Mechanisms already covered

| Mechanism family | Specific channels | Status | Verdict (evidence) |
|---|---|---|---|
| Risk premia / classic factors | value (B/M, E/P, CF/P, intangible-adjusted), size, market beta / low risk, momentum, investment, accruals, issuance, F-score-type quality | TESTED (v1, v3/Study A, ML features) | decay after 1990 and 2014; only gpa survives costs, weakly |
| Profitability / quality | gross profitability, ROA, Novy-Marx / quality themes, long-only tilt | TESTED (v1, v2; ledger running) | dev +0.18 %/mo (t 1.9), tilt holdout +0.06 (t 0.4) |
| Composite of everything in JKP | 13 themes, `jkp_all13`, `jkp_slow10`, long-short | TESTED (Study A, diagnostic, v6 one shot) | IC real, net ~0, edge in the short leg, inconclusive out of sample |
| Intangibles / innovation | R&D capital, patents, KPSS value, Research Quotient | TESTED (v1 candidates, Study B) | nothing promoted |
| Expectation change from earnings | announcement-window return, SUE, revenue SUE, composite | TESTED (v11 / Study D) | gross effect real before 2015 (IC t 5-6), mostly gone after, net ~0 |
| Ownership / governance / text | blockholders, classified boards, culture, 10-K risk text, XBRL complexity, federal litigation | TESTED (v9 / Study C) | nothing promoted |
| Short selling implementation | single-stock inverse ETFs; direct margin short and break-even borrow | TESTED (v5, v6) | ETFs not viable; margin short inconclusive |
| Cross-market | same signals in 22 developed ex-US markets | TESTED (v10, descriptive) | IC persists (not Japan), spreads halve after 2014, net ~0, 2021-22 mostly US |
| Machine-learning nonlinearity | gradient boosting on 20 fixed features | PENDING (v8, forward only, decision not before ~2029) | walk-forward: IC higher but net +0.01 vs composite +0.24 |
| Implementation | cost model, turnover buffer, position cap, tax-lot harvesting, passive core | TESTED (cost model, v2, v7) | harvesting ~+0.17 %/yr best case, ~0 if liquidated |
| Mean reversion / seasonality / lottery / liquidity | 1-month reversal, seasonality, MAX/idio-vol, Amihud and spread measures | INSIDE JKP | reversal and seasonality strongly negative net (turnover 60-84 %/mo) |
| Distress / leverage | O-score, Z-score, leverage themes | INSIDE JKP | Low Leverage theme: nothing |

JKP's 153 characteristics span most published characteristic anomalies, so the unexplored space is mostly **not** more firm characteristics. Ranking the best individual JKP characteristics would be data mining, not a new mechanism.

## 2. Mechanisms NOT yet tested

| # | Mechanism | Idea | Data | Status | Effort | Prior | Decision it could change |
|---|---|---|---|---|---|---|---|
| 1 | Short interest | informed shorts predict poor returns; usable as a long-side veto (no borrow needed) | Compustat `comp.sec_shortint` (twice monthly) | PRE-REGISTERED v12 (probe confirmed 2006-07..2026-09; I wrongly called it absent earlier) | low | low-moderate (effect is partly borrow-fee driven) | whether to exclude heavily shorted names from the tilt |
| 2 | Industry momentum / lead-lag | industry returns predict firm returns, distinct from firm momentum | CRSP + SIC (in hand) | UNTESTED-DATA OK | low-medium | low (decayed) | none unless it beats gpa |
| 3 | Customer-supplier links | news at customers reaches suppliers slowly (Cohen-Frazzini) | Compustat `comp.seg_customer` (names, no customer gvkey) | BLOCKED in practice: probe shows only 2020-06..2026-07 | high (fuzzy name matching) | low-moderate | only if net of costs in large caps |
| 4 | Mutual-fund holdings and flows | flow-induced price pressure, fund herding (Coval-Stafford) | `crsp_q_mutualfunds` (holdings, flows, TNA) | UNTESTED-DATA? | high (fund-to-stock aggregation, report-date timing) | low (decayed after 2008) | none likely |
| 5 | Corporate-event drift beyond earnings | dividend initiations/omissions, spin-offs, new listings (IPO age), stock splits | CRSP distributions (`dsedist`, accessible to 2024-12-31) and listing dates | UNTESTED-DATA OK | medium | low (small, illiquid names) | none likely |
| 6 | Overnight vs intraday returns | the two halves of the day carry different premia (retail vs institutions) | CRSP daily open/close in `dsf_v2` | UNTESTED-DATA OK | medium | low; daily turnover makes it cost-dominated | none (not implementable at USD 35k in a taxable account) |
| 7 | Bond-market information | corporate-bond trades/spreads lead equities | `trace_enhanced` (455M trades), `contrib_bond_firm_link` | UNTESTED-DATA? | very high | unknown, distinct information | would need a liquid-name implementation |
| 8 | Executive compensation / incentives | pay-for-performance, option grants | ExecuComp | UNTESTED-DATA? | medium | low | none likely |
| 9 | Index membership changes | S&P 500 additions/deletions | CRSP S&P 500 history / SPDJI | UNTESTED-DATA? | low | very low (arbitraged since ~2000) | none |
| 10 | Macro regime conditioning | factor premia vary with uncertainty, cay, valuation spreads | `macrofin` (JLN uncertainty, cay to 2019), FF, our factors | UNTESTED, deliberately | medium | very low (few independent regimes) | would only be a simple vol-managed rule judged forward |
| 11 | Quarterly timeliness | update profitability at the filing date (`rdq`, `comp.filings`) instead of a 4-month lag | Compustat `fundq` (`comp.filings` is permission-denied, so only `rdq`) | PARTLY BLOCKED | low | not a new mechanism; shifts timing only | slightly better PIT for existing signals |

## 3. Mechanisms blocked by data (confirmed absent or sample only)
analyst revisions and forecasts (I/B/E/S, Zacks); insider trades and 13F institutional ownership (Thomson); option-implied information and borrow rates (OptionMetrics, Cboe: permission denied); intraday order flow and microstructure (TAQ; `wrdssec_midas` covers only 2012-2025 and was not used); news and call-transcript sentiment (RavenPack, FactSet); credit default swaps; ISS/BoardEx governance; Audit Analytics; PitchBook; SDC deals (M&A, IPO, SEO). These are the places where the literature finds the least-arbitraged information, and exactly where we cannot test.

## 4. Reading the map
- The covered rows are not "failed ideas of equal quality": most carried a real **gross** effect before about 2015 that turned out to be cost- or decay-limited in the tradable (value-weighted, ex-microcap, long-only) book.
- The remaining untested mechanisms are, on my honest read, low prior: the two with the highest expected information per unit effort are #1 (short interest, now pre-registered as v12) and #11 (a timing upgrade, not a new signal).
- Anything on the list is a new family and must go through the usual gate: data probe first, power estimate, pre-registration, one judged run.
- Completeness caveat: this taxonomy follows the published anomaly literature and our data inventory; it is not a proof that no other mechanism exists.
