# Alpha research registry (do not re-test closed rows)

One line per tested family: what was tested, the verdict, what sample was spent, where the evidence lives. Counts are unique logged trials (76 so far, plus two forward-only ML models).
New work must add a row BEFORE running and must state the data check, the power estimate and the decision it could change.

| Family | Tested (pre-registration) | Verdict | Sample spent | Evidence |
|---|---|---|---|---|
| Replicated factors (value, gpa, asset growth, composite, momentum) | v1 | Decayed; only gpa survives costs, weakly (dev +0.18 %/mo t 1.9) | 1963-2025 (2020+ shown once) | ROADMAP status log, `docs/` |
| Candidate quality/profitability/intangible variants (6) | v1 | Nothing promoted vs gpa | dev, validation | ROADMAP v1 |
| Long-only gpa tilt (buffer + 3% cap) | v2 | Not validated (holdout +0.056 %/mo, t 0.36) | 2020+ for gpa models SPENT | ROADMAP v2 |
| JKP themes + composites (15) | v3 / Study A | Nothing promoted; composite IC decays after 2014, net ~0, short-leg driven | dev, validation | `docs/study_a_report.txt` |
| Innovation family (R&D, patents, KPSS, RQ) | v3 / Study B | Nothing promoted | dev, validation | `docs/study_b_report.txt` |
| Conversion diagnostic (why IC does not become return) | v4(b) | Costs + short leg + decay; not a small-cap effect | pre-2020 only | `docs/diagnostic_report.txt` |
| Single-stock leveraged/inverse ETFs as the short leg | v5 | NOT VIABLE (2025 coverage of bottom quintile 0.1-5.6%) | no signal returns used | ROADMAP v5 |
| Long-short `jkp_slow10`, borrow break-even | v6 | INCONCLUSIVE (net 0.50 %/mo, t 0.89; all in 2021-22) | 2020+ for JKP composites SPENT | ROADMAP v6 |
| Tax-loss harvesting on a DIY index | v7 | Not worth it (+0.17 %/yr hold, ~0 liquidated) | full history, mechanical | ROADMAP v7 |
| ML cross-sectional model | v8 | PENDING forward (walk-forward: net +0.01 vs composite +0.24 vs gpa +0.14; prior: no evidence) | forward only | `paper_ledger/ml_manifest.jsonl` |
| Alternative data (arc, risk text, blockholders, boards, culture, litigation) | v9 / Study C | Nothing promoted | dev, validation | ROADMAP v9 |
| Earnings-announcement expectation change (EAR, SUE, RSUE, EAR+SUE) | v11 / Study D | Nothing promoted; gross drift real pre-2015 (IC t 5-6), gone after; short-horizon (months 0-1), net ~0 | dev, validation | ROADMAP v11 |
| gpa paper-trading ledger | v4(a) | RUNNING (implementation check) | forward | `paper_ledger/manifest.jsonl` |
| International replication of the JKP findings | v10 | DONE (descriptive): H1 MIXED, H2 NOT SHORT-LEG, H3 US-SPECIFIC; ex-US spreads roughly halve after 2014 (not Japan), net ~0; no follow-up licensed | non-US all periods | ROADMAP v10 |

## Candidate families: data feasibility (verified against the WRDS inventory/probe)
| Family | Needs | Accessible? | Note |
|---|---|---|---|
| Analyst revisions / forecast changes | I/B/E/S, Zacks | NO (confirmed absent) | cannot be tested as written |
| Earnings surprise / PEAD | Compustat quarterly (`rdq`) | YES | SUE-type JKP members already sit inside the Profit Growth theme (Study A: dev IC t 2.6 but net negative); an announcement-window return (EAR) from daily CRSP is the untested variant |
| Insider purchases, 13F changes | Thomson insiders/13F | NO (confirmed absent) | `blocksv` (13D/G) tested in Study C |
| Short interest | Compustat `comp.sec_shortint` | PROBED OK: 5.37M rows, 2006-07-14 .. 2026-09-30 | DONE as v12 / Study E: nothing promoted, veto not licensed (see ROADMAP) |
| Order flow / retail-institutional imbalance | TAQ, OptionMetrics, Cboe | NO (permission denied) | |
| Option-market information: index level | `cboe.cboe` / `cboe_all.cboe` | YES: daily VIX (OHLC, 1990+), VXO (1986+), VXN (2001+), VXD (1997+), to 2026-09-30 | verified by `scripts/wrds_access_map.py` 2026-10-12; enables the index-level variance-risk-premium test |
| Option-market information: stock level | `optionm_all`, `optionm_europe`, `cboe_eod` | NO (permission denied; only samples) | needs a paid external feed, see docs/option_data_sources.md |
| Short volume / FINRA short interest | `wrds_short_volume`, `wrds_finra_short_interest` | NO (permission denied) | Compustat `sec_shortint` was the usable source (v12) |
| Volatility forecasting (Study F) | CRSP daily returns, FF daily | YES | DONE: v13 gjr_t licensed in dev/validation and confirmed in the holdout, which is now SPENT (see ROADMAP) |
| Liquidity / spread / turnover / volatility | CRSP daily, JKP | YES | already inside JKP themes (Size, Low Risk); `wrdssec_midas` (2012-2025) is the only microstructure source; cost-dominated per the research report |
| International replication | JKP non-US rows | YES | done as v10 |
| Filing dates (PIT upgrade) | `comp.filings` | NO (permission denied for schema comp_filings) | `rdq` from `fundq` is the only timeliness source |
| Customer-supplier links | `comp.seg_customer` | PARTLY: accessible but only 2020-06..2026-07 and customer names only (no customer gvkey) | effectively infeasible; a pre-2020 test is impossible |
| Dividend / distribution events | `crsp.dsedist` | YES (declaration dates to 2024-12-31) | untested; low prior |
