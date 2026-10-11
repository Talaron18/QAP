# Research directions given the WRDS access (Oct 2026)

Basis: the user's subscription list, the first schema dump (which only covered libraries whose names matched a few
hints), and a few web checks. Items marked VERIFY need `scripts/wrds_inventory.py` output before anything is built.

## What the evidence so far says
Published, well-known anomalies (value, investment, momentum, accruals, issuance, F-score) did not survive costs
and the post-1990 period in this pipeline; gross profitability survived weakly and its long-only tilt decayed to
+0.056%/mo (t 0.36) in 2020+. More data helps only if it carries information that is (a) not yet widely arbitraged,
(b) available at the time, (c) implementable in liquid names. Every new signal is a logged trial; expect most to fail.

## Access, by usefulness
**Usable now (visible in the schema dump):** CRSP (CIZ + legacy, S&P 500 history, mutual funds), Compustat NA
(fundamentals, quarterly, segments, short interest, SEC filing dates, daily/monthly prices), Total Q (intangible
capital), Fama-French/liquidity factors, WRDS Financial Ratios, WRDS characteristic signals (`signals_raw_*`),
DGTW benchmark returns (`wrdsapps.mastertable`), US patents (KPSS-linked: `wrdsapps_patents`), EX-21 subsidiaries,
ExecuComp, event-study returns.
**Probably usable, VERIFY:** Thomson/Refinitiv (13F s34 holdings, insider filings: WRDS lists both as Thomson Reuters
products), Blockholders (13D/G), Cboe indexes, Contributed Data (33 datasets), TRACE (corporate bonds), Macro Finance
Society (WRDS lists a single commodity-trade product), Historical SPDJI (index membership), Beta Suite, Intraday
Indicators, Public Data, Federal Reserve Board reports, Research Quotient, OTC Markets.
**Sample only (not usable for research):** OptionMetrics, TAQ, RavenPack, FactSet, ISS, MSCI, BoardEx, Audit Analytics,
WRDS Insiders, Short Volume, ETF Global, HFR, Zacks, PitchBook and the other "| Sample" items.
**Irrelevant for US equity alpha:** Bank Regulatory, Bureau van Dijk (private firms), CSMAR (China), MSRB (munis),
DMEF, Federal Judicial Center, NielsenIQ/IRI (samples or consumer panels without a firm link), European short data.

## Candidate directions (rank = expected information per unit effort, our judgement)
1. **Innovation / patents** (`wrdsapps_patents`: grant date, forward/backward citations, gvkey link score).
   Signals: citation-weighted patents per unit of R&D capital (Hirshleifer-Hsu-Li "innovative efficiency"), R&D
   intensity (we already have xrd, and Total Q's `k_int_know`). Information sits in intangibles that accounting
   expenses; complements gross profitability. Risks: coverage after the late 2010s, truncation of citations,
   link quality. Effort: medium. VERIFY coverage by grant year first.
2. **Short interest as a long-side veto** (`comp.sec_shortint`, twice monthly). Report: much of this predictability
   is borrow-fee driven, so use only as a filter on a long-only book. Effort: low. Expected gain: small.
3. **Mutual-fund flows and holdings** (`crsp_q_mutualfunds.holdings` with permno, `fund_flows`, `monthly_tna`).
   Flow-induced price pressure (Coval-Stafford; Lou). Effort: high (fund-to-stock aggregation, report-date vs filing-date
   timing); likely decayed after 2008.
4. **13F institutional ownership/breadth and insider trades** (Thomson, VERIFY). Chen-Hong-Stein breadth; insider
   opportunistic trades. Filing-date PIT (13F up to 45 days late). Effort: medium. Probably crowded.
5. **Economic links** (`comp.seg_customer`: customer names and sales, no customer gvkey). Cohen-Frazzini customer
   momentum. Needs fuzzy name matching to gvkeys: high effort, noisy.
6. **Better point-in-time fundamentals** (`comp.filings` gives 10-K/10-Q filing dates; `fundq` has `rdq`). Replace the
   4-month lag with true filing dates and update profitability quarterly. A methodology upgrade for ALL existing
   signals; low risk; shifts timing, not information.
7. **Market-level overlays** (Cboe indexes, Fed reports, S&P index history): risk budgeting only, per the research report.
8. **Reversal / liquidity-provision**: strong gross, cost-dominated; the cost model now exists, so it can be tested
   honestly, but the report calls it the most cost-constrained anomaly. Needs a fast-book design.

## Rules for any new direction
Pre-register (candidate list, formulas, rule, exclusions) in its own commit BEFORE results; judge new signals
relative to the incumbent (gpa); test on 1990-2014, validate on 2015-2019; 2020+ stays sealed and is only "clean" for
genuinely new ingredients; log every spec as a trial.


## Update after the full library inventory (236 libraries; `data/wrds_inventory.txt`)
The inventory overturns two assumptions. Libraries come in pairs (`x` = views with unknown row counts, `x_all` = counts), so
"0 rows" means unknown, not empty.

**Confirmed absent (samples only):** Thomson 13F and insider filings, I/B/E/S, OptionMetrics (2014 trial only), TAQ,
RavenPack (trial), FactSet, Zacks, ISS, MSCI, BoardEx, Audit Analytics, Compustat Snapshot, Capital IQ, SDC.

**Newly found, likely full access (VERIFY with `scripts/wrds_probe.py`):**
- **`cboe` (90 tables):** names (`eqmaster`, `eqprice`, `eqhvol`, `ivlisted_1998`..`ivlisted_2020+`, `ivcmpr`, `ivcmdr`,
  `ivexpr`, `ivborrowrate`, `deltanumber`) look like an equity-option database with implied volatilities, historical
  volatility and STOCK BORROW RATES. If confirmed, this is the biggest unlock: option-implied signals (IV spread, skew,
  term structure, IV-RV) and, crucially, borrow costs to judge them net, which the research report says decides whether
  they are tradable at all.
- **`contrib_global_factor`:** `global_factor` (16M rows) is almost certainly the Jensen-Kelly-Pedersen Global Factor Data
  (153 characteristics, stock-level), plus `ctff_*` (103M daily return rows). Gives (a) an independent cross-check of our
  own signal construction and (b) the 13-theme composite the report recommends, without rebuilding every characteristic.
- **`contrib_kpss` + `contrib_patent_firm_link`:** KPSS patent values and a patent-firm link (2.95M patents).
- **`contrib_intangible_value`:** intangible-adjusted value characteristics and factors (`int_xsec`, `int_factors`).
- **`contrib_general`:** `better_beta`, `liva`, `arc`, `classified_boards`, `marginal_tax`, `as_firm_risks`, `blocksv`,
  `common_own_firm` (1.1B rows of common ownership), `factors`.
- **`macrofin`:** Hou-Xue-Zhang q-factors (daily to annual), consumption-wealth `cay`, Jurado-Ludvigson-Ng uncertainty.
- **Also:** `rq_all.rq_data` (Research Quotient, 52,779 rows), `trace_enhanced` (455M bond trades), `wrdsapps.bondret`,
  `contrib_bond_firm_link`, `otc_endofday` (77M OTC rows), `wrdssec_midas.security` (21M, SEC market-structure metrics),
  `fjc_litigation`, `block_all` (small), `frb`, `djones`, `comp_na_daily_all.secd` (161M daily prices).

**Revised ranking (pending the probe):** (1) Cboe option-implied signals net of borrow rates; (2) JKP characteristics for
a pre-registered theme composite and as a construction cross-check; (3) patents/innovation with KPSS values;
(4) intangible-adjusted value; (5) short interest as a veto; then the rest. Option signals are the only family here the
report flags as a genuinely different information source, and the only one the cost model cannot yet judge (borrow).
