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
