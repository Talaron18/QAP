# Roadmap

Spec: [docs/research_report.md](docs/research_report.md). Phases follow its §9.3.

**Data boundary:** WRDS extraction runs wherever the credentials live and writes parquet
under `data/` (git-ignored). Everything else reads only parquet. WRDS data and credentials
never enter git (licence).

1. **Foundation (done in this repo):** PIT-safe primitives with unit tests: CCM link filter,
   label builder (t+1 lag), purged/embargoed walk-forward splits, rank IC/ICIR, deflated Sharpe.
2. **Extraction (user runs it locally; start with `scripts/wrds_inspect.py`):** CRSP (CIZ), Compustat (+ Snapshot if licensed),
   CCM link, I/B/E/S. Table names live in config until verified against the live schema.
3. **Replicate, then build Tier-1 signals** (`scripts/run_signals.py`): gate 1 market vs FF Mkt-RF (corr >= 0.99),
   gate 2 canonical FF 2x3 SMB/HML (corr >= 0.95), gate 3 UMD (todo); signals only if gates pass.
   Original item: (value, profitability, investment, issuance, residual momentum)
   and compare with Chen-Zimmermann / JKP returns. This validates PIT alignment and links.
4. **Walk-forward evaluation** with a trial registry feeding DSR/PBO. Splits: dev 1990-2014,
   validation 2015-2019, sealed holdout 2020+.
5. **Slow-book composite + cost model**, then fast book, probabilistic layer, optimizer.
   Option/TAQ/Markit/RavenPack layers depend on which licences exist.
6. **Paper trade** 3+ months, then staged capital 10% -> 30% -> 100%.

## Licences / what the WRDS account actually exposes (confirmed from the inspector output)

- Present: CRSP (CIZ v2 plus legacy), CCM links, Compustat NA (funda, fundq, sec_shortint),
  Fama-French factors, Total Q (totalq.total_q: intangible capital k_int, q_tot), WRDS Financial
  Ratios (wrdsapps_finratio), WRDS characteristic signals (wrdsapps.signals_raw_basic/plus), link
  tables to TAQ/IBES/OptionMetrics identifiers.
- **Absent as data** (sample libraries or links only): I/B/E/S, OptionMetrics, TAQ, RavenPack,
  Thomson 13F, Compustat Snapshot, Markit. So analyst revisions, option signals, retail flow, news
  and 13F signals are out of scope. Without Snapshot, use the conservative filing lags from
  report section 7.4 (rdq for quarterly; fiscal year-end + 4 months for annual).
- Useful for replication (§9.3 step 1): wrdsapps.signals_raw_* can cross-check my own Tier-1 signals.
- CIZ delisting flags confirmed (crsp.metaflaginfo): mthret already includes the delisting return
  for mthdelflg A/P/V and excludes it for G (price >10 days from delisting) and M (missing). Of
  4.84M monthly rows, 3,746 are G/M. `qap.crsp.adjust_delisting_returns` fixes those: compound the
  stkdelists delret when usable, else -30% for performance-related reasons (Shumway), else 0.
- Monthly labels: signals formed at month-end t earn month t+1, i.e.
  `forward_return_from_returns(ret, horizon=1, lag=0)`; the daily default (lag=1) skips a day.

## Status log

- Gates 1-2 passed on real data (market corr 0.99999; SMB 0.995; HML 0.983). Gate 3 (UMD) added,
  awaiting a run.
- Governance note: the first signal run reported the FULL sample, including 2020+. The three signals
  (BM, GP/A, asset growth) are published, untuned replications, so little is lost, but the 2020+
  holdout has now been seen for them. Any NEW signal, model or tuning must be selected on 1990-2014
  (validated on 2015-2019) with 2020+ untouched; `scripts/run_signals.py` now reports each split
  separately and appends every evaluation to `data/trials.jsonl` (trial count for deflated Sharpe).
