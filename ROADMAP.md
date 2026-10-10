# Roadmap

Spec: [docs/research_report.md](docs/research_report.md). Phases follow its §9.3.

**Data boundary:** WRDS extraction runs wherever the credentials live and writes parquet
under `data/` (git-ignored). Everything else reads only parquet. WRDS data and credentials
never enter git (licence).

1. **Foundation (done in this repo):** PIT-safe primitives with unit tests: CCM link filter,
   label builder (t+1 lag), purged/embargoed walk-forward splits, rank IC/ICIR, deflated Sharpe.
2. **Extraction (user runs it locally; start with `scripts/wrds_inspect.py`):** CRSP (CIZ), Compustat (+ Snapshot if licensed),
   CCM link, I/B/E/S. Table names live in config until verified against the live schema.
3. **Replicate Tier-1 signals** (value, profitability, investment, issuance, residual momentum)
   and compare with Chen-Zimmermann / JKP returns. This validates PIT alignment and links.
4. **Walk-forward evaluation** with a trial registry feeding DSR/PBO. Splits: dev 1990-2014,
   validation 2015-2019, sealed holdout 2020+.
5. **Slow-book composite + cost model**, then fast book, probabilistic layer, optimizer.
   Option/TAQ/Markit/RavenPack layers depend on which licences exist.
6. **Paper trade** 3+ months, then staged capital 10% -> 30% -> 100%.

## Licences (from the user's WRDS account, Oct 2026)

- Full access: CRSP, Compustat-Capital IQ, Thomson/Refinitiv (13F; I/B/E/S to confirm),
  Cboe Options, Intraday Indicators, Beta Suite, Factors, Financial Ratios Suite,
  Event Study, Linking Queries, Fama-French, Pastor-Stambaugh, Peters-Taylor Total Q.
- **Sample only** (not usable for research): OptionMetrics, TAQ, RavenPack, Short Volume.
- Possibly covered, confirm in the inspector output: Intraday Indicators by WRDS (realized
  variance and spreads, so cost model and HAR vol layer may not need TAQ), Peters-Taylor Total Q
  (intangible capital for the value composite), Compustat Snapshot (not clearly in the list; if
  absent, use the conservative filing lags from report section 7.4).
- Out of scope without more licences: option-implied signals (OptionMetrics), TAQ retail flow,
  RavenPack news. Build the slow book (Tier 1) first.
