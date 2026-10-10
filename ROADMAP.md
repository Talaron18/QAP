# Roadmap

Spec: [docs/research_report.md](docs/research_report.md). Phases follow its §9.3.

**Data boundary:** WRDS extraction runs wherever the credentials live and writes parquet
under `data/` (git-ignored). Everything else reads only parquet. WRDS data and credentials
never enter git (licence).

1. **Foundation (done in this repo):** PIT-safe primitives with unit tests: CCM link filter,
   label builder (t+1 lag), purged/embargoed walk-forward splits, rank IC/ICIR, deflated Sharpe.
2. **Extraction (blocked on WRDS access):** CRSP (CIZ), Compustat (+ Snapshot if licensed),
   CCM link, I/B/E/S. Table names live in config until verified against the live schema.
3. **Replicate Tier-1 signals** (value, profitability, investment, issuance, residual momentum)
   and compare with Chen-Zimmermann / JKP returns. This validates PIT alignment and links.
4. **Walk-forward evaluation** with a trial registry feeding DSR/PBO. Splits: dev 1990-2014,
   validation 2015-2019, sealed holdout 2020+.
5. **Slow-book composite + cost model**, then fast book, probabilistic layer, optimizer.
   Option/TAQ/Markit/RavenPack layers depend on which licences exist.
6. **Paper trade** 3+ months, then staged capital 10% -> 30% -> 100%.
