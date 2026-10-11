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
- All three gates passed on real data (UMD corr 0.999). Replicated-signal results by split (LS = VW
  quintile spread, % per month, ex-microcaps): BM-timely 0.60 (t 2.7) pre-1990, 0.06 (t 0.2) 1990-2014,
  -0.60 2015-19; GP/A 0.32 / 0.33 / 0.18 (positive in every split); neg asset growth 0.42 / 0.29 / -0.28.
  Expect decay, not new alpha (McLean-Pontiff). Holdout is now SEALED by default in the runner
  (`--open-holdout` to show 2020+, once per major model version).
- Next candidates: composite and 12-1 momentum evaluation (added), intangibles-adjusted value using
  `totalq.total_q` (k_int), industry neutralisation (needs siccd in the panel), cost model, optimizer.
- Composite and momentum (pre-2020, gross): composite_vpi LS 0.52%/mo (t 4.15) but -0.48%/mo (t -1.16)
  in 2015-2019; momentum flat in 2015-2019. The composite failed validation, and its 2020+ is not a clean
  holdout (a fixed function of signals whose 2020+ was already shown); only NEW ingredients get a clean one.
- Cost model built (`qap.costs`, `qap.portfolio`, `scripts/run_costs.py`): half-spread (quoted, else
  Corwin-Schultz, else month x size-decile median) + square-root impact (Y=1) + optional commission;
  stress 1x/1.5x/2x, AUM $1M/$10M/$100M, break-even multiple, share of trades above 5%/10% ADV. Books:
  LS (borrow NOT modelled) and long-minus-universe. Awaiting the `crsp_daily_agg` extraction and a run.
  Partial-rebalance / no-trade-band rules are later trials, chosen on 1990-2014 only.
- First costed run (nominal constant AUM) is NOT reliable: a fixed $10M book is huge against 1960s-70s
  volumes, so early-era impact was overstated. Fixed: AUM is now a constant share of total equity market
  cap, quoted in 2019 market-size dollars (`--nominal-aum` restores the old behaviour). Also known:
  quoted and Corwin-Schultz spreads agree poorly (corr 0.15-0.58; CS is far above quoted for large caps
  after 2000, far below for small caps before 1990), so pre-1990 cost levels are the least trustworthy.
  Quoted spreads supply 86-99% of rows after 1990. Re-run `scripts/run_costs.py` for decision numbers.
- Costed run with market-scaled AUM (impact Y=1, no commission, 1x/1.5x/2x; $10M in 2019 market-size
  dollars; net of trading costs, borrow not modelled). Costs are small (gpa ~0.015-0.035%/mo, composite
  ~0.08-0.14%, momentum ~0.18-0.41%). Post-1990 (1990-2019) net at 1.5x, long-minus-universe: gpa +0.17%/mo
  (t 1.98), composite +0.07 (t 0.63), neg_ag +0.01 (t 0.16), mom +0.07 (t 0.36), bm_timely -0.08 (t -0.54).
  Long-short post-1990: gpa +0.24 (t 1.36), composite +0.14 (t 0.81). Conclusion: after 1990 and after costs
  only gross profitability survives; the composite's edge sits mostly in pre-1990, where cost estimates are
  least reliable (1960s spreads are 92% Corwin-Schultz and look too low vs the quoted-spread 1970s).
  35 unique trials logged. Next: profitability/quality variants chosen on 1990-2014 only, intangibles-
  adjusted value from totalq.total_q, then a long-only tilt portfolio with an optimizer.

## Pre-registration v1: profitability / quality / intangible-value candidates

Committed BEFORE any candidate result exists. Nothing below is revisited after seeing results.

**Fixed protocol (as in the gpa costed run):** formation universe = common equity mask ex-NYSE-20th-percentile
microcaps; quintiles on value-weighted ME; long = top quintile, 'L-U' book = long minus the VW universe
(benchmark costless); AUM $10M in 2019 market-size dollars, impact Y=1, no commission, cost stress 1.5x;
annual fundamentals available fiscal year-end + 4 months; Total Q available the same way. Splits: dev 1990-2014,
validation 2015-2019, 2020+ SEALED (opened once, later, for promoted signals only).

**Incumbent:** gpa = (revt - cogs) / at. On dev its net L-U at 1.5x has t about 1.9, so the bar is RELATIVE to gpa.

**Candidates (six, formulas fixed):**
1. `op_ff` = (revt - cogs - xsga - xint) / be. Missing cogs/xsga/xint count as 0 only if at least one is present; be > 0.
2. `neg_acc` = -(ib - oancf) / at (cash-flow accruals, low is good). Needs oancf (sparse before ~1988).
3. `fscore` = Piotroski (2000) nine binary signals on ALL stocks (the high-BM-only variant is excluded):
   ni/at > 0; oancf > 0; ni/at rose vs last year; oancf/at > ni/at; dltt/at fell; act/lct rose; no equity issuance
   (sstk missing or 0 counts as no issuance; sstk > 0 counts as issuance); (revt-cogs)/revt rose; revt/at rose.
   ROA uses end-of-year assets (not beginning). Score = sum of available components, needs >= 8 of 9, else missing.
4. `neg_issuance` = -ln(adj_shares_t / adj_shares_{t-12}), adj_shares = shrout * mthcumfacshr (CRSP, monthly).
5. `bm_int` = (be + k_int) / me_co with k_int = Peters-Taylor total intangible capital (totalq.total_q.k_int), be > 0.
6. `quality3` = equal-weight average of within-month rank scores of gpa, op_ff, neg_acc (needs >= 2 of 3).

**Excluded on purpose:** cash-based operating profitability (needs a new funda pull); Piotroski restricted to high
book-to-market; Total Q's q_tot standalone; any parameter tuning, alternative weighting or alternative quantiles.

**Promotion rule (coded in the script, which prints PASS/FAIL):** regress the candidate's monthly net L-U return on
gpa's net L-U return over dev: r_c = a + b * r_gpa + e. Promote iff (i) alpha t-stat >= 2.0 on dev (plain OLS) and
(ii) validation alpha = mean(r_c - b_dev * r_gpa) over 2015-2019 is > 0. Information only (never a gate): standalone
net mean/t, correlation with gpa, equal-weight gpa+candidate vs gpa alone (dev and validation), NW(6) alpha t.
bm_int is also compared with bm_timely on the rows where both exist. Promoted signals are provisional: the sealed
holdout is opened once for them. If nothing is promoted, that is a valid result and the next step is the gpa-only
long-only tilt.

**Checks printed before results:** Total Q coverage by year and join rate; median k_int/at in [0.05, 2] (asserted);
share of |monthly d ln adj_shares| > 0.4 below 0.5% (asserted; catches broken split adjustment); oancf coverage per
signal per split; long-leg names per month for each candidate (F-score ties).

**Amendment to pre-registration v1 (made before any candidate result exists; result-independent):** the first real run
failed the split-adjustment check, which turned out to be too crude (it counted fund/ETF share jumps and used a flat
threshold). It is replaced by a spike test at exact split ratios on common equity, with the alternatives printed. A failed
DATA check now drops only the candidate that depends on that data (`neg_issuance` for the split check, `bm_int` for the
Total Q checks) and says so; the promotion rule, thresholds and other candidates are unchanged.
