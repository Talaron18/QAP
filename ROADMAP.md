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

- Candidate test result (pre-registration v1, commit e81cb8a + amendment): NOTHING PROMOTED. quality3 standalone net L-U dev
  +0.165%/mo (t 2.12), validation +0.163%, but 0.73 correlated with gpa and alpha t 1.09; op_ff alpha t 0.84; fscore 0.86;
  neg_acc 0.23; neg_issuance -0.21; bm_int 0.99. Split-adjustment convention shrout*mthcumfacshr confirmed (spike ratio 1.3 vs
  3.9 and 13.1). bm_int vs bm_timely on common rows: +0.06% vs -0.07% (both insignificant). 43 unique trials logged.
  IC vs VW-book gap (fscore/op_ff/neg_issuance/quality3: dev IC t 3.5-6.3, weak VW L-U) is information only; not pursued.

## Pre-registration v2: gpa-only long-only tilt (course change, stated plainly)

I told the user the next step would be a long-only tilt "with an optimizer". I am NOT building a factor risk model and a
mean-variance optimizer now: the top-quintile long-minus-universe book already IS the long-only tilt (net active return
about +0.18%/mo, t about 1.9), so a new construction can only change tracking error, turnover and implementability, and a
risk model plus optimizer add many knobs (risk aversion, TE target, model choices), each a trial, with no evidence they
help for one weak signal. Committed before any tilt result exists.

**Spec (final spec = (c); fixed a priori):** universe, signal (gpa), value weights, costs ($10M in 2019 market-size dollars,
Y=1, 1.5x stress, benchmark costless) as before. Signal rank = within-month percentile among eligible names.
(a) baseline: hold names with rank pct > 0.80, value-weighted (this is the measured top-quintile book);
(b) + turnover buffer: enter at pct > 0.80, an existing holding stays while pct > 0.70;
(c) + position cap: as (b), no weight above 3% (excess redistributed pro rata; water-filling). FINAL SPEC = (c).
(a) and (b) are ablations for information, NOT for selection. Benchmark = value-weighted eligible universe.
**Reported by split (pre-1990, dev 1990-2014, validation 2015-2019):** net active return (mean %/mo, t), tracking error and
information ratio (annualised), active share, one-way monthly turnover, holdings count, max weight, top-10 weight.
**Information only (nothing neutralised):** regression of net active return on mktrf/smb/hml/umd; industry (SIC2)
concentration vs benchmark. 2020+ opens ONCE, for (c). It is not fully clean: gpa's gross 2020+ long-short was printed in
the first signals run.
**Not decided here:** whether to deploy (edge about 0.1-0.2%/mo net, t below 2; taxes on short-term gains from monthly
turnover are not modelled). Paper trading needs a different price source: CRSP here ends 2025-12 (annual update) while
Compustat runs to 2026-09; Compustat security price tables or another feed would be needed.

- Tilt result (pre-registration v2, 0fef4ed; holdout sealed; $10M 2019 market-size, 1.5x costs, benchmark VW eligible
  universe, costless): FINAL SPEC (c) buffer + 3% cap: net active +0.216%/mo (t 2.56, TE 5.1%, IR 0.51) on dev 1990-2014,
  +0.096% (t 0.78) on validation 2015-19, +0.124% (t 1.35) pre-1990; ~490 holdings, max weight 3%, top-10 weight ~29%,
  active share ~0.73-0.76, one-way turnover 1.2-1.9%/mo, cost@1x 0.001-0.012%/mo. Ablations (information only):
  (a) baseline dev +0.184 (t 1.88), (b) buffer only +0.148 (t 1.65). Information only: FF4 alpha dev +0.305%/mo (t 4.05),
  validation +0.066 (t 0.68); HML loading -0.22 to -0.33 (t -5 to -10) i.e. a strong growth tilt; industry active weights
  dev: -7% finance (SIC 60), +10% chemicals/pharma (SIC 28). 46 unique trials logged. Next: open 2020+ ONCE for (c)
  (`--open-holdout`); then the deployment questions (capital, account type/taxes, price source for paper trading).

- HOLDOUT OPENED once for the final tilt spec (c): 2020+ net active +0.056%/mo (t 0.36, TE 4.65%, IR 0.15), gross +0.058%.
  Decay across splits: dev +0.216 -> validation +0.096 -> holdout +0.056 %/mo. Full-sample t 2.78 comes mostly from before
  2015; pooled 2015-2025 is roughly +0.07%/mo (t about 0.7, rough pooling of the two splits). VERDICT: the long-only gpa
  tilt is NOT validated out of sample; treat as a documented baseline, not as deployable alpha. 2020+ is now spent for
  gpa-based models. User context: USD 35,000 taxable account, simulation-first (taxes ignored for now). At that size a
  ~490-name, 3%-cap portfolio is not implementable without fractional shares; expected edge ~0.06-0.1%/mo is about USD 20-35
  a month against about USD 1,600/yr of tracking noise. Forward simulation is possible with Compustat daily/monthly
  prices (comp.secd: prccd, cshoc, ajexdi, trfd; comp.secm: prccm, trt1m) which run past CRSP's 2025-12 end.

- Data access verdict (inventory + probe): Cboe option/borrow data and wrdsapps.bondret are NOT queryable (permission denied);
  no I/B/E/S, 13F, insiders, OptionMetrics, TAQ, RavenPack. Accessible new data: JKP global_factor (to 2025-12), KPSS patent
  values, patent firm-year panel, Research Quotient, intangible value (to 2018), blockholders, XBRL complexity, 10-K risk text.
  See docs/research_directions.md. No new research pre-registered yet.

## Pre-registration v3: Study A (JKP themes through 2025) and Study B (innovation family)

Committed BEFORE any test code or result exists. Data facts, the JKP theme table (`src/qap/jkp_themes.py`, from Table 9 of
the supplied Global Factor Data Documentation: 13 themes, 153 characteristics, original signs) and the access verdict are in
`docs/research_directions.md`. Both studies reuse the validated pipeline (CRSP CIZ panel, delisting-adjusted returns,
eligibility mask, ex-microcap rule, point-in-time handling, costs at $10M 2019-market-scaled, Y=1, 1.5x stress, benchmark = VW
eligible universe, costless). Labels are OUR next-month delisting-adjusted returns (not JKP's).

**Judging (both studies, fixed):** net long-minus-universe monthly series (top quintile VW minus VW universe, 1.5x costs) per
signal; incumbent = gpa. PROMOTE iff dev (1990-2014) alpha t >= 2.0 of the signal's net series regressed on gpa's, AND validation
(2015-2019) alpha, computed with the DEV beta, is > 0. Information only: standalone net mean/t, IC, correlation with gpa,
equal-weight gpa+signal vs gpa. Promoted = provisional; 2020+ opens ONCE for promoted signals only (not clean for anything
tied to gpa, whose 2020+ gross spreads were printed earlier). With 19 trials, about 0.4 false promotions are expected by chance
at t >= 2; the sealed holdout is what decides. 65 cumulative unique trials after this study (46 logged so far + 19).

### Study A: JKP themes (15 trials)
- **Data:** `contrib_global_factor.global_factor`, US only (`excntry='USA'`), CRSP-sourced rows (`permno` not null), JKP screens
  `common=1, primary_sec=1, exch_main=1, obs_main=1`, `size_grp` in {mega, large, small} (micro/nano dropped at extraction),
  months 1963-01..2025-12; keys `permno`, `eom` (information available at `eom`; JKP says lag characteristics one month or use the
  next-month return, which is exactly our label convention). Only the 153 theme members are fetched.
- **Theme score** (per stock-month): mean over available members of `sign_k * (percentile_rank_k - 0.5)`, where the rank is within
  the month among stocks that pass OUR eligibility (common-equity mask, positive company ME, ex NYSE-20th-percentile microcaps)
  and have JKP data; needs >= max(3, ceil(half of the theme's members)) available members, else missing.
- **Signals (15):** the 13 theme scores (Accruals, Debt Issuance, Investment, Low Leverage, Low Risk, Momentum, Profit Growth,
  Profitability, Quality, Seasonality, Size, Short-Term Reversal, Value); `jkp_all13` = equal-weight mean of the 13 theme-score
  percentile ranks (needs >= 10 themes); `jkp_slow10` = the same without Size, Short-Term Reversal and Seasonality (needs >= 8).
- **Checks printed before results:** extraction rows by year and share with `permno`; `size_grp` values; missing theme-member
  columns (reported, never guessed); our `gpa` vs JKP `gp_at` and our be/ME vs JKP `be_me` monthly rank correlations; JKP
  `ret_exc_lead1m` vs our next-month return minus risk-free (correlation); theme availability by decade.

### Study B: innovation family (4 trials)
- **Sources:** `rq_all.rq_data` (gvkey-fyear, `rawrq`), `contrib_kpss.kpss_patents` (permno, `issue_date`, `xi_nominal`),
  `contrib_patent_firm_link.patent_level` aggregated by (gvkey, `grant_year`) on the server (count, sum and count of
  `num_cite_3y`; the firm-year panel's `year` is ambiguous between filing and grant year, so it is not used), Compustat `xrd`.
- **Signals (4):** (1) `rq` = latest `rawrq`, available June 30 of fyear+1; (2) `kpss_xi_me` = sum of `xi_nominal` over patents
  issued in the 36 months to month-end t, divided by company ME at t (missing before a firm's first patent, 0 after if none in
  window; evaluated only through the last month with complete KPSS issue dates); (3) `pat_flow_rd` = patents granted in year y
  / R&D capital at y, R&D capital = sum_{j=0..4} (1 - 0.2 j) xrd_{y-j} (needs >= 3 observed years, stock > 0), available June
  30 of y+1; (4) `pat_quality3` = sum `num_cite_3y` / count of patents with `num_cite_3y` for the grant cohort y-3 (its 3-year
  window is complete at year-end y), available June 30 of y+1. Linked to permno through CCM, date-bounded, as before.
- **Excluded on purpose:** forward citation totals (look-ahead), exploration/exploitation measures, R&D-to-market (it is a JKP Size
  member, tested in Study A), `int_xsec` (ends 2018-05; overlaps the earlier `bm_int` test), any tuning of windows or weights.
- **Universe note:** these signals exist only for innovating/R&D firms, so each is ranked within the firms that have it and
  benchmarked to that covered universe; this differs from gpa's universe (information only).
- **Checks printed first:** coverage by decade; last KPSS issue date; patents per grant year; link hit rate; `rawrq` scale.

## Results: Study A and Study B (reports in docs/study_a_report.txt, docs/study_b_report.txt; holdout still sealed)
All data checks passed (JKP next-month return vs ours: rank corr 1.000; JKP gp_at vs our gpa 0.971; be/ME 0.977; 0 of 153 theme
columns missing; coverage 99.1%). **Nothing promoted in either study.** Cumulative unique trials: 65.
- **Study A (15 trials):** the only dev alpha t >= 2.0 was Value (+0.30%/mo, t 2.13, NW6 t 1.67), and its validation alpha was
  -0.04, so it fails; with 15-19 tests a t of 2.13 is what chance produces. Quality is the best standalone (dev t 1.99, val
  +0.29) but has correlation 0.76 with gpa, so its alpha over gpa is 0.05 (t 0.85): it is gpa again. Short-Term Reversal and
  Seasonality are significantly negative net of costs (turnover).
- **The composites are the informative result:** jkp_all13 gross rank IC 0.039 (t 7.2) on dev but only 0.0098 (t 0.8) on
  validation, and its net long-minus-universe return is about zero in both. The cross-sectional ranking information is real
  and decayed after 2014, and a value-weighted long-only top-quintile ex-microcaps after 1.5x costs does not capture it.
- **Study B (4 trials):** no innovation signal has dev alpha over gpa (|t| < 1.4); RQ and patent quality ~0. kpss_xi_me is the
  best (dev +0.084, t 1.34; val +0.155) and is not distinguishable from noise.
- **gpa remains the sole survivor, but weaker than the 2.0 bar the challengers had to clear:** dev (1990-2014) net L-U +0.184%/mo,
  t 1.88; validation (2015-2019) +0.098, t 0.60. The earlier +0.17 (t 1.98) was the pooled 1990-2019 window, not a changed number.
  Separately, the long-only gpa *tilt* (a different construct: active return vs the benchmark) had holdout 2020+ +0.056, t 0.36,
  opened once. Verdict unchanged: no validated out-of-sample edge in long-only net-of-cost equity factor tilts.
- **Reading the composites:** most JKP characteristics' original publication samples overlap 1990-2014, so a dev IC t of 7.2 is
  roughly what in-sample data produces; the finding is the validation drop (IC t 0.8) together with net L-U ~ 0.
- **Pre-registration deviations/omissions (recorded after seeing results, not backfilled):** two promised pre-result checks were
  not printed: Study A "extraction rows by year and share with permno" and Study B "link hit rate". Study B's U is the set of
  firms with a non-missing signal (covered universe), as pre-registered; the L-U is therefore not on gpa's universe.
- Compute note: user has 36 cores / 31.5 GB; per-signal costed backtests are independent and should be parallelised next time.

## Pre-registration v4: (b) conversion diagnostic and (a) forward paper-trading harness (written before either runs)

User chose "b alongside with a" after Studies A/B. No new signal is tested, so **no new trials**; nothing here can promote anything.

### (b) Conversion diagnostic (information only; 2020+ stays sealed; nothing is logged as a trial)
Question: why does `jkp_all13` have dev rank IC t 7.2 yet net long-minus-universe ~0? Fixed signal list: `gpa`, `jkp_all13`,
`jkp_slow10`, `Quality`, `Profitability`, `Value`, `Momentum` (themes as in Study A). Same eligible universe, ex-microcaps, same
quintile definition, by split (dev 1990-2014, validation 2015-2019). Reported per signal and split:
1. Gross monthly return (%) of each quintile Q1..Q5, value-weighted AND equal-weighted, and the spreads Q5-Q1, Q5-U, Q1-U (U =
   VW covered universe); decile D10-D1 (VW and EW).
2. Costs (1.5x, $10M 2019 market-size): L-leg and S-leg cost, one-way turnover, net L-U and net long-short (borrow NOT modelled).
3. Size dependence: within-month size terciles of company ME among eligible names: rank IC and EW Q5-Q1 gross inside each tercile.
Reading rule fixed in advance: if EW Q5-Q1 is clearly larger than VW Q5-Q1 and the IC lives in the smallest tercile, the
information is a small-cap/EW effect that the VW long-only construction cannot hold; if the short leg (Q1-U) carries most of Q5-Q1,
the information is a short-leg effect (needs borrow, which we do not have data for); if costs explain the gap, turnover is the
cause. Any construction designed from this can be validated only on the still-sealed 2020+ data (never opened for the JKP
composites), as a new pre-registered trial.

### (a) Forward paper-trading harness for the gpa tilt (spec (c) of v2, unchanged: entry 0.80 / exit 0.70 / 3% cap)
Purpose: the only clean out-of-sample test left. CRSP and JKP end 2025-12 but Compustat runs to 2026-09, so (i) the months
2026-01..2026-09 are **retrospective-unseen** (no result of ours has used them, but they were realised before the harness
existed) and (ii) every formation from the 2026-09 month-end on is **live** (weights frozen and committed before the month realises).
- Price source: `comp.secm` (monthly; total return `trt1m`), common stock (`tpci='0'`), `fic='USA'`, `curcdm='USD'`; company ME =
  sum over a gvkey's issues of prccm*cshom (scale calibrated against CRSP ME in the overlap, must come out a power of ten),
  NYSE = `exchg` 11. Fundamentals: Compustat annual, same 4-month PIT lag and 548-day staleness as before. gpa = (revt-cogs)/at.
- **Check printed before any forward result:** on the CRSP overlap (2024-01..2025-12) the Compustat-built universe and tilt
  must reproduce the CRSP-built ones: ME scale, share of CRSP-eligible gvkeys found, rank correlation of gpa, weight overlap
  (sum of min weights) of the tilt and of the benchmark, and the correlation and mean difference of monthly tilt-vs-benchmark
  active returns. Tolerance fixed now: weight overlap >= 0.80 and active-return correlation >= 0.90, else the harness is
  reported as not reproducing the research and is not used for evaluation.
- The buffered tilt is path dependent: holdings state starts 2024-01 (24-month burn-in).
- Ledger: each formation writes `data/paper/forms/{yyyymm}.parquet` (spec weights, benchmark weights, spec commit) and a sha256
  manifest line; because `data/` and parquet are git-ignored (licensed data), the tracked evidence is `paper_ledger/manifest.jsonl`
  (sha256, month, formed-on date, spec commit; no data), committed and pushed before each live month starts.
- Information-only implementable variant for the USD 35k account: the 50 largest spec weights, renormalised, 3% cap (tracking
  difference to spec reported; fractional shares assumed).
- Settlement: next-month `trt1m` of each held (gvkey, iid); missing returns counted 0 and their weight reported. Reported:
  active return = tilt minus benchmark (costless, gross; costs from the research model are informational), by group
  retrospective-unseen / live, with t-stats and the plain statement that 9 months have no power (a 0.2%/mo edge with ~5% annual
  tracking error needs decades for t = 2).

- **(b) implementation constants, fixed before any real result (the pre-registration gave no numbers):** "clearly larger" EW vs VW
  Q5-Q1 = gap >= 0.10 %/mo AND ratio >= 1.5; "IC concentrated in the smallest tercile" = its share of the summed tercile ICs >= 0.5
  and it is the largest; "short leg carries" = -(Q1-U)/(Q5-Q1) >= 0.5 (VW); "costs explain the gap" = L-leg cost / gross L-U >= 0.5.
  Flags only; no conclusion is drawn from them mechanically. In (b) all seven signals use the JKP-merged eligible universe (gpa merged
  on permno, month), so its numbers differ slightly from the Study A/B gpa series. `inp` and `aum_t` are cut at 2020 before costing.
  Implemented in `qap.diagnostics` / `scripts/run_diagnostic.py` (`--jobs N`; jobs 1 and 4 give byte-identical reports on synthetic data).

## Results: (b) conversion diagnostic (docs/diagnostic_report.txt; information only, 2020+ sealed, no trials)
- **Why `jkp_all13`'s IC (dev t 7) does not become net return, in three parts.** (1) Costs: one-way turnover ~39%/mo; L-leg cost 0.300
  vs gross L-U 0.346 (87%), so net L-U 0.046 (t 0.45). (2) The information sits in the short leg: Q1-U -0.48 (t -3.7) vs Q5-U +0.35
  (dev); after 2014 the long leg is nothing (Q5-U +0.049) and the short leg carries 84% of Q5-Q1 (0.31, t 1.0). A long-only book cannot
  use it and we cannot model borrow. (3) Decay: validation gross Q5-Q1 falls from 0.83 to 0.31 and gross L-U from 0.35 to 0.05.
- **Small-cap hypothesis rejected for dev:** IC is spread across size terciles (T1 0.047, T2 0.034, T3 0.034); EW Q5-Q1 is 1.35x VW,
  below the "clearly larger" rule. On validation the all13 IC is ~0 in the largest tercile.
- `jkp_slow10`: dev gross L-U +0.258 (t 2.8) at 17% turnover, net +0.131 (t 1.4); gross long-short Q5-Q1 0.90 (t 3.75), net long-short 0.545
  (t 2.28), but validation gross L-U +0.038 and the short leg carries 91%. `Quality` is the most stable long-only: net L-U +0.190 (t 1.99) dev,
  +0.293 (t 1.68) validation at ~5.5% turnover, but it is 0.76 correlated with gpa. `Value` reverses on validation (Q5-U -0.08),
  `Momentum` loses to its own turnover (cost 0.256 > gross 0.172). gpa on the JKP universe: dev Q5-U +0.196 (t 2.0), net +0.169 (t 1.74);
  validation +0.156 (t 0.97).
- Reading: nothing here justifies a new long-only composite; the only remaining gross information is on the short side (no borrow data).
  Any construction designed from this (e.g. a turnover-buffered slow10) is a new pre-registered trial that can be validated only on the
  still-sealed 2020+ data. Prior odds are low.

## Results: (a) paper-trading harness check (user run, 2026-10-11)
comp.secm 384,559 rows 2022-01..2026-10 (the last month had 1 row: partial, now excluded automatically); ME scale 1e6 (shares are
raw units; median ratio 0.99976 of a power of ten). CRSP-overlap check over 2024-01..2025-11 (23 months): tilt weight overlap mean 0.884
(min 0.872), benchmark overlap 0.944, gpa rank corr 1.000, active-return correlation 0.974, mean difference +0.089 %/mo: both
pre-registered tolerances PASS. Ten frozen formations written (2025-12..2026-09, ~470-493 names each).

## Note: single-stock leveraged/inverse ETFs as a way to reach the short leg (user raised 2026-10-11; assessment, no test run yet)
Examples: Tradr 2X Long SNDK (SNXX, 1.49% expense ratio, launched 2026-01) and Tradr 2X Short LITE (LITZ, launched 2026-04). Each wraps ONE
issuer-chosen stock with daily reset, so they are access tools, not signals. Why they do not rescue the short leg on current evidence:
(1) the product universe is a few hundred popular, volatile stocks chosen by issuers, while the signal's bottom quintile is broad and
largely unlisted-for; (2) history is ~3-4 years (products launched 2022+), all inside the sealed 2020+ window, so no historical validation
is possible; (3) cost: a secondary-source estimate puts rebalancing drag at ~0.49%/mo for -2x single-stock funds plus 1-1.5%/yr fees,
which is the same order as the short-leg alpha measured in diagnostic (b) (slow10 Q1-U -0.64 dev / -0.41 val; all13 -0.48 / -0.26 %/mo).
Cheap, decisive check if wanted (information only, v5 to be pre-registered first): from CRSP, list single-stock ETFs with launch dates and
underlyings, measure (i) monthly share of each signal's bottom quintile with a product available, (ii) realised shortfall of each ETF vs
-L x underlying monthly return (fees, financing and decay measured, not assumed).

## Pre-registration v5: single-stock leveraged/inverse ETF check (information only; written before it runs)
No signal is tested and **no alpha is estimated**: the check uses signal *ranks* (to see who sits in the bottom quintile) and ETF and
underlying *returns*, never signal returns, so the sealed 2020+ window is not opened for any signal. No trials are logged.
1. **Inventory (data: `crsp.stocknames_v2` via `select *`, CRSP monthly rows we already hold).** ETFs are found by name: contains "daily" and a
   leverage multiple (`<n>X`), direction negative if the name has short/bear/inverse. Index/sector/commodity products are excluded by a fixed
   word list (S&P, Nasdaq, Russell, Dow, Index, Sector, Treasury, Gold, Silver, Oil, Bitcoin, Ether, Semiconductor, China, ...). The underlying is
   the common stock whose ticker equals an upper-case token of the name (stop-words removed), active at the ETF's first month; fallback:
   a word of the name equal to the first word of an eligible stock's issuer name (largest ME). The full parsed list (ETF, L, underlying, method,
   first/last month) is printed so mis-parses are visible; unmatched ETFs are listed. Parser rules are fixed before the first real run; if the
   first real run shows mis-parses, the fix is made and recorded as a deviation (parsing is data cleaning, not a result).
2. **Coverage.** For gpa, Quality, Profitability, `jkp_slow10`, `jkp_all13` (same eligible universe as the diagnostic): by calendar year, the
   share (by count and by value weight) of the bottom quintile whose underlying has an inverse product live that month, the same for the top
   quintile and long products, and the number of live inverse products. Also the count of bottom-quintile names covered in the last month.
3. **Realised shortfall.** For each ETF-month (the ETF's first and last months dropped; both returns present): shortfall = r_ETF - L x r_underlying
   (monthly, delisting-adjusted CRSP). Reported: Fama-MacBeth style monthly cross-sectional means with t over months, overall and by L bucket
   (<= -2, -1.x, +), the std (tracking noise), per unit of exposure (divided by |L|). Plus the theory check: regress shortfall on
   (L^2 - L)/2 x (R_u^2 - v), where R_u is the underlying's monthly return and v = sigma_d^2 x days from `crsp_daily_agg`: slope should be near 1
   (path convexity, mean zero) and the intercept is the realised fee + financing carry per month.
4. **Decision rule fixed in advance.** Compare the intercept-per-unit-exposure (carry, %/mo) with the short-leg gross gain already measured in
   diagnostic (b) (slow10 0.64 dev / 0.41 val; all13 0.48 / 0.26; Quality 0.18 / 0.31; gpa 0.08 / 0.22 %/mo of Q1-U). If carry per unit exposure is
   >= 50% of a signal's validation short-leg gain, or bottom-quintile coverage (VW) averages < 30% in the last full year, single-stock inverse ETFs
   are recorded as NOT a viable route to the short leg. Otherwise a forward-only, pre-registered pilot would be considered (no historical test is
   possible: products start 2022).

- **v5 deviations/readings fixed BEFORE the first real run (found in review of the implementation):** (1) shortfall is measured in excess-of-rf
  terms, (r_ETF - rf) - L (r_u - rf), because swap-based funds earn (1 - L) rf on their cash, so the raw gap would show a large positive rf term for
  inverse funds (about +1%/mo at 2023-25 rates) and make the carry clause unfireable; excess form isolates fee, swap spread and path effects, which
  is also what a margin short costs relative to its rebate. rf from `ff_monthly`. (2) Decision rule: the route is NOT VIABLE only if the rule holds
  for EVERY listed signal (gpa, Quality, jkp_slow10, jkp_all13); the 'any' reading would call it dead if one signal fails. Per-signal lines printed.
  (3) The parsed ETF list is printed in full; first-run mis-parses, if any, are fixed and recorded as data-cleaning deviations.

- **v5 first real run: INVALID, verdict retracted.** `crsp.stocknames_v2` has no fund-name column (only ticker, issuernm = trust/issuer name,
  securitytype/subtype), so the name parser found 0 ETFs. The report then printed "NOT VIABLE" purely because an empty inventory gives 0% coverage;
  that verdict carries NO information and is withdrawn. Fixes recorded before the second run: (i) the inventory is now return-based:
  non-common-equity securities with >= 12 monthly returns since 2019 whose monthly returns correlate >= 0.90 with one US common stock and
  whose slope is within 25% of an allowed leverage (-3, -2, -1.5, -1, 1.5, 2, 3); the name parser is kept only as a cross-check; (ii) the decision
  rule returns INVALID (never NOT VIABLE) when fewer than 10 ETFs are matched. Limits to remember: ETFs with < 12 months of history are missed and
  underlyings that are not in our common-equity universe (ADRs) cannot match.

- **v5 second real run (return-based inventory): contaminated, verdict NOT used** (raw report kept as docs/etf_check_report_run2_contaminated.txt).
  254 'ETFs' were detected, but most are false positives (SPACs, ADRs, closed-end funds, ordinary stocks flagged non-eq, and 1.5x index ETFs
  matched to obscure tiny banks at n=12-19 where spurious |rho| reaches ~0.9); the ETF shortfall noise of 17%/mo confirms it. The ~100 genuine
  single-stock products (e.g. AAPB, TSLL, NVDL, MSFU) are in the list with rho >= 0.97 and large underlyings. Printed numbers (carry drag
  0.73%/mo; coverage 1-8% VW of the bottom quintile) are therefore unreliable as stated. Cleaning rules fixed before the third run, from
  inspecting false positives only: candidates must be names-flagged FUND/ETF; rho >= 0.95; underlying company ME >= $1,000M; 1.25 added to the
  allowed leverages; a funnel is printed. These filters can only REMOVE products, so they cannot raise coverage: the coverage clause
  (bottom-quintile VW coverage < 30%) is robust to the cleaning (even the contaminated list covered 2 of ~265 bottom-quintile names in 2025-12);
  the carry estimate is what will change.
  The 1.25x family uses a tight band (beta 1.125-1.375) so that plain 1x funds are not labelled 1.25x; other leverages keep the 25% band.

## Result: v5 single-stock ETF check, third run (cleaned inventory) - single-stock inverse ETFs are NOT a viable route to the short leg
Funnel: 9,317 candidates -> 6,063 flagged FUND/ETF -> 4,733 with >= 12 months -> 231 with |rho| >= 0.95 -> 77 with a leverage in tolerance -> 72 with
underlying ME >= $1bn. The 72 are mostly genuine 2022+ single-stock products (AAPB, TSLL, NVDL, MSFU, ...); about a dozen are legacy sector/index
ETFs matched to arbitrary stocks (e.g. MIDZ, GASL, FINU, WDRW, EUFL, OILU, UCOM, TENG, GUSH, DPST, SKRE), mostly pre-2020.
- **Coverage (decisive, robust to the inventory noise):** 2025 value-weighted bottom-quintile coverage by a live inverse product = 3.2% (slow10), 5.6%
  (all13), 0.1% (Quality), 0.1% (gpa); 1 of ~265 bottom-quintile names in 2025-12; only ~16 inverse products live, all on mega-cap, high-attention
  names (AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, NFLX, AVGO, MU, BRK, ...). The signals' bottom quintile is the opposite kind of stock. Long products cover
  45-66% of the top-quintile weight (mega caps), which is irrelevant here because leverage on the long leg adds noise, not edge.
- **Carry estimate NOT trusted:** intercept -0.73%/mo per unit of short exposure (and -0.8% for longs) is far above fees plus swap spread (~0.2%/mo expected),
  the convexity slope is 0.44 (theory 1), shortfall noise is 8%/mo, and nominal leverage labels are snapped from betas, so mislabels and the residual
  non-single-stock funds contaminate it. The verdict does not depend on it: the coverage clause alone holds for every signal.
- Verdict under the pre-registered rule (NOT VIABLE only if the rule holds for EVERY signal): **NOT VIABLE**, on the coverage clause. No signal returns were
  used; 2020+ signal performance was not opened; no trials logged. A forward-only pilot is not warranted.

## Result: break-even borrow cost for a direct margin short (computed from docs/diagnostic_report.txt; no new data, 2020+ sealed, no trial)
Break-even annual cost on the short notional = net long-short return (after the model's 1.5x trading costs, borrow not modelled) x 12. VW top vs bottom quintile,
$ neutral (capital 1 on each leg).
| signal | dev net LS %/mo (t) | break-even %/yr dev | val net LS %/mo (t) | break-even %/yr val | idx-long + short Q1 only, net %/mo dev / val |
|---|---|---|---|---|---|
| jkp_slow10 | 0.545 (2.28) | 6.5 | 0.386 (1.10) | 4.6 | 0.413 / 0.362 |
| jkp_all13 | 0.225 (1.06) | 2.7 | 0.210 (0.70) | 2.5 | 0.179 / 0.203 |
| gpa | 0.222 (1.17) | 2.7 | 0.370 (0.91) | 4.4 | 0.053 / 0.217 |
| Quality | 0.301 (1.49) | 3.6 | 0.594 (1.35) | 7.1 | 0.112 / 0.300 |
| Profitability | 0.352 (1.26) | 4.2 | 0.460 (1.37) | 5.5 | 0.264 / 0.318 |
Reading: the all-in hurdle is (borrow fee) + (cash rebate shortfall vs the T-bill rate, broker-specific, assumed ~0.5-2%/yr at retail size, UNVERIFIED) + short squeeze and
locate risk. With easy-to-borrow names only (~0.3-1%/yr) the hurdle is ~1-3%/yr: slow10 clears it on both splits, all13 does not, gpa/Quality/Profitability are marginal.
Blended-basket tolerance: with easy names at 0.5% and a hard-to-borrow share h at fee F, break-even h = (BE - 0.5)/(F - 0.5): for slow10 validation (BE 4.6) h <= 43% at F=10%
and 17% at F=25%; development (BE 6.5) 63% and 24%. The weight share of hard-to-borrow names in the bottom quintile is UNKNOWN and is the decisive unmeasured input.
Statistical caveats: slow10 net LS validation t is only 1.10 (95% CI roughly +-8%/yr), pooled dev+val t ~2.5 but slow10 is one of 15 Study A trials chosen for a new use (long-short),
65 cumulative trials, and its gross edge decayed (validation gross L-U 0.04). The sealed 2020+ window is unopened for JKP composites, so a long-short slow10 would be a new
pre-registered trial there, with borrow fees taken from the broker's live shortable list for a sample basket.

- **Borrow fees: none measured (user cannot look them up now), so a labelled ASSUMPTION schedule is used for sensitivity only** (`scripts/short_basket.py --assume`).
  General-collateral fee by company-ME tier (<$500M / $0.5-2B / $2-10B / >=$10B): 3.0 / 1.5 / 0.6 / 0.3 %/yr; hard-to-borrow weight share by tier and fee: optimistic
  5/2/0/0% at 10%/yr, base 12/5/1/0% at 15%, pessimistic 25/10/3/1% at 25%; plus an assumed 1.0%/yr rebate shortfall. These numbers are common-sense guesses (no data),
  fixed here before seeing the real basket. Outcome on a real basket is a sensitivity table, never a validation.

## Result: assumed-fee sensitivity on the real jkp_slow10 bottom-quintile basket (2025-12; user run; fees ASSUMED, not measured)
266 names; weight by company-ME tier <$0.5B / $0.5-2B / $2-10B / >=$10B = 0% / 0% / 18% / 82% (value weighting concentrates the short basket in large, easy-to-borrow names).
Weighted assumed fee: optimistic 0.36%, base 0.38%, pessimistic 0.69% per year (+1.0% assumed rebate shortfall => hurdle 1.4-1.7%/yr), against break-evens 6.5% (dev) and 4.6%
(validation): clears in all three scenarios. Tolerated hard-to-borrow share at 15%/yr: 35% of weight (dev) / 22% (validation). Reading: borrow cost is NOT the binding constraint for a
value-weighted slow10 short basket on these assumptions; the binding constraints are statistical (validation net long-short t 1.10, gross edge decaying), the decay itself, implementability
at USD 35k across 266 names, and the dependence on JKP data that ends 2025-12 on WRDS (no forward updates without rebuilding the 153 characteristics).
Not yet done: the sealed 2020-2025 window is unopened for JKP composites. A one-shot long-short test there would have low power (monthly sd ~3.9%, 71 months => SE ~0.46%/mo; if the
true net edge equals the pooled 0.52%/mo the expected t is ~1.1 and P(t >= 2) is ~20%), so it can refute, not confirm.

## Pre-registration v6: one-shot sealed-window test of a long-short jkp_slow10 (written and committed BEFORE the script exists or runs)
User chose to spend the sealed 2020-2025 window once for this. It has never been opened for any JKP composite (the earlier 2020+ openings were gpa models only).
- **Strategy (fixed, no tuning):** signal `jkp_slow10` exactly as Study A (10 themes: all except Size, Short-Term Reversal and Seasonality; needs >= 8 themes). Formation universe =
  JKP-merged eligible universe, ex-microcaps (NYSE 20th percentile of company ME). Value-weighted top quintile LONG and bottom quintile SHORT (the `quintile_legs` definition), capital 1 on
  each leg, monthly rebalance. Formation months 2019-12 .. 2025-11, realised months 2020-01 .. 2025-12 (72).
- **Costs:** the same cost model as the diagnostic (half spread, square-root impact, $10M in 2019 market-size dollars, 1.5x stress, benchmark irrelevant) on BOTH legs. Borrow: each month
  the short leg pays sum_i w_i fee_i / 12 where fee_i = the BASE-scenario assumed fee for the name's company-ME tier (`qap.basket.assumed_fee(me, "base")`) plus an assumed 1.0%/yr rebate shortfall
  on the short notional. Optimistic and pessimistic scenarios are printed as sensitivity only. The fees are assumptions (no measured borrow data), fixed here.
- **Primary statistic:** net long-short return per month, `gross L-S - 1.5 x (L cost + S cost) - borrow`, mean (%/mo) and OLS t over the 72 months (Newey-West 6 lags also printed).
- **Decision rule (fixed now):** REFUTED if base-scenario net mean <= 0; SUPPORTS FURTHER WORK only if net mean >= 0.20 %/mo AND t >= 1.0; otherwise INCONCLUSIVE. No result of this test can
  promote the strategy to deployment: at best it licenses a further pre-registered step. A failure retires long-short slow10; there is no second look at 2020+ for JKP composites.
- **Information only (not decision inputs):** by-calendar-year returns, first vs second half, max drawdown, beta and alpha on the FF market factor (and FF4 loadings), the long-only
  (L-U) and short-only (U-S) legs separately, turnover per leg, and the gross rank IC.
- **Power, stated honestly:** monthly sd of net L-S ~3.9% => SE ~0.46%/mo over 72 months. If the true edge equals the pooled dev+validation estimate (0.52%/mo) the expected t is ~1.1 and
  P(t >= 2) is ~20%; the test can refute a large failure, it cannot confirm a modest edge. Counted as trial 66 (a new use of a Study A signal).
- **One shot:** the script writes a marker file and refuses to run again; the result is recorded in this file whatever it is.
- **v6 implementation notes (recorded before the run):** the information-only gross rank IC was not implemented (not a decision input); FF alpha/loadings regress the raw net long-short on the FF
  factors with plain OLS standard errors; everything shown from the sealed window is computed in memory BEFORE the marker file is written, so a crash cannot burn the one shot unseen. A
  rehearsal gate must first reproduce the diagnostic (b) numbers (tolerance 0.01 %/mo) on 1990-2019 or the real run aborts without reading 2020+.
- **v6 rehearsal (user run): gate PASSED** - gross long-short 0.899 (dev) / 0.450 (val) and net before borrow 0.545 / 0.386 %/mo reproduce diagnostic (b) exactly. With the base-scenario borrow
  (assumed fees + 1.0%/yr rebate shortfall) the pre-2020 net long-short is 0.393 (dev) and 0.258 (val) %/mo, i.e. borrow costs ~0.13-0.15 %/mo (1.5-1.8%/yr); validation after borrow is only
  just above the pre-registered 0.20 %/mo support threshold and its t is ~0.7. 2020+ not read. Next: the one-shot run.

## RESULT v6 (one shot used, user run 2026-10-11): long-short jkp_slow10 on the sealed window 2020-01..2025-12 - INCONCLUSIVE
Gate passed (reproduced diagnostic (b) exactly). 72 realised months.
- **Primary (base borrow): net L-S mean 0.500 %/mo, t 0.89 (NW6 0.84), sd 4.75 %/mo.** gross L-S 0.705, L cost 0.017, S cost 0.066, borrow 0.121 %/mo. Optimistic borrow 0.504 (t 0.90),
  pessimistic 0.470 (t 0.84). Rule: mean >= 0.20 but t < 1.0 => **INCONCLUSIVE** (not refuted, not supported). Nothing is licensed by this result; per the pre-registration there is no second look.
- **Heterogeneity:** compounded net L-S by year: 2020 -13.2%, 2021 +20.4%, 2022 +35.4%, 2023 -9.5%, 2024 -1.7%, 2025 +4.8%. First half 1.091 %/mo, second half -0.090 %/mo. Cumulative +32.0%,
  max drawdown -30.4%. The whole result comes from 2021-2022 (the junk/growth unwind); 2023-2025 is flat to negative.
- **Factor information (not decision inputs):** CAPM alpha 0.945 %/mo (t 1.84), market beta -0.42 (t -4.2); FF4 alpha 0.390 (t 1.15) with SMB -0.83 (t -6.8), UMD +0.42 (t 4.4), HML 0.01. So a large part of the
  return is explained by size and momentum exposures plus a net short-market tilt.
- **Legs (info):** long leg L-U net of cost +0.051 %/mo (nothing); short leg U-S net of cost and borrow +0.449 %/mo: the edge is entirely on the short side, as in diagnostic (b). One-way
  turnover about 20% (L) and 30% (S) per month (the script printed 0.2/0.3: a display bug, fraction not x100; fixed after the run, no effect on results).
- **Pooled view (not a test):** dev 0.393, validation 0.258, holdout 0.500 %/mo after base borrow; weighted mean about 0.39 %/mo (4.7%/yr) over 432 months, pooled t about 2 before any correction
  for choosing slow10 among 15 Study A signals and then diagnostic (b); 66 cumulative trials.
- Reading: weak, regime-dependent evidence of a short-side premium that did not persist after 2022, carried by size/momentum tilts, with 30% drawdowns and a data feed (JKP) that ends 2025-12.
  Not a basis for deploying USD 35k.

- **Ledger timing note (recorded 2026-10-11):** all ten frozen formations were created on 2026-10-11, so `paper_settle.py` (prospective = formed within 7 days of the target month's start) will label them retrospective.
  The rule is left unchanged. The first formation that can be prospective is 2026-10 (needs formation and a pushed manifest by about 2026-11-07). If WRDS posts the month-end row later, report zero prospective months
  honestly rather than loosening the rule after the fact.

## Pre-registration v7: tax-aware direct indexing (loss harvesting) simulation for a USD 35k taxable account (written before any code or run)
Question: is the mechanical after-tax benefit of tax-loss harvesting on a ~150-name DIY index, with regular cash additions, large enough to matter against its tracking error
and complexity? This is NOT an alpha test (no signal is used; nothing predictive is estimated), so no trial is logged; the sealed 2020+ window is irrelevant and the full 1963-2025
history is used with rolling 10-year windows. The result is a measurement under stated assumptions, not a validation of a strategy.
**User inputs (2026-10-11):** federal bracket 32% or higher; realises some other capital gains every year; PRIMARY end treatment = hold indefinitely / bequeath (unrealised gains
never taxed); fractional shares; regular cash additions.
**Tax assumptions (constants, current-law rates applied to all eras; to be confirmed with a tax adviser):** federal short-term 32% (ordinary), long-term 15%, NIIT 3.8% on both (assumes
MAGI above the unindexed USD 200k threshold), assumed state 5%: short-term 40.8%, long-term and qualified dividends 23.8%. Long-term = held more than 12 months. Net losses offset gains
(ST against ST first, LT against LT, then cross), then up to USD 3,000/yr against ordinary income at 40.8%, remainder carries forward indefinitely (cap and wash-sale rule are from general
knowledge, not re-verified here). Outside realised gains assumption: USD 3,000/yr long-term (absorbs harvested losses at 23.8%); sensitivities 0 and USD 10,000. Rates checked via a public source
for 2026 (15%/20% LT brackets unchanged, NIIT 3.8% with a frozen USD 200k threshold; the 20% bracket starts well above the assumed income).
**Arms (all start USD 35,000 at the start of each window; contributions USD 500/month, sensitivities 0 and 1,500):**
- *ETF benchmark:* value-weighted total-market portfolio (all common-equity names, CRSP VW return incl. dividends) held like a broad ETF: no realised gains; dividends taxed each year at
  23.8%; costless.
- *DIY, no harvest:* the 150 largest company-ME common stocks at window start, value-weighted at purchase; held; new cash and dividends buy the most underweight names (no sells); the
  universe is reconstituted only by delistings/mergers (treated as taxable sales at the delisting return, proceeds reinvested pro rata); no other sells.
- *DIY, harvest:* as above plus: each month, any tax lot with unrealised loss >= 10% of its basis and >= USD 50 is sold (lot-level, dollars); the proceeds are reinvested pro rata in the
  other holdings; the sold name is blocked from all purchases (new cash, dividends, substitution) that month and the next (30-day wash-sale rule at monthly steps), and a name with any purchase
  in the previous month is not harvested. Single spec: threshold 10%, universe 150, pro-rata substitution; ablations (5%/15%, 100/250 names) are information only.
**Dividends/returns:** price return from `mthretx`, dividend yield from `mthret - mthretx`; delisting month uses the delisting-adjusted return. Lots are tracked in dollars so splits do not matter.
**Costs:** flat 5 bp per side on traded notional for both DIY arms (sensitivities 0 and 15 bp); no commissions.
**Windows:** 10-year windows starting every January 1963..2015 (53, overlapping), plus a monthly-start robustness; the 150-name universe is re-picked at each window start.
**Primary statistic:** after-tax terminal wealth under HOLD-INDEFINITELY (unrealised gains untaxed): (harvest - no-harvest)/wealth, annualised %, median / IQR / 10th percentile across windows.
Also reported: harvest vs the ETF benchmark; liquidate-at-end after-tax wealth (deferred tax counted) as co-reported, because much of the harvest benefit is deferral; tracking error of each DIY arm vs the
ETF benchmark (annualised, from monthly pre-tax returns); realised loss harvested per year, ST/LT split, cumulative carryforward, turnover; number of names and lots.
**Decision rule (fixed now):** "worth pursuing" only if, under hold-indefinitely with the primary assumptions, the median annualised after-tax benefit of harvest over no-harvest is >= +0.5 %/yr of wealth,
at least 75% of windows are positive, and the harvest arm's tracking error vs the ETF benchmark is <= 2.0 %/yr; the liquidate-at-end figure is reported next to it and, if it is <= 0, the result is
labelled "deferral only". Otherwise "not worth the complexity". Either way this is an informational result about an implementation choice, not alpha.
**Known biases:** monthly steps miss intramonth harvesting (biases the benefit DOWN); constant current-law tax rates across eras; wash sales across the user's other accounts are ignored; CRSP merger/delisting
tax treatment is assumed taxable; fractional shares assumed; sectors/industry matching in substitution is not modelled (pro-rata) which adds tracking error.

- **v7 amendments made BEFORE any real-data run (found in review of the implementation on a synthetic world; no real result seen):**
  (1) The 'name with any purchase last month is not harvested' rule was far too coarse: under the actual wash-sale rule only the portion matched by replacement shares is disallowed, and
  contributions go to the most underweight names, i.e. the losers, so the literal rule blocked ~2.6 qualifying lots for every one harvested. New rule (the single spec): a name's loss lots may be
  sold up to (their current value minus dollars of that name bought in the previous month); the matched portion is not sold. The literal rule is kept as an information-only ablation.
  (2) The tracking-error criterion compared the harvest arm with the total-market ETF, which mixes the harvest effect with the cost of holding only 150 names. It now applies to harvest vs DIY-no-harvest
  (median <= 1.0 %/yr, the deviation harvesting itself causes); DIY-vs-ETF tracking error is reported separately as the universe cost. Other criteria unchanged (median benefit >= +0.5 %/yr of wealth, >= 75% of
  windows positive).

## RESULT v7 (user run): tax-loss harvesting on a 150-name DIY index - NOT WORTH THE COMPLEXITY (primary assumptions)
53 rolling 10-year windows 1963..2015, USD 35k start, USD 500/month, ST 40.8% / LT 23.8%, outside LT gains USD 3,000/yr, 5 bp per side, 10% threshold, proportional wash-sale rule.
- **Primary (hold-indefinitely), harvest minus no-harvest: median +0.167 %/yr of wealth, IQR [-0.026, +0.406], p10 -0.142, 68% of windows positive.** Liquidate-at-end: median +0.033 (IQR
  [-0.084, +0.176]): essentially all of the benefit is deferral. Tracking error harvest vs no-harvest: median 1.08 (max 3.31) %/yr against the 1.0 limit.
  The label fails all three criteria (benefit < 0.5, share positive < 0.75, TE > 1.0): **NOT WORTH THE COMPLEXITY**.
- **By start decade (primary):** 1960s +0.22 (86% positive), 1970s 0.00, 1980s +0.13, 1990s -0.07 (30% positive), **2000s +0.65 (100% positive)**, 2010s +0.26. The benefit appears in drawdown-heavy
  windows (2000s) and is nil in others.
- **Sensitivities (information only):** no sensitivity reaches +0.5 %/yr or the WORTH label. Zero contributions: +0.30 hold (TE 1.81); contributions USD 1,500: +0.12; outside gains 0: +0.26, USD 10,000:
  +0.09 (liquidate -0.02); cost 0 / 15 bp: +0.18 / +0.16; threshold 5% / 15%: +0.13 / +0.17; 100 / 250 names: +0.24 (TE 1.39) / +0.12 (TE 0.81); literal wash rule: +0.19, 79% positive, TE 0.94,
  but USD 57k of loss lots withheld.
- **DIY vs ETF:** median terminal after-tax wealth (hold): ETF 186k, no-harvest 201k, harvest 200k. Holding 150 large names without reconstitution tracks the total market with ~4 %/yr tracking error
  (p10 relative wealth -1.2 %/yr, IQR -0.5..+0.75): a universe/size outcome, not an edge. Harvest vs ETF median +0.51 %/yr is mostly that universe effect (no-harvest vs ETF +0.24).
- **Reading:** under the most favourable end treatment (never taxed), harvesting adds a median ~0.17 %/yr (roughly USD 100-300/yr at this account size) with ~1 %/yr extra tracking error and thousands
  of lots to manage; with liquidation the benefit is ~0. Not a reason to hold individual stocks; the cheaper levers are a broad low-cost fund, no unnecessary turnover and asset location.
  Biases against harvesting: monthly steps, pro-rata (not industry-matched) substitution; in favour: hold-indefinitely treatment, today's high tax rates applied to all eras.
- Report-line bug found in review: the printed 'universe cost: harvest median TE vs ETF' reused the harvest-vs-no-harvest median (fixed in code after the run; the DECISION line and all benefit
  statistics were unaffected).

## Pre-registration v8: forward-only machine-learning cross-sectional check (written before any code or run)
User chose to run this alongside cheap alternative-data probes. All historical windows are spent, so this is built to be judged FORWARD ONLY. No claim can come from any backtest here.
- **Pipeline independence:** features, labels and the universe are built only from Compustat (`comp.secm` monthly prices/shares/total return `trt1m`, annual fundamentals with the same 4-month PIT lag and 548-day
  staleness), so the identical code produces training data (1990-2025) and forward data (2026-). Universe: common stock (`tpci` '0'), US, ME >= NYSE 20th percentile, as in the paper harness.
- **Features (fixed list, 20, each cross-sectionally rank-transformed to [-0.5, 0.5] every month, missing -> 0):** log ME, book-to-market, earnings/price, cash-flow/price, gross profit/assets, ROA, asset growth,
  capex/assets, accruals (Sloan), leverage, cash/assets, sales growth, R&D/assets, net share issuance (12m change in shares), momentum 12-1, 1-month reversal, 12-month return volatility, 36-month market beta,
  dividend-free price level proxy is NOT used, 12-month turnover is NOT used (not in the fixed list). Features that cannot be built for a month are missing, never imputed from the future.
- **Target:** next-month return rank-transformed within the month. **Model (fixed, no tuning, no cross-validation, no architecture search):** `HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
  max_iter=200, min_samples_leaf=500, l2_regularization=1.0, random_state=0)`, expanding training window from 1990-01, retrained each January on formation months up to the previous November (labels through
  December), predictions for the next 12 formation months are a deterministic function of the frozen model and that month's features. Trial count: 2 (the ML model and its comparator).
- **Comparators:** (i) equal-weight composite of the same 20 features with literature signs (value-type +, gpa +, ROA +, asset growth -, accruals -, issuance -, leverage -, momentum +, reversal +, volatility -,
  beta -, size -, ...; signs fixed in code before any run), (ii) gpa alone (incumbent).
- **Evaluation (forward only):** formation months from 2025-12 on. Months 2026-01..2026-09 are PSEUDO-FORWARD (model frozen with data through 2025-12; these months were realised before the harness but have not been
  used by any result so far); later months are LIVE (predictions hashed into a tracked manifest before the month realises, as in the paper ledger). Metrics per signal: mean monthly rank IC (t), VW top-quintile minus
  VW universe gross return and net of an assumed 10 bp per one-way turnover, turnover. Checkpoints at 12, 24, 36 forward months; DECISION only at 36 months: 'ML ADDS VALUE' if its net long-minus-universe exceeds the
  composite's by >= 0.15 %/mo with a paired t >= 2 and also exceeds gpa's; 'NO EVIDENCE' otherwise. Interim checkpoints are reported but never trigger deployment.
- **Information only (not evidence):** a walk-forward 1995-2025 run of the same frozen protocol, with the explicit caveat that its inputs, features and model class were chosen with knowledge of this period; useful only
  to catch gross implementation errors (an ML IC far above the composite's in-sample would signal leakage, not skill).
- **Power, honestly:** with ~5% tracking error a 0.15 %/mo edge needs many years to confirm; the realistic outcome of the 36-month decision is 'NO EVIDENCE'. The value of this protocol is that it cannot fool us.

- **v8 implementation notes (recorded before any real run):** (1) the 12-month share-issuance feature (`issu`) is the change in log `cshom` with split months removed (|dlog cshom| > 0.10 and dlog ME within 25% of the jump
  implied by the return), because `comp.secm` shares are not split-adjusted; (2) model-to-month mapping: formation months of year Y are scored by the model trained on formation months through November of Y-1
  (labels through December); (3) CLARIFICATION of the decision window: the 36-month decision counts only PROSPECTIVE forward months (formed within 7 days of the target month's start); pseudo-forward and retrospective
  months (2026-01..2026-09 and any formed late) are reported at the checkpoints but do not enter the decision - stricter than the pre-registration text and chosen before any real formation exists; since 2026-10 is the
  first month that can be prospective, the decision cannot arrive before about late 2029; (4) walk-forward information run stops at formation 2025-11 so the pseudo-forward months stay unspent; (5) turnover for the
  first formation month is 1.0; one value-weighted benchmark of all eligible names is used for all three signals; (6) `spec_commit` in the manifest is `git rev-parse HEAD`, so the code is committed before the first real formation.

- **v8 deviation recorded before any real formation (first real walk-forward run failed on coverage):** `comp_secm_hist` yields usable feature months only from 1998-04 (cause to be read from the printed coverage block;
  suspects: `trt1m` missing before 1998, or no `exchg`=11 rows so the NYSE 20th-percentile cut is undefined). The pre-registered training start of 1990-01 is therefore replaced by a DATA-DRIVEN start: the first month
  from 1990-01 whose 12-month span holds >= 1,000 usable rows (forward return and >= 15 of 20 features); the first walk-forward year is the first with >= 1,000 usable rows over >= 36 formation months. Explicit
  infeasible windows exit with a clear message. The chosen train_start is recorded in each manifest line. Everything else in v8 is unchanged.

## Result: v8 walk-forward information run (user run; NOT evidence) and the coverage cause
Coverage block: `cshom` (shares) is missing for 100% of comp.secm rows before 1998 and ~71% in 1998, so market equity (prccm x cshom) is undefined and the universe is empty until 1998-04; trt1m and prccm are fine back to 1985.
Chosen windows (data-driven): train_start 1998-04, first scored January model 2002 (24 models 2002..2025; 287 scored formation months through 2025-11). Leakage sanity passed (ML IC 0.0324 vs composite 0.0275, ratio 1.18, below 0.10).
Full-sample 2002-2025 (VW top quintile minus VW eligible universe, 10 bp per one-way turnover): ML IC +0.032 (t 5.5), gross +0.063, net +0.009 %/mo (t 0.1), one-way turnover 0.54;
equal-weight composite of the same 20 features IC +0.027, gross +0.268, net +0.242 (t 2.7), turnover 0.26; gpa IC +0.016, net +0.136 (t 1.5), turnover 0.03. ML minus composite: -0.233 %/mo (t -2.5);
by blocks the ML never beats the composite on a VW net basis (2012-2016 +0.02, all other blocks negative; 2022-2025: ML -0.40 vs composite +0.02 vs gpa +0.38).
Reading: the boosted model has a slightly higher rank IC than the simple composite but its VW top-quintile return is far lower and its turnover double: the information it adds sits in small, hard-to-trade names the
VW book does not hold. This is an in-sample-contaminated comparison (features, signs and model class were chosen knowing this period; composite signs are standard literature signs) and changes nothing in the v8 forward
protocol (no tuning after seeing it); prior for the forward test is 'NO EVIDENCE'. The composite's net +0.24 %/mo (t 2.7, turnover 0.26) is itself a hindsight-contaminated number, not a finding.

## Pre-registration v9: alternative-data probes (cheap; written before any code or run)
Datasets (all verified accessible; columns from the user's probe, docs/wrds_probe_altdata.txt): contrib_general.arc (XBRL reporting complexity, by cik and filing), contrib_general.as_firm_risks (10-K Item 1A text measures, by cik),
contrib_general.blocksv (13D/G blockholder positions, by company cik-year), contrib_general.classified_boards (cbi, by gvkey), contrib_corporate_culture.corp_culture (call-transcript culture scores, by gvkey-year),
contrib_corp_fed_litigation.firmyear_level (federal cases, by gvkey-year, 2006-2021). Judged exactly like Studies A and B: net long-minus-universe vs the incumbent gpa (same cost model, $10M 2019 market-size, 1.5x),
promotion = dev alpha t >= 2.0 and validation alpha > 0, 2020+ sealed and opened once only for promoted signals. SIX signals only (6 trials), direction fixed a priori:
1. `arc10k` = -(arc of the latest 10-K filed on or before month-end; complexity mispricing, higher complexity -> lower returns); stale > 15 months -> missing; PIT = date_filed.
2. `rf_len` = -log(1 + item_1a_word_count) of the latest 10-K (more disclosed risk text -> lower returns); PIT = filing_date; stale > 15 months -> missing.
3. `blocks` = + total blockholder `position` (sum over blockholders of company cik-year; more monitoring -> higher returns); PIT = June 30 of year+1; stale > 18 months; firms with no row = 0 only in years the dataset is populated.
4. `cbi_neg` = -cbi (classified board indicator, entrenchment -> lower returns); PIT = datadate + 6 months; stale > 18 months.
5. `culture` = + mean of the five z-scored culture scores (s_integrity, s_teamwork, s_innovation, s_respect, s_quality), z-scored within year; PIT = June 30 of year+1; stale > 18 months.
6. `litig` = -log(1 + case_f) (cases as defendant); PIT = June 30 of year+1; stale > 18 months; dataset ends 2021, so the signal is missing afterwards.
Linking: cik -> gvkey via the `cik` column of Compustat annual (a cik mapping to more than one gvkey is dropped), then gvkey -> permno through the date-bounded CCM links as before. Universe: the usual eligible universe
(common equity, ex-microcaps). Signals exist only in the years the datasets cover (arc/risk text mostly 2006+, litigation 2007-2021, culture 2002+, blocks 1994+, classified boards 1991-2024), so dev/validation windows are SHORT
for several of them; the promotion rule still applies unchanged and low power is accepted (a signal with fewer than 60 dev months cannot promote). Checks printed first: row counts per dataset, cik->gvkey hit rate, coverage by decade.
Counted as 6 trials. Honest prior: all are academic datasets with small, mined samples; the most likely outcome is that nothing promotes.

- **v9 amendment before the first real run (found in the synthetic smoke):** binary or heavily tied signals (cbi_neg, litig: every case_f == 0 row ties at the maximum) leave the top quintile EMPTY in any month where the tied group holds
  >= 40% of ranked names (average-rank quintiles), which would fail the signal for a structural reason and still log a trial. Fix: add 1e-9 x a fixed per-permno hash key (Knuth multiplicative) to every signal so exact ties are ordered the same way
  every month (the long/short legs then hold a fixed pseudo-random subset of the tied group; unbiased, less efficient; distinct values keep their order). Applied to all six signals for consistency. A promotion minimum of 60 dev months
  is implemented via `promotion_stats(min_dev_months=60)` as pre-registered.

## RESULT v9 / Study C (user run): alternative-data probes - nothing promoted (holdout not opened)
Checks: cik->gvkey linking unique for 69-72% of each dataset's ciks (rest not in Compustat; 0 ambiguous); coverage by decade: arc10k 80% (2010s), rf_len 91%, blocks 97%, cbi_neg 92%, culture 88%, litig 78% (2010s).
The tie amendment mattered: before it cbi_neg's long leg would have been empty in 310 of 337 pre-2020 months (median tied share 45%); litig's tied share was 30%.
Net long-minus-universe, % per month (t), dev / validation (dev months differ by signal: arc10k 66, rf_len 118, blocks 245, cbi_neg 275, culture 145, litig 88):
arc10k -0.009 (-0.04) / +0.354 (1.29); rf_len -0.120 (-0.92) / -0.015 (-0.15); blocks +0.027 (0.22) / -0.381 (-2.67); cbi_neg +0.078 (1.10) / +0.133 (1.50); culture -0.115 (-1.05) / +0.329 (1.86); litig +0.040 (0.18) / +0.077 (0.42).
Promotion (alpha over gpa, dev t >= 2.0 and validation alpha > 0): every dev alpha t is below 1.3 in absolute value (best: litig 1.06, cbi_neg 1.16) => **PROMOTED: NONE**. 72 cumulative unique trials.
Reading: no signal has development evidence; the two with positive validation alpha (arc10k, culture) had ~zero or negative development alpha, i.e. sign instability not signal; blocks is significantly negative in validation (the
opposite of the registered direction); cbi_neg is the only one with a consistent positive sign in both windows but t 1.1-1.5 on a 1991-2024 governance indicator that is widely known. The 2020+ window was not opened.

## Pre-registration v10 (SCOPED, not built): international replication of the JKP findings (written before any code or extraction; descriptive, NOT a tradability test)
Purpose: calibrate priors. The US results (decay after 2014, short-leg concentration, the 2021-22 episode) rest on one market; `global_factor` holds the same 153 characteristics for other countries, which is the only independent
cross-market sample available without waiting. It does not create an investable idea for a USD 35k account and no trial is promoted or deployed from it. Global factors are correlated, so this is cross-market, not independent-in-time, evidence.
- **Data:** `contrib_global_factor.global_factor`, same screens as the US extraction (obs_main = 1, common = 1, exch_main = 1, primary_sec = 1, size_grp in mega/large/small; note size_grp uses NYSE breakpoints in USD), `excntry <> 'USA'`, 1990-2025,
  returns `ret_exc_lead1m` (USD excess, per the documentation). Regions fixed from the documentation's MSCI table: **Developed Europe** (AUT BEL CHE DEU DNK ESP FIN FRA GBR IRL ITA NLD NOR PRT SWE), **Japan** (JPN), **Other developed**
  (AUS CAN HKG ISR NZL SGP). Emerging/frontier markets are excluded (information-only IC at most). A country-month needs >= 100 eligible stocks to enter. Extraction is one parquet per year (resumable), as for `jkp_us`.
- **Signals (no new ones):** the 15 Study A signals exactly (13 theme scores + `jkp_all13` + `jkp_slow10`) plus `gp_at` as the gpa analogue; theme scores use percentile ranks within COUNTRY-month; portfolios pool stocks across countries within the region,
  value-weighted by USD market equity; the US is run through the identical code path (from `jkp_us`, JKP returns) as the reference.
- **Reported per region and period (dev 1990-2014, validation 2015-2019, 2020-2025; the international 2020+ window has never been looked at):** mean monthly rank IC with NW6 t; gross VW Q5-Q1, Q5-U, Q1-U; one-way turnover of the top quintile;
  information-only net long-minus-universe using half-spread = `bidaskhl_21d`/2 and a 1.5x stress. No promotion rule applies.
- **Three labelled questions (fixed now; composites `jkp_all13` and `jkp_slow10` only):**
  H1 *Is the decay global?* ratio = mean IC in (validation + 2020-2025) / mean IC in dev. GLOBAL DECAY if ratio <= 0.5 for both composites in >= 2 of 3 regions; PERSISTS OUTSIDE THE US if ratio >= 0.75 in >= 2 of 3; otherwise MIXED.
  H2 *Is the edge in the short leg?* share of Q5-Q1 coming from Q1-U in validation + 2020-2025, per region, compared with the US (0.84 / 0.91): SHORT-LEG if >= 0.6 in >= 2 of 3 regions.
  H3 *Was 2021-2022 global?* slow10 gross Q5-Q1 averaged over 2021-2022 versus its mean over all other months since 2015: GLOBAL EPISODE if the 2021-22 mean exceeds 3x the other-months mean in >= 2 of 3 regions, else US-SPECIFIC.
- **What each answer changes:** GLOBAL DECAY + SHORT-LEG supports the post-publication-decay / crowded-short-leg reading and closes the signal search for good. PERSISTS OUTSIDE THE US would mean the US decay may be sample-specific; the only follow-up allowed is a NEW
  pre-registration of a long-only ex-US factor-tilt via broad ETFs, judged forward. Nothing here touches the US sealed windows.
- **Limits:** Compustat-sourced returns outside the US are lower quality than CRSP; USD returns include currency noise; short selling and costs abroad differ (the spread proxy is crude); thin countries; common global shocks.
- **Cost of doing it:** one extraction (about 4-5M screened rows x ~160 columns, per-year files) and one script run of minutes on 36 cores; no new data sources.

## Pre-registration v11: expectation-change family from earnings announcements (EAR, SUE), without analyst data (written before any code, extraction or run)
Economic hypothesis: prices under-react to earnings news (post-earnings-announcement drift), so the market's reaction to, and the size of, the latest earnings surprise predict the next months' returns. Analyst-revision data (I/B/E/S, Zacks)
is NOT accessible (confirmed absent), so expectation change is measured with the announcement-window return and a seasonal-random-walk surprise from Compustat. Decision it could change: whether an earnings-news tilt is worth adding to the passive
core (prior: no - PEAD is concentrated in small, costly names and the SUE-type JKP members inside the Profit Growth theme already failed net of costs in Study A).
**Data (access verified by the earlier probes; checks printed first):** `compustat_quarterly` (gvkey, datadate, fyearq, fqtr, rdq, ibq, saleq; already extracted), daily CRSP returns `crsp.dsf_v2` (permno, dlycaldt, dlyret; new per-year extraction, resumable),
daily FF market factor `ff.factors_daily` (mktrf, rf; new small extraction), CCM links, the usual eligible universe (common equity, ex-microcaps).
**Events:** each quarter with a non-missing `rdq` (Compustat report date; rows without rdq are dropped). Day 0 = the first CRSP trading day on or after rdq. Linked gvkey -> permno through the date-bounded CCM links at rdq.
**Signals (4 trials; high = good, long the top quintile):**
1. `ear` = sum over trading days -1, 0, +1 of (stock return - (mktrf + rf)); all three returns required. Known from the close of day +1.
2. `sue` = (ibq_q - ibq_{q-4}) / sd of the last 8 seasonal changes (ibq_k - ibq_{k-4}, k = q-7..q; at least 6 required); q-4 means same fqtr, fyearq-1. Known from the close of day +1 of the announcement.
3. `rsue` = the same with saleq.
4. `ear_sue` = mean of the cross-sectional percentile ranks (within month, among eligible names) of `ear` and `sue`; needs both.
**Carry rule:** at each month-end use the latest event whose day +1 is on or before that month-end; if it is more than 130 days after rdq the signal is missing. Exact ties are broken with the fixed per-permno hash used in v9.
**Judging (exactly Studies B and C):** net long-minus-universe vs the incumbent gpa (the usual cost model, $10M in 2019 market-size dollars, 1.5x), promotion = dev (1990-2014) alpha t >= 2.0 and validation (2015-2019) alpha > 0, minimum 60 dev months; 2020+ is
sealed and opened once, only for promoted signals, by `--open-holdout`. Cumulative trials become 76.
**Information only (not decision inputs):** equal-weighted Q5-Q1 gross, rank IC by size tercile (does the drift live in small caps?), IC by months since the announcement (0-1, 2, 3), turnover of the top quintile.
**Checks printed first:** rdq coverage by decade (share of quarters with rdq), events per decade, share of events with a complete 3-day window, SUE availability, daily-file row counts, merge hit rate to permno.
**Power / honesty:** the VW ex-microcap long-only book is where PEAD is weakest; the most likely outcome is that nothing promotes. The information-only block is there so that, if nothing promotes, we can see whether the effect exists in small names we cannot trade.

- **v11 implementation notes (recorded before any real run):** SUE/RSUE history uses all de-duplicated quarters (a quarter without rdq still serves as the q-4 comparison and as one of the 8 changes; rdq is required only to form the event);
  an event is attached to every permno whose CCM link is in force on rdq (share classes all receive it); the latest event is used even if its own value is missing (no fall-back to an older quarter); availability = the first month-end on or after the calendar date of
  day +1; the months-since-announcement buckets are 0-1, 2 and 3+; events outside the daily calendar get no EAR. Daily file estimated at 55-75M rows (a few hundred MB of parquet), peak memory 3-5 GB.

- **v10 implementation notes (recorded before any real run):** series are indexed by realised-return month (formation month + 1), so a December-2014 formation counts in validation; an eligible stock has `me > 0` and a non-missing next-month return (ranks are over eligible stocks);
  EXTRA rule not in the pre-registration: a country-month enters a signal's portfolio only with >= 20 names carrying that signal (the >= 100 eligible-stock rule for the country-month is as registered); turnover = 0.5 x sum|dw|, proxy cost = 1.5 x sum|dw| x half-spread
  (leavers charged at their previous-month spread; a country-month with no spread data gets 0, which understates cost, and is counted in CHECKS); H1 ratio = mean IC 2015-2025 / mean IC dev, n/a when dev mean <= 0 (an n/a region counts toward neither label), a region counts only if BOTH
  composites meet the threshold; H2 share = -mean(Q1-U)/mean(Q5-Q1) over 2015-2025 (n/a if the Q5-Q1 mean is not positive; the non-SHORT-LEG branch is printed NOT SHORT-LEG); H3 uses strict >, denominator = all months from 2015-01 except 2021-22, n/a counts as not global.

## RESULT v11 / Study D (user run): earnings-announcement expectation change (EAR, SUE, RSUE) - nothing promoted (holdout not opened)
Checks: daily returns 70,975,951 rows (1989-2025); 2,137,984 quarters, 64.9% with rdq; events 1,386,903; permno link at rdq 74% overall (60-64% in the 2000s-2010s, i.e. many Compustat quarters have no CCM-linked common stock); complete 3-day windows 98-99% in the 1990s-2010s
(12% in the 1980s, 0% in the 1970s because the daily file starts in 1989); SUE available for 88.6% of events. Signal coverage of the eligible universe from the 1990s: ear 92-98%, sue 86-96%, rsue 86-95%, ear_sue 86-96%.
Net long-minus-universe %/mo (t), dev / validation: ear -0.054 (-0.46) / +0.154 (0.79); sue +0.063 (0.71) / -0.087 (-0.58); rsue +0.041 (0.42) / +0.352 (2.11); ear_sue +0.031 (0.35) / -0.008 (-0.05); gpa +0.184 (1.88) / +0.098 (0.60).
Gross rank IC dev / validation: ear 0.0156 (5.0) / 0.0032 (0.4); sue 0.0231 (4.9) / 0.0074 (0.9); rsue 0.0119 (2.2) / 0.028 (2.1); ear_sue 0.0256 (6.0) / 0.0074 (0.8).
Promotion (alpha over gpa, dev t >= 2 and validation alpha > 0): every dev alpha t is between -0.62 and -0.02 => **PROMOTED: NONE**. 76 cumulative unique trials.
Information only: (1) the gross effect is real before 2015 and mostly gone after: EW Q5-Q1 dev 0.42 / 0.45 / 0.19 / 0.52 %/mo (t 3.9 / 2.8 / 1.2 / 3.8) vs validation 0.21 / -0.01 / 0.38 / 0.13 (all |t| < 1.2); (2) it is a SHORT-HORIZON effect: IC is 0.023-0.034
(t 5.6-7.1) in the first two months after the announcement, about zero in month 3, and for ear NEGATIVE after month 3 (dev -0.032, t -2.2), so the pre-registered 130-day carry dilutes it (a shorter carry would be a new trial, not tested); (3) NOT a small-cap-only effect: dev IC by size tercile
0.016 / 0.017 / 0.012 for ear (similar across terciles), all ~0 in validation; (4) turnover of the top quintile is 12-33% a month (rsue lowest), which together with the value-weighted ex-microcap book leaves net ~0. rsue is the only signal with a positive validation (+0.352, t 2.11) but its
dev alpha is -0.05 (t -0.54): sign instability, one of four trials, not promoted and not to be chased.
Reading: another family whose gross historical effect decayed after about 2014 and does not survive costs in the tradable book; consistent with the Study A/C pattern. Expectation-change via announcements is closed (see registry).

## RESULT v10 (user run): international replication of the JKP findings - H1 = MIXED, H2 = NOT SHORT-LEG, H3 = US-SPECIFIC (descriptive, no trials)
**Coverage (check before reading):** with size_grp in mega/large/small (NYSE breakpoints in USD) and the >= 100-stock rule, AUT BEL DNK FIN IRL NOR PRT NZL had no qualifying month; ESP 11, NLD 31 and ISR 32 months only. "Developed Europe" is therefore CHE (from 1995), DEU, FRA, GBR, ITA (to 2023),
SWE (from 1999) plus a few NLD/ESP months; "Japan" is JPN; "Other developed" is AUS, CAN, HKG (from 1992), SGP (to 2023) plus ISR from 2021. 189 of 4,326 kept country-months (1990s Europe/Other) have no spread data, which flatters net there. Scored stock-months: 2,231,322.
**H1 (is the decay global?):** IC ratio (2015-2025 / dev) for jkp_all13 / jkp_slow10: US 0.50 / 0.71 (reference); Europe 0.68 / 0.78; Japan 0.10 / 0.01; Other developed 0.88 / 0.93. Both ratios <= 0.5 in 1 of 3 regions, both >= 0.75 in 1 of 3 => MIXED.
Rank IC itself barely decays outside Japan (e.g. Other developed slow10 IC 0.062 dev, 0.057 val, 0.058 later; US slow10 0.047 / 0.020 / 0.045); the decline in tradable spreads is partly global: slow10 gross VW Q5-Q1 %/mo, dev / validation / later: US 1.04 / 0.55 / 0.75; Europe 0.96 / 0.63 / 0.51;
Japan 0.36 / -0.13 / -0.05; Other developed 1.37 / 0.48 / 0.81 (roughly halves after 2014 in three markets, absent in Japan).
**H2 (short leg?):** share of Q5-Q1 from Q1-U, 2015-2025, all13 / slow10: US 1.04 / 0.84; Europe 0.42 / 0.71; Japan 1.10 / n/a; Other 0.29 / 0.80. Rule needs both composites >= 0.6 in >= 2 regions: 0 of 3 => NOT SHORT-LEG. But for slow10 alone the short-leg
dominance replicates in all three developed non-Japan markets (Q1-U -0.37 / -0.64 later vs Q5-U +0.08 / +0.18 in Europe / Other): the long leg of slow10 is nearly worthless outside the US too.
**H3 (2021-22 global?):** slow10 gross Q5-Q1 2021-22 vs other months since 2015: US 2.59 vs 0.22 (11.5x); Europe 0.97 vs 0.45 (2.1x); Other 1.42 vs 0.47 (3.00x, not strictly > 3); Japan n/a (non-positive baseline) => US-SPECIFIC by the rule: the episode was largest in the US and elevated elsewhere.
**Net of the (crude) spread proxy:** composites are about zero everywhere after 2014 (slow10 validation / later: Europe +0.09 / -0.13, Japan -0.27 / -0.35, Other -0.11 / -0.06; all13 negative), the only positive net numbers are development-period ones (Other developed slow10 +0.44, t 2.4; value in Japan/Other).
**Reading:** the US pattern is not a US-only artefact: the cross-sectional information (IC) persists in Europe and Other developed, tradable spreads roughly halved after 2014 in three markets, the slow10 edge is short-leg heavy in the same three, and costs remove the rest. "Are factor premia stronger outside the US?" - no, not after 2014 and not net of costs.
Per the pre-registration neither GLOBAL DECAY + SHORT-LEG (closes the search) nor PERSISTS OUTSIDE THE US (licenses a new ex-US tilt pre-registration) was met, so NO follow-up is licensed. Limits: thin countries dropped, Compustat-sourced foreign returns, USD currency noise, crude spread proxy,
cross-market not independent-in-time evidence.

## Pre-registration v12: short interest (information diffusion; written before any code, extraction or run)
Economic hypothesis: informed short sellers act on negative information that prices incorporate slowly, so high short interest predicts low returns (Asquith-Pathak-Ritter; Boehme-Danielsen-Sorescu; Rapach-Ringgenberg-Zhou). The long-only decision it can change: **should heavily shorted names be excluded from
the gpa tilt?** (a veto needs no borrow). Standalone long-only signals are weak by construction (the information is in the high-SI leg) and are included only as the cheap, standard cross-check.
**Data (access verified by the user's `scripts/wrds_probe.py` run; columns as probed):** `comp.sec_shortint` (gvkey, iid, shortint, shortintadj, datadate, splitadjdate; 5,372,584 rows, 2006-07-14 .. 2026-09-30), `crsp.msf_v2` columns already in the panel (`shrout`, plus `mthvol` which the
panel does not select today and the build adds), CCM links with `liid`. Extraction: gvkey, iid, datadate, shortint, shortintadj (the latter ONLY for a check line) for datadate >= 2006-01-01.
**Traps fixed now (so none can be tuned later):**
1. SIR = raw `shortint` / (CRSP `shrout` x 1000) using the formation month's own `shrout`. `shortintadj` (adjusted as of a later `splitadjdate`) is never a signal input; it appears in one check line comparing shortint and shortintadj.
2. Linking is at the SECURITY level: (gvkey, iid) -> permno through the date-bounded CCM links filtered as in the research, matching `iid` to `liid`. A share class is never given another class's short interest; records whose (gvkey, iid) has no link with that liid are dropped and counted.
3. Point in time: each record is available at `datadate` + 14 calendar days (settlement date to publication); at month-end m the latest record with availability <= m is used (often the mid-month record; in short months the prior month-end one). A record older than 62 days from its availability date is
   missing. Missing is missing, never zero. Exact ties are broken with the fixed v9 per-permno hash.
4. Rows: duplicates of (gvkey, iid, datadate) are asserted to be zero. Units do not matter for ranks; the check prints median SIR, p99 and the share of SIR > 1 as a sanity check only.
**Signals (3 trials; cumulative 76 -> 79; high = good, long the top quintile):**
1. `sir_neg` = -SIR.
2. `dtc_neg` = -days to cover = -(shortint / (mthvol / 21)), mthvol of the formation month (rank-invariant to the volume unit; the post-2004 Nasdaq volume convention is the same across the sample).
3. `gpa_veto` = the incumbent `gpa` signal with every name in the top SIR DECILE of that month (highest SIR among eligible names with a SIR value) set to missing BEFORE ranking (so the quintile breakpoints are those of the remaining names); names with no SIR are NOT vetoed (the share is reported).
**Judging:** as Studies B-D: net long-minus-universe, the usual cost model ($10M in 2019 market-size dollars, 1.5x), vs the incumbent gpa; dev (the data start 2006-07, so about 2006-08 .. 2014-12, ~100 months) alpha t >= 2.0 AND validation (2015-2019) alpha > 0, minimum 60 dev months;
2020-2025 is sealed (never looked at for any short-interest signal) and opened once with `--open-holdout`, only for promoted signals. **Extra requirement for `gpa_veto` only (size floor):** its mean net long-minus-universe improvement over gpa must be >= 0.05 %/month in development AND > 0 in validation,
because the veto is almost collinear with gpa and a tiny effect can clear t >= 2.0 without being worth implementing. The veto's decision rule: "exclude the top SIR decile from the tilt" is licensed only if `gpa_veto` is PROMOTED under this full rule (and, if the holdout is opened, its holdout improvement is > 0).
**Information only (no decision weight):** gross VW and EW return of the top-SIR decile minus the universe (the short-leg effect) by dev / validation, rank IC by size tercile, rank IC by year, turnover of the top quintile, share of eligible names with a SIR by year.
**Checks printed first:** records per year; median days between records per gvkey by year (the cadence check - monthly in the early years would justify the 62-day staleness); duplicate count (asserted 0); iid->liid link hit rate; share of eligible names with a value by year; the SIR distribution; share of vetoed rows.
**Honest prior:** the documented effect is concentrated in small, hard-to-borrow stocks that the VW ex-microcap book excludes, and it decayed after 2005; the most likely outcome is nothing promoted and a veto worth well under 0.05 %/mo. A negative result is the expected one and is recorded as such.

- **v12 implementation notes (recorded before any real run):** (a) signals are attached to the rows of the incumbent gpa frame (`build_research_signals` output), so `sir_neg` and `dtc_neg` exist only for names with a gpa row and `gpa_veto` and `gpa` are ranked on the identical universe; the veto decile is formed among
  `eligible_universe` names that have a SIR (top decile = within-month percentile rank > 0.9). (b) Record linking requires the CCM link to be in force on `datadate`; (gvkey, iid) -> permno through `liid`. (c) The registered size-floor measure is net long-minus-universe, where each book's "universe" is the set of names with its own signal, so
  `gpa_veto`'s universe excludes the vetoed names; if high-SIR names underperform this UNDERSTATES the veto's improvement. The registered measure stays binding for promotion; the runner prints, information only, the direct net long-book difference (veto long book minus gpa long book) beside it and any disagreement is reported, not used to relax the rule.
- **v12 amendment before the first real run (found in review, aligns code with the registered window):** `gpa_veto` is set to MISSING in every month where no eligible name carries a SIR (before 2006-07). Otherwise it would equal gpa row for row in 1990-2006, adding ~200 zero-difference months to the
  development window, diluting the 0.05 %/mo size floor by about two thirds and letting `min_dev_months = 60` pass trivially. With the amendment all three signals and the floor live on the months with short-interest data (about 2006-08 .. 2014-12 development). No threshold or rule changed.

## RESULT v12 / Study E (user run): short interest - nothing promoted, veto NOT licensed (holdout not opened)
Checks: 5,372,584 records, cadence 15 days in every year 2006-2026 (so the 62-day staleness was ample), 0 duplicates; only 50.4% of records link at the security level (iid = liid) to a CRSP permno (the file covers many OTC/foreign/non-common securities), but 99-100% of eligible names have a SIR from 2007
(49% in 2006, the half-year of data); SIR median 1.6% of shares, p99 25%, 0.02% above 1; DTC median 2.5 days; 3.2% of eligible formation rows vetoed. Development 100 months (2006-08..2014-12), validation 60.
Net long-minus-universe %/mo (t), dev / validation: sir_neg +0.008 (0.11) / +0.071 (1.03); dtc_neg -0.012 (-0.19) / +0.184 (2.08); gpa_veto +0.174 (1.23) / +0.128 (0.79); gpa (same dates) +0.201 (1.39) / +0.098.
Promotion (alpha over gpa, dev t >= 2 and validation alpha > 0): dev alpha t -0.25 / 0.00 / -1.16 => **PROMOTED: NONE**. Size floor for gpa_veto (registered measure): dev -0.028 %/mo (t -1.53), validation +0.030 (t 1.50) => FAIL; direct long-book difference (information only): dev -0.022, validation +0.036, same conclusion. 79 cumulative unique trials.
Information only: (1) gross rank IC of sir_neg is real and has NOT decayed (0.0175, t 2.1 dev; 0.0257, t 2.4 validation; similar across size terciles), unlike every other family tested; (2) the information sits where predicted - the highest-SIR decile underperforms the universe by 0.31 / 0.19 %/mo VW (t -1.06 / -0.64) and 0.32 / 0.40 %/mo EW (t -1.83 / -1.51) -
right sign in both windows, never significant; (3) a long-only book cannot use it: the low-SIR quintile is almost the universe (net +0.01 / +0.07), and vetoing 3% of rows changes the tilt by about 0.03 %/mo (around USD 10 a month on USD 35k) with no statistical support; (4) dtc_neg's validation t of 2.08 comes with a dev alpha of zero: sign instability, one of three trials, not to be chased.
Reading: short interest carries a persistent but small ranking signal whose payoff is in the short leg, i.e. exactly the borrow-constrained place we have no fee data for. For a long-only account the veto is worth about nothing. Short-interest family closed; the 2020+ window remains unused for it and no follow-up is licensed.

## Pre-registration v13 (Study F, Stage A only): out-of-sample volatility and tail-risk forecasting, GJR-GARCH-t versus simple benchmarks (written before any code or run)
Purpose: forecasting, NOT alpha. Question 1 of the three separate questions in the options discussion: does the heavier model (GJR-GARCH(1,1), Student-t innovations, winsorised input, Monte Carlo) forecast variance and downside risk better than simple benchmarks, out of sample? The decision it can change: which volatility model sizes positions
(including the daily-reset drag of leveraged ETFs, drag ~ (L^2 - L) sigma^2 / 2 per year) and whether a heavier model earns a place. **Stages B-D (forecast versus option-implied variance, option P&L) are NOT authorised and NOT run**: OptionMetrics is absent and Cboe is permission-denied, so no implied volatility exists in our data; they need a separate pre-registration if per-stock option data is obtained.
**No trials are logged and the trial count stays at 79**: no return series or signal is promoted. The governance (written spec first, dev / validation / holdout opened once, simple benchmarks, record the outcome) is kept.
**Data (only):** `data/raw/crsp_daily_ret/{year}.parquet` (permno, dlycaldt, dlyret; 1989-2025, extracted for v11 by `scripts/extract_earnings.py`), `data/raw/ff_daily.parquet` (market return = mktrf + rf, decimals via `ff_decimals`), `crsp_monthly`/`crsp_delistings` for the universe, `crsp_names` for the information-only ticker block. No price, volume or high/low, so range-based or intraday realised-variance estimators are out. No new dependency: GARCH QMLE is implemented with scipy.
**Assets:** (i) primary = a ROLLING universe of the 100 largest US common stocks: each calendar year Y, the top 100 companies (permco) by prior-December company market equity among common equity (largest permno per permco); forecasts for stock i are made only during Y while i trades; no survivor filter (a stock that delists stops being forecast);
(ii) the FF market (mktrf + rf) reported separately, information only (one series); (iii) information only: the user's tickers (below).
**Return series:** daily decimal returns, zero conditional mean, calendar = sorted union of all daily dates. Forecast date t means information through the close of t; targets use returns on t+1... only.
**Estimation window and refitting:** at each refit date (every 21st trading day of the market calendar, calendar index multiple of 21) estimate on the asset's last 1000 trading days ending at t (inclusive); require >= 950 non-missing returns in them (missing days are dropped from the window; the window is the asset's own non-missing returns). Parameters are held for forecast dates t_r .. t_r + 20; the variance recursion is
re-run from the window start with those parameters (initial variance = window sample second moment) so each forecast uses all data through its own date and nothing later. Zero mean (no demeaning).
**Models (six; forecast of average variance over the next h days, h = 1 and h = 21):**
- M0 `hist`: mean of the last 252 squared returns (flat). M1 `ewma`: lambda = 0.94 (flat). 
- M2 `garch`: GARCH(1,1), Gaussian QMLE, variance targeting (omega = (1 - p) s^2, s^2 = window second moment, p = alpha + beta). The h-step mean is analytic: E sigma^2_{t+k} = s^2 + p^(k-1) (sigma^2_{t+1} - s^2), averaged over k = 1..h.
- M3 `gjr_t` (the method under test): GJR-GARCH(1,1), sigma^2_{t+1} = omega + (alpha + gamma 1[r_t < 0]) r_t^2 + beta sigma^2_t, variance targeting with p = alpha + gamma/2 + beta, STANDARDISED Student-t innovations (unit variance, scale sqrt((nu - 2)/nu), nu in [2.1, 100]), exact ML. Bounds alpha, gamma in [0, 1], beta in [0, 1). The h-step mean uses the same analytic formula with p = alpha + gamma/2 + beta.
- M4 `gjr_t_wins` (ablation of the winsorisation concern): M3 with the window's returns clipped at the window's own 0.5th and 99.5th percentiles before BOTH estimation and filtering (forecasts are for the clipped series, as the method does).
- M5 `har`: HAR on squared returns, y = mean of r^2 over t+1..t+h regressed by OLS in LEVELS on [1, r_t^2, mean r^2 (t-4..t), mean r^2 (t-21..t)] using the window rows whose target lies inside the window; coefficients fixed between refits; forecast floored at 0.1 x the window second moment (floor hits counted).
**Fallback (as described by the method, counted and scored):** if M3 or M4 fails to converge, returns a non-finite value, or has p >= 0.999, that date's forecast is the M2 (Gaussian GARCH) forecast; if M2 itself fails or has p >= 0.999 the forecast is M1. Fallback forecasts STAY in the evaluation, scored as the heavy model's forecast (dropping them would flatter it); the fallback share is reported per model.
**Targets and loss (primary):** proxy = r_{t+1}^2 (h = 1) and sum of r^2 over t+1..t+21 divided by 21 (h = 21; only forecast dates with a complete 21-day window, the count and year of dropped incomplete windows are reported). Primary loss QLIKE = ln(sigma_hat^2) + proxy / sigma_hat^2 (robust to the noisy proxy; defined at r^2 = 0). MSE of variance is information only.
**Inference:** per forecast date, average the loss difference (model - benchmark) across the stocks forecast that day, then a Diebold-Mariano test on that time series with Newey-West lags 5 (h = 1) and 30 (h = 21); mean difference and t reported on dev (forecast dates 1995-01..2014-12), validation (2015-01..2019-12), holdout (2020-01..2025-12, sealed until `--open-holdout`, opened once).
**Fixed comparison set:** M3 versus M2 (GARCH) and M3 versus M5 (HAR), at h = 1 and h = 21, on the primary stock universe. Everything else (M0, M1, M4 versus M3, the market series, tickers) is descriptive.
**Decision rule (M3 is LICENSED as the position-sizing volatility model only if ALL four comparisons satisfy):** dev mean QLIKE difference <= -0.01 (materiality) AND dev DM t <= -2.0 AND validation mean difference < 0; then the holdout is opened once and the licence STANDS only if all four holdout differences are < 0, else it is recorded "licensed in dev/validation, failed in the holdout".
If M3 is not licensed, the volatility model for sizing is the model with the lowest dev mean QLIKE at h = 21 among M0, M1, M2, M3, M5 (M4 is an ablation, not a sizing candidate). The winsorisation question is answered descriptively by M4 versus M3 (mean QLIKE difference and DM t, dev and validation).
**Information only (no decision weight):** predictive log score of r_{t+1} (Normal for M0, M1, M2, M5; standardised Student-t with the fitted nu for M3, M4; fallback uses Normal) and 1-day 1% and 5% VaR violation rates (nominal vs realised, pooled; Kupiec p-values flagged as optimistic because of cross-sectional dependence); 21-day return VaR coverage (1% and 5%) from Monte Carlo (20,000 paths, fixed seed per (asset, refit date), GJR dynamics with standardised-t shocks for M3 and M4, Gaussian shocks for M2) evaluated only at refit dates (non-overlapping 21-day returns);
QLIKE by calendar year and in 2008-09 and 2020 stress windows; fitted parameter summaries (median alpha, gamma, beta, nu, persistence, share of nu at its bounds). A unit test checks that the Monte Carlo mean variance equals the analytic mean within sampling error: more paths shrink Monte Carlo error, not model error.
**Information only: the user's tickers** (ORCL, INTU, LITE, BE, STX, AMD, CELH, ALAB, SNDK; SKHY is not a CRSP security): a ticker maps to the permno whose CRSP name record is in force on the last daily-data date; an ambiguous or missing match is refused and printed (SNDK has two securities over time; the one in force in 2025 has under 1,000 days of history, as does ALAB, so they are reported as "insufficient history", not forecast). For the others, over their full available history (not restricted to the top-100): mean QLIKE by model at h = 21, mean ratio of forecast to realised 21-day variance by model over the last 3 years of data, and the implied annualised daily-reset drag of a 2x product (L = 2: (L^2 - L) sigma^2 / 2 = sigma^2 per year) from the M2 and M3 forecasts versus realised.
**Checks printed first:** daily rows per year and calendar length; ff coverage; universe size per year; stock-years forecast; share of windows meeting the history rule; fit success and fallback shares per model; optimiser summary; incomplete h = 21 windows dropped; a recomputation of the vectorised GARCH/GJR variance recursion against a slow reference loop on a sample (must agree to 1e-10 or the run aborts); the standardised-t density check (integrates to 1, unit variance).
**Honest prior:** GARCH(1,1) is famously hard to beat out of sample; I expect M3 to be roughly tied with M2 on QLIKE, possibly slightly better at h = 1, and M5 (HAR) competitive at h = 21; winsorisation probably makes forecasts slightly too low in crises. The most likely outcome is "not licensed", with the default sizing model chosen by the dev QLIKE ranking.
Compute: fitted models per refit: M2, M3, M4 (L-BFGS-B on a vectorised recursion, tens of milliseconds each); roughly 3,100 stock-years x 12 refits; minutes on 36 cores with `--jobs`.

- **v13 implementation notes (recorded before any real run):** (1) SEAL: without `--open-holdout` no forecast dated 2020 or later is computed for the primary or market series (the h = 21 target of a December-2019 forecast reaches into January 2020; this one-month overlap is accepted). The holdout is computed with the flag but its numbers are printed only if all four comparisons pass the dev/validation rule, per the licence rule.
  (2) The user-ticker block is information only; its "last 3 years of data" lie inside the sealed window and show model QLIKE by model, which would reveal holdout-era model comparisons, so by default it is capped at 2019-12-31 (a printed note says so) and runs to the end of the data only with `--open-holdout` (constant `TICKER_BLOCK_IGNORES_SEAL` in `scripts/run_volfc.py` prints it as literally pre-registered). For current-volatility questions about the user's own names, use the journal analyzer's explicit `--vol` inputs, not this block.
  (3) Window = non-missing returns among calendar days t-999..t (>= 950 required); representative security of a company = largest `me` (ties: smaller permno); a forecast date needs a return on t; h = 1 proxy = return on calendar day t+1; h = 21 needs all 21 calendar days t+1..t+21; one common valid-row mask per horizon for all models; non-finite or non-positive forecasts are excluded and counted.
  (4) Optimiser: L-BFGS-B on log nu, unit-scaled returns, p projected to 0.9999 inside the objective, breach judged at 0.999, "converged" = scipy success; EWMA fallback in the Monte Carlo is IGARCH(0.06, 0.94) with omega 0; MC seeds from (asset, refit index) so `--jobs 1` equals `--jobs N`; BLAS threads are pinned inside workers.
  Review sanity check (simulated GJR-t, 6,000 days, known parameters): fitted medians alpha 0.047 / gamma 0.069 / beta 0.880 / nu 6.6 against truth 0.04 / 0.08 / 0.88 / 6; mean QLIKE ranking at h = 1 and 21 puts the true model first (gjr_t < garch < har < gjr_t_wins < ewma < hist), as it should.

## RESULT v13 / Study F (user run): out-of-sample volatility forecasting - gjr_t LICENSED IN DEV/VALIDATION, holdout still sealed
Checks: 70,975,951 daily rows (1989-2025), calendar 9,319 days, FF market on every day; rolling top-100 universe, 2,421 of 2,500 stock-years forecast (600,465 forecast rows, dev+val); history rule met by 96.4% of refit blocks; fallback share of forecast rows: garch 8.3% (to EWMA), gjr_t 5.0%, gjr_t_wins 4.7%; recursion check 5.5e-16; t-density check passes; 1,082 incomplete h=21 windows dropped.
Fixed comparisons, mean QLIKE difference gjr_t minus benchmark (DM t), dev / validation: vs garch h=1 -0.0163 (-7.86) / -0.0117 (-2.11); vs garch h=21 -0.0131 (-4.80) / -0.0080 (-1.63); vs har h=1 -0.0614 (-9.62) / -0.0237 (-2.55); vs har h=21 -0.0554 (-5.71) / -0.0126 (-2.11). All four meet dev <= -0.01, dev t <= -2 and validation < 0 => **LICENSED IN DEV/VALIDATION; the holdout (2020-2025) has not been opened.** Default sizing model by dev QLIKE at h=21: gjr_t (-6.978) ahead of garch (-6.965), ewma (-6.964), hist (-6.923), har (-6.922).
Reading, with the caveats that matter: (1) the improvements are small (about 0.01-0.02 QLIKE units, i.e. 1-2% in log-variance terms) and SHRINK from development to validation in all four comparisons (e.g. vs garch h=21 -0.0131 -> -0.0080, validation t only -1.6): they clear the pre-registered bar but are not a large effect; (2) on the market series alone the edge exists at h=1 (vs garch -0.045 dev, -0.064 val, t -6.4 / -2.6) but is ABSENT at h=21 (vs garch -0.006, t -0.6; -0.001, t -0.1), so for monthly-horizon sizing of a market position, plain GARCH(1,1) is as good; (3) winsorisation HURTS: M4 minus M3 = +0.0132 (t 3.6) at h=1 and +0.0148 (t 2.0) at h=21 in dev, same sign in validation, and in 2008 h=1 QLIKE -5.74 versus -5.91 (the clipped model under-forecasts in crises) - the concern about combining winsorisation with fat-tailed innovations is supported: keep raw returns; (4) heavy tails matter for calibration: predictive log score gjr_t 2.718 vs garch 2.660 vs hist 2.618; 1-day 1% VaR violation rate gjr_t 1.12% (garch 1.45%, hist 1.61%), 5% rate 4.80% (nominal 5%); 21-day Monte Carlo VaR coverage is conservative (1%: gjr_t 0.86%, 5%: 3.82%); (5) HAR in levels is worst on MSE in dev (3.3e-5 vs 2.3e-5) and ties hist on QLIKE - the daily-squared-return HAR is not competitive here; (6) fitted gjr_t medians: alpha 0.022, gamma 0.084 (leverage effect), beta 0.897, persistence 0.969, nu 6.8.
User tickers (sealed, ends 2019-12-31): ORCL, INTU, AMD, STX and LITE (146 dates only) were forecast; BE and CELH show "insufficient history" only because of the seal (they have > 1,000 days by 2025); ALAB and the current SNDK genuinely lack history; SKHY is not a CRSP security. Implied 2x daily-reset drag (sigma^2 per year) over the last 3 years to 2019: AMD 41% (realised 34%), ORCL 4%, INTU 5%, STX 14-17%, LITE 25% (realised 17%): daily-reset drag on a 60%+ volatility stock is tens of percent per year.
Next in the pre-registration: the one-time holdout opening (`--open-holdout`) decides whether the licence stands; it also extends the ticker block to the end of the data. Nothing is logged to the trial registry (still 79).

## Pre-registration v14 (Study G, Stage B at the INDEX level): does the model forecast carry information beyond VIX, and is there a variance-risk-premium signal? (written before any code, extraction or run)
Purpose: question 2 of the three options questions, at the index level only, with data already in the subscription: `cboe.cboe` (daily VIX, verified by the access map). The decision it can change: whether there is any case for BUYING stock-level option data (ORATS / Alpha Vantage / Cboe DataShop) to test single-stock implied volatility against the forecasts. **It licenses no trade of any kind**: there is no tradable instrument, cost or margin data in our data; nothing here
licenses VIX futures, VIX ETNs or options (SVXY lost about 90% in February 2018), and the user's leveraged inverse single-stock products are a different risk entirely. The unconditional variance risk premium (VIX^2 above realised variance on average) is compensation for crash risk, not an edge; the report must say so. An index-level "VIX already encompasses the forecast" is only a weak prior for single stocks, whose implied volatility is usually less efficient.
**Data:** `cboe.cboe` (date, vix [annualised percentage points, close], plus vxo, vxn, vxd extracted for checks; new dataset `cboe_vix` in `wrds_extract.py`), the FF market return mktrf + rf on the Study F calendar (the SAME series, calendar, engine and forecasts as the v13 market series: `volfc.market_series`, the CRSP-union calendar, `run_asset(..., mc=False)`, `FIRST_FC_YEAR = 1995`). Forecast dates therefore start 1995-01 (VIX for 1990-94 would need a different calendar and is not used; reported as information only if available). S&P 500 daily returns (`crsp.dsp500`) are an INFORMATION-ONLY robustness input: tried by an optional extraction `sp500_daily`, skipped cleanly if the table is missing; the primary index is the FF market, fixed now.
**Units (window variance, one convention for everything):** IV2_win(t) = (VIX_t / 100)^2 x 30/365 (VIX's own convention, close of day t); RV_win(t) = sum of r^2 over trading days t+1..t+21 (= 21 x the v13 proxy y21; complete windows only); F_win(t, m) = 21 x the model's h = 21 average-variance forecast. Days with a null VIX or a date mismatch are dropped and counted. Sanity lines printed first: median VIX in the teens or twenties; mean IV2_win above mean RV_win; correlation of log IV2_win with log F_win(gjr_t).
**Sampling (fixed now, no free choice later):** the PRIMARY sample is the refit dates of the v13 engine, i.e. forecast dates whose calendar index is a multiple of 21 (non-overlapping 21-day windows; about 240 development and 60 validation observations); regressions use Newey-West with 3 lags. The daily overlapping version (all forecast dates, Newey-West 30 lags) is information only.
**Windows (identical to v13):** development forecast dates 1995-01..2014-12, validation 2015-01..2019-12, holdout 2020-01..2026 (sealed). With only about 60 validation observations the validation sign checks have little power; say so in the report.
**Models:** M3 `gjr_t` is primary; M2 `garch` is reported beside it (the v13 market table already shows them equal at h = 21, t -0.6); `ewma`, `hist`, `har`, `gjr_t_wins` are computed because the reproduction check below needs all six on one common valid-row mask. If M3 and M2 disagree on a test, report both; the rule is not relaxed.
**Tests (separate; T1 and T3 are not both counted as evidence of the same thing):**
- **T1, information content (not a trial):** on the primary sample, log RV_win = a + b log IV2_win + c log F_win(M3) + e. Rule "THE FORECAST ADDS INFORMATION BEYOND VIX": development c >= 0.10 AND t(c) >= 2.0 AND validation c > 0 (the validation regression re-estimated on its own window). Reported beside it: b and its t (does VIX add beyond the forecast), the single-regressor fits, QLIKE of the constant-scaled implied variance k x IV2_win (k = mean RV_win / mean IV2_win estimated on development ONLY and frozen) against M3 and M2, Diebold-Mariano on the primary sample.
- **T3, economic payoff (ONE trial, logged: cumulative 79 -> 80):** Y_t = (RV_win - IV2_win) / IV2_win, the return per unit strike of a long 30-day variance swap struck at VIX^2 (before any cost); D_t = log F_win(M3) - log IV2_win. Regress Y_t on D_t, primary sample, Newey-West 3. Rule "A VARIANCE-RISK-PREMIUM SIGNAL EXISTS": development slope t >= 2.0 AND validation slope > 0. Information only beside it: mean Y by development-tercile of D (breakpoints frozen on development), the level version D = F_win - IV2_win, and a rerun that excludes the windows touching 2008-09..2008-12 (development) so that a single crash cannot decide it.
- **T2, unconditional premium (information only):** mean IV2_win - RV_win (annualised volatility points too), share of positive months, Newey-West t, by window.
- **T4, market return predictability (information only):** next-21-day FF market excess return on D_t and on IV2_win - F_win, t with Newey-West 3; low power, no weight.
**Decision rule:** the case for purchasing stock-level option data is made ONLY if BOTH T1 and T3 pass in development and validation. Then the 2020+ window is opened once with `--open-holdout`, and the case STANDS only if c > 0 (T1) and the T3 slope > 0 there; otherwise it is recorded as "passed dev/validation, failed in the holdout". If T1 or T3 fails, the record says: the index-level implied variance encompasses the model forecast; no case for buying stock-level data on this evidence (the stock-level question stays open but unfunded).
**Holdout gate (new):** `run_vrp.py --open-holdout` REFUSES to run unless the v13 holdout has already been opened, because the 2020+ window scores the same market gjr_t forecasts that Study F's sealed holdout covers. `run_volfc.py --open-holdout` therefore writes a one-shot marker file (`data/holdout_v13.done`, written only after its holdout statistics have been computed, refusing a second opening) that `run_vrp.py` checks. Without the flag no forecast dated 2020 or later is computed, as in v13, and the report has no 2020+ number.
**Reproduction check (aborts the run if it fails):** the v14 pipeline must reproduce the v13 market-table dev / validation mean QLIKE (tolerance 5e-4) for gjr_t and garch: h = 1 gjr_t -8.2590 / -8.8061, garch -8.2144 / -8.7425; h = 21 gjr_t -8.1053 / -8.5679, garch -8.0991 / -8.5666. This ties the new pipeline to the already reviewed numbers and catches any calendar or index misalignment.
**Checks printed first:** VIX rows, date range, null days dropped, calendar overlap, units sanity lines above, the reproduction check, number of primary-sample observations per window, share of RV windows incomplete.
**Honest prior:** VIX is a strong variance forecast and GARCH adds little to it; the most likely outcome is that T1 fails (c near zero) and T3 fails, i.e. "no case for buying stock-level data on this evidence". The unconditional premium will be large and positive and must not be read as an opportunity.

## RESULT v13 / Study F holdout (user run, 2026-10-12): the licence STANDS - gjr_t is the position-sizing volatility model (opened once; do not run `--open-holdout` again)
Holdout forecast dates 2020-01..2025-12, mean QLIKE difference gjr_t minus benchmark (DM t): vs garch h=1 -0.0131 (-1.79), h=21 -0.0393 (-1.40); vs har h=1 -0.0608 (-2.98), h=21 -0.1545 (-1.44). All four < 0, as the rule requires => **LICENSED (dev/validation and holdout)**. Dev and validation numbers are identical to the sealed run.
Reading, with the caveat that matters: the holdout advantage is concentrated in 2020. By year at h=21, gjr_t beats garch clearly in 2020 (-5.434 vs -5.267) and mildly in 2021, 2023 and 2025, ties in 2024 and loses slightly in 2022 (-6.655 vs -6.659); the evidence is "better in crashes (2008, 2020), about equal in calm years", the same pattern as in dev/validation. Winsorisation hurts again (M4 minus M3 +0.036 at h=1, +0.044 at h=21; 2020 h=21 -5.176 vs -5.434).
Calibration (holdout): predictive log score gjr_t 2.652 vs garch 2.565; 1-day 1% VaR violation rate gjr_t 1.31% (garch 1.74%, hist 1.98%), 5% rate 5.33%; 21-day Monte Carlo VaR coverage 1%: 0.81%, 5%: 4.31%. Fitted gjr_t medians in the holdout: alpha 0.031, gamma 0.106, beta 0.872, nu 4.9 (fatter tails than before 2020).
User tickers, full history to 2025-12 (information only; ratio of means = forecast / realised 21-day variance over the last 3 years, <1 means the forecast was too low): ORCL 0.59 (gjr_t), INTU 1.16, LITE 0.70, BE 0.87, STX 0.91, AMD 0.94, CELH 1.29; ALAB and the current SNDK have too little history; SKHY is not a CRSP security. Implied annualised drag of a +2x daily-reset product (sigma^2 per year; gjr_t / realised): AMD 25.8% / 27.5%, BE 63.6% / 73.0%, CELH 47.4% / 36.8%, LITE 25.9% / 37.3%, STX 15.3% / 16.8%, ORCL 11.4% / 19.3%, INTU 10.0% / 8.6%.
IMPORTANT for inverse products such as the user's 2x short ETFs: the daily-reset drag is (L^2 - L) sigma^2 / 2 = 3 sigma^2 per year for L = -2 (three times the +2x figure printed above), and the underlying must FALL by 1.5 sigma^2 T (log) just for the product to break even: for AMD (sigma^2 ~ 0.27/yr) about 41% a year, about 3.4% a month.
Implementation note: this run used the pre-marker `run_volfc.py`, so `data/holdout_v13.done` was not written; v14 (`run_vrp.py --open-holdout`) requires it. Create it by hand once, recording the fact: `echo "v13 holdout opened 2026-10-12 (user run, licence stands)" > data/holdout_v13.done`. The CHECKS block of an `--open-holdout` run counts 10 more refit blocks (28,877 vs 28,867) than the sealed run because December-2019 refit blocks whose forecast days fall in January 2020 become reachable; no dev or validation number changes.

## RESULT v14 / Study G (user run): index-level forecast versus VIX - T1 FAIL, T3 FAIL; no case for buying stock-level option data on this evidence (holdout not opened; 80 trials)
Checks: reproduction of the v13 market table within 5e-5 for all eight numbers (PASS); VIX from 1990-01-02, median 17.2; mean IV2_win 0.00366 vs mean RV_win 0.00282 (implied 1.30x realised); corr(log IV2_win, log F_win gjr_t) 0.838; primary sample dev 240 / validation 59 non-overlapping 21-day windows (daily-overlap 5,034 / 1,258). Data-quality note: the extraction contained 507 DUPLICATE VIX dates (kept: last) and 1,013 null/non-positive VIX days (1986-89, before the VIX starts); whether the duplicates carry identical values has not been checked (see below).
T1 (log RV_win on log IV2_win and log F_win, Newey-West 3): M3 gjr_t dev b 0.678 (t 9.6), c 0.407 (t 5.6), R2 0.664 vs 0.628 for VIX alone; validation b 1.480 (t 4.7), c -0.557 (t -1.9), R2 0.343 => rule needs dev c >= 0.10 and t >= 2 (met) AND validation c > 0 (NOT met) => FAIL. M2 garch: dev c 0.252 (t 2.9), validation c -0.637 (t -1.5) => FAIL. S&P 500 returns instead of the FF market (robustness): dev c 0.306 (t 3.4), validation c -0.491 (t -1.6) => FAIL. Daily-overlap versions agree (dev c 0.347, validation c -0.129).
T3 (payoff of a long variance swap at VIX^2 on D = log F - log IV2): dev slope 0.266 (t 3.12), validation slope -0.482 (t -1.20) => FAIL. The development slope is not a 2008 artefact (ex-2008 rerun 0.190, t 3.66), but the sign reverses in validation. Even the top tercile of D loses on average (dev mean payoff -0.45 / -0.23 / -0.17 by tercile, t -12.0 / -4.3 / -1.6): in no state did buying variance pay; D only changes how much the sellers earn.
Information only: T2 premium dev +0.00093 (t 3.4), VIX 20.7 vs realised 16.8 vol points (+3.8, t 7.7), implied above realised in 83% of windows (validation +2.8 vol points, 81%); the only years with realised above implied are 2000, 2008 and 2018 - the premium is crash insurance, as pre-registered. T4: next-month market return on IV2 - F: dev slope 7.06 (t 4.1), validation -4.12 (t -0.3). Constant-scaled VIX (k = 0.770 from development) is a BETTER 21-day variance forecast than the model: QLIKE model minus scaled IV dev gjr_t +0.031 (t 1.35), garch +0.052 (t 2.39); validation gjr_t +0.068 (t 2.02).
Reading: in 1995-2014 the model forecast carried information beyond VIX (c 0.41), in 2015-2019 VIX dominated and the model's weight turned negative; with 59 validation windows the power is low, but the pre-registered rule is applied as written. Together with the scaled-VIX QLIKE result the conclusion is: at the index level, the implied variance encompasses the model; the variance-risk-premium signal is not stable; nothing here is tradable. DECISION (as registered): no case for buying stock-level option data; the stock-level question stays open but unfunded. The 2020+ window of v14 stays sealed (its opening needs both tests to pass).
For position sizing: the licensed model (gjr_t, v13) remains the choice for individual stocks, where no implied volatility is available; for a MARKET volatility estimate, VIX itself (free, from `cboe.cboe`) forecasts the next 21 days at least as well as any model tested here.
Open data item: check the 507 duplicate VIX dates in `data/raw/cboe_vix.parquet` (do duplicate rows carry different values?). If they differ materially, the primary sample could change; the preregistered handling (keep the last row) would then need a documented amendment, not a silent change.

- **v14 data item closed (user check, 2026-10-12):** all 507 duplicate VIX dates carry IDENTICAL values (max-minus-min of vix per duplicated date is 0 for every one; the first duplicates are 1990-01-02 onward, count 2 each), so "keep the last row" changed nothing and the v14 result stands unchanged. No amendment needed.

## Pre-registration v15 (Study H): the mechanical skeleton of the user's band-switch and momentum rules on simulated 2x daily-reset products (written before any code or run)
Origin: the user (account USD 45,000, cash account: long stocks and 2x ETFs only, shorts only through inverse ETFs) trades SNDK, SKHY, ALAB and LITE through 2x long and 2x inverse ETFs (SNXX / SNDQ, SKHX / SKDD, LABX, LITX, DAMD, AXTQ) with price-level rules, a -20% stop per trade and a -15%-of-deposit account stop followed by a one-trading-day pause. Confirmed rule shapes: (1) a BAND SWITCH: buy the 2x long below the buy level, sell it and buy the 2x inverse at the sell level, sell the inverse and buy the long back at the buy level (shorts and longs are never held together); (2) ALAB-style MOMENTUM: buy the 2x long after two up closes, sell at +10-20% on the 2x product; (3) an INTU-style asymmetric band with a discretionary exit. The discretion (thesis, conviction sizing, options and analyst views) cannot be tested; this study tests only the MECHANICAL SKELETON on history, so it can say whether the structure of these rules (levels, 2x leverage with daily reset, stops, flips) has any edge in volatile large-cap tech, not whether the user's judgement does.
**Decision it can change:** whether the user should keep trading the band switch / momentum structure mechanically (with the stops as stated), or treat it as entertainment-sized risk. No trade is licensed by a pass; a pass licenses at most a forward paper test.
**Data:** `data/raw/crsp_daily_ret` (dlyret, 1989-2025), `data/raw/ff_daily.parquet` (rf via `ff_decimals`), `crsp_monthly` / `crsp_delistings` (universe), and for the checks and the information-only block `data/raw/secd_tickers.parquet` (Compustat daily prices for the user's names and ETFs through 2026-10-14). Closes only: no OHLC.
**Universe (fixed now):** each calendar year Y: among the 300 largest companies (permco, representative permno, common equity) by prior-December company market equity, keep those whose trailing-252-day annualised volatility at the last trading day of December is >= 45% (>= 200 non-missing returns); one SLEEVE per (stock, year), starting FLAT on the first trading day of Y, positions open at the last close of Y are closed there (counted as exits, with costs). Reported in CHECKS: universe size per year and the volatility distribution next to the user's names (EWMA vol from `secd`). The v13 top-100 universe and trailing-volatility terciles are INFORMATION ONLY.
**Simulated products:** for leverage L in {+2, -2}, daily instrument return = L r_t + (1 - L) rf_t - f/252, with r the underlying's total return, rf the FF daily risk-free rate, f = 1.5% a year fee and swap-spread drag; value floored at 0. (The (1 - L) rf term is the financing a daily-reset product earns or pays; omitting it biases longs up and inverses down by several points a year when rates are high.) Trading cost: 10 bp of the traded value per side (a flip is two sides). Sensitivities (information only): fee 0% and 3%, trading cost 0 and 25 bp.
**Replication check printed first and ABORTING on failure:** for each pair with at least 60 common days in `secd` (SNXX / SNDK +2, LABX / ALAB +2, LITX / LITE +2, SKHX / SKHY +2, SNDQ / SNDK -2, DAMD / AMD -2, SKDD / SKHY -2, AXTQ / AXTI -2), regress the actual ETF's daily return on the underlying's: the slope must lie within 15% of L (else the mapping or data is wrong: abort); print the slope, tracking error and the cumulative log gap actual minus simulated per year of common data; abort if the median absolute gap per year over the pairs exceeds 0.15.
**Rules (parameters taken from the user's words, ONE set, no tuning; grids are information only):**
- **R1 band switch (every sleeve):** anchor M_t = mean of the last N = 21 closing values of the underlying's total-return price index (the user: a cycle is about a month); bands B_t = M_t (1 - d), S_t = M_t (1 + d) with d = 0.059 (the user's 1,600 / 1,800 around 1,700); at close t: if P_t < B_t the target state is LONG2, if P_t >= S_t it is SHORT2 (the -2x inverse), otherwise the previous state is kept (hysteresis); a state change flips the whole position.
- **R2 momentum:** signal at close t if r_t > 0 and r_(t-1) > 0 (two up closes of the underlying): enter LONG2; exit when the 2x product has gained +15% since entry or lost 20% (stop); no time exit; signals are ignored while in the trade.
- **Execution:** signal at close t, trade at the close of t+1 (one-day lag, primary: CRSP gives closes only and a same-close mean-reversion fill picks up bid-ask bounce); the position earns the instrument return from day t+2. Same-close execution is information only.
- **Stop (both rules):** -20% on the instrument from the entry close, checked on closes and filled at the breaching close (overshoot included); after a stop the sleeve stays flat: no entry may be executed at the next close, the earliest new entry executes the close after (one trading day pause), then the rules re-evaluate and re-entry is allowed.
- **Information-only variants:** R3 INTU-style asymmetric band on the stock itself (L = 1, buy below 0.967 M, take profit at 1.133 M, stop -20%); lag 0; d in {0.03, 0.10}; N = 63.
**Evaluation:** a sleeve's daily net return is the instrument return while in position (after trading costs on the days of a trade) and rf while flat. Primary statistic per rule: for each date, the cross-sectional mean over the active sleeves of (sleeve net return - rf); mean and Newey-West t (10 lags) on the dev window 1995-01..2014-12, then validation 2015-01..2019-12; 2020-01..2025-12 is sealed and opened once (marker `data/holdout_v15.done`, written after the statistics are computed) only if a rule passes dev and validation.
**Placebo (the control for tech's drift and for leaning net long):** circularly shift each sleeve-year's final position series (after stops and pauses) against its returns by a random k of at least 63 trading days, 200 draws with a fixed seed; this keeps the fractions of time long, short and flat, the persistence and the number of transitions, and breaks the timing. A rule needs its dev statistic above the 95th percentile of the placebo distribution of the same statistic.
**Licence (per rule; nothing is promoted for use):** dev mean net excess return over rf > 0 with t >= 2 AND dev above the placebo 95th percentile AND validation mean > 0; then the holdout is opened once and the licence stands only if its mean is > 0 and above the placebo median. **Two trials (R1, R2): the cumulative count goes 80 -> 82.**
**Information only, no decision weight:** per-trade statistics (win rate, mean win, mean loss, expectancy, stop frequency, holding days, transitions per year); the distribution of sleeve-year maximum drawdowns and the share beyond -15% (a one-day pause after a stop is a timeout, not a loss limit; the account stop cannot be tested by single-name sleeves because the user sizes by conviction); benchmarks (cash, buy-and-hold underlying, buy-and-hold 2x long, buy-and-hold 2x inverse); by volatility tercile and by year; the top-100 universe; the sensitivities above; and the user's own names on `secd` data 2024-2026 (SNDK, ALAB, LITE, SKHY, AMD, INTU, STX, ORCL, CELH, BE, AXTI) labelled IN-SAMPLE (the rules were designed on exactly those names and period).
**Honest prior:** one-month reversal was strongly negative net of costs in Study A; adding 2x daily-reset drag (about sigma^2 per year for a long, 3 sigma^2 for an inverse), fees, a stop that sits inside normal daily noise for these volatilities, and a one-day lag makes a negative net result the expected one; the placebo is there so that a positive number from tech's drift or from net-long exposure cannot pass. A bare positive mean is not a pass.

## RESULT v15 / Study H (user run): band switch R1 and momentum R2 on simulated 2x daily-reset products - NEITHER LICENSED (holdout not opened; 82 trials)
Checks: replication of the user's real ETFs PASS (slopes 1.984-1.995 for +2x, -1.986 to -2.000 for -2x; median absolute log gap 0.032 per year over 9 pair-year cells; AXTQ skipped, 40 days), so the simulated product tracks the real ones (the real ETFs trail the simulation by 0.01-0.07 in log terms per year of common data, i.e. extra real-world costs of a few %/yr). rf coverage 100%. Universe: 1,086 sleeve-years, 266,382 sleeve-days, but very uneven (2001: 185 sleeves, 2009: 211, 2006: 2, 2018: 0; validation years 6/7/18/0/12), so the validation window (1,007 dates) is thin.
R1 (N=21, d=0.059): dev -2.07 bp/day (-5.2%/yr, t -0.57), validation -2.74 bp/day (-6.9%/yr, t -0.39); placebo dev 95th pct +0.91, median -2.56 bp/day, rule above 57% of draws => no timing information; 22.6 transitions per sleeve-year, 52% of time short, 48.4% of trades end at the stop.
R2 (two up closes, +15%/-20%): dev -0.81 bp/day (-2.0%/yr, t -0.21), below every placebo draw (95th pct +7.70, median +5.56 bp/day: shifting the same long exposure at random earns more than the rule's timing); validation +11.26 bp/day (+28.4%/yr, t 1.54) on 1,007 dates and 0-19 sleeves a year, not distinguishable from buy-and-hold 2x long (+34.5%/yr, t 1.42). Failed: dev mean > 0, t >= 2, placebo 95th pct.
Information only: per-trade expectancy R1 -0.41% (long legs +1.13%, inverse legs -1.89%), R2 +0.31%, stop frequency 48% / 41%; sleeve-year maximum drawdown median -68.5% (R1) and -56.7% (R2), 98% of sleeve-years beyond -15% (single-name sleeves, so not the account stop); buy-and-hold 2x long beat both rules in both windows (dev +14.5%/yr, validation +34.5%/yr) and the 2x inverse lost -17.5% / -36.0%/yr; sensitivities (fee 0-3%, cost 0-25 bp, lag 0, d 0.03/0.10, N 63) all leave R1 negative and R2 near zero in dev (R2 cost 0: +0.4%/yr); top-100 universe R1 -2.8%/yr dev, R2 +7.5%/yr dev (t 1.25) and +17.2%/yr validation (t 1.80), no pass; R3 (INTU-style stock band, not a trial) +6.0%/yr dev (t 1.01), +11.7%/yr validation (t 1.03), expectancy +2.25% per trade, stop frequency 41%.
Where the user trades: the top volatility tercile (>= 64% trailing vol; the user's names are SNDK 74%, ALAB 85%, LITE 86%, SKHY 69%) is the worst one in development for both rules (R1 -22.8%/yr, R2 -26.7%/yr); R3 also trails buy-and-hold of the stock itself (6.0 vs 8.4%/yr dev, 11.7 vs 17.9%/yr validation).
Reading: the mechanical skeleton of the band switch has no edge on volatile large caps (the inverse leg loses, the long leg is only slightly positive); the momentum rule neither beats random placement of the same exposure nor buying and holding the 2x long. Whatever profit the user's rules produce has to come from the discretionary parts (which levels, which names, how much size), which this test cannot see. The stop inside daily noise explains the high stop frequency; the account stop is a timeout, not a loss limit. User-names block withheld (`--open-holdout` only prints it; no rule passed, so it was not run).
Trial count: 80 -> 82 (R1, R2 logged).

