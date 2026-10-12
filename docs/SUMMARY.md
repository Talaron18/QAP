# QAP research wrap-up (as of 2026-10-11)

Plain-language summary of what was built, what was tested, what held up and what did not. Details and every pre-registered
spec live in [ROADMAP.md](../ROADMAP.md); the original brief is [research_report.md](research_report.md).

## Bottom line

After 76 logged trials on 1963-2025 US equity data (CRSP + Compustat + JKP characteristics, WRDS), **no strategy has been validated
out of sample net of costs.** The one signal that survives costs, gross profitability (gpa), has a small, decaying edge
(+0.18 %/mo, t 1.9 on 1990-2014; +0.10, t 0.6 on 2015-19; the long-only tilt version was +0.06 %/mo, t 0.4 on the 2020+ holdout).
What *is* valuable and reusable: a leakage-safe research pipeline, a cost model, a trial registry, pre-registered studies with sealed
windows, and a forward paper-trading ledger. For a USD 35,000 taxable account the defensible position is a passive core, optionally
with a small quality/profitability tilt, with the ledger running as an implementation check - not a deployed alpha strategy.

## What was built (all in `src/qap/`, scripts in `scripts/`, 235 tests)

- **Data/PIT plumbing:** CCM link filter, CIZ delisting-return handling, point-in-time annual fundamentals (4-month lag), month-grid and
  unit-calibration assertions, label builder with t+1 lag, purged/embargoed splits, rank IC, deflated Sharpe.
- **Replication gates (passed):** market vs FF Mkt-RF corr 0.99999, SMB 0.995, HML 0.983, UMD 0.999 - the data and alignment are sound.
- **Cost model:** half spread (quoted, then Corwin-Schultz, then size-decile median), square-root impact, market-scaled AUM
  ($10M in 2019 market-size dollars), 1.5x stress; benchmark costless; borrow modelled only as an assumption (see below).
- **Studies and diagnostics:** candidate test, long-only tilt, JKP themes (Study A), innovation family (Study B), conversion diagnostic,
  single-stock ETF check, borrow-cost break-even and basket sheet, one-shot sealed-window long-short test.
- **Governance:** trial registry (`data/trials.jsonl`, unique spec dedupe), splits dev 1990-2014 / validation 2015-19 / holdout 2020+,
  written pre-registrations committed before each run, rehearsal gates, one-shot markers.

## What was tested and what happened

| Step | Result |
|---|---|
| Replicated factors (BM, GP/A, asset growth) | Decay as the literature predicts: BM about 0 after 1990, asset growth +0.29 (1990-2014) then -0.28 %/mo (2015-19); GP/A positive in every split |
| Cost model | Costs are small for slow books (gpa ~0.02-0.04 %/mo), large for fast ones (momentum 0.2-0.4) |
| 6 pre-registered candidates vs gpa (v1) | Nothing promoted |
| Long-only gpa tilt (v2: buffer + 3% cap) | dev +0.216 %/mo (t 2.56), validation +0.096 (t 0.78), **holdout +0.056 (t 0.36)** - not validated |
| Study A: 13 JKP themes + 2 composites (v3) | Nothing promoted. Composites had dev rank IC t of 6-7, validation ~1, net long-minus-universe ~0 |
| Study B: R&D/patent/KPSS/RQ signals (v3) | Nothing promoted (all |t| < 1.4 vs gpa) |
| Diagnostic (b), why IC does not become return | Costs (turnover ~39%/mo for the all-13 composite), information sits in the **short** leg, decay after 2014; not a small-cap effect |
| Single-stock leveraged/inverse ETFs for the short leg (v5) | **Not viable**: 2025 coverage of the bottom quintile 0.1-5.6% (only ~16 inverse products live, all mega-caps) |
| Direct margin short, break-even borrow cost | `jkp_slow10` break-even 4.6-6.5 %/yr; assumed fees for the real basket ~0.4-0.7 %/yr (82% of weight in >= $10B names) - borrow is not the binding constraint |
| Alternative-data probes (v9): arc complexity, 10-K risk text, blockholders, classified boards, culture, litigation | Nothing promoted: all dev t < 1.3 vs gpa; blocks significantly wrong-signed in validation (t -2.7); cbi_neg consistent but weak (t 1.1-1.5); 6 trials |
| Earnings-announcement expectation change (v11): EAR, SUE, RSUE, EAR+SUE | Nothing promoted; gross IC t 5-6 pre-2015 but ~0 after; effect lives in months 0-1; net ~0; 4 trials |
| Forward-only ML check (v8; frozen, running) | Walk-forward (information only): boosted trees IC 0.032 but net L-U +0.01 %/mo vs simple composite +0.24 vs gpa +0.14; forward decision not before ~late 2029, prior 'no evidence' |
| Tax-aware direct indexing / loss harvesting, 150-name DIY index, 53 ten-year windows (v7) | median benefit **+0.17 %/yr** (hold-indefinitely), ~0.03 %/yr if liquidated; TE 1.1 %/yr; **not worth the complexity**; DIY index vs total market adds ~4 %/yr tracking error |
| **One-shot sealed window, long-short `jkp_slow10` (v6)** | net 0.50 %/mo, **t 0.89 -> INCONCLUSIVE**. All of it in 2021-22 (2022: +35%); 2023-25 flat/negative; max drawdown -30%; short leg only; FF4 alpha 0.39 (t 1.15) with large size and momentum loadings |

Reading across all of it: the characteristics carry real cross-sectional information historically, but it is concentrated in places a small
long-only investor cannot use (short leg, small and illiquid names, high turnover), and it decayed after about 2014.

## What has been spent (do not re-open)

- 2020+ for gpa-based models (signals run, tilt, candidates): spent.
- 2020+ for JKP composites (`jkp_slow10` long-short): spent by v6. There is no second look.
- 2015-2019 validation: seen many times; anything designed from it needs forward data.
- 76 unique trials logged (plus the two forward-only ML models); any new search should be judged against that count (deflated Sharpe) and, ideally, forward in time only.

## Live: the paper-trading ledger (keep running)

Forward test of the gpa tilt (spec: entry 0.80 / exit 0.70 / 3% cap, value-weighted, benchmark = value-weighted eligible universe), built on
Compustat `comp.secm` because CRSP and JKP stop at 2025-12. It reproduced the CRSP research on the overlap (weight overlap 0.884, active-return
correlation 0.974). Ten formations (2025-12 .. 2026-09) are frozen. All ten were formed on 2026-10-11, so by the settle script's rule (formed within 7 days of the target month's start) they settle as **retrospective**; the first formation that can count as prospective is 2026-10, and only if it is formed and its manifest pushed by about Nov 7. If WRDS posts the month-end `comp.secm` row later than that, the prospective count stays 0 and the weaker claim is only that weights are a deterministic function of month-end data under an unchanged spec commit.
9-12 months have essentially no power (a +0.2 %/mo edge at ~5% tracking error needs decades for t = 2): it is an implementation check.

Monthly routine (after the month closes and WRDS has the new `comp.secm` month):

```
python scripts/wrds_extract.py compustat_annual comp_secm
python scripts/paper_form.py                      # CRSP-overlap check first, then writes the new frozen formation
git add paper_ledger && git commit -m "paper ledger" && git push     # tracked evidence is the hash manifest only (no licensed data)
python scripts/paper_settle.py                    # verifies hashes, settles realised months, writes data/paper_report.txt
```

## The USD 35k decision (my read, not advice)

- The investable edge found is at most a small profitability/quality tilt, with an uncertain sign out of sample and tracking noise of ~USD 1,600/yr
  on this account size (5% tracking error) against an expected benefit of roughly USD 20-35 a month.
- Long-short `jkp_slow10` is not a basis to deploy: 30% drawdown, factor-explained returns, regime concentration, a data feed that ends 2025-12, and margin/short
  operational risk. Its break-even borrow cost (4.6-6.5 %/yr) is comfortable against assumed fees, but so are those of Quality (3.6 dev / 7.1 val), Profitability (4.2 / 5.5) and gpa (2.7 / 4.4); the binding constraints are statistical and structural, not borrow.
- Taxes: the research ignored them, so I tested the main mechanical lever (loss harvesting on a DIY index) with your inputs (32%+ bracket, outside gains, hold-indefinitely): about +0.17 %/yr in the best case,
  nearly all deferral, with extra tracking error. A broad low-cost fund with low turnover captures most of the tax benefit with none of that complexity. Monthly-turnover strategies would realise short-term gains
  at ~41% and are untenable in a taxable account at this edge size.
- The highest-value non-research levers: savings rate, fees, diversification (including international), asset location, and not trading.

## Known limits and caveats

- Borrow fees were never measured (Cboe, bondret, I/B/E/S, 13F, TAQ, OptionMetrics, RavenPack were not accessible); the assumed schedule in `qap.basket` is a guess.
- Pre-1990 cost estimates are the least reliable (spreads 92% Corwin-Schultz in the 1960s).
- JKP characteristics on WRDS end 2025-12; running any JKP-based signal forward means rebuilding about 120 characteristics from Compustat/CRSP.
- The ETF inventory is return-matched (names table has no fund names): ETFs with < 12 months of history and ADR underlyings are missed; the ETF carry estimate
  was contaminated and is not used - the coverage result alone decides the question.
- All backtests are value-weighted quintiles of ex-microcap names; equal-weighted, microcap and international results are not covered.
- Several reported numbers are pooled across selected signals; treat any t near 2 as weak given 76 trials.

## How to resume

1. `git pull`; the venv needs numpy, pandas, scipy, pyarrow, pytest, duckdb and `pip install -e .`.
2. Raw data is regenerated with `scripts/wrds_extract.py` (see its docstring for dataset names); `data/` is git-ignored (licensed).
3. Read the latest pre-registration in ROADMAP.md before running anything; write the next one *before* the code, and keep the same governance
   (trial log, sealed windows, rehearsal gates, record the outcome whatever it is).
4. Candidate next steps, in my order (tax-lot simulation is done, see v7): a bounded forward-only ML check; cheap probes of the accessible alternative datasets (blockholders, XBRL complexity, risk-factor text); volatility-managed exposure last.
