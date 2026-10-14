# Mathematical model of the current strategy (as of 2026-10-12, 79 logged trials)

The "strategy" is a passive core plus an optional long-only gross-profitability (gpa) tilt. No active alpha strategy is deployed. Everything below mirrors the code in `src/qap/` (`tilt.py`, `costs.py`, `portfolio.py`, `candidates.py`).
Plain-text math (renders anywhere). The estimates in section 5 are from the logged runs; the shrinkage in section 5 is a rough illustration, not a registered result.

## 1. Setup
Months are indexed by t. Formation at month-end t, return earned in t+1 (no look-ahead).
- Universe U_t: US common equity, ex-microcaps, ME_i,t > 0.
- Benchmark weights: b_i,t = ME_i,t / sum_j in U ME_j,t (value-weighted universe).
- Signal: gpa_i,t = GP_i / AT_i from the latest annual report available 4 months after fiscal year-end (staleness cut 548 days).
- Percentile rank p_i,t = rank of gpa_i,t within U_t, scaled to (0, 1].

## 2. The tilt portfolio (long-only, no leverage)
```
Held set H_t  = { i : p_i,t > 0.80 }  U  { i in H_t-1 : p_i,t > 0.70 }          (entry 0.80, exit 0.70)
Raw weights     w~_i,t = ME_i,t / sum_j in H_t ME_j,t
Capped weights  w_i,t  = min(w~_i,t, kappa) + pro-rata redistribution, kappa = 3%   (water-filling, sum w = 1)
```
The buffer makes holdings sticky (lower turnover); the cap bounds single-name risk.

## 3. Returns, active return and costs
```
Gross portfolio return   R^P_t+1 = sum_i w_i,t * r_i,t+1          (delisting returns included; missing returns count as 0)
Benchmark return         R^B_t+1 = sum_i b_i,t * r_i,t+1
Trade size               D_i,t = | w_i,t - w_i,t-1 * (1 + r_i,t) / (1 + R^P_t) |
Cost of trade i          C_i,t = D_i,t * [ 0.5 * s_i,t + Y * sigma_i,t * sqrt( D_i,t * A_t / ADV_i,t ) ]
Net active return        a_t+1 = R^P_t+1 - R^B_t+1 - m * sum_i C_i,t
```
- s = bid-ask spread (quoted, then Corwin-Schultz, then size-decile median); sigma = daily volatility; ADV = average daily dollar volume; Y = 1.
- A_t = book size, a constant share of the equity market (USD 10M in 2019 market-size dollars). m = 1.5 is the cost stress multiplier. The benchmark is costless (generous to the tilt).
- For a USD 45k account (the user's stated starting capital; earlier work assumed 35k), A_t is about 0.4% of the research book, so impact is negligible and only the half-spread matters.

## 4. How a signal is judged (pre-registered)
Work on the series a_t of net long-minus-universe returns of the top quintile.
```
Promote signal c  <=>  t_dev(alpha_c) >= 2.0   AND   alpha_val > 0
  a_c,t = alpha_c + beta * a_gpa,t + e_t        (OLS on dev 1990-2014; beta frozen; alpha_val = mean(a_c - beta * a_gpa) on 2015-19)
```
Holdout (2020+) is opened once, only for promoted signals. Newly tested datasets need >= 60 development months.

## 5. Where we are
| Quantity | Value |
|---|---|
| gpa net L-U, dev 1990-2014 | alpha = +0.184 %/mo, t 1.88 (se ~ 0.098) |
| gpa net L-U, validation 2015-19 | +0.098 %/mo, t 0.60 (se ~ 0.163) |
| Long-only tilt, holdout 2020+ | +0.056 %/mo, t 0.36 (se ~ 0.156) |
| Tracking error of tilt vs benchmark | ~5 %/yr (sigma_m ~ 1.44 %/mo) |
| Challengers tested | 79 trials, none promoted |

Rough Bayesian shrinkage (prior alpha ~ N(0, tau^2); the three estimates treated as independent normals, which is crude because the tilt and quintile series overlap):
```
posterior mean = (data mean) * P_data / (P_data + 1/tau^2),   P_data = sum 1/se^2 = 104 + 38 + 41 = 183,  data mean ~ 0.138 %/mo
tau = 0.10 %/mo -> alpha ~ +0.09 +- 0.06 %/mo   (~ +1.1 %/yr)
tau = 0.05 %/mo -> alpha ~ +0.04 +- 0.04 %/mo   (~ +0.5 %/yr)
```
Treat +0.04 to +0.09 %/mo as an upper bound: gpa was picked as the incumbent after seeing data (selection not corrected), and with 79 trials the null maximum t is near sqrt(2 ln 79) ~ 3.0, so a t of 1.9 is not distinguishable from a lucky draw.

## 6. What the data can tell you
```
Months needed for t = 2:   n = (2 * sigma_m / alpha)^2
alpha = 0.20 %/mo ->   208 months (17 years)
alpha = 0.10 %/mo ->   833 months (69 years)
alpha = 0.09 %/mo -> 1,028 months (86 years)
```
The paper ledgers (gpa, ML) are implementation checks, not evidence.

## 7. In dollars on USD 45,000 (starting capital stated by the user on 2026-10-15; earlier versions of this note used 35,000)
```
Expected gain     alpha * 12 * W ~ 0.04-0.09 %/mo * 12 * 45,000 ~ USD 215-485 per year
1-sigma noise     TE * W = 5% * 45,000 ~ USD 2,250 per year
Information ratio ~ 1.07% / 5% ~ 0.2  ->  P(beat benchmark in a given year) ~ Phi(0.2) ~ 58%
```

## 8. After tax (inputs: 32%+ bracket, hold-indefinitely, regular cash additions)
```
After-tax edge ~ alpha - tau_LT * g_LT - tau_ST * g_ST + deferral value of unrealised gains,   tau_ST = 40.8%, tau_LT = 23.8%
```
The buffer keeps turnover low, but rebalancing realises gains. v7 found loss harvesting adds only ~ +0.17 %/yr on a DIY index. The tax leak of the tilt itself was not measured and could be a large fraction of a +0.5 to +1.1 %/yr edge.

## 9. Bottom line
```
Terminal wealth ~ passive core W0 * (1 + r_mkt)^T  +  tilt: ~5% annual noise around a +0.5 to +1.1 %/yr upper-bound edge,
statistically undetectable for decades
```
Hold the passive core; the tilt is a small, uncertain positive whose sign cannot be confirmed within a working lifetime.
