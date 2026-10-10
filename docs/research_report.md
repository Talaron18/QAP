# Designing a Compound Quantitative + Probabilistic US Equity Trading System: A Literature-Grounded Research Report (Qlib + WRDS)

The most robust design is a two-book system: a slow, low-turnover "long-horizon" book built on value-with-quality, profitability, investment and volatility-managed momentum, and a separately risk-budgeted "fast" book of short-horizon signals (short-term reversal, analyst revisions, earnings-event and option-market signals). Both feed a single cost-aware optimizer, and a distributional (volatility and quantile) layer governs sizing rather than direction. The evidence behind this design is strong but humbling. Anomaly returns average roughly half their published size after publication.\[1\] Many option-implied "alphas" shrink by about two-thirds once stock-borrow fees are accounted for. Machine-learning gains are real but modest in R² terms and fragile to costs. Your edge is therefore more likely to come from careful point-in-time engineering, signal combination, cost-aware portfolio construction and honest validation than from any single exotic predictor.

## TL;DR

- **Build two books with one optimizer.** The slow book (value+quality, gross profitability, investment, momentum with crash protection) has the best-documented, most scalable premia. The fast book (reversal, revisions, earnings/news events, option-derived signals) has higher gross IC but is cost- and borrow-constrained. Combine them through a Gârleanu–Pedersen-style "trade partially toward an aim portfolio" rule instead of netting raw signals.
- **Discount published effects heavily.** McLean–Pontiff (2016) find returns fall 26% out-of-sample and 58% post-publication. Hou–Xue–Zhang (2020, *RFS*) find 65% of 452 anomalies fail |t|≥1.96 under value-weighting with NYSE breakpoints, and 82% fail the 2.78 multiple-testing hurdle. Jensen–Kelly–Pedersen (2023) argue most factors do replicate once properly clustered into 13 themes. Martineau (2022, *Critical Finance Review*) reports that "for large stocks, PEAD have been non-existent since 2006." Muravyev–Pearson–Pollet (2025) show option IV-spread/skew predictability falls by at least two-thirds once high-borrow-fee stocks are excluded.\[2\]
- **Use probability for sizing, risk and gating, not as a directional oracle.** A GARCH/HAR/realized-vol and option-implied layer forecasts variance, tails and event risk. Quantile/conformal models give calibrated intervals. The ATM (≈50-delta) point of the option-implied distribution is just the forward price and carries no alpha. Validate everything with purged/embargoed walk-forward CV, deflated Sharpe ratios and a t≥3 hurdle for new signals.\[3\]

---

## Chapter 1 — Foundations: Concepts and Terminology

### 1.1 Returns

- **Simple return:** $R_{t+1} = \frac{P_{t+1} + D_{t+1}}{P_t} - 1$, where $P$ is price and $D$ is the dividend/distribution paid in $(t, t+1]$. Simple returns aggregate across assets: a portfolio return is $R_p = \sum_i w_i R_i$.
- **Log return:** $r_{t+1} = \ln(1+R_{t+1})$. Log returns aggregate across time ($r_{t\to t+h} = \sum_{k=1}^{h} r_{t+k}$) but not across assets. Use simple returns for portfolio accounting and log returns for volatility modeling and multi-period compounding.
- **Excess return:** $R^e_{t+1} = R_{t+1} - R^f_t$, where $R^f_t$ is the risk-free rate known at $t$ (e.g., 1-month T-bill from Ken French/FRED). In CRSP, use `RET` (holding-period return incl. dividends) and incorporate delisting returns (`DLRET`), or you will inject survivorship bias. Note CRSP's 2024–25 switch from SIZ to CIZ format. A JFQA study ("Rewriting CRSP's History") reports that the transition rewrites 9.62% of monthly returns by more than 1 bp, mainly via the dividend-reinvestment assumption, so pin your data vintage.
- **Forward return (the label):** $y_{i,t}^{(h)} = \prod_{k=1}^{h}(1+R_{i,t+k}) - 1$, the return from the first tradable price after the signal time to $t+h$. In Qlib the default label is `Ref($close, -2)/Ref($close, -1) - 1`. This deliberately skips one day so that a signal computed at close $t$ is traded at $t+1$ and earns $t+1 \to t+2$. Preserve this lag discipline in custom handlers.
- **Residual/abnormal return:** $\epsilon_{i,t} = R^e_{i,t} - \hat\beta_i' f_t$, the return unexplained by factor returns $f_t$. It is often the better label for cross-sectional models because it removes market/sector noise.

### 1.2 Cross-sectional vs. time-series prediction

- **Cross-sectional:** at each date $t$, predict the *relative* ranking of $N$ stocks: $E_t[R_{i,t+1} - \bar R_{t+1}]$. The market component is differenced out. Almost all equity anomaly literature and Qlib's model zoo work in this setting. Breadth is large (thousands of bets per period).
- **Time-series:** predict the level of one asset's (or the market's) return: $E_t[R_{m,t+1}]$. Signal-to-noise is far lower and breadth is tiny. Examples are market timing using the variance risk premium or valuation ratios.

### 1.3 Alpha, beta, factor, anomaly

- **Factor model:** $R^e_{i,t} = \alpha_i + \beta_i' f_t + \epsilon_{i,t}$. $\beta_i$ are exposures (loadings), $f_t$ are factor (excess) returns, and $\alpha_i$ is the average return unexplained by the factors.
- **Factor:** a systematic source of co-movement and/or priced risk, typically implemented as a long-short characteristic-sorted portfolio (e.g., HML, SMB, UMD, RMW, CMA).
- **Anomaly:** a characteristic that predicts returns *not* explained by a benchmark model (non-zero $\alpha$). Whether it is "risk" or "mispricing" matters for persistence. Risk premia should persist; mispricing should decay as arbitrage capital arrives.

### 1.4 Information coefficient (IC), rank IC, ICIR

- **IC:** $IC_t = \mathrm{corr}(s_{\cdot,t}, y_{\cdot,t+1})$, the cross-sectional Pearson correlation between signals $s$ and subsequent returns $y$ at date $t$.
- **Rank IC:** $\mathrm{RankIC}_t = \mathrm{corr}(\mathrm{rank}(s_{\cdot,t}), \mathrm{rank}(y_{\cdot,t+1}))$ (Spearman). It is robust to outliers and preferred for heavy-tailed returns.
- **ICIR:** $\mathrm{ICIR} = \overline{IC}/\sigma(IC)$, the consistency of the signal. $\overline{IC}\cdot\sqrt{T}/\sigma(IC)$ is approximately the t-stat of the mean IC.
- Typical magnitudes: daily-rebalanced single-factor ICs are about 0.01–0.03, and good composite ML models reach about 0.04–0.06. Qlib's published CSI300 benchmark shows LightGBM on Alpha158 with IC 0.0448 and ICIR 0.3660.\[4\]

### 1.5 Sharpe ratio, information ratio, fundamental law

- **Sharpe ratio:** $SR = \frac{E[R_p - R_f]}{\sigma(R_p - R_f)}$, annualized by $\sqrt{252}$ (daily) or $\sqrt{12}$ (monthly) *only if* returns are i.i.d. Serial correlation requires Lo (2002) adjustments.
- **Information ratio:** $IR = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)} = \alpha/\omega$, active return over tracking error versus benchmark $b$.
- **Fundamental Law of Active Management** (Grinold 1989; Grinold–Kahn): $IR \approx IC \cdot \sqrt{BR}$, where $BR$ is breadth, the number of *independent* bets per year. The extended version (Clarke, de Silva, Thorley 2002) adds the transfer coefficient $TC$ (correlation between ideal and actually-held active weights): $IR \approx TC \cdot IC \cdot \sqrt{BR}$. Implication: an IC of 0.02 over 500 independent monthly bets gives $IR \approx 0.02\sqrt{6000} \approx 1.5$ *before costs and constraints*. In practice bets are correlated and TC is 0.3–0.7, so expect far less.

### 1.6 Turnover, transaction costs, market impact, slippage

- **Turnover:** $TO_t = \frac{1}{2}\sum_i |w_{i,t} - w_{i,t^-}|$, where $w_{i,t^-}$ is the drifted pre-trade weight. Annual one-way turnover of 1000% or more is typical for daily reversal books.
- **Cost decomposition:** commissions + half bid-ask spread + market impact + opportunity cost (unfilled orders) + borrow fees (short side).
- **Square-root impact law:** $\Delta P / P \approx Y \cdot \sigma_d \sqrt{Q/V_d}$. Here $\sigma_d$ is daily volatility, $Q$ the order size, $V_d$ average daily volume, and $Y$ an empirical constant of order 1 (Torre/Barra; Almgren et al. 2005; Tóth et al. 2011; Bouchaud et al.). Impact is concave in size, so cost per dollar grows like $\sqrt{Q}$ and total cost like $Q^{3/2}$.
- **Slippage:** realized fill price minus decision (arrival) price; implementation shortfall (Perold 1988) is the total.
- **Evidence:** Frazzini, Israel and Moskowitz ("Trading Costs of Asset Pricing Anomalies", AQR working paper, 2012/2015) used nearly $1 trillion of live institutional trades across 19 developed markets from 1998–2011.\[5\] They found actual costs much smaller than earlier academic estimates.\[6\] Value and momentum were more scalable than size, and short-term reversal was the most cost-constrained.\[5\] Their later extension ("Trading Costs") uses $1.7 trillion across 21 markets over 19 years.\[7\] Caveat: these are costs for a large, sophisticated execution desk. A new entrant should assume higher costs.

### 1.7 Biases in data and research

- **Survivorship bias:** using only currently-listed firms (or ignoring delisting returns) inflates backtests. CRSP includes dead firms and delisting returns.
- **Look-ahead bias:** using information not available at decision time. Common forms are fiscal-period fundamentals aligned to period end instead of filing date, restated data, index membership as of today, and I/B/E/S "actual" dates misaligned with announcement time (pre-open vs. post-close).
- **Point-in-time (PIT) data:** each value is stamped with the date it became knowable. Compustat Snapshot/PIT (S&P describes point-in-time snapshots beginning in 1987 for North America) and I/B/E/S detail history with announcement timestamps are the tools. Without PIT, use conservative lags (e.g., 4 months after fiscal year-end for annual data, 45–60 days for quarterly `RDQ` if missing).
- **Data snooping / multiple testing:** when $M$ strategies are tried, the maximum t-stat grows roughly like $\sqrt{2\ln M}$ under the null. Harvey, Liu and Zhu (2016, *RFS* 29(1):5–68) catalogued 316 factors and argued a new factor needs t > 3.0, concluding "most claimed research findings in financial economics are likely false."
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014, *JPM* 40(5)): $DSR = \Phi\left(\frac{(\widehat{SR} - SR_0)\sqrt{T-1}}{\sqrt{1 - \hat\gamma_3 \widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right)$. $\hat\gamma_3, \hat\gamma_4$ are the skewness and kurtosis of returns and $T$ the sample length. The expected maximum Sharpe under the null across $N$ trials is $SR_0 = \sqrt{V[\widehat{SR}_n]}\left((1-\gamma)\Phi^{-1}(1 - \tfrac{1}{N}) + \gamma\Phi^{-1}(1 - \tfrac{1}{N}e^{-1})\right)$, with $\gamma \approx 0.5772$ (Euler–Mascheroni). DSR is the probability the true SR exceeds zero after accounting for selection across $N$ trials and non-normality.

### 1.8 Validation designs

- **Walk-forward (rolling/expanding) validation:** train on $[t_0, t]$, validate on $(t, t+v]$, test on $(t+v, t+v+h]$, then roll forward. Qlib's `RollingGen`/`task` workflow supports this.
- **Purged k-fold CV** (López de Prado 2018, *Advances in Financial Machine Learning*): when labels span $h$ periods, remove ("purge") training observations whose label windows overlap the test fold. Add an **embargo** of $\approx h$ periods after each test fold to kill leakage from serial correlation. **Combinatorial purged CV (CPCV)** produces many backtest paths for a distribution of out-of-sample Sharpe ratios.

### 1.9 Measures: risk-neutral (Q) vs. physical (P)

- Under **Q**, discounted asset prices are martingales: $S_0 = e^{-rT}E^Q[S_T]$ (with dividends, $E^Q[S_T] = F_{0,T}$). Option prices reveal Q-distributions.
- Under **P** (real world), $E^P[S_T] = S_0 e^{(\mu - q)T}$ with $\mu > r$ for risky assets. The two are linked by the pricing kernel (stochastic discount factor) $M_T$: $\frac{dQ}{dP} = \frac{M_T}{E^P[M_T]}$, so $q(S_T) = p(S_T) \cdot \frac{M(S_T)}{E[M]}$. Q puts more weight on bad states, so Q-implied volatility and left-tail probability systematically exceed P-realized ones.
- **Forward price:** $F_{0,T} = S_0 e^{(r - q)T}$, with $q$ the continuous dividend yield (or $F = (S_0 - PV(D))e^{rT}$ for discrete dividends).

### 1.10 Implied volatility, surface, skew, term structure, greeks

- **Implied volatility (IV):** the $\sigma$ that equates the Black–Scholes(–Merton) price to the market price: $C^{mkt} = C^{BS}(S, K, T, r, q, \sigma^{imp})$.
- **Volatility surface:** $\sigma^{imp}(K, T)$ or $\sigma^{imp}(k, T)$ with log-moneyness $k = \ln(K/F)$. OptionMetrics' `vsurfd` gives a standardized surface by delta (±0.10…0.90) and maturity (10–730 days).
- **Skew/smirk:** the slope of IV in moneyness, e.g., $IV(\text{25}\Delta\text{ put}) - IV(\text{25}\Delta\text{ call})$ or $IV_{OTM put} - IV_{ATM call}$. Equity skew is typically negative (downside puts rich).
- **Term structure:** $IV(T_2) - IV(T_1)$ at fixed moneyness. It is usually upward-sloping in calm markets and inverted in stress or before known events (earnings).
- **Greeks:** $\Delta = \partial V/\partial S$, $\Gamma = \partial^2 V/\partial S^2$, $\mathcal{V}$ (vega) $= \partial V/\partial\sigma$, $\Theta = \partial V/\partial t$, $\rho = \partial V/\partial r$. For BS calls, $\Delta = e^{-qT}N(d_1)$ with $d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt T}$ and $d_2 = d_1 - \sigma\sqrt T$.

### 1.11 Realized volatility and variance risk premium

- **Realized variance:** $RV_t = \sum_{j=1}^{n} r_{t,j}^2$ over intraday returns $r_{t,j}$ (e.g., 5-min from TAQ), a consistent estimator of integrated variance absent noise and jumps.
- **Variance risk premium (VRP):** $VRP_t = E^Q_t[\text{Var}_{t,t+\tau}] - E^P_t[\text{Var}_{t,t+\tau}]$. Operationally this is often $IV^2_t - RV_t$ (backward-looking realized) or $IV^2_t - \widehat{RV}_{t+\tau}$ (forecast). Carr–Wu (2009, *RFS*) measure it by synthetic variance swaps. Sign conventions vary: some papers define $RV - IV$, which is negative on average.

### 1.12 Risk models, neutralization, optimization, Kelly, drawdown

- **Factor risk model:** $\Sigma = B F B' + D$. $B$ is the $N\times K$ exposure matrix, $F$ the $K\times K$ factor covariance, and $D$ the diagonal specific variance.
- **Factor neutralization:** replace a signal $s$ by its residual $\tilde s = (I - B(B'WB)^{-1}B'W)s$, the cross-sectional regression of $s$ on exposures (industry dummies, size, beta, etc.) with weights $W$. In portfolio form, impose $B'w = 0$.
- **Mean–variance optimization** (Markowitz 1952): $\max_w\ w'\mu - \frac{\lambda}{2}w'\Sigma w$, giving $w^* = \frac{1}{\lambda}\Sigma^{-1}\mu$.
- **Risk parity:** choose $w$ so each asset's risk contribution $RC_i = w_i(\Sigma w)_i/\sqrt{w'\Sigma w}$ is equal.
- **Kelly criterion:** maximize $E[\ln(1 + f R)]$. For a single asset with Gaussian excess returns, $f^* = \mu/\sigma^2$. For a portfolio, $f^* = \Sigma^{-1}\mu$ (the $\lambda = 1$ MV solution). Full Kelly is extremely sensitive to estimation error in $\mu$, so practitioners use fractional Kelly (¼–½).
- **Drawdown:** $DD_t = 1 - \frac{V_t}{\max_{s\le t}V_s}$; max drawdown $MDD = \max_t DD_t$. Under Gaussian returns, expected MDD scales roughly with $\sigma^2/\mu$ for long horizons. Heavy tails and regime shifts make real drawdowns worse.

---

## Chapter 2 — Equity Factors and Anomalies

**Reading the evidence first.** Three landmark meta-studies frame everything below:

1. **Hou, Xue & Zhang (2020, *RFS* 33(5):2019–2133), "Replicating Anomalies":** with microcaps mitigated via NYSE breakpoints and value-weighted returns, 65% of 452 anomalies fail |t|≥1.96, including 96% of the trading-frictions category. At the multiple-testing hurdle of 2.78, failure rises to 82%. Critics note many of the 452 were never significant in the original papers.
2. **Chen & Zimmermann (2022, *Critical Finance Review*), "Open Source Cross-Sectional Asset Pricing":** of 161 characteristics clearly significant in the original papers, 98% of their long-short replications have t > 1.96. The Open Source Asset Pricing data (October 2025 release) provides 212 predictors.
3. **Jensen, Kelly & Pedersen (2023, *JF* 78(5):2465–2518), "Is There a Replication Crisis in Finance?":** a Bayesian replication framework. Most factors replicate, cluster into 13 themes, work out of sample across 93 countries, and are strengthened (not weakened) by the large number of related factors. The Global Factor Data (jkpfactors.com) covers 153 characteristics.

**Decay:** McLean & Pontiff (2016, *JF* 71(1):5–32) study 97 predictors. The average in-sample long-short return was 58.2 bp/month, 26% lower out-of-sample and 58% lower post-publication.\[1\]\[8\] Declines are larger for predictors with higher in-sample returns, and returns stay higher in high-idiosyncratic-risk, illiquid stocks (i.e., where limits to arbitrage bind).\[1\] **Planning rule: haircut any published anomaly's in-sample premium by about 50–60% and assume it lives mostly where it is expensive to trade.**

Below, "Horizon" means the signal's natural holding period; "WRDS" names the primary source tables.

### 2.1 Value

- **Formulas:**
  - $B/M = \frac{\text{Book equity}}{\text{Market equity}}$ with $BE = SEQ + TXDITC - PS$ (Fama–French). $SEQ$ is stockholders' equity, $TXDITC$ deferred taxes and investment tax credit, and $PS$ preferred stock (redemption, liquidating or par value).
  - $E/P = \frac{EPS \times Shares}{ME}$.
  - $S/P = \frac{Sales}{ME}$.
  - $EBITDA/EV$ with $EV = ME + Debt + PS + MI - Cash$.
  - $CF/P$ uses (earnings + depreciation).
- **Mechanism:** a risk explanation (distress, costly reversibility of assets, Zhang 2005) and/or a behavioral one (extrapolation of past growth, Lakonishok–Shleifer–Vishny 1994).
- **Horizon:** long (quarters to years); monthly or slower rebalancing.
- **Performance and decay:** HML's US performance was poor from about 2007 to 2020, with a deep drawdown into 2020 and a partial rebound after. EV/EBITDA and composite value measures have held up better than B/M.
- **Known flaws:**
  - **Timeliness ("The Devil in HML's Details", Asness & Frazzini 2013, *JPM*):** Fama–French HML uses book and market equity lagged to the prior December. Using *current* market equity (updated monthly) with lagged book produces a better value measure: higher alpha against momentum and more negative correlation to UMD.
  - **Intangibles:** B/M ignores internally generated intangible capital, which is expensed. Capitalizing R&D (perpetual-inventory: $K^{R\&D}_t = (1-\delta)K^{R\&D}_{t-1} + R\&D_t$, $\delta \approx 15\%$–20%) and a share of SG&A as organization capital (Peters & Taylor 2017, *JFE*; Eisfeldt & Papanikolaou 2013) repairs much of value's recent underperformance (Eisfeldt, Kim & Papanikolaou 2022 "Intangible Value", *CFR*; Lev & Srivastava 2019).
  - **Industry composition:** raw value is a large sector bet. Industry-neutral value is cleaner.
- **Data:** Compustat Fundamentals Annual/Quarterly (`comp.funda`/`fundq`), CRSP for ME, CCM link. Use Compustat Snapshot for PIT if licensed.

### 2.2 Quality and profitability

- **Gross profitability** (Novy-Marx 2013, *JFE*): $GP/A = \frac{REVT - COGS}{AT}$. Gross profits are "cleanest" because they sit high on the income statement. They predict returns as strongly as B/M and hedge value (negatively correlated), so value + profitability is one of the best-documented combinations.
- **Operating profitability** (Fama–French 2015 RMW): $OP = \frac{REVT - COGS - XSGA - XINT}{BE}$. **Cash-based operating profitability** (Ball, Gerakos, Linnainmaa & Nikolaev 2016, *JFE*) strips accruals and subsumes accruals and OP.
- **ROIC:** $\frac{EBIT(1-\tau)}{\text{Invested capital}}$, with invested capital = debt + equity − cash. It is intuitive but noisy; ROE (Hou–Xue–Zhang) is the q-factor version.
- **Piotroski F-score** (2000, *JAR*): the sum of nine binary signals.
  - Profitability: ROA > 0, CFO > 0, ΔROA > 0, CFO > ROA (accrual).
  - Leverage/liquidity: Δleverage < 0, Δcurrent ratio > 0, no equity issuance.
  - Efficiency: Δgross margin > 0, Δasset turnover > 0.
  - It works best within cheap (high-B/M) stocks, with a mild effect in large caps.
- **Accruals** (Sloan 1996, *TAR*): $ACC = \frac{(\Delta CA - \Delta Cash) - (\Delta CL - \Delta STD - \Delta TP) - Dep}{\text{avg } AT}$. Low accruals outperform because investors fixate on earnings rather than cash flows. The effect has **decayed substantially since the early 2000s** (Green, Hand & Soliman 2011, "Going, Going, Gone?", *Mgmt Sci*).
- **Quality Minus Junk** (Asness, Frazzini & Pedersen 2019, *RAST*): a composite z-score of profitability, growth, safety (low beta, low leverage, low earnings volatility) and payout. It has a strong negative market beta and a flight-to-quality character.
- **Horizon:** long.
- **Data:** Compustat; CRSP for betas.

### 2.3 Dividend yield and shareholder yield

- **Formulas:** $DY = \frac{\sum_{12m} D}{P}$. **Net payout yield** (Boudoukh, Michaely, Richardson & Roberts 2007, *JF*) is $\frac{Div + Repurchases - Issuance}{ME}$. It predicts returns better than dividend yield because firms substitute buybacks for dividends.
- **Related:** net share issuance (Pontiff & Woodgate 2008; Daniel & Titman 2006), $\ln(Shares_t/Shares_{t-12})$, is negative-signed and among the most robust anomalies.
- **Horizon:** long.
- **Data:** CRSP (`shrout`, `cfacshr`), Compustat `PRSTKC`, `SSTK`, `DVC`.

### 2.4 Momentum family

- **Price momentum 12-1** (Jegadeesh & Titman 1993, *JF*): $MOM_{i,t} = \prod_{k=2}^{12}(1 + R_{i,t-k}) - 1$. It skips the most recent month to avoid short-term reversal. The long-short winners-minus-losers portfolio earned about 1%/month historically.
- **Momentum crashes** (Daniel & Moskowitz 2016, *JFE* 122(2):221–247): crashes are partly forecastable. They occur in "panic" states, after market declines and when volatility is high, at the market rebound, because the loser portfolio behaves like a call option on the market.\[9\]\[10\]\[11\] Examples are 1932 and 2009, when WML lost more than 70% in a few months. Their dynamic strategy scales exposure by forecast mean/variance ($w_t \propto \hat\mu_t/\hat\sigma^2_t$) and "approximately doubles the alpha and Sharpe ratio" of static momentum.\[9\] Barroso & Santa-Clara (2015, *JFE* 116(1):111–120, "Momentum Has Its Moments") show constant-volatility scaling ($w_t = \sigma_{target}/\hat\sigma_t$, using 6-month realized daily vol) raises momentum's Sharpe ratio from 0.53 to 0.97 and cuts its maximum drawdown from −96.69% to −45.20%. **Action:** always vol-manage momentum. Note, however, that Cederburg et al. (2020) find vol-managing *other* factors adds little out-of-sample, and after-cost evidence for vol-managing factors other than the market is weak.\[12\]
- **Residual (idiosyncratic) momentum** (Blitz, Huij & Martens 2011, *JEF*; Gutierrez & Pirinsky 2007): rank on the cumulative sum of FF3 residuals $\sum_{k=2}^{12}\hat\epsilon_{i,t-k}/\sigma(\hat\epsilon)$. It has roughly half the volatility of raw momentum, a similar return, and much smaller crash exposure. It is a strong candidate for the slow book.
- **Industry momentum** (Moskowitz & Grinblatt 1999, *JF*): winners/losers at the industry level. It explains a large share of stock momentum and has a strong 1-month industry effect (the opposite sign of stock-level reversal).
- **Other variants:** 52-week-high momentum (George & Hwang 2004), $P_t/\max_{252}P$, and earnings momentum (see PEAD/revisions). Time-series momentum (Moskowitz, Ooi & Pedersen 2012) is more relevant to index timing.
- **Horizon:** medium (3–12 months), monthly rebalancing.
- **Data:** CRSP monthly/daily.

### 2.5 Short-term reversal

- **Formula** (Jegadeesh 1990; Lehmann 1990): $STR_{i,t} = -R_{i,t-1}$ (past month), or weekly/daily variants: $-R_{i,t-5\to t}$.
- **Mechanism:** liquidity provision; prices overshoot due to order-flow pressure and revert (Nagel 2012, *RFS*, "Evaporating Liquidity": reversal returns are like liquidity-provision returns and spike when VIX is high).
- **Improvements:**
  - Residual/industry-adjusted reversal (Da, Liu & Schaumburg 2014, *Mgmt Sci*) strips industry and news-driven moves.
  - Reversal is concentrated in no-news moves. News-driven moves tend to *continue* (Chan 2003; Jiang et al.). This is a natural interaction with RavenPack.
- **Horizon:** days to 1 month.
- **Decay:** the gross premium is still visible in daily/weekly data, but it is **the most cost-constrained anomaly** (Frazzini–Israel–Moskowitz; Novy-Marx & Velikov 2016, *RFS*).\[5\] It is profitable only with cost-aware construction, liquidity-weighted universes and patient execution.
- **Data:** CRSP daily; TAQ for intraday refinements.

### 2.6 Low volatility / betting against beta

- **Formulas:**
  - $\hat\beta_i^{BAB} = \hat\rho_{i,m}\frac{\hat\sigma_i}{\hat\sigma_m}$ (Frazzini & Pedersen 2014, *JFE*: correlations from 5-yr 3-day overlapping returns, volatilities from 1-yr daily, shrunk toward 1: $\beta^{shrunk} = 0.6\hat\beta + 0.4$).
  - The BAB factor is $r^{BAB} = \frac{1}{\beta_L}(r_L - r_f) - \frac{1}{\beta_H}(r_H - r_f)$, leveraged to be beta-neutral.
- **Mechanism:** leverage constraints push investors toward high-beta stocks (Black 1972; Frazzini–Pedersen). Benchmarking is another channel (Baker, Bradley & Wurgler 2011).
- **Performance and caveats:** strong long-run Sharpe, but Novy-Marx & Velikov (2022, "Betting against betting against beta", *JFE*) argue BAB's construction (rank-weighting, the beta-shrinkage timing mismatch, and implicit small-cap tilt) inflates its performance. Value-weighted low-vol portfolios deliver more modest alpha. There are crowding concerns post-2012 (low-vol ETFs) and heavy rate sensitivity. JKP note that the low-beta factor has positive alpha but raw return near zero.\[13\]
- **Horizon:** long.
- **Data:** CRSP daily.

### 2.7 Size

- **Formula:** $\ln(ME)$ (Banz 1981).
- **Status:** weak since the early 1980s, concentrated in microcaps and January. It revives only when controlling for quality/junk (Asness, Frazzini, Israel, Moskowitz & Pedersen 2018, "Size Matters, If You Control Your Junk", *JFE*). **Treat size as a risk-model exposure, not an alpha.**

### 2.8 Investment (FF5 and q-factor)

- **Formula:** $I/A = \frac{AT_t - AT_{t-1}}{AT_{t-1}}$ (asset growth, Cooper, Gulen & Schill 2008, *JF*). Conservative-minus-aggressive (CMA) is the Fama–French (2015, *JFE*) 5-factor version.
- **q-factor model** (Hou, Xue & Zhang 2015, *RFS*): $E[R^e_i] = \beta_{MKT}\lambda_{MKT} + \beta_{ME}\lambda_{ME} + \beta_{I/A}\lambda_{I/A} + \beta_{ROE}\lambda_{ROE}$, derived from investment-based asset pricing. With the expected-growth factor it is the $q^5$ model (Hou, Mo, Xue & Zhang 2021).
- **Mechanism:** a lower discount rate means more investment. Alternatively, empire-building overinvestment is punished later.
- **Horizon:** long.
- **Data:** Compustat. Factor returns are free at global-q.org and in Ken French's library.

### 2.9 Post-earnings announcement drift (PEAD) and earnings surprises

- **Formulas:**
  - $SUE_{i,q} = \frac{E_{i,q} - E_{i,q-4}}{\sigma(E_{i,q} - E_{i,q-4})_{8q}}$ (seasonal random walk, Bernard & Thomas 1989).
  - Analyst-based: $SUE^{IBES} = \frac{Actual - \text{Consensus}}{P}$.
  - Earnings-announcement abnormal return: $CAR_{[-1,+1]}$.
- **Mechanism:** underreaction and limited attention (DellaVigna & Pollet 2009: Friday announcements drift more).
- **Horizon:** days to about 60 trading days.
- **Decay, and it is serious:** Martineau (2022, *Critical Finance Review* 11:613–646, "Rest in Peace Post-Earnings Announcement Drift") finds that prices now fully reflect earnings surprises on the announcement date. PEAD is "non-existent since 2006" for large stocks and has only recently disappeared for microcaps.\[14\]\[15\] Two 2025 papers dispute this; the disagreement is largely about research design (surprise measures, earnings-factor construction). With all stocks included, one study's drift t-stat is about 2.18, barely significant.\[16\] **Verdict:** classic SUE-sorted drift is a weak, capacity-limited small-cap effect. Richer surprise measures remain promising: text-based surprise from calls, revenue surprises (Jegadeesh & Livnat 2006), expectation-ladder surprises, and "fast" intraday reaction features. PEAD should be a feature in the fast book rather than a standalone strategy.
- **Data:** I/B/E/S detail/summary (`ibes.det_epsus`, `statsum_epsus`, `act_epsus` with `anndats`/`anntims`), Compustat `RDQ`, CRSP daily; the ICLINK/`wrdsapps_link_crsp_ibes.ibcrsphist` link.

### 2.10 Analyst revisions and recommendations

- **Formulas:**
  - $REV_{i,t} = \sum_{k=0}^{2}\frac{FY1_{i,t-k} - FY1_{i,t-k-1}}{P_{i,t-k-1}}$ (Chan, Jegadeesh & Lakonishok 1996).
  - Up-minus-down revision breadth: $\frac{\#up - \#down}{\#analysts}$.
  - Changes in recommendation level (Womack 1996; Jegadeesh et al. 2004).
- **Mechanism:** slow diffusion; analysts herd and revise gradually.
- **Horizon:** 1–6 months. It is one of the more robust fast-to-medium signals. Note McLean–Pontiff-style decay, and that dispersion (Diether, Malloy & Scherbina 2002) has weakened.
- **Data:** I/B/E/S detail history (PIT by `actdats`/`anndats`), recommendations (`recddet`).

### 2.11 Short interest and borrow fees

- **Formulas:**
  - $SIR_{i,t} = \frac{\text{Shares short}}{\text{Shares outstanding}}$.
  - Days-to-cover $DTC = \frac{\text{Shares short}}{ADV}$ (Hong, Li, Ni, Scheinkman & Yan 2016). DTC is a better predictor than SIR because it normalizes by liquidity.
  - Loan fee / utilization from securities lending data.
- **Mechanism:** short sellers are informed (Boehmer, Jones & Zhang 2008; Asquith, Pathak & Ritter 2005). High fees also signal overpricing that is costly to arbitrage.
- **Critical caveat:** Muravyev, Pearson & Pollet (2022 *JF*; 2025 *JFE*) and related work ("Anomalies and Their Short-Sale Costs") show that many anomaly short-leg profits are consumed by borrow fees. Much of the return predictability is concentrated in high-fee stocks, about 12% of stock-dates in one sample.\[17\] **Long-only or long-tilted implementations of short-side signals are much more realistic.**
- **Horizon:** 2 weeks to months.
- **Data:** Compustat supplemental short interest (`comp.sec_shortint`, mid-month and end-of-month).\[18\] The start date is reported inconsistently across sources (1973 vs. later), so verify on WRDS.\[19\]\[20\]\[21\] Markit/S&P Global Securities Finance for fees/utilization (daily, from about 2002–2006).\[22\] FINRA/exchange short-interest files are open.

### 2.12 Institutional ownership and 13F flows

- **Signals:**
  - Breadth of ownership change (Chen, Hong & Stein 2002): $\Delta Breadth = \frac{\#holders_t - \#holders_{t-1}}{\#managers}$.
  - Institutional demand shocks and crowding.
  - Hedge-fund "best ideas" (Cohen, Polk & Silli).
  - Flow-induced price pressure (Coval & Stafford 2007; Lou 2012).
- **Horizon:** quarterly (13F lags up to 45 days after quarter-end, so use the filing date, not the report date).
- **Data:** Thomson/Refinitiv s34 (from 1980; known data-quality issues, some fixed by WRDS), WRDS SEC 13F, EDGAR raw 13F-HR.

### 2.13 Liquidity (Amihud)

- **Formula:** $ILLIQ_{i} = \frac{1}{D}\sum_{d=1}^{D}\frac{|R_{i,d}|}{DVOL_{i,d}}$ (Amihud 2002, *JFM*), the price impact per dollar traded. Pástor–Stambaugh (2003) liquidity *risk* (beta to aggregate liquidity innovations) is a separate concept.
- **Status:** the level premium is mostly a microcap/cost-compensation effect, and HXZ's trading-frictions category fails 96% of the time under VW. **Use it as a cost and capacity input, not an alpha.**

### 2.14 Seasonality (Heston–Sadka)

- **Formula** (Heston & Sadka 2008, *JFE*): $SEAS_{i,t} = \frac{1}{K}\sum_{k=1}^{K} R_{i,t-12k}$, the average return in the same calendar month in prior years. Keloharju, Linnainmaa & Nyberg (2016, *JF*) show seasonalities exist across anomalies and factors too.
- **Horizon:** 1 month.
- **Status:** it replicates reasonably and is low-correlated with other signals, but its magnitude is modest.

### 2.15 Lottery / MAX and the idiosyncratic volatility puzzle

- **MAX** (Bali, Cakici & Whitelaw 2011, *JFE*): $MAX_{i,t} = \max_{d\in t}R_{i,d}$, the maximum daily return in the prior month. It is negatively related to future returns (lottery preference).
- **IVOL** (Ang, Hodrick, Xing & Zhang 2006, *JF*; 2009 *JFE*): $IVOL = \sigma(\epsilon^{FF3}_{i,d})$ over the prior month; the high-IVOL quintile underperforms badly.\[23\]
- **Caveats:** both are concentrated in small, illiquid, hard-to-short stocks and are sensitive to microstructure (Bali & Cakici 2008; Fu 2009 using EGARCH conditional IVOL finds a *positive* relation). Stambaugh, Yu & Yuan (2015) show the effect is concentrated in overpriced stocks. **Use as a short-side filter / risk feature.**

### 2.16 The factor zoo and dimension reduction

Beyond HXZ/JKP/Chen–Zimmermann:

- Feng, Giglio & Xiu (2020, *JF*, "Taming the Factor Zoo") use double-selection LASSO to find which factors add pricing information.
- Kozak, Nagel & Santosh (2020, *JFE*) take a shrinkage SDF approach: a few principal components of anomaly portfolios capture most of the pricing.
- DeMiguel, Martín-Utrera, Nogales & Uppal (2020, *RFS* 33(5):2180–2222) find that "transaction costs increase the number of significant characteristics from 6 to 15," because combining characteristics nets trades and reduces costs.

**Practical takeaway:** do not hunt for single anomalies. Build a combined, cost-aware composite from the 13 JKP themes, weighting robust themes (value, profitability, investment, momentum, low risk, seasonality, accruals-lite).

### 2.17 Summary table: equity factor evidence status

| Factor | Horizon | Core data | Evidence strength | Status/decay |
|---|---|---|---|---|
| Value (B/M, EV/EBITDA, composite) | Long | Compustat, CRSP | Strong long-run; weak 2007–2020 | Improved by timely ME and intangibles |
| Gross/cash profitability | Long | Compustat | Strong | Robust; hedges value |
| Investment/asset growth | Long | Compustat | Strong | Robust; in FF5/q models |
| Net share issuance | Long | CRSP | Strong | Robust |
| Accruals | Long | Compustat | Weakened | Decayed post-2000s |
| Momentum 12-1 | Medium | CRSP | Strong but crash-prone | Vol-manage; residual momentum better |
| Industry momentum | Medium | CRSP | Good | Moderate |
| Short-term reversal | Days–1m | CRSP/TAQ | Strong gross | Cost-dominated |
| Low beta/low vol | Long | CRSP | Good, construction-sensitive | Crowded; rate-sensitive |
| Size | — | CRSP | Weak | Use as risk factor |
| PEAD (SUE) | Days–60d | I/B/E/S, Compustat | Contested | Largely gone in large caps since about 2006 |
| Analyst revisions | 1–6m | I/B/E/S | Good | Moderate decay |
| Short interest/DTC | Weeks–months | Compustat SI, Markit | Good gross | Borrow-fee constrained |
| 13F breadth/flows | Quarter | Thomson s34 | Moderate | Lagged data |
| Amihud illiquidity | Long | CRSP | Weak under VW | Cost input only |
| Seasonality | 1m | CRSP | Moderate | Modest, low correlation |
| MAX/IVOL | 1m | CRSP | Moderate, small-cap | Short-side filter |

---

## Chapter 3 — Option-Implied Signals and Probabilistic Information

### 3.1 Breeden–Litzenberger risk-neutral density (RND)

- **Formula** (Breeden & Litzenberger 1978, *J. Business*): $q(K) = e^{rT}\frac{\partial^2 C(K,T)}{\partial K^2}$. Also $Q(S_T \le K) = 1 + e^{rT}\frac{\partial C}{\partial K}$. $C(K,T)$ is the call price as a function of strike.
- **Implementation:**
  1. Convert quotes to IVs.
  2. Fit a smooth, arbitrage-free smile (SVI/SSVI, or splines in IV space: Shimko 1993; Figlewski 2010 with GEV tails).
  3. Reprice calls on a dense strike grid.
  4. Take numerical second differences.
  5. Check $q \ge 0$ and $\int q = 1$.
- For single-stock American options, first back out European-equivalent IVs (OptionMetrics already does this with a Cox–Ross–Rubinstein binomial model with discrete dividends).

### 3.2 The ATM/50-delta point carries no directional information

For a lognormal-ish RND, the 50-delta strike satisfies $N(d_1) = 0.5$, so $K_{50\Delta} = F e^{\sigma^2 T/2} \approx F = S e^{(r-q)T}$. The center of the risk-neutral distribution is pinned by no-arbitrage to the forward. It reflects the carry ($r$, $q$, and borrow fee), not anyone's expectation of the stock going up. **Any "signal" computed as "ATM strike vs. spot" is just the cost of carry.** Directional information, if any, lives in the *shape* (skew, tails, call–put IV deviations), in *flows* (volume, open interest), and in *risk premia* (the Q–P wedge). Even these, as shown below, largely proxy for short-sale costs.

### 3.3 From Q to P: pricing-kernel adjustments

$p(S_T) = \frac{q(S_T)/M(S_T)}{\int q(s)/M(s)\,ds}$. The common parametric choice is a power (CRRA) kernel $M(S_T) \propto S_T^{-\gamma}$, giving $p(S_T) \propto q(S_T)S_T^{\gamma}$, with $\gamma$ estimated by maximizing the likelihood of realized outcomes (Bliss & Panigirtzoglou 2004, *JF*; Liu, Shackleton, Taylor & Xu 2007). Alternatives:

- Calibrate the Q→P map statistically, e.g., a beta-transformation of the Q-CDF fitted to realized PITs (Fackler & King 1990; Kitsul & Wright 2013).
- Ross (2015, *JF*) "Recovery Theorem." Empirically fragile: Borovička, Hansen & Scheinkman 2016; Jackwerth & Menner 2020 reject it.
- Martin (2017, *QJE*) SVIX lower bound on the equity premium: $E_t[R_{m,t\to T}] - R_f \ge \frac{1}{R_f}\text{SVIX}^2_t$. Martin & Wagner (2019, *JF*) give the stock-level version: $E_t\frac{R_{i}-R_f}{R_f} = SVIX^2_t + \frac{1}{2}(SVIX^2_i - \overline{SVIX^2})$. **This is the most theoretically disciplined way to turn option prices into expected-return estimates**, and it is a good "long-horizon option" signal to test.

### 3.4 Model-free implied variance (CBOE VIX methodology)

- **Formula:** $\sigma^2_{MF} = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2}e^{rT}Q(K_i) - \frac{1}{T}\left(\frac{F}{K_0} - 1\right)^2$ (Demeterfi, Derman, Kamal & Zou 1999; Britten-Jones & Neuberger 2000; Jiang & Tian 2005).
  - $Q(K_i)$ is the OTM option midquote at strike $K_i$ (puts below $K_0$, calls above, the average at $K_0$).
  - $K_0$ is the first strike below the forward $F$.
  - $\Delta K_i$ is the strike spacing.
  - VIX interpolates two maturities to 30 days: $VIX = 100\sqrt{\sigma^2_{30}}$.
- For single stocks, sparse strikes require smile interpolation/extrapolation before integration. OptionMetrics' standardized surface is a pragmatic input.

### 3.5 Model-free implied skewness and kurtosis (Bakshi–Kapadia–Madan 2003)

- **Contracts** (BKM 2003, *RFS*): with $V$, $W$, $X$ the prices of variance, cubic and quartic contracts:
  - $V = \int_{S}^{\infty}\frac{2(1 - \ln(K/S))}{K^2}C(K)dK + \int_0^{S}\frac{2(1 + \ln(S/K))}{K^2}P(K)dK$
  - $W = \int_{S}^{\infty}\frac{6\ln(K/S) - 3\ln(K/S)^2}{K^2}C\,dK - \int_0^{S}\frac{6\ln(S/K) + 3\ln(S/K)^2}{K^2}P\,dK$
  - $X = \int_{S}^{\infty}\frac{12\ln(K/S)^2 - 4\ln(K/S)^3}{K^2}C\,dK + \int_0^{S}\frac{12\ln(S/K)^2 + 4\ln(S/K)^3}{K^2}P\,dK$
- **Moments:** with $\mu = e^{rT} - 1 - \frac{e^{rT}}{2}V - \frac{e^{rT}}{6}W - \frac{e^{rT}}{24}X$:
  - $SKEW = \frac{e^{rT}W - 3\mu e^{rT}V + 2\mu^3}{(e^{rT}V - \mu^2)^{3/2}}$
  - $KURT = \frac{e^{rT}X - 4\mu e^{rT}W + 6e^{rT}\mu^2V - 3\mu^4}{(e^{rT}V - \mu^2)^2}$
- **Cross-sectional evidence:** Conrad, Dittmar & Ghysels (2013, *JF*) find more negative (and higher-kurtosis) risk-neutral skew predicts *higher* returns, while higher skew predicts lower returns (the lottery analogue). Results depend on truncation and strike coverage, so use liquid names only.

### 3.6 Variance risk premium

- **Index level:** Carr & Wu (2009, *RFS*, "Variance Risk Premiums") find that synthetic variance-swap returns are strongly negative for indices, a priced premium for bearing variance risk. Bollerslev, Tauchen & Zhou (2009, *RFS* 22(11):4463–4492) use $VRP_t = IV^2_t - RV_t$ with model-free IV and 5-min realized variance. VRP explained more than 15% of the variation in quarterly excess market returns over 1990–2005 (more than 25% combined with P/E). Predictability peaks at the quarterly horizon and dominates P/E, the default spread and cay.\[24\] Later work (Bollerslev, Marrone, Xu & Zhou 2014) confirms it internationally.\[25\] **Use it for market-exposure/risk-budget timing, not stock selection.** Note that out-of-sample post-2009 performance has been weaker and noisy.
- **Stock level:** Han & Zhou (2011) and Bali & Hovakimian (2009, *Mgmt Sci*) relate the stock-level IV–RV spread to future stock returns. Goyal & Saretto (2009, *JFE* 94:310–326) sort on $HV - IV$ and find long-short positions in straddles and delta-hedged options earn economically large monthly returns. This is primarily an *option-return* predictor. For stock returns, Bali–Hovakimian find the RV–IV spread negatively related to returns and the call–put IV spread positively related.

### 3.7 Cross-sectional option-based stock predictors

1. **IV spread / put–call parity deviations** (Cremers & Weinbaum 2010, *JFQA* 45(2):335–367): $VS_{i,t} = \sum_j w_{j}(IV^{call}_{j} - IV^{put}_{j})$, averaged over matched strike/maturity pairs weighted by open interest. Per the paper's abstract, stocks with relatively expensive calls outperformed those with relatively expensive puts by 50 basis points per week. Predictability "decreases over the sample period", consistent with gradual elimination of mispricing. Changes in VS also predict.
2. **Volatility smirk** (Xing, Zhang & Zhao 2010, *JFQA* 45(3):641–662): $SKEW_{i,t} = IV^{OTM\,put}_{i,t} - IV^{ATM\,call}_{i,t}$ (OTM put: $K/S \approx 0.8$–0.95; ATM call: $K/S \approx 0.95$–1.05). Per the paper's abstract, stocks with the steepest smirks underperform those with the least pronounced smirks by 10.9% per year on a risk-adjusted basis. Predictability persists for at least 6 months, and steep-smirk firms have the worst earnings surprises next quarter.
3. **Implied–realized spread** (Bali & Hovakimian 2009; Goyal & Saretto 2009): $IV - RV$ (or $RV - IV$). High IV relative to RV predicts lower option returns, with a weaker stock-return link.
4. **Option-to-stock volume ratio** (Johnson & So 2012, *JFE* 106(2):262–286): $O/S_{i,w} = \frac{\text{Option volume (contracts × 100)}}{\text{Stock volume}}$. The lowest-O/S decile beat the highest by 0.34%/week (about 19.3% annualized). The mechanism is short-sale costs pushing negative-information traders into options. Predictability is short-lived: strategy returns "rapidly decline from 0.34% in the first week following portfolio formation and become statistically insignificant beyond the sixth week." Ge, Lin & Pearson (2016, *JFE*) show signed volume components that embed synthetic short positions drive it.
5. **Put–call ratio / signed option volume** (Pan & Poteshman 2006, *RFS*): the ratio of open-buy puts to open-buy puts plus open-buy calls predicts next-day and next-week returns (about 40 bp next day, 1%+ next week for extreme deciles).\[26\] It requires *signed, open/close-flagged* CBOE/ISE data. The public unsigned P/C ratio carries much less information.
6. **Open interest changes:** Fodor, Krieger & Doran (2011) find changes in call vs. put open interest predict returns. It is a weak-to-moderate signal and is measured with OptionMetrics `opprcd` OI (the prior-day timing convention matters).
7. **Term-structure slope** (Vasquez 2017, *JFQA*): $IV_{long} - IV_{short}$. The slope predicts option (straddle) returns positively. For stocks, an inverted term structure flags event or stress risk.
8. **IV around earnings:** the implied earnings-move $\sigma_{EA}^2 \approx \frac{IV^2_{T_2}T_2 - IV^2_{T_1}T_1}{1}$ is extracted by differencing variances across expiries bracketing the announcement (Dubinsky, Johannes, Kaeck & Seeger 2019, *RFS*). IV run-up before earnings and the crush after are well documented. Straddle returns around earnings are on average negative (earnings VRP) but vary cross-sectionally. It is useful as a *risk/size* input for the fast book and as a PEAD conditioning variable.
9. **Forward implied volatility:** $\sigma^2_{fwd}(T_1,T_2) = \frac{\sigma^2_2T_2 - \sigma^2_1T_1}{T_2 - T_1}$. It isolates expected variance over a future window and cleanly strips known events.
10. **Option-implied borrow fee:** from put–call parity with borrow cost $h$, $C - P = Se^{-(q+h)T} - Ke^{-rT}$, so $\hat h = -\frac{1}{T}\ln\frac{C - P + Ke^{-rT}}{S} - q$. Muravyev, Pearson & Pollet (JF 2022) show option-implied fees equal quoted fees on average and that the borrow-fee *risk premium* is small.

### 3.8 The critical reinterpretation: options signals ≈ borrow fees

Muravyev, Pearson & Pollet, "Why does options market information predict stock returns?" (*JFE*, October 2025), derive a formula linking the IV spread to the borrow fee. They show "return predictability from implied volatility spread and skew decreases by at least two-thirds if high-fee stocks are excluded."\[2\]\[27\] Gies College of Business coverage of the paper (January 2026) reports that net-of-fee returns on several options-based anomalies are close to zero, and the SSRN version states that "the predictability decreases by a similar amount after stock returns are adjusted for borrow fees." This does not make options data useless. But it implies:

- Options signals are largely *short-side* signals in hard-to-borrow names, where your net return depends on your borrow cost and availability.
- For a long-biased or long-short book with realistic borrow, expect a fraction of published option-signal alphas.
- Use them to *avoid* longs in stocks with steep skew or high O/S (a "do-no-harm" long filter), and as inputs to risk/event models, more than as a standalone long-short alpha.

### 3.9 Volatility surface parameterization and no-arbitrage

- **SVI** (Gatheral 2004): total implied variance $w(k) = a + b\left(\rho(k - m) + \sqrt{(k-m)^2 + \sigma^2}\right)$, with log-moneyness $k = \ln(K/F)$, level $a$, wing slope $b \ge 0$, rotation/skew $|\rho| < 1$, translation $m$, and ATM curvature $\sigma > 0$.
- **SSVI** (Gatheral & Jacquier 2014, *Quant. Finance*, "Arbitrage-free SVI volatility surfaces"): $w(k, \theta_t) = \frac{\theta_t}{2}\left(1 + \rho\varphi(\theta_t)k + \sqrt{(\varphi(\theta_t)k + \rho)^2 + 1 - \rho^2}\right)$. $\theta_t$ is ATM total variance and $\varphi$ a curvature function, e.g., power-law $\varphi(\theta) = \eta\theta^{-\lambda}$.
- **No-arbitrage conditions:**
  - No calendar-spread arbitrage: $\partial_t w(k,t) \ge 0$.
  - No butterfly arbitrage, i.e., a non-negative density: $g(k) = \left(1 - \frac{kw'}{2w}\right)^2 - \frac{w'^2}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{w''}{2} \ge 0$.
  - Sufficient SSVI conditions: $\theta\varphi(\theta)(1 + |\rho|) < 4$ and $\theta\varphi(\theta)^2(1 + |\rho|) \le 4$.
- **Use:** fit SSVI per stock per day (liquid names), then extract features such as ATM level $\theta$, skew $\rho\varphi$, curvature and term slope. These are *smoothed, arbitrage-free* versions of the raw XZZ/CW signals, and they are much less noisy.

### 3.10 American options, dividends, and implied forwards

- US single-stock options are American. Early exercise matters for deep ITM puts and calls before ex-dividend dates. OptionMetrics computes IVs with a binomial tree and projected discrete dividends. For model-free integrals use **OTM options only** (little early-exercise premium). Or de-Americanize: compute American IV, then price a European option at that IV (Burkovska et al. 2018).
- **Implied forward from put–call parity (European):** $F = K + e^{rT}(C(K) - P(K))$ at the strike where $|C - P|$ is minimal (CBOE method). For American options, parity becomes bounds: $S - K \le C - P \le S - Ke^{-rT}$ (no dividends). Use the implied forward to absorb dividends and borrow fees consistently rather than assuming $q$.

### 3.11 WRDS OptionMetrics implementation notes

- IvyDB US covers "all US exchange-listed and NASDAQ equities and market indices, as well as all US-listed index and equity options, starting from January 1996."
- **Key tables:**
  - `opprcd` (option prices, IV, greeks, OI, volume)
  - `vsurfd` (standardized surface)
  - `stdopd` (standardized ATM options)
  - `secprd` (underlying prices)
  - `distrd` (dividends)
  - `zerocd` (zero curve)
  - `fwdprd` (forward prices)
- **Identifier:** SECID. WRDS provides a historical SECID↔PERMNO link table with a match score (0 = best to 6; the OCLINK macro). Use score ≤ 1.
- **Timing:** OptionMetrics' end-of-day quotes are closing (≈4:00 pm) best bid/offer, while stocks close at 4:00 pm. Stale-quote asynchronicity can create spurious IV-spread predictability, so **lag option signals one day** before trading.

---

## Chapter 4 — Volatility and Probabilistic Return-Distribution Modeling

Volatility is far more predictable than returns. This chapter supplies the "probabilistic" half of the system: forecast distributions for sizing, risk, stop/gating and option-signal normalization.

### 4.1 GARCH family

- **GARCH(1,1)** (Bollerslev 1986): $r_t = \mu + \epsilon_t$, $\epsilon_t = \sigma_t z_t$, $\sigma^2_t = \omega + \alpha\epsilon^2_{t-1} + \beta\sigma^2_{t-1}$.
  - $\omega > 0$ is the constant, $\alpha$ the ARCH (shock) coefficient and $\beta$ the persistence.
  - Stationarity requires $\alpha + \beta < 1$. The unconditional variance is $\bar\sigma^2 = \omega/(1 - \alpha - \beta)$ and the shock half-life is $\ln 0.5/\ln(\alpha + \beta)$. Typical daily equity estimates are $\alpha \approx 0.05$–0.10 and $\beta \approx 0.88$–0.94.
- **GJR-GARCH** (Glosten, Jagannathan & Runkle 1993, *JF*): $\sigma^2_t = \omega + (\alpha + \gamma I_{t-1})\epsilon^2_{t-1} + \beta\sigma^2_{t-1}$, with $I_{t-1} = 1$ if $\epsilon_{t-1} < 0$.
  - $\gamma > 0$ captures the **leverage effect**: negative returns raise future volatility more than positive ones (Black 1976; Christie 1982). Volatility feedback is an alternative explanation.
  - Persistence is $\alpha + \beta + \gamma/2$ (symmetric $z$).
- **EGARCH** (Nelson 1991, *Econometrica*): $\ln\sigma^2_t = \omega + \beta\ln\sigma^2_{t-1} + \alpha(|z_{t-1}| - E|z|) + \gamma z_{t-1}$. It needs no positivity constraints and $\gamma < 0$ gives the leverage effect.
- **Realized GARCH** (Hansen, Huang & Shek 2012) adds a measurement equation linking RV to $\sigma^2_t$. GARCH-MIDAS (Engle, Ghysels & Sohn 2013) adds a macro-driven slow component.
- **Tooling:** Python `arch` (Sheppard) implements all of these.

### 4.2 HAR-RV (Corsi 2009)

- **Formula:** $RV_{t+1}^{(d)} = \beta_0 + \beta_d RV_t^{(d)} + \beta_w RV_t^{(w)} + \beta_m RV_t^{(m)} + \epsilon_{t+1}$. $RV^{(w)}_t = \frac{1}{5}\sum_{j=0}^{4}RV_{t-j}$ and $RV^{(m)}_t = \frac{1}{22}\sum_{j=0}^{21}RV_{t-j}$.
- It is a simple OLS "cascade" of heterogeneous-horizon traders that mimics long memory and is hard to beat.
- **Extensions:**
  - HAR-J / HAR-CJ separate jumps (Andersen, Bollerslev & Diebold 2007).
  - SHAR uses semivariances $RS^+$, $RS^-$ (Patton & Sheppard 2015).
  - HARQ corrects for measurement error via realized quarticity (Bollerslev, Patton & Quaedvlieg 2016).
  - Log-HAR, and adding option IV as a regressor (improves forecasts: Busch, Christensen & Nielsen 2011).
- **Cross-sectional panel HAR** with pooled coefficients works well for thousands of stocks (Bollerslev, Hood, Huss & Pedersen 2018).

### 4.3 Realized measures from TAQ

- **Realized variance** from 5-min returns balances discretization against microstructure noise (the volatility signature plot; Andersen, Bollerslev, Diebold & Labys 2001/2003).
- **Realized kernels** (Barndorff-Nielsen, Hansen, Lunde & Shephard 2008, *Econometrica*): $RK = \gamma_0 + \sum_{h=1}^{H}k\left(\frac{h-1}{H}\right)(\gamma_h + \gamma_{-h})$, with $\gamma_h = \sum_j r_j r_{j-h}$ and a Parzen kernel $k$. They are noise-robust.
- **Other estimators:** pre-averaging (Jacod et al. 2009); two-scale RV (Zhang, Mykland & Aït-Sahalia 2005); bipower variation $BV = \frac{\pi}{2}\sum|r_j||r_{j-1}|$ (jump-robust), with jumps given by $\max(RV - BV, 0)$.
- **Data:** WRDS TAQ daily product (09/10/2003–present, millisecond) and monthly TAQ (1993–2014, second resolution). WRDS provides TAQ–CRSP links and pre-computed intraday indicators (WRDS IID) including realized variance and spreads, which saves heavy compute.

### 4.4 Stochastic volatility and jumps

- **Heston (1993):** $dS = \mu S dt + \sqrt{v}S dW_1$, $dv = \kappa(\theta - v)dt + \xi\sqrt{v}dW_2$, $d\langle W_1, W_2\rangle = \rho dt$. $\kappa$ is the mean-reversion speed, $\theta$ the long-run variance, $\xi$ the vol-of-vol and $\rho < 0$ the leverage.
- **Merton (1976) jump diffusion:** $\frac{dS}{S} = (\mu - \lambda k)dt + \sigma dW + (J - 1)dN$, with jump intensity $\lambda$, $\ln J \sim N(\mu_J, \sigma_J^2)$ and $k = E[J - 1]$. Bates (1996) combines SV and jumps. Discrete-time SV via MCMC/particle filters (Jacquier, Polson & Rossi 1994) gives full posterior predictive distributions.
- **Rough volatility** (Gatheral, Jaisson & Rosenbaum 2018, *Quant. Finance*): log-vol behaves like fractional Brownian motion with $H \approx 0.1$. Simple rough forecasts are competitive with HAR.

### 4.5 Heavy tails and skewed distributions

Standardized residuals remain fat-tailed. Use Student-t, Hansen's (1994) skewed-t, NIG or GED innovations. Tail risk can be modelled with EVT (peaks-over-threshold, GPD) on standardized residuals (McNeil & Frey 2000). For single-stock daily returns, $\nu \approx 3$–6 degrees of freedom is typical, with strong earnings-day jumps. **Model earnings days separately**, e.g., an event-day variance drawn from the option-implied earnings move.

### 4.6 Regime switching and HMMs

- **Hamilton (1989) Markov switching:** $r_t = \mu_{s_t} + \sigma_{s_t}\epsilon_t$, $P(s_t = j | s_{t-1} = i) = p_{ij}$. Estimate by EM/Hamilton filter; filtered probabilities $P(s_t | \mathcal{F}_t)$ are the usable (non-look-ahead) quantities. Never use smoothed probabilities in backtests.
- **Applications:** market-regime detection (calm/stress) to scale gross exposure, switch momentum risk (the Daniel–Moskowitz panic state), and choose between model sets. Newer "statistical jump models" (Nystrup et al. 2020; Shu, Yu & Mulvey 2024) penalize regime switching for more persistent, less noisy regimes.

### 4.7 Quantile regression and distributional forecasting

- **Quantile regression** (Koenker & Bassett 1978): $\hat\beta_\tau = \arg\min_\beta\sum_t\rho_\tau(y_t - x_t'\beta)$ with the check (pinball) loss $\rho_\tau(u) = u(\tau - \mathbb{1}\{u < 0\})$.
  - "Growth-at-risk" style conditional quantiles (Adrian, Boyarchenko & Giannone 2019) apply directly to equity returns.
  - LightGBM supports `objective='quantile'`, so you can train a family of $\tau \in \{0.05, 0.25, 0.5, 0.75, 0.95\}$ models inside Qlib. Enforce non-crossing by sorting or by monotone composite quantile nets.
- **Other distributional models:** distributional regression (GAMLSS), NGBoost (Duan et al. 2020), mixture-density networks and deep quantile nets. For cross-sectional ranking, distributional outputs add risk-adjusted scores such as $\hat\mu/\hat\sigma$ or the downside quantile.

### 4.8 Conformal prediction for finance

- **Split conformal:** on a calibration set, compute scores $s_i = |y_i - \hat y_i|$ and the $\lceil(n+1)(1-\alpha)\rceil/n$ quantile $\hat q$. The interval $[\hat y - \hat q, \hat y + \hat q]$ has finite-sample marginal coverage $\ge 1 - \alpha$ under exchangeability.
- **Conformalized quantile regression** (Romano, Patterson & Candès 2019) adapts interval width.
- Financial data are not exchangeable, so use **adaptive conformal inference** (Gibbs & Candès 2021): $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \mathrm{err}_t)$, which guarantees long-run coverage under distribution shift. Weighted conformal (Barber et al. 2023) is another option.
- **Uses:** calibrated risk bands for position sizing, abstention (trade only when the interval excludes zero or the signal-to-width ratio is high), and monitoring (coverage breaches signal model decay).

### 4.9 Bayesian approaches

- Bayesian shrinkage of expected returns toward priors: Black–Litterman (1992), $\mu_{BL} = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\pi + P'\Omega^{-1}q]$, where $\pi$ is the equilibrium prior, $P$ the view matrix, $q$ the views and $\Omega$ the view uncertainty.
- Hierarchical/empirical-Bayes shrinkage of factor premia (JKP 2023; Kozak–Nagel–Santosh).
- Bayesian model averaging over signals.
- Posterior predictive distributions from Bayesian SV models.
- **Practical recommendation:** apply empirical-Bayes shrinkage of each signal's IC toward its theme mean (JKP-style) before combination.

### 4.10 Combining physical-model forecasts with option-implied densities

- **Volatility:** regress $RV_{t+h}$ on HAR terms plus $IV^2_t$ ("encompassing"). IV is typically biased upward (the VRP) but informative, so the optimal combination uses both.
- **Densities:**
  - Linear opinion pool $p = w\,p^{P\text{-model}} + (1 - w)\,\tilde p^{Q\to P}$.
  - Logarithmic pool $p \propto (p^{P})^{w}(\tilde p^{Q\to P})^{1-w}$.
  - Weights chosen by CRPS or log score on rolling windows.
  - Or treat option-implied moments (IV, skew, kurtosis, implied earnings move) as regressors in quantile/GARCH-X models.
- **Events:** blend the option-implied jump distribution for known events (earnings) with a physical diffusive component.

### 4.11 Forecast evaluation

- **QLIKE** (robust to noisy variance proxies; Patton 2011, *J. Econometrics*): $L(\hat\sigma^2, RV) = \frac{RV}{\hat\sigma^2} - \ln\frac{RV}{\hat\sigma^2} - 1$. Also MSE on variance. Compare models with Diebold–Mariano tests or the Model Confidence Set (Hansen, Lunde & Nason 2011).
- **Pinball loss** for quantiles (above), and **CRPS** for full distributions: $CRPS(F, y) = \int_{-\infty}^{\infty}(F(z) - \mathbb{1}\{z \ge y\})^2dz = E|X - y| - \frac{1}{2}E|X - X'|$.
- **Calibration:** PIT histograms $u_t = F_t(y_t)$ should be i.i.d. U(0,1) (Berkowitz 2001 test). For VaR, use Kupiec/Christoffersen backtests.

---

## Chapter 5 — Machine Learning and Modern Methods

### 5.1 Gu–Kelly–Xiu (2020): the benchmark

"Empirical Asset Pricing via Machine Learning" (*RFS* 33(5):2223–2273) uses about 30,000 US stocks from 1957–2016 with 94 characteristics × 8 macro variables plus 74 industry dummies (900+ predictors).\[28\]\[29\]

- **Findings:**
  - OLS with all predictors overfits badly (negative out-of-sample R²).
  - Penalized/dimension-reduced linear models recover a modest R², and trees and neural nets reach roughly 0.33–0.40% monthly stock-level out-of-sample R².\[28\]
  - A value-weighted long-short decile portfolio on NN forecasts earns an annualized out-of-sample Sharpe of 1.35 (2.45 equal-weighted). S&P 500 timing with NN forecasts earns Sharpe 0.77 vs. 0.51 buy-and-hold.\[28\]\[30\]
  - Dominant predictors are price trends (momentum, reversal), liquidity and volatility.
- **Caveats for practice:** returns are *gross* of costs, equal-weighted results are microcap-heavy, and turnover is high. Later work (Avramov, Cheng & Metzker 2023, *Mgmt Sci*, "Machine Learning vs. Economic Restrictions") shows ML profits concentrate in distressed, illiquid and hard-to-arbitrage stocks and shrink substantially after excluding microcaps and applying costs. **Expect ML to improve IC by perhaps 20–50% over a good linear composite, not to multiply it.**

### 5.2 Conditional factor models: IPCA and autoencoders

- **IPCA** (Kelly, Pruitt & Su 2019, *JFE*): $r_{i,t+1} = \alpha_{i,t} + \beta_{i,t}'f_{t+1} + \epsilon_{i,t+1}$ with $\beta_{i,t} = z_{i,t}'\Gamma_\beta$ and $\alpha_{i,t} = z_{i,t}'\Gamma_\alpha$. $z_{i,t}$ is the $L$-vector of characteristics and $\Gamma_\beta$ is $L\times K$. It is estimated by alternating least squares. With $K = 4$–5 latent factors, characteristics predict returns mostly through *betas* (compensated risk), and $\Gamma_\alpha = 0$ is often not rejected. It produces strong out-of-sample tangency Sharpe ratios.
- **Conditional autoencoder** (Gu, Kelly & Xiu 2021, *J. Econometrics*, "Autoencoder Asset Pricing Models"): the beta network $\beta_{i,t} = g(z_{i,t})$ is a nonlinear NN of characteristics, and the factor network builds $f_{t+1}$ from characteristic-managed portfolios. It beats IPCA out-of-sample.
- **Also relevant:**
  - Chen, Pelger & Zhu (2024, *Mgmt Sci*), "Deep Learning in Asset Pricing": a GAN-style SDF with no-arbitrage moment conditions and LSTM macro states.
  - Kelly, Malamud & Zhou (2024, *JF*), "The Virtue of Complexity in Return Prediction": random-feature ridge regressions with more parameters than observations improve out-of-sample market timing. This is debated (Buncic 2024; Nagel 2025 critiques).
  - Kelly, Kuznetsov, Malamud & Xu (2025, NBER WP 33351), "Artificial Intelligence Asset Pricing Models": transformer-based cross-asset attention.
- **Use in your system:** IPCA or an autoencoder serves as the *risk model* and as a *beta-vs-alpha decomposition* of your characteristic signals.

### 5.3 Gradient boosting: the workhorse

LightGBM/XGBoost/CatBoost on engineered features remain the most reliable tabular learners. They handle missing values, monotone constraints and interactions, and they train quickly for rolling retraining. Recommended practice:

- Cross-sectional rank-normalize features and labels per date.
- Train on a residualized or rank label.
- Use early stopping on a purged validation fold.
- Use Huber/rank objectives (LambdaRank for top-k selection).
- Ensemble across seeds and time windows.

### 5.4 The Qlib model zoo: what the published benchmarks show

Qlib's benchmark README reports mean ± std over 20 random seeds on **China A-shares (CSI300)**, forecasting daily excess returns, with a top-k dropout strategy.

**Alpha158 (158 hand-engineered price/volume features), CSI300:**

| Model | IC | ICIR | Rank IC | Ann. excess return | IR | Max DD |
|---|---|---|---|---|---|---|
| DoubleEnsemble (LGBM base) | 0.0521 | 0.4223 | 0.0502 | 11.58% | 1.34 | −9.2% |
| LightGBM | 0.0448 | 0.3660 | 0.0469 | 9.01% | 1.02 | −10.4% |
| MLP | 0.0376 | 0.2846 | 0.0429 | 8.95% | 1.14 | −11.0% |
| XGBoost | 0.0498 | 0.3779 | 0.0505 | 7.80% | 0.91 | −11.7% |
| TRA | 0.0440 | 0.3535 | 0.0540 | 7.18% | 1.08 | −7.6% |
| Linear | 0.0397 | 0.3000 | 0.0472 | 6.92% | 0.92 | −15.1% |
| ALSTM (20 feats) | 0.0362 | 0.2789 | 0.0463 | 4.70% | 0.70 | −10.7% |
| Transformer | 0.0264 | 0.2053 | 0.0407 | 2.73% | 0.40 | −11.0% |

**Alpha360 (raw 60-day price/volume windows), CSI300:**

| Model | IC | Rank IC | Ann. excess return | IR |
|---|---|---|---|---|
| HIST (concept-graph) | 0.0522 | 0.0667 | 9.87% | 1.37 |
| IGMTF | 0.0480 | 0.0606 | 9.46% | 1.35 |
| TRA | 0.0485 | 0.0587 | 9.20% | 1.28 |
| TCTS (GRU base) | 0.0508 | 0.0599 | 8.93% | 1.23 |
| GATs | 0.0476 | 0.0598 | 8.24% | 1.11 |
| GRU | 0.0493 | 0.0584 | 7.20% | 0.97 |
| ALSTM | 0.0497 | 0.0599 | 6.26% | 0.87 |
| LightGBM | 0.0400 | 0.0499 | 5.58% | 0.76 |
| Transformer | 0.0114 | 0.0327 | −2.70% | −0.34 |

**Interpretation:**

1. On *engineered* features, boosted trees (and DoubleEnsemble, a sample-reweighting plus feature-selection ensemble of LightGBMs; Zhang et al. 2020, ICDM) are best, and a *linear* model is close behind. Feature engineering matters more than architecture.
2. On *raw* sequences, temporal/relational deep nets (HIST, TRA, IGMTF, GATs) beat trees.
3. Vanilla Transformers underperform.
4. These are **A-share results**. The Chinese market is retail-dominated with strong short-term reversal and different microstructure, so US ICs will be lower and the ranking of models may differ. Treat the table as a *relative* guide and re-benchmark on your US data.

**Model notes:**

- **ALSTM:** attention-LSTM (Qin et al. 2017).
- **TRA** (Temporal Routing Adaptor; Lin et al. 2021, KDD): multiple predictors plus a router that assigns samples to latent trading "patterns" via optimal transport, a regime-aware mixture of experts.
- **HIST** (Xu et al. 2021): a graph over predefined concepts (industries/themes) and hidden concepts, separating shared from idiosyncratic information.
- **TCTS:** temporally correlated task scheduling (auxiliary horizons).
- **AdaRNN:** distribution-shift-aware RNN.

### 5.5 Graph neural networks for stock relations

Relations can come from industry/supply-chain links (Cohen & Frazzini 2008, "Economic Links": customer returns predict supplier returns monthly), analyst co-coverage (Ali & Hirshleifer 2020, *JFE*: co-coverage momentum subsumes industry and customer momentum), common institutional ownership (13F), text similarity (Hoberg–Phillips TNIC industries) and return correlation. Models include GAT/GCN (HIST, IGMTF in Qlib), temporal relational ranking (Feng et al. 2019) and dynamic graphs. **The economic lead-lag relations** (customer→supplier, co-coverage, TNIC peers) are documented, interpretable signals in their own right. Build them as features first, then let a GNN learn the rest.

### 5.6 Concept drift and adaptation

- **DDG-DA** (Li et al. 2022, AAAI; in Qlib): learns to predict the future data distribution and resamples/reweights training data accordingly. Qlib reports gains over rolling retraining on the A-share benchmarks.
- **Other approaches:**
  - Rolling retraining with exponentially decaying sample weights.
  - Meta-learning.
  - Regime-conditional models (TRA).
  - Online ensembles with performance-based weights (exponentiated gradient).
- **Monitoring:** feature-distribution drift (PSI, KS), rolling IC decay, and conformal coverage breaches.

### 5.7 NLP and LLM signals

- **Classic textual finance:**
  - Tetlock (2007, *JF*) media pessimism.
  - Loughran & McDonald (2011, *JF*) finance-specific sentiment dictionaries.
  - Cohen, Malloy & Nguyen (2020, *JF*), "Lazy Prices": year-over-year changes in 10-K/10-Q language (low similarity) predict negative returns, with very slow diffusion. This is a strong slow-book text signal from EDGAR.
  - Ke, Kelly & Xiu (2019), SESTM supervised sentiment.
- **LLM-era:**
  - Lopez-Lira & Tang ("Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models", *JFE* forthcoming; arXiv 2304.07619) score post-knowledge-cutoff headlines with GPT-4. They achieve about 90% portfolio-day hit rates for the *non-tradable* initial reaction. GPT-4 scores also "significantly predict the subsequent drift, especially for small stocks and negative news." Forecasting ability rises with model size, and strategy returns "decline as LLM adoption rises."\[31\]
  - The critical methodological point: **look-ahead contamination**. An LLM trained on data after the test period "knows" outcomes. Use models with training cutoffs strictly before test periods, or point-in-time/"time-locked" models, and anonymize entities where possible (Sarkar & Vafa 2024; Glasserman & Lin 2023 "distraction effect").
  - Embeddings of earnings-call transcripts (e.g., FinBERT or LLM embeddings fed to boosted trees) and LLM-extracted structured facts (guidance changes, risk-factor changes) are promising.
- **Data:** RavenPack on WRDS (Dow Jones edition from January 2000; entity-level relevance, sentiment and novelty scores); EDGAR full-text filings (free); earnings call transcripts (S&P Capital IQ Transcripts on WRDS, if licensed).

### 5.8 LLM-agent factor mining (RD-Agent / R&D-Agent-Quant)

Microsoft's R&D-Agent-Quant (Li et al., NeurIPS 2025 Datasets & Benchmarks; arXiv 2505.15155) is a multi-agent loop with Research (hypothesis generation), Development (the Co-STEER code agent implements factors/models and runs Qlib backtests) and Feedback (a multi-armed bandit scheduler). It reports "up to 2× higher annualized returns than classical factor libraries using 70% fewer factors." Secondary summaries cite IC 0.0532 and annualized return 14.21% on CSI300 for joint factor–model optimization.\[32\]\[33\] It plugs natively into Qlib.

**Caveat, and it is large:** agentic search is industrialized multiple testing. Every hypothesis the agent tries is a trial for the DSR/PBO calculation. Its reported results are on Chinese data with standard Qlib cost assumptions. Use it as an idea generator inside a *locked* validation protocol: the agent sees train and validation only, with a sealed holdout period and preregistered acceptance criteria.

### 5.9 Alternative data

Candidates include web traffic, app usage, credit-card panels, satellite imagery, job postings, patents (Kogan et al. 2017), supply-chain data (FactSet Revere), ESG controversies, Google search volume (Da, Engelberg & Gao 2011, "In Search of Attention"), options flow and retail order flow (Boehmer, Jones, Zhang & Zhang 2021, "Tracking Retail Investor Activity" via TAQ sub-penny trade prints: retail net buying predicts returns over weeks). **The retail order-imbalance measure is implementable from TAQ alone** and is a good fast-book candidate. Be aware that the post-2021 rule changes and payment-for-order-flow dynamics affect it.

### 5.10 Triple barrier, meta-labeling, and sample weights (López de Prado)

- **Triple-barrier labeling:** for an event at $t_0$, label $+1$ if price hits the upper barrier $P_{t_0}(1 + k_u\hat\sigma_{t_0})$ first, $-1$ if it hits the lower $P_{t_0}(1 - k_l\hat\sigma_{t_0})$ first, and $0$ or the sign of the return at the vertical barrier $t_0 + h$. Barriers scale with forecast volatility.
- **Meta-labeling:** a primary model gives the *side*. A secondary classifier predicts $P(\text{primary is correct} | x)$ and is used for *sizing*: $m = 2\Phi(z) - 1$ with $z = \frac{p - 0.5}{\sqrt{p(1-p)}}$. It improves precision and lets you size by confidence.
- **Other tools:** sample uniqueness weights for overlapping labels, fractional differentiation (stationarity with memory), and CUSUM event sampling.
- **Fit for this system:** triple-barrier and meta-labeling are most useful for the *event-driven* part of the fast book (earnings, news, options-flow events) rather than for daily cross-sectional ranking, where rank-IC objectives are more natural.

### 5.11 Alpha combination and ensembling

- **Methods:**
  - Equal-weighted z-score composites (hard to beat).
  - IC-weighted: $w \propto \Sigma_{IC}^{-1}\overline{IC}$, i.e., mean-variance on signal ICs with shrinkage.
  - Regression-stacked (ridge on signal portfolios, i.e., "characteristic-managed portfolios").
  - Nonlinear stacking (GBM on signals plus conditioning variables, e.g., regime).
- **Rules:**
  1. Orthogonalize to risk factors first.
  2. Shrink weights toward equal.
  3. Combine *before* optimization (one alpha per stock) and let the optimizer handle costs. Netting trades across signals is the single biggest cost saver (DeMiguel et al. 2020).
  4. Horizon-align signals by decay: scale each signal's forecast to the rebalancing horizon.

---

## Chapter 6 — Portfolio Construction, Risk and Execution

### 6.1 Risk models

- **Fundamental (Barra-style):** $r_{i,t} = \sum_k X_{i,k,t}f_{k,t} + u_{i,t}$ with exposures $X$ (industries, style z-scores: size, beta, momentum, value, volatility, liquidity, leverage, growth). Factor returns $f_t$ come from daily WLS cross-sectional regressions (weights $\propto\sqrt{ME}$).
  - Factor covariance: EWMA with half-lives around 90 days for volatility and 180+ days for correlation, a Newey–West adjustment, and eigenfactor and volatility-regime adjustments.
  - Specific risk: EWMA of residuals with Bayesian shrinkage by size.
- **Statistical (PCA):** $\Sigma = V_K\Lambda_KV_K' + D$ from the leading $K$ eigenvectors of the return covariance. It captures unnamed transient factors, and it is useful to run both models and constrain exposure to the union.
- **IPCA/autoencoder** conditional betas (Chapter 5) are the modern option.
- **Shrinkage covariance** (Ledoit & Wolf 2004, *J. Multivariate Analysis*, "A well-conditioned estimator…"): $\hat\Sigma = \delta F + (1 - \delta)S$, shrinking the sample covariance $S$ toward a structured target $F$ (constant correlation or identity) with analytically optimal $\delta$. Nonlinear shrinkage (Ledoit & Wolf 2017/2020) is better for large $N$. Use it for the specific/residual block or when $N$ is comparable to $T$.

### 6.2 Factor neutralization

- **Signal level:** residualize alpha on industry plus style exposures (Chapter 1).
- **Portfolio level:** constraints $|X'w| \le \epsilon$ plus beta-neutrality $\beta'w = 0$.
- Neutralize to *unintended* exposures (industry, size, beta, and momentum for a value book) while keeping *intended* ones. Over-neutralization kills legitimate alpha when a signal's value lies partly in its factor tilt (e.g., quality's low-beta tilt).

### 6.3 Mean–variance with transaction costs

- **Objective:** $\max_{w}\ \alpha'w - \frac{\lambda}{2}w'\Sigma w - \kappa\sum_i\left[c_i|w_i - w_i^0| + \eta_i\frac{\sigma_i}{\sqrt{ADV_i}}|w_i - w_i^0|^{3/2}\right]$.
  - The linear term covers spread and commission.
  - The $3/2$-power term is the square-root-impact total cost (per-dollar impact $\propto\sqrt{\text{trade}}$).
  - Constraints: budget, neutrality, position limits ($|w_i| \le$ min(x% NAV, y% ADV)), and a gross-leverage cap.
- **Solvers:** CVXPY with power-cone constraints, or MOSEK. Qlib's `EnhancedIndexingStrategy` uses a CVXPY optimizer and can be extended.

### 6.4 Hierarchical risk parity (HRP)

López de Prado (2016, *JPM*):

1. Tree-cluster assets by correlation distance $d_{ij} = \sqrt{\frac{1}{2}(1 - \rho_{ij})}$.
2. Quasi-diagonalize the covariance.
3. Recursively bisect, allocating inversely to cluster variance.

HRP needs no matrix inversion and is robust out-of-sample for *risk allocation*. It is best used to allocate across **sleeves/signals** (e.g., between fast and slow books, or signal families) rather than across 3,000 stocks with an alpha view.

### 6.5 Combining short- and long-horizon signals: Gârleanu–Pedersen

Gârleanu & Pedersen (2013, *JF*, "Dynamic Trading with Predictable Returns and Transaction Costs") use quadratic costs $\frac{1}{2}\Delta x'\Lambda\Delta x$ with $\Lambda \propto \Sigma$ and signals $f^k$ with mean-reversion rates $\phi_k$.

- **Optimal policy:** $x_t = (1 - \frac{a}{\lambda})x_{t-1} + \frac{a}{\lambda}\,\text{aim}_t$. Trade partially toward an *aim portfolio*, a weighted average of the current Markowitz portfolio and expected future ones.
- **Weighting of signals:** in the aim portfolio, each signal is weighted by $\frac{1}{1 + \phi_k a/\gamma}$, where $\gamma$ is risk aversion and $a$ solves a quadratic in $\lambda$, $\gamma$ and $\rho$ (the discount rate). **Fast-decaying signals receive lower weight** relative to their raw Markowitz weight, and persistent signals receive more. The trading rate $a/\lambda$ falls as costs rise.
- **Practical translation:**
  1. Express each signal as an expected-return forecast over its own horizon, with an estimated half-life from the IC decay curve.
  2. Build the aim portfolio as a decay-weighted blend.
  3. Move a fraction $\tau = a/\lambda$ (e.g., 10–30% per day for the combined book) toward it.
- Extensions: Collin-Dufresne, Daniel & Sağlam (2020) handle regime-dependent costs, and Kakushadze-style multi-alpha frameworks are another option.

### 6.6 Multi-book design: fast book vs. slow book

- **Why separate:**
  - Signals differ by an order of magnitude in half-life (reversal: days; value: years).
  - Their cost sensitivities, capacities and risk behaviours (reversal pays in volatility spikes; momentum crashes then) are different.
  - Keeping sleeves separate makes attribution, risk budgets and kill-switches clean.
- **Why integrate at execution:** netting. A reversal sell and a value buy on the same name should cross internally. The best practice is separate alpha research and risk budgets per book, but *one* combined optimizer and trade list per day, with sleeve-level risk contribution constraints.
- **Risk-budgeting across books:** the allocation between books can be set by HRP/risk parity on sleeve returns, adjusted by forecast Sharpe and capacity. Rebalance the allocation monthly.

### 6.7 Position sizing: Kelly and volatility targeting

- **Fractional Kelly on the portfolio:** $w = c\cdot\Sigma^{-1}\hat\mu$ with $c \in [0.25, 0.5]$, justified by parameter uncertainty (MacLean, Thorp & Ziemba 2011). Under estimation error, half-Kelly sacrifices about 25% of growth for about 50% less variance.
- **Volatility targeting:** scale gross exposure by $\sigma^*/\hat\sigma_{p,t}$ with $\hat\sigma_{p,t}$ from the risk model or HAR/GARCH. Moreira & Muir (2017, *JF*) show volatility-managed factor portfolios raise Sharpe ratios. Cederburg, O'Doherty, Wang & Yan (2020, *JFE*) and Barroso & Detzel (2021, *JFE*) show out-of-sample and after-cost gains are unreliable except for the market and momentum. **Use vol-targeting at the book level as risk control and for momentum; do not assume it creates alpha elsewhere.**
- **Drawdown control:** Grossman–Zhou (1993) style exposure reduction as drawdown approaches a limit, plus hard kill-switches per book.

### 6.8 Execution and market impact: Almgren–Chriss

Almgren & Chriss (2000, *J. Risk*) liquidate $X$ shares over $T$ in $N$ intervals.

- **Price dynamics:** permanent impact $g(v) = \gamma v$ and temporary impact $h(v) = \epsilon\,\mathrm{sgn}(v) + \eta v$.
- **Objective:** minimize $E[\text{cost}] + \lambda\,\mathrm{Var}[\text{cost}]$. The optimal trajectory is $x_j = X\frac{\sinh(\kappa(T - t_j))}{\sinh(\kappa T)}$ with $\kappa \approx \sqrt{\lambda\sigma^2/\eta}$. Higher risk aversion front-loads trading.
- **Practical rules:**
  - Implement via VWAP/TWAP or POV algos, capping participation around 5–10% ADV.
  - For the slow book, spread trades over days.
  - For the fast book, trade at the open/close auctions or with passive limit orders where signals allow.
  - Calibrate cost models from your own fills (implementation shortfall) and from TAQ (effective spreads, Kyle's lambda).

### 6.9 Capacity

- **Break-even capacity:** the AUM $A^*$ at which expected alpha equals expected costs, $\alpha(A^*) = TO\times c(A^*)$ with $c(A) \propto \sigma\sqrt{A\cdot w/ADV}$.
- Fast books have capacity in the tens to low hundreds of millions for a US small/mid universe. Value/momentum sleeves scale to billions (Frazzini–Israel–Moskowitz).\[5\] For an individual or small fund, capacity is not the binding constraint. *Costs and borrow at small size* are, particularly retail borrow availability for short legs.

---

## Chapter 7 — Pitfalls and Validation Methodology

### 7.1 Backtest overfitting

- **Probability of Backtest Overfitting** (Bailey, Borwein, López de Prado & Zhu 2017, *J. Computational Finance*):
  1. Split the performance matrix of $N$ configurations × $T$ periods into $S$ blocks.
  2. For every combination of $S/2$ in-sample blocks (CSCV), pick the best in-sample configuration and record its out-of-sample relative rank $\bar\omega$.
  3. Compute the logit $\lambda = \ln\frac{\bar\omega}{1 - \bar\omega}$. $PBO = P(\lambda \le 0)$.
  - PBO above about 0.5 means the selection process is no better than random.
- **Deflated Sharpe** (Chapter 1) and **minimum backtest length:** $MinBTL \approx \frac{2\ln N}{E[\max SR]^2}$ years. With 1,000 trials you need many years of data for an annualized SR around 1 to be meaningful.
- **Keep a trial log.** Every model, feature set and hyperparameter configuration counts toward $N$, including RD-Agent's.

### 7.2 Multiple-testing adjustments

- **Bonferroni:** $p_i < \alpha/M$. It controls the family-wise error rate but is too conservative with correlated tests.
- **Holm** step-down.
- **Benjamini–Hochberg FDR:** sort $p_{(1)} \le \dots \le p_{(M)}$ and reject $H_{(i)}$ for $i \le \max\{k : p_{(k)} \le \frac{k}{M}q\}$. Benjamini–Yekutieli handles dependence.
- **HLZ recommendation:** t ≥ 3.0 for new factors.
- **Bootstrap reality check:** White (2000) and Hansen's SPA test (2005) test whether the best of many strategies beats a benchmark.
- **Bayesian/empirical-Bayes shrinkage** (JKP) is a more informative alternative to binary hurdles.
- **Rule for you:**
  - Genuinely new signals must clear t ≥ 3 on a pre-specified construction.
  - Well-established signals (value, momentum, profitability) can use a lower bar but a haircut premium.

### 7.3 Live vs. backtest decay

Sources of decay:

- Post-publication arbitrage (McLean–Pontiff: 58%).
- Selection bias (the "winner's curse").
- Crowding: factor returns fall and co-movement rises when crowded (Lou & Polk "comomentum"; the arXiv "Not All Factors Crowd Equally" 2025).
- Regime changes (decimalization 2001, Reg NMS 2005–07, 2020s retail/options boom).
- Implementation shortfall.

**Planning rule:** expect live Sharpe ≈ 0.5× backtest Sharpe for a careful process, and less for ML-heavy fast books. Paper-trade for 3–6 months, then scale capital in stages conditional on live-vs-backtest tracking.

### 7.4 Point-in-time pitfalls

1. **Restated fundamentals:** Compustat's standard files overwrite originals with restated values. Use Compustat Snapshot/PIT (from 1987) or lag conservatively. Quarterly data: use `RDQ` (report date) plus 1 trading day. Annual: the filing date from EDGAR (or 90 days after FYE plus buffer).
2. **Earnings timing:** I/B/E/S `anndats` vs. `anntims`. After-close announcements must map to the next trading day. There are known misdating issues pre-2000s, which can be cross-checked with Compustat `RDQ` (DellaVigna–Pollet method: take the earlier date).
3. **Index membership:** use historical constituents (CRSP index constituents, Compustat `idxcst_his`) and never current S&P 500 lists. Qlib's `instruments` files must be PIT (start/end dates per ticker).
4. **Identifier changes:** tickers get recycled, so key on PERMNO internally and map to tickers only at execution. CUSIPs change on corporate events, so use NCUSIP history.
5. **Corporate actions:** use CRSP adjustment factors (`cfacpr`, `cfacshr`). Qlib's `$factor` field must be consistent with how you compute returns and trade prices.
6. **Analyst data revisions:** I/B/E/S history is occasionally altered (Ljungqvist, Malloy & Marston 2009, *JF*, "Rewriting History"). Use the unadjusted detail files and your own snapshots going forward.
7. **Option data timing:** see Chapter 3.11; lag by one day.
8. **Delisting returns:** include CRSP `DLRET`. When missing for performance-related delistings, impute about −30% (Shumway 1997).
9. **13F timing:** use the filing date (up to 45 days lag), not the period-end.

### 7.5 Cost realism

- **Model every component:** spread (TAQ effective spreads or the Corwin–Schultz / Abdi–Ranaldo high-low estimators), impact (square-root), borrow fees (Markit; assume 25–50 bp general collateral and much higher for specials, and exclude hard-to-borrow shorts or charge them), and financing.
- **Stress costs** at 1.5–2× baseline. A fast-book signal that dies at 1.5× cost should not be deployed.
- **Qlib's default** cost settings (`open_cost` 0.05%, `close_cost` 0.15% incl. stamp duty, `min_cost`) are *China-specific*. Reset them for the US: no stamp tax, SEC fee (Section 31) on sells, spread-based costs by liquidity bucket, and an impact term.
- **Tradability:** Qlib's `limit_threshold` for price-limit bars is irrelevant in the US. Instead filter on price > $5, a minimum ADV, and halted-stock handling.

---

## Chapter 8 — Data Source Map

### 8.1 WRDS databases

| Database | Contents / signal families | Coverage | Identifiers & links |
|---|---|---|---|
| **CRSP US Stock (CIZ format)** | Prices, returns incl. dividends, delisting returns, shares, volume → momentum, reversal, low-vol, beta, size, MAX, IVOL, Amihud, seasonality, issuance | NYSE from 12/31/1925 (first monthly return Jan 1926); AMEX from 1962; NASDAQ from 12/1972. Legacy SIZ discontinued after Dec-2024 data | PERMNO (security), PERMCO (company), historical CUSIP/NCUSIP, ticker |
| **Compustat North America (Fundamentals A/Q)** | Balance sheet, income, cash flow → value, profitability, investment, accruals, F-score, payout, R&D capitalization | 1950s+ (annual), 1960s+ (quarterly) | GVKEY (company), IID (issue); CCM link (`crsp.ccmxpf_lnkhist`: linktype LC/LU[/LS], linkprim P/C) to PERMNO. The older `ccmxpf_linktable` is deprecated |
| **Compustat Snapshot / PIT** | As-first-reported fundamentals with change dates | North America from 1987 | GVKEY |
| **Compustat supplemental short interest** (`comp.sec_shortint`) | Short interest mid-month/end-of-month → SIR, DTC | Start date reported inconsistently (1973 vs. later); verify | GVKEY/IID |
| **I/B/E/S** (detail, summary, actuals, recommendations, guidance) | EPS forecasts, revisions, surprises (SUE), recommendations, dispersion | US from 1976 (summary), detail from ~1980s–1990s | IBES TICKER; WRDS IBES–CRSP link (`wrdsapps_link_crsp_ibes.ibcrsphist`, ICLINK score 0–6) |
| **OptionMetrics IvyDB US** | Option prices, IVs, greeks, OI, volume, standardized surface, forward prices, zero curve → IV spread, skew, O/S, VRP, BKM moments, term slope, implied earnings move, implied borrow | January 1996–present | SECID; WRDS OptionMetrics–CRSP link (SECID↔PERMNO with score; OCLINK macro) |
| **TAQ (daily / monthly)** | Trades & quotes → realized variance/kernels, effective spreads, order imbalance, retail flow (sub-penny), intraday reversal | Daily TAQ 09/10/2003–present; monthly TAQ 1993–2014 | Symbol + suffix; WRDS TAQ–CRSP link (1993–present); WRDS Intraday Indicators |
| **Thomson/Refinitiv 13F (s34)** | Institutional holdings → breadth, ownership changes, crowding | 1980–present (known quality issues, partly fixed) | CUSIP → PERMNO via NCUSIP |
| **Markit / S&P Global Securities Finance** | Loan fees, utilization, lendable supply → borrow cost, short-crowding | ~2002+ (vendor), daily; fuller US coverage mid-2000s | CUSIP/ISIN/SEDOL |
| **RavenPack (on WRDS)** | Entity-level news events, sentiment, novelty, relevance | Dow Jones edition from January 2000 | RP_ENTITY_ID → CUSIP/ticker mappings |
| **Capital IQ Transcripts / Key Developments** (if licensed) | Earnings call text, event dates | ~2000s+ | Company ID → GVKEY |
| **WRDS Beta Suite / Factor data / JKP characteristics on WRDS** | Precomputed betas, factor returns, 153 JKP characteristics at stock level | Varies | PERMNO/GVKEY |

### 8.2 Open-source databases

| Source | Contents | Notes |
|---|---|---|
| **Ken French Data Library** | FF3/FF5, momentum, reversal, industry portfolios, breakpoints | Monthly/daily; the standard benchmark for factor alphas |
| **Open Source Asset Pricing (Chen–Zimmermann)** | 212 predictor long-short portfolios (Oct 2025 release) and 209 firm-level characteristic signals with code | Most through Dec-2024; option-based signals through Dec-2022; ideal for replication and as a feature library |
| **Global Factor Data (Jensen–Kelly–Pedersen)** | 153 factors in 13 themes, 93 countries; stock-level characteristics via WRDS | Updated through Dec-2025; the canonical cluster taxonomy for signal combination |
| **global-q.org (Hou–Xue–Zhang)** | q-factor and q⁵ returns, replicated anomalies | Robustness checks |
| **FRED** | Rates, term/default spreads, macro series → regime and conditioning variables | Use ALFRED vintages for PIT macro data |
| **SEC EDGAR** | 10-K/10-Q/8-K full text, XBRL financial data (2009+), Form 4 insider trades, 13F-HR, 13D/G | Free; filing timestamps allow true PIT; "Lazy Prices", insider-trading signals (Cohen, Malloy & Pomorski 2012 "opportunistic insiders") |
| **CBOE** | VIX, VIX term structure (VIX9D, VIX3M), SKEW index, put/call ratios | Market-level VRP and stress regime inputs |
| **FINRA** | Bi-monthly short interest, daily short-sale volume (Reg SHO) | Free complement to Compustat SI |
| **Robert Shiller / Amit Goyal (Welch–Goyal) data** | Market predictors (CAPE, dividend yield, etc.) | Time-series timing research |

### 8.3 Linking-table cookbook

- **CRSP ↔ Compustat:** CCM `lnkhist`. Keep `linktype in ('LC','LU')`, `linkprim in ('P','C')`, and date between `linkdt` and `linkenddt` (NULL = current).
- **CRSP ↔ I/B/E/S:** `ibcrsphist`, score ≤ 1, date-bounded.
- **CRSP ↔ OptionMetrics:** WRDS SECID–PERMNO link, score ≤ 1, date-bounded.
- **CRSP ↔ TAQ:** WRDS TAQ–CRSP link (daily).
- **CRSP ↔ 13F:** 8-digit historical CUSIP (NCUSIP) with date bounds.
- **Qlib integration:** build a master PIT security master keyed on PERMNO with daily validity. Export Qlib's `instruments` (start/end per PERMNO) and `features` bins from a custom `DataHandlerLP`. Store fundamentals via Qlib's PIT database feature (`P()` operator, which supports `(period, date)` pairs). Otherwise pre-align fundamentals to availability dates before dumping to bin format.

---

## Chapter 9 — Synthesis: Candidate Architecture and Prioritized Signal List

### 9.1 Architecture

```
┌──────────────────────── DATA LAYER (PIT) ────────────────────────┐
│ CRSP (PERMNO master) · Compustat PIT/Snapshot · I/B/E/S detail    │
│ OptionMetrics (SECID→PERMNO) · TAQ/WRDS IID · 13F · Short int/    │
│ Markit · RavenPack/EDGAR text · FRED/CBOE macro-vol               │
│ → Custom Qlib DataHandlerLP + PIT fundamentals + availability lag │
└───────────────────────────────────────────────────────────────────┘
          │                                   │
┌──── SLOW BOOK (1–12 m) ────┐   ┌──── FAST BOOK (1 d – 1 m) ────┐
│ Value composite (timely ME,│   │ Residual/industry-adj. STR    │
│  intangibles-adjusted)     │   │  (no-news moves; news filter) │
│ Profitability (GP/A, cash  │   │ Analyst revisions/breadth     │
│  OP), investment, issuance │   │ Earnings-event: SUE+text+     │
│ Residual & industry mom.   │   │  implied-move normalization   │
│  (vol-managed)             │   │ Option layer: IV spread, skew,│
│ Low-risk (VW, beta-neutral)│   │  O/S, implied borrow (mostly  │
│ Lazy Prices text, seasonal │   │  as long-side veto)           │
│ Model: linear/IPCA + GBM   │   │ Short interest/DTC, retail    │
│  on JKP themes, monthly    │   │  flow (TAQ), LLM news scores  │
└────────────────────────────┘   │ Model: GBM/DoubleEnsemble/    │
          │                      │  TRA/HIST, daily, rank label  │
          │                      └───────────────────────────────┘
          └──────────────┬──────────────────┘
┌──────── PROBABILISTIC LAYER ─────────────────────────────────────┐
│ Per-stock σ̂ (HAR-RV + IV, GJR for daily), quantile GBMs,          │
│ adaptive conformal intervals, event-variance from options,        │
│ regime filter (HMM/jump model on market; VRP; VIX term slope)     │
│ → converts alphas to risk-adjusted scores, sets confidence/size   │
└───────────────────────────────────────────────────────────────────┘
          │
┌──────── PORTFOLIO & EXECUTION ───────────────────────────────────┐
│ Barra-style + PCA risk model (Ledoit-Wolf specific)               │
│ Gârleanu-Pedersen aim portfolio: decay-weighted alpha blend       │
│ Cost-aware MVO (spread + √impact + borrow), neutrality, ADV caps  │
│ Book-level vol targeting; HRP across sleeves; fractional Kelly    │
│ Almgren-Chriss/POV execution; TCA feedback into cost model        │
└───────────────────────────────────────────────────────────────────┘
          │
┌──────── GOVERNANCE ──────────────────────────────────────────────┐
│ Trial registry → DSR/PBO · purged CPCV · sealed holdout · paper   │
│ trading · staged capital · drift/coverage monitors · kill-switch  │
└───────────────────────────────────────────────────────────────────┘
```

**Key design choices:**

1. **Universe:** US common stocks (share codes 10/11), price > $5, top ~1,500–3,000 by market cap with ADV filters. Exclude microcaps (below NYSE 20th percentile) from the traded universe, but keep them in training with lower weight if desired.
2. **Labels:**
   - Slow book: 1-month and 3-month forward *residual* returns (vs. risk model).
   - Fast book: 1-day, 5-day and 20-day residual returns, with separate models or multi-task heads (TCTS-style).
3. **Probabilistic role:** the model output is $\hat\mu_{i}$ (rank/alpha). The distributional layer supplies $\hat\sigma_i$, quantiles and event flags. The optimizer consumes $\hat\mu$ and $\Sigma$; sizing confidence comes from conformal width/meta-labels. **Do not treat option-implied densities as directional forecasts; use them for risk and for the Q–P wedge.**
4. **Regime overlay:** reduce momentum weight and gross exposure in "panic states" (market drawdown plus high vol, per Daniel–Moskowitz). Increase reversal weight when VIX is high (Nagel 2012). Use the market VRP and VIX term structure for the overall risk budget.

### 9.2 Prioritized signal list (test in this order)

**Tier 1 — Well-documented, robust, scalable (slow book core)**

1. **Value composite with timely prices and intangibles adjustment** (B/M with current ME, EV/EBITDA, CF/P, R&D-and-SG&A-capitalized B/M). Industry-neutral. Robust long-run, but expect long droughts.
2. **Profitability** (gross profitability, cash-based operating profitability), combined with value. The combination is one of the most replicated results.
3. **Investment/asset growth and net share issuance** (negative). Replicate in HXZ, JKP and Chen–Zimmermann.
4. **Residual (idiosyncratic) momentum plus industry momentum, volatility-managed** with a Daniel–Moskowitz panic-state brake.
5. **Low-risk (value-weighted low-beta/low-vol, beta-neutral construction)**, with rate-sensitivity monitoring.

**Tier 2 — Good evidence, moderate decay (medium/fast)**

6. **Analyst revisions and recommendation changes** (I/B/E/S PIT).
7. **Short-term residual reversal restricted to no-news moves** (RavenPack/EDGAR filter), traded with cost-aware GP dynamics. Gross alpha is strong but it is the most cost-sensitive, so build the cost model first.
8. **Days-to-cover / short interest**, primarily as a long-side veto and underweight. Short only where borrow is cheap.
9. **Text-change signals ("Lazy Prices") and LLM-scored news/transcripts**, time-locked models only. Lopez-Lira–Tang's own evidence shows returns decline as adoption rises, so treat this as a decaying edge requiring continuous refresh.
10. **Economic-link momentum** (customer→supplier, analyst co-coverage peers).
11. **Seasonality (Heston–Sadka)**: small, but uncorrelated.

**Tier 3 — Documented but decayed, contested or friction-bound (test with skepticism)**

12. **Option-implied signals** (IV spread, smirk, O/S, BKM skew, IV–RV). Published effects are large (CW 50 bp/week; XZZ 10.9%/yr; Johnson–So about 0.34%/week). Muravyev–Pearson–Pollet show at least two-thirds of the IV-spread/skew predictability is borrow-fee-driven, and CW's effect declined over their sample.\[2\]\[34\] **Best use:** long-side filters, event-risk inputs, and interactions (e.g., steep skew plus negative revision). Martin–Wagner SVIX expected returns are a theoretically grounded slow alternative worth testing.
13. **PEAD (classic SUE drift)**: "non-existent since 2006" for large stocks per Martineau (2022, *CFR*) (contested). Use earnings-event information as features in the fast book (announcement return, revenue surprise, text tone, implied-move-normalized surprise $\frac{CAR_{[-1,1]}}{\sigma^{implied}_{EA}}$) rather than a standalone drift strategy.
14. **MAX/IVOL, Amihud, accruals, size**: use as risk features, short-side filters or cost inputs.
15. **13F-based signals**: slow, lagged and crowded, but ownership-crowding measures are useful as *risk* features (fire-sale exposure).

**Market-level (risk-budget only):** variance risk premium ($VIX^2 - RV$ with TAQ-based RV), VIX term slope, and CBOE SKEW. Use these to scale gross exposure, not for stock selection.

### 9.3 Validation and deployment protocol

1. **Replicate first:** reproduce Chen–Zimmermann/JKP portfolio returns for your Tier-1 signals from your own WRDS pipeline. This is the check that your PIT alignment and links are correct.
2. **Sample split:**
   - Development: 1990–2014 (options from 1996; TAQ daily from 2003).
   - Validation: 2015–2019.
   - Sealed holdout: 2020–present, opened once per major model version.
3. **Walk-forward with annual retraining** (monthly for the fast book), purged/embargoed by label horizon.
4. **Report for each signal:**
   - Rank IC, ICIR and the IC decay curve (1–60 days).
   - Long-short and long-only-tilt returns, value-weighted, ex-microcaps.
   - Turnover, net-of-cost returns at 1× and 2× cost, and borrow-adjusted returns.
   - DSR with the full trial count, and PBO for model-selection steps.
5. **Combination tests:** marginal contribution of each signal to the combined, cost-aware portfolio (not standalone Sharpe).
6. **Paper trade** for at least 3 months with production data feeds (daily delivery timing differs from WRDS history, so check that every feature is available *before* the decision time). Then stage capital: 10% → 30% → 100% of the target, gated on live-vs-expected tracking within confidence bands.

### 9.4 Honest expectations

- A carefully built slow book on Tier-1 signals might deliver a long-short gross Sharpe of 0.5–0.8 historically. With post-publication haircuts, a long-only tilt and costs, expect about 0.3–0.6 of active IR.
- A fast book can show gross backtest ICs of 0.03–0.06 and Sharpe of 2+, but net-of-cost live Sharpe is commonly less than half of backtest.
- The combined book's advantage comes from low correlation between the books and trade netting.
- The probabilistic layer's main payoff is *drawdown and sizing quality*, not extra alpha.

---

## Caveats

- **Qlib benchmark numbers are A-share (CSI300) results** with China-specific costs. Re-benchmark on US data before drawing conclusions about model choice.
- **RD-Agent(Q) performance figures** come from the authors' paper and secondary summaries, on Chinese data. Its 14.21% annualized return and IC 0.0532 are from a LinkedIn/blog summary rather than a verified table. Treat them as indicative only.
- **PEAD status is contested** (Martineau 2022 vs. 2025 papers). The disagreement reflects research design choices.
- **Some coverage dates need verification in your WRDS account:** the Compustat short-interest start date, Markit US coverage start, and exact OptionMetrics–CRSP link table names. They are reported inconsistently across sources.
- **Several magnitudes quoted from the literature are in-sample and pre-cost.** Examples are the XZZ 10.9%/yr (from the paper's JFQA abstract) and GKX Sharpe ratios. Apply the haircuts discussed.
- **LLM-based return predictability** is highly vulnerable to look-ahead contamination and is decaying with adoption, per the authors' own evidence.
- This report is research support, not investment advice. Real-money deployment requires legal/compliance review (e.g., pattern-day-trader rules, short-sale regulation SHO locate requirements, and broker borrow availability).

## Sources

1. [Does Academic Research Destroy Stock Return Predictability? - MCLEAN - 2016 - The Journal of Finance - Wiley Online Library](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)
2. [Replication package for "Why Does Options Market Information Predict Stock Returns?" by Muravyev, Pearson, and Pollet - Mendeley Data](https://data.mendeley.com/datasets/n73cyx89gs/2)
3. […and the Cross-Section of Expected Returns by Campbell R. Harvey, Yan Liu, Caroline Zhu :: SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314)
4. [qlib/examples/benchmarks/README.md at main · microsoft/qlib](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md)
5. [Trading Costs of Asset Pricing Anomalies by Andrea Frazzini, Ronen Israel, Tobias J. Moskowitz :: SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2294498)
6. [Trading Costs of Asset Pricing Anomalies](https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies)
7. [Trading Costs ANDREA FRAZZINI, RONEN ISRAEL, AND TOBIAS J. MOSKOWITZ](https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2021/08/Trading-Cost.pdf)
8. [Does Academic Research Destroy Stock Return Predictability?\*](https://tevgeniou.github.io/EquityRiskFactors/bibliography/AcademicReviewFactor.pdf)
9. [Momentum Crashes | Columbia Business School](https://business.columbia.edu/faculty/research/momentum-crashes)
10. [Momentum crashes - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304405X16301490)
11. [Journal of Financial Economics Momentum crashes](https://www.kentdaniel.net/papers/published/jfe_16.pdf)
12. [What you see is not what you get: The costs of trading market anomalies - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X20300453)
13. [Is There a Replication Crisis in Finance? - JENSEN - 2023 - The Journal of Finance - Wiley Online Library](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249)
14. [Rest in Peace Post-Earnings Announcement Drift | Critical Finance Review | Emerald Publishing](https://www.emerald.com/cfr/article-abstract/11/3-4/613/1324916/Rest-in-Peace-Post-Earnings-Announcement-Drift?redirectedFrom=fulltext)
15. [Extracting the Structure of Press Releases for Predicting Earnings Announcement Returns](https://arxiv.org/pdf/2509.24254)
16. [Research Brief](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/)
17. [Trading Costs of Asset Pricing Anomalies | Request PDF](https://www.researchgate.net/publication/324657588_Trading_Costs_of_Asset_Pricing_Anomalies)
18. [Where can I find short interest by company? - FAQ](https://faq.library.upenn.edu/business/faq/45338)
19. [Where can I find short interest by company? - Library Help](https://manchester-uk.libanswers.com/teaching-and-learning/faq/193106)
20. [Short interest by company - How-To](https://libhowto.iese.edu/faq/77133)
21. [www.library.wisc.edu](https://www.library.wisc.edu/business/services/services-for-faculty/research-databases/compustat/)
22. [Securities Finance: Program Management Dataset | S&P Global Marketplace](<https://www.marketplace.spglobal.com/en/datasets/securities-finance-program-management-(1709139421)>)
23. [The Adaptive Multi-Factor Model and the Financial Market](https://arxiv.org/pdf/2107.14410)
24. [Expected Stock Returns and Variance Risk Premia](https://ideas.repec.org/a/oup/rfinst/v22y2009i11p4463-4492.html)
25. [Tail Risk Premia and Return Predictability](https://haas.berkeley.edu/wp-content/uploads/tvt.pdf)
26. [Option-Implied Signals and Crash Risk: Predictability and Machine-Learning Evidence from U.S. Equity Options](https://arxiv.org/pdf/2608.26115)
27. [Study: Borrowing fee is the variable that debunks theory that options predict future stock returns](https://giesbusiness.illinois.edu/news/2026/01/07/study--borrowing-fee-is-the-variable-that-debunks-theory-that-options-predict-future-stock-returns)
28. [Empirical Asset Pricing via Machine Learning](https://www.research-center.mirova.com/en/research-library/Empirical-Asset-Pricing-via-Machine-Learning)
29. [Empirical Asset Pricing via Machine Learning | NBER](https://www.nber.org/papers/w25398)
30. [NBER WORKING PAPER SERIES EMPIRICAL ASSET PRICING VIA MACHINE LEARNING](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)
31. [arxiv.org](https://arxiv.org/pdf/2304.07619v1)
32. [RD-Agent(Q): A framework for automated quant R&D | Arman Khaledian posted on the topic | LinkedIn](https://www.linkedin.com/posts/armankhaledian_rd-agentq-data-centric-multi-agent-factormodel-activity-7336326541053767681-qJHN)
33. [Automated Quant Research with AI Agents: How Microsoft's RD-Agent Achieves 2x Returns with 70% Fewer Factors | Saulius blog](https://saulius.io/blog/automated-quant-research-ai-agents-rd-agent)
34. [IDEAS home Printed from https://ideas.repec.org/a/cup/jfinqa/v45y2010i02p335-367\_00.html](https://ideas.repec.org/a/cup/jfinqa/v45y2010i02p335-367_00.html)
