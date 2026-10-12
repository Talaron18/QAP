# Option / implied-volatility data: what we can reach (survey of 2026-10-12)

Context: OptionMetrics (Ivy DB) is NOT in our WRDS subscription (2014 trial only) and the `cboe` option tables answered "permission denied for schema cboe_eod". The researcher agent could not reach several primary sites (DNS errors), so **every price below is unverified** and must be confirmed with the vendor before buying.
Local access is being re-checked with `scripts/wrds_access_map.py` (notably `cboe.cboe` and a possible `cboe_all` VIX table, never queried).

| Source | Content | Cost | Answers | Status |
|---|---|---|---|---|
| FRED VIXCLS / VXVCLS / VXNCLS | daily VIX (30d), VIX3M, VXN from ~1990 | free, cite CBOE | index variance risk premium vs our forecasts (30-day horizon) | usable now (download CSV) |
| Cboe historical page | VIX, VIX9D, VIX3M, VVIX (2007+), OVX, GVZ, single-stock VIX for ~5 names | free | term structure, vol-of-vol, sanity checks | usable now |
| WRDS `cboe` / `cboe_all` | possibly VIX history (`cboe.cboe`); option tables denied | in subscription | index VRP | to be tested by the access map |
| ORATS historical file | EOD chains with IV/Greeks from 2007, 5,000+ symbols | one-time ~USD 399 or subscription (sources conflict) | stock-level IV vs forecast (100 names, 10+ years) | unverified; ask coverage, per-strike vs summary, licence |
| Alpha Vantage HISTORICAL_OPTIONS | per-date chains with IV and Greeks, "15+ years" | premium plan (price unverified) | same | verify start date |
| Cboe DataShop Option EOD Summary (+ Calcs) | EOD chains from ~2012; IV needs the calcs add-on | ~USD 350 per symbol-year with calcs (mirror, unverified); academic discount 50% with a USD 500 minimum | same | expensive for 100 names |
| Intrinio historical options | standard plan from 2021, bulk back to 2008 | ~USD 2,800/yr | same | too expensive |
| iVolatility | pay per ticker-day (USD 0.20-0.60) | ~USD 75-150k for 100 names x 10 years | same | ruled out |
| Databento OPRA | trades/OHLCV from 2013-04, NBBO from 2023-03 | pay per GB | not an EOD IV panel | poor fit |
| Polygon | no historical Greeks/IV | USD 29-199/mo | no | no |
| Interactive Brokers | historical IV bars not confirmed; expired options unavailable | free with account | test only | unlikely 5-10 years |
| Open Source Asset Pricing | pre-built monthly predictors (some option-based, names unconfirmed), derived from licensed data | free | cross-section signals, not daily IV | check which option signals exist |
| Goyal-Welch, BTZ | equity premium predictors; VRP series to replicate | free | benchmark forecasts | usable |

Licence caveat: OptionMetrics and Cboe academic feeds are research-only; trading a signal built from academically licensed data in a personal account may breach the terms. Prefer a vendor with a retail licence and get it in writing.

Reading for this project: (a) the INDEX-level variance risk premium test (our market-return GARCH/GJR/HAR forecast of 30-day variance versus VIX^2, 1990-2025) needs only free data and can be pre-registered as Stage B-index; (b) a STOCK-level panel needs paid data (ORATS or Alpha Vantage are the candidates) and a licence check, and is only worth buying if (a) and Study F (v13) show that forecasts carry information.
