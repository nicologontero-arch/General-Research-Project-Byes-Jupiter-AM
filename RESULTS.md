# Bond risk premia and the timing of the value factor: every result, in one file

Generated from `outputs/sections/*.json`, the structured tables the scripts write when they
run, so nothing here is retyped and nothing can drift from the code. Every block below can be
reproduced on its own with `cd src && python <script>.py`.

## Conventions

Scripts appear in the order the argument is made, not alphabetically. Each is introduced by a
description of what it computes; the numbers below each introduction are exactly what the
script printed.

Percentages are annualised unless the row says otherwise. `NW t` is a Newey-West
*t*-statistic, with lags set to the overlap in the data: 12 monthly, 4 quarterly, 2 annual,
and 18 for the 12-month predictive regressions.

The decomposition `E[active] = Cov(w,h) + (E[w]-1)E[h] - c*E[turnover]` is exact. Its three
terms are labelled timing, tilt and cost throughout. Mean weight `E[w]` is reported beside
every alpha: it is about 1.00 in the US and runs 0.86 to 1.25 across the four foreign markets.

## Contents

**The data** (§3)  
[`data_summary`](#data_summary)

**Mechanism and controls** (§5)  
[`duration_test`](#duration_test)  [`robustness`](#robustness)  [`hybrid`](#hybrid)

**The traded strategy** (§6)  
[`results`](#results)  [`decomposition`](#decomposition)  [`integrity_audit`](#integrity_audit)  [`bond_link`](#bond_link)  [`factor_alpha`](#factor_alpha)

**The value spread** (§7)  
[`p2_driver`](#p2_driver)  [`spread_checks`](#spread_checks)

**International** (§8)  
[`country_cp`](#country_cp)  [`cp_intl`](#cp_intl)  [`decomposition_intl`](#decomposition_intl)

**Estimation choices** (§4)  
[`r2_investigation`](#r2_investigation)  [`rolling_cp`](#rolling_cp)

---

# The data

<a id="data_summary"></a>

## `data_summary.py` — Input data coverage

*every series as its loader returns it, with the raw row count beside the count that survives filtering*

Coverage of every input series as its loader returns it: source file, frequency, units, first and last observation, the row count in the file and the count that survives filtering. The closing block lists where each sample ends.

| series | source file | freq | units | first | last | rows | used |
|---|---|---|---|---|---|---|---|
| GSW zero-coupon 1-5y | gsw_yields_daily.csv | daily | percent p.a. | 1961-06-14 | 2026-05-29 | 16948 | 16203 |
| Fama-French 5 factors | F-F_Research_Data_5_Factors_2x3_daily.csv | daily | percent | 1963-07-01 | 2026-04-30 | 15813 | 15813 |
| 6 size-B/M portfolios | 6_Portfolios_2x3_Daily.csv | daily | percent | 1926-07-01 | 2026-04-30 | 26233 | 26233 |
| BE/ME characteristic | Portfolios_Formed_on_BE-ME.csv | annual | log ratio | 1927-06-30 | 2025-06-30 | 99 | 99 |
| UK gilt curve 1-5y | intl/yields_UK.csv | daily | percent p.a. | 1979-01-02 | 2026-05-29 | 11966 | 11966 |
| Canada curve 1-5y | intl/yields_CA.csv | daily | percent p.a. | 1986-01-02 | 2026-06-03 | 9923 | 9923 |
| Japan JGB curve 1-5y | intl/yields_JP.csv | daily | percent p.a. | 1974-09-24 | 2026-05-29 | 12578 | 12578 |
| Germany curve 1-5y | intl/yields_DE.csv | monthly | percent p.a. | 1972-09-30 | 2026-05-31 | 645 | 645 |
| UK value factor | intl/hml_daily_intl.csv | daily | percent | 1988-07-01 | 2026-03-31 | 10892 | 9848 |
| Canada value factor | intl/hml_daily_intl.csv | daily | percent | 1984-07-02 | 2026-03-31 | 10892 | 10892 |
| Japan value factor | intl/hml_daily_intl.csv | daily | percent | 1988-07-01 | 2026-03-31 | 10892 | 9848 |
| Germany value factor | intl/hml_daily_intl.csv | daily | percent | 1990-07-02 | 2026-03-31 | 10892 | 9327 |

*'rows' counts what the file holds, 'used' what the loader returns. GSW carries blank rows on US market holidays; the four value factors share one wide file with NaNs preserved, so each market drops on its own date.*

### Where each sample stops

| | |
|---|---|
| GSW yields end | 2026-05-29 |
| US value factor ends | 2026-04-30 |
| foreign value factors end | 2026-03-31 |
| BE/ME characteristic ends | 2025-06-30 |
| US traded quarters | 204 (1975-09-30 to 2026-06-30) |

---

# Mechanism and controls

<a id="duration_test"></a>

## `duration_test.py` — Duration vs business-cycle predictive regressions

*12m-ahead bond, growth-leg, value-leg and HML returns regressed on CP, the leg contrast, and a horizon profile*

Predictive regressions of 12-month-ahead returns on the CP factor, for bonds, the growth leg, the value leg and HML. Block 2 estimates the contrast `b_v - b_g` directly, so it carries its own standard error; block 2c repeats that contrast on the real-time factor. Block 3 varies the forecast horizon from 1 to 24 months. The final block is the same regression in raw units.

| | |
|---|---|
| CP construction R2 (forwards -> 12m bond excess return) | 0.153 |
| coefs f1..f5 | -2.46  +7.11  -11.36  +8.13  -1.18 |

### Predictive regression on standardized CP (per 1 SD), NW 18 lags

| target | beta/SD | NW t | R2 |
|---|---|---|---|
| 12m BONDS (excess) | +1.36 | 3.88 | 0.153 |
| 12m GROWTH leg (excess) | +0.40 | 0.22 | 0.000 |
| 12m VALUE leg (excess) | +2.67 | 1.73 | 0.022 |
| 12m HML (value-growth) | +2.34 | 1.91 | 0.034 |

*Loadings are per standard deviation of CP. NW 18 lags for the 12m overlap.*

### 2) The leg contrast b_v - b_g

The contrast VALUE - GROWTH is H - L, so it is itself a regressand and carries its own standard error.

| specification | beta/SD | NW t | one-sided p vs duration | two-sided p |
|---|---|---|---|---|
| GROWTH leg  b_g | +0.40 | +0.22 | 0.413 | 0.827 |
| VALUE leg   b_v | +2.67 | +1.73 | 0.042 | 0.084 |
| CONTRAST    b_v - b_g | +2.26 | +1.87 | 0.030 | 0.061 |

*One-sided p is P(beta <= 0); two-sided p tests beta = 0 against any alternative. The contrast row is the test, the two leg rows its components.*

**The contrast, restated**

| | |
|---|---|
| contrast b_v - b_g (per 1 SD of CP) | +2.264 pp over 12m |
| NW t on the contrast | +1.87 |
| one-sided p against the duration null | 0.030 |
| growth leg, two-sided p (is it moving at all?) | 0.827 |

### 2c) The same contrast, using the real-time CP factor instead

Sections 1 and 2 fit CP once over the full sample. Here CP is rebuilt by expanding, realised-only regression and the contrast re-estimated on it.

| specification | n | beta/SD | NW t | one-sided p vs duration | two-sided p |
|---|---|---|---|---|---|
| GROWTH leg  b_g | 601 | -1.39 | -0.58 | 0.719 | 0.563 |
| VALUE leg   b_v | 601 | +2.22 | +1.09 | 0.139 | 0.277 |
| CONTRAST    b_v - b_g | 601 | +3.61 | +2.21 | 0.014 | 0.027 |

*Real-time CP, standardised by an expanding z-score. Shorter sample than section 2.*

### 3) Horizon profile, h = 1 to 24 months

CP forecasts the 12m-ahead bond return, so h=12 is its own horizon. The cumulative loading grows with h by construction; the annualised column and the t do not.

| horizon | n | NW lags | cumulative beta/SD | annualised beta/SD | NW t | R2 | note |
|---|---|---|---|---|---|---|---|
| 1m | 753 | 1 | +0.12 | +1.41 | +0.97 | 0.002 |  |
| 3m | 751 | 4 | +0.38 | +1.54 | +1.06 | 0.005 |  |
| 6m | 748 | 9 | +0.74 | +1.49 | +1.06 | 0.008 |  |
| 12m | 742 | 18 | +2.34 | +2.34 | +1.91 | 0.034 | CP's fitting horizon |
| 24m | 730 | 36 | +3.51 | +1.75 | +2.16 | 0.037 |  |

*Overlapping beyond h=1, so n overstates the effective sample; NW lags are 1.5h.*

### Raw regression r_HML(t+1:t+12) = a + b*CP(t) + e [percent]

| | |
|---|---|
| a (intercept) | +2.335  (t +1.25) |
| b (slope on CP) | +1.724  (t +1.91) |
| R2 | 0.0343 |
| n | 742 |

<a id="robustness"></a>

## `robustness.py` — Robustness checks

*yield-curve controls, multiple-testing adjustment, stub-period fix, cost sensitivity*

Four checks. Competing term-structure signals run through identical machinery, with a spanning regression in both strategy and predictive form. White's Reality Check across the seven CP x spread designs. The trailing partial period at the end of the sample. And sensitivity of alpha and IR to the transaction-cost assumption, with the break-even cost.

### 1) Competing term-structure signals

Each signal runs through identical machinery: expanding z, 1+tanh(z), lag, net 10bps.

**Each term-structure signal used as the timing signal**

| timing signal | n | alpha%/yr | IR | NW t | meanW | turn/yr |
|---|---|---|---|---|---|---|
| CP (baseline) | 204 | +1.78 | +0.26 | +1.55 | 1.003 | 1.22 |
| Slope  y5 - y1 | 206 | +1.30 | +0.15 | +0.96 | 1.113 | 0.80 |
| Level  y1 | 206 | +0.09 | +0.01 | +0.07 | 0.724 | 0.35 |
| Curvature 2y2-y1-y5 | 206 | -1.19 | -0.15 | -1.24 | 0.767 | 1.16 |

**Spanning regression: CP active return on slope/level/curvature active returns**

| | |
|---|---|
| CP-timed alpha not explained by slope/level/curvature timing | +1.14 %/yr |
| NW t on that intercept | +2.03 |
| R2 of CP active on the three controls | 0.613 |
| raw CP-timed alpha, for comparison | +1.78 %/yr |

**Predictive regression: 12m-ahead HML on standardized signals (NW 18 lags)**

| specification | b(CP)/SD | t(CP) | b(slope) | b(level) | R2 | n |
|---|---|---|---|---|---|---|
| CP alone | +2.12 | +1.74 |  |  | 0.025 | 636 |
| slope alone |  |  | +1.47 |  | 0.014 | 636 |
| level alone |  |  |  | +1.90 | 0.023 | 636 |
| CP + slope + level | +1.42 | +1.28 | +1.88 | +2.84 | 0.069 | 636 |

Two spanning tests are reported: the strategy form above (CP active return on the three control active returns) and the predictive form here (12m-ahead HML on the standardised signals). The strategy uses tanh(z), a nonlinear function of the curve.

### 2) Multiple testing across the seven CP x spread designs

Selecting the best of 7 designs makes a naive p conditional on that selection. The Reality Check bootstraps the maximum statistic across all 7 under the null of no skill.

**All seven designs (n=204 quarters)**

| design | alpha%/yr | t (iid) | naive p |  |
|---|---|---|---|---|
| CP-only | +1.78 | +1.87 | 0.042 |  |
| Spread-only | +1.38 | +0.97 | 0.178 |  |
| Additive | +1.59 | +2.23 | 0.013 |  |
| Gated | +1.07 | +1.32 | 0.100 |  |
| A monotonic | +1.92 | +1.37 | 0.105 |  |
| B U-shaped | +2.33 | +1.66 | 0.064 |  |
| S agreement | +2.34 | +2.67 | 0.004 | <- selected |

*t (iid) is the plain mean/SE used to rank designs. 'naive p' is the one-sided bootstrap p for each design on its own, ignoring that six others were tried.*

**White Reality Check, 5000 block bootstraps (block length 4)**

| | |
|---|---|
| best design by t-stat | S agreement |
| its naive one-sided p | 0.004 |
| Reality Check p, adjusted for searching 7 designs | 0.042 |

### 3) Trailing partial-period ('stub') contamination

Daily HML ends 2026-04-30. resample() labels each period with its END date, so the final quarterly bucket is stamped 2026-06-30 while holding only April, and the final annual bucket is stamped 2026-12-31 while holding only Jan-Apr. The annual row carries the project's largest t-stat, so this needs checking rather than assuming.

**Headline frequency table, before and after dropping the trailing stub period**

| frequency / treatment | n | end | alpha%/yr | IR | NW t |
|---|---|---|---|---|---|
| Monthly - as published | 611 | 2026-04-30 | +1.36 | +0.23 | +1.22 |
| Monthly - complete periods only | 611 | 2026-04-30 | +1.36 | +0.23 | +1.22 |
| Quarterly - as published | 204 | 2026-06-30 | +1.78 | +0.26 | +1.55 |
| Quarterly - complete periods only | 203 | 2026-03-31 | +1.79 | +0.26 | +1.56 |
| Annual - as published | 51 | 2026-12-31 | +1.91 | +0.26 | +2.32 |
| Annual - complete periods only | 50 | 2025-12-31 | +1.90 | +0.25 | +2.26 |

### 4) Transaction-cost sensitivity

Headline results assume 10bps per unit of turnover; quarterly turnover is ~1.2x/yr.

**Alpha and IR versus the cost assumption (quarterly, 1975+)**

| cost per unit turnover | CP alpha | CP IR | CP t | S alpha | S IR | S t |
|---|---|---|---|---|---|---|
| 0 bps | +1.90 | +0.28 | +1.66 | +2.47 | +0.39 | +2.59 |
| 10 bps | +1.78 | +0.26 | +1.55 | +2.34 | +0.37 | +2.47 |
| 25 bps | +1.60 | +0.24 | +1.40 | +2.16 | +0.35 | +2.28 |
| 50 bps | +1.29 | +0.19 | +1.13 | +1.85 | +0.30 | +1.97 |
| 100 bps | +0.68 | +0.10 | +0.60 | +1.24 | +0.20 | +1.33 |

**Cost at which the alpha is fully eaten**

| | |
|---|---|
| CP-only break-even cost | 156 bps per unit turnover |
| S agreement break-even cost | 201 bps per unit turnover |

<a id="hybrid"></a>

## `hybrid.py` — Hybrid CP-HML placebo

*forwards fit directly to 12m-ahead HML instead of to bond returns, real-time*

The same five forward rates fit directly to 12-month-ahead HML instead of to bond returns, then traded through the identical machinery. Reports the full-sample regression against the standard bond target, the correlation between the two signals, and the traded result at quarterly and annual frequency.

### Diagnostic: full-sample regression of 12m target on 5 forwards

| target | R2 | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|---|
| bonds (standard CP) | 0.153 | -2.46 | +7.11 | -11.36 | +8.13 | -1.18 |
| HML (hybrid) | 0.053 | -1.53 | -3.86 | +17.13 | -18.74 | +7.80 |

| | |
|---|---|
| corr(CP signal, hybrid signal), quarterly | 0.34 |

### HML timing: standard CP vs hybrid CP-HML (net 10bps, 1975+)

**Quarterly**

| design | n | alpha%/yr | IR | NW t | meanW |
|---|---|---|---|---|---|
| standard CP | 204 | +1.78 | +0.26 | +1.55 | 1.00 |
| hybrid CP-HML | 196 | -1.71 | -0.20 | -1.16 | 0.63 |

**Annual**

| design | n | alpha%/yr | IR | NW t | meanW |
|---|---|---|---|---|---|
| standard CP | 51 | +1.91 | +0.26 | +2.32 | 1.03 |
| hybrid CP-HML | 49 | -2.35 | -0.23 | -1.81 | 0.66 |

---

# The traded strategy

<a id="results"></a>

## `results.py` — Problem 1 - US CP-timed value

*frequency comparison, quarterly sub-periods, tilt-vs-timing decomposition, look-ahead audit + bootstrap*

The CP overlay `w = 1 + tanh(z_CP)` at monthly, quarterly and annual frequency, then over four quarterly sub-periods. Block 3 reports mean weight, the correlation between active and static returns, and next-quarter value returns sorted into weight buckets. Block 4 is a block bootstrap of the quarterly active return.

Real-time CP: 649 obs, 1972-05-31 -> 2026-05-31

### 1) Frequency comparison (start 1975, calendar-consistent 3y z-burn-in)

| period | n | start | end | staticSR | timedSR | alpha%/yr | IR | NW t | meanW | corr(a,v) | turn/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Monthly | 611 | 1975-06-30 | 2026-04-30 | 0.32 | 0.42 | +1.36 | 0.23 | 1.22 | 0.980 | -0.15 | 2.21 |
| Quarterly | 204 | 1975-09-30 | 2026-06-30 | 0.28 | 0.41 | +1.78 | 0.26 | 1.55 | 1.003 | -0.21 | 1.22 |
| Annual | 51 | 1976-12-31 | 2026-12-31 | 0.26 | 0.36 | +1.91 | 0.26 | 2.32 | 1.028 | -0.07 | 0.55 |

### 2) Quarterly sub-periods

| period | n | start | end | staticSR | timedSR | alpha%/yr | IR | NW t | meanW | corr(a,v) | turn/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1975-1989 | 58 | 1975-09-30 | 1989-12-31 | 0.63 | 0.74 | +3.18 | 0.50 | 1.62 | 1.074 | +0.19 | 1.36 |
| 1990-2004 | 60 | 1990-03-31 | 2004-12-31 | 0.43 | 0.49 | -0.07 | -0.01 | -0.03 | 0.946 | -0.53 | 1.67 |
| 2005-2019 | 60 | 2005-03-31 | 2019-12-31 | -0.09 | -0.04 | +0.47 | 0.11 | 0.39 | 1.049 | +0.20 | 0.76 |
| 2020-2026 | 26 | 2020-03-31 | 2026-06-30 | 0.11 | 0.50 | +5.98 | 1.03 | 2.99 | 0.869 | -0.45 | 0.92 |

### 3) Tilt-vs-timing decomposition + asymmetry (quarterly)

| | |
|---|---|
| mean weight (~1 => no static tilt) | 1.003 |
| corr(active, value) | -0.21 |
| active alpha OVERWEIGHT | +3.09%/yr (n=110) |
| active alpha UNDERWEIGHT | +0.25%/yr (n=94) |

**Next-quarter value return by weight bucket**

| weight bucket | next-qtr value return (ann.%) |
|---|---|
| deep-UW | -1.49 |
| UW | +0.97 |
| OW | +5.00 |
| deep-OW | +9.57 |

### 4) Look-ahead audit + bootstrap (quarterly)

| | |
|---|---|
| corr(w_lag_t, hml_t)  (timing skill, weight set at t-1) | +0.150 |
| block-bootstrap quarterly alpha 95% CI | [-0.65, +3.82] %/yr |
| P(alpha <= 0) | 0.072 |

<a id="decomposition"></a>

## `decomposition.py` — Timing vs tilt decomposition, with t-stats

*E[active] = Cov(w,h) + (E[w]-1)E[h] - costs; the exact split behind both alpha definitions*

The exact split `E[active] = Cov(w,h) + (E[w]-1)E[h] - cost*E[turnover]`, with a HAC *t*-statistic on each term and a residual column. Reported by frequency, by sub-period, for the 1990-2004 block split at the dot-com peak, and for the seven CP x spread designs. Blocks 1b and 3c bootstrap the timing term.

timing = Cov(w,h). tilt = (E[w]-1)E[h]. total = the benchmark-difference alpha. |resid| is total minus the three terms, and is zero to machine precision.

### 1) CP-only overlay w = 1 + tanh(z_CP), by frequency (1975+, net 10bps)

| label | n | meanW | TIMING %/yr | t | t(slope) | tilt %/yr | t(W=1) | cost %/yr | total %/yr | t | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Monthly | 611 | 0.980 | +1.65 | +1.55 | +1.57 | -0.07 | -0.36 | -0.22 | +1.36 | +1.22 | 5.2e-16 |
| Quarterly | 204 | 1.003 | +1.89 | +1.69 | +1.72 | +0.01 | +0.05 | -0.12 | +1.78 | +1.55 | 3.5e-16 |
| Annual | 51 | 1.028 | +1.86 | +2.22 | +2.11 | +0.10 | +0.33 | -0.05 | +1.91 | +2.32 | 0.0e+00 |

*All terms annualised percent. HAC lags 12 monthly, 4 quarterly, 2 annual.*

### 1b) Block bootstrap on the timing term (quarterly)

Resamples 4-quarter blocks and recomputes Cov(w,h) on each draw.

**5,000 block bootstraps, block length 4 quarters**

| | |
|---|---|
| timing term (point estimate) | +1.89 %/yr |
| HAC t | +1.69 |
| block-bootstrap 95% CI | [-0.48, +3.87] %/yr |
| P(timing <= 0) | 0.056 |

**The same, annual holding period, block length 2 years**

| | |
|---|---|
| timing term (point estimate) | +1.86 %/yr |
| HAC t | +2.22 |
| block-bootstrap 95% CI | [-0.10, +3.55] %/yr |
| P(timing <= 0) | 0.031 |

### 2) CP-only overlay, quarterly sub-periods

| label | n | meanW | TIMING %/yr | t | t(slope) | tilt %/yr | t(W=1) | cost %/yr | total %/yr | t | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1975-1989 | 58 | 1.074 | +2.82 | +1.70 | +1.95 | +0.50 | +0.56 | -0.14 | +3.18 | +1.62 | 6.9e-16 |
| 1990-2004 | 60 | 0.946 | +0.41 | +0.16 | +0.15 | -0.32 | -0.62 | -0.17 | -0.07 | -0.03 | 2.2e-16 |
| 2005-2019 | 60 | 1.049 | +0.59 | +0.50 | +0.52 | -0.05 | +0.46 | -0.08 | +0.47 | +0.39 | 0.0e+00 |
| 2020-2026 | 26 | 0.869 | +6.33 | +3.75 | +4.43 | -0.26 | -0.95 | -0.09 | +5.98 | +2.99 | 0.0e+00 |

*The four blocks partition the sample.*

### 2b) 1990-2004 split at the dot-com peak

| label | n | meanW | TIMING %/yr | t | t(slope) | tilt %/yr | t(W=1) | cost %/yr | total %/yr | t | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1990-1999 | 40 | 0.988 | +2.23 | +1.54 | +1.50 | -0.01 | -0.12 | -0.16 | +2.06 | +1.46 | 0.0e+00 |
| 2000-2004 | 20 | 0.861 | -1.91 | -0.37 | -0.36 | -2.26 | -0.90 | -0.18 | -4.34 | -0.56 | 0.0e+00 |

**Timed wealth divided by static wealth, quarterly compounding**

| | |
|---|---|
| timed / static wealth at end-1989 | 1.51 |
| at end-1999 | 1.84 |
| at end-2001 (the trough of the episode) | 1.32 |
| at end-2004 | 1.54 |
| minimum of the ratio over the whole sample | 1.010 |
| ever below 1 (i.e. ever behind the benchmark)? | False |
| worst quarter | 2000-12-31 |
| its active return | -24.0% |
| weight held into it | 0.15 |
| HML that quarter | +28.1% |

### 3) The seven CP x spread designs, quarterly

| label | n | meanW | TIMING %/yr | t | t(slope) | tilt %/yr | t(W=1) | cost %/yr | total %/yr | t | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CP-only | 204 | 1.003 | +1.89 | +1.69 | +1.72 | +0.01 | +0.05 | -0.12 | +1.78 | +1.55 | 3.5e-16 |
| Spread-only | 204 | 1.188 | +0.74 | +0.53 | +0.54 | +0.67 | +1.91 | -0.03 | +1.38 | +0.83 | 0.0e+00 |
| Additive | 204 | 1.095 | +1.32 | +1.92 | +2.15 | +0.34 | +1.72 | -0.07 | +1.59 | +2.06 | 3.5e-16 |
| Gated | 204 | 0.998 | +1.15 | +1.20 | +1.20 | -0.01 | -0.06 | -0.07 | +1.07 | +1.10 | 3.5e-16 |
| A monotonic | 204 | 0.997 | +2.07 | +1.24 | +1.22 | -0.01 | -0.04 | -0.14 | +1.92 | +1.12 | 0.0e+00 |
| B U-shaped | 204 | 0.956 | +2.64 | +1.62 | +1.61 | -0.16 | -0.60 | -0.15 | +2.33 | +1.36 | 3.5e-16 |
| S agreement | 204 | 1.042 | +2.32 | +2.47 | +2.80 | +0.15 | +0.63 | -0.12 | +2.34 | +2.47 | 3.5e-16 |

*The timing share of each design's gross alpha is in the next table.*

**Timing and tilt by design**

| design | meanW | timing %/yr | tilt %/yr | timing share of gross |
|---|---|---|---|---|
| CP-only | 1.003 | +1.89 | +0.01 | 99% |
| Spread-only | 1.188 | +0.74 | +0.67 | 53% |
| Additive | 1.095 | +1.32 | +0.34 | 79% |
| Gated | 0.998 | +1.15 | -0.01 | 101% |
| A monotonic | 0.997 | +2.07 | -0.01 | 100% |
| B U-shaped | 0.956 | +2.64 | -0.16 | 106% |
| S agreement | 1.042 | +2.32 | +0.15 | 94% |

### 3b) S agreement, quarterly sub-periods

| label | n | meanW | TIMING %/yr | t | t(slope) | tilt %/yr | t(W=1) | cost %/yr | total %/yr | t | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1975-1989 | 58 | 0.899 | +3.52 | +1.98 | +2.51 | -0.68 | -0.68 | -0.15 | +2.69 | +1.39 | 0.0e+00 |
| 1990-2004 | 60 | 1.149 | +2.25 | +1.22 | +1.30 | +0.87 | +1.74 | -0.16 | +2.96 | +1.67 | 3.5e-16 |
| 2005-2019 | 60 | 1.046 | +0.48 | +0.34 | +0.35 | -0.05 | +0.34 | -0.09 | +0.34 | +0.24 | 4.3e-17 |
| 2020-2026 | 26 | 1.108 | +4.62 | +2.20 | +3.71 | +0.21 | +0.80 | -0.07 | +4.76 | +1.95 | 2.1e-15 |

### 3c) Block bootstrap on the timing term, CP-only vs S agreement (quarterly)

| design | timing %/yr | HAC t | 95% CI | P(timing <= 0) |
|---|---|---|---|---|
| CP-only | +1.89 | +1.69 | [-0.48, +3.87] | 0.056 |
| S agreement | +2.32 | +2.47 | [+0.41, +3.97] | 0.007 |

*5,000 block bootstraps, block length 4 quarters.*

<a id="integrity_audit"></a>

## `integrity_audit.py` — Upstream integrity audit

*signal distribution, alpha decomposition, real-time truncation test, placebos, OOS bond R2*

Six checks on the CP to weight to return chain. The signal's distribution; the alpha decomposition; a truncation test rebuilding the chain on data cut at 2000, 2010 and 2018; randomisation placebos and the alpha at weight lags 0 to 4; the Campbell-Thompson out-of-sample bond R2 against an expanding historical mean; and alpha by first traded quarter.

Baseline quarterly strategy: n=204 1975-09-30..2026-06-30, alpha=+1.78%/yr, IR=0.26, NW t=1.55

### A) Signal distribution

| | |
|---|---|
| mean / SD of z | -0.024 / 0.717 |
| mean / SD of weight | 1.003 / 0.504 |
| weight min .. max | 0.001 .. 1.931 |
| share of quarters \|z\| > 1 | 0.11 |
| share of quarters \|w-1\| > 0.5 | 0.41 |
| weight AR(1) | +0.638 |

**Weight distribution**

| quantile | 0% | 10% | 25% | 50% | 75% | 90% | 100% |
|---|---|---|---|---|---|---|---|
| weight | 0.00 | 0.32 | 0.59 | 1.03 | 1.41 | 1.69 | 1.93 |

*Weight is 1 + tanh(z), so it is bounded in [0, 2] by construction.*

### B) Alpha decomposition

**Alpha decomposition (quarterly, 1975+)**

| component | ann. % | share of gross |
|---|---|---|
| timing  Cov(w, r) | +1.893 | 99.4% |
| tilt    (E[w]-1) E[r] | +0.010 | 0.6% |
| cost    -c E[turnover] | -0.122 |  |
| total | +1.782 |  |

| | |
|---|---|
| reported alpha | +1.7818 %/yr |
| decomposition total | +1.7818 %/yr |
| identity holds (< 1e-8) | True |
| mean weight | 1.0029 |
| => tilt contribution | +0.010 %/yr of +1.78 (0.6%) |

### C) Real-time truncation test

Rebuild the chain using only data available up to T, then compare every value dated before T against the full-sample run.

| truncated at T | CP obs | max \|dCP\| | max \|dz\| | max \|d active\| | verdict |
|---|---|---|---|---|---|
| 2000-12-31 | 344 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| 2010-12-31 | 464 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| 2018-12-31 | 560 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |

*Zero difference => no information from after T influences any value dated before T.*

**Estimation-window structure**

| | |
|---|---|
| estimation dates checked | 649 |
| smallest gap between last training obs and CP date | 12 months |
| required (12m return must be realized) | >= 12 months |
| assertion max(elig) <= t-12 | PASS |
| last 12 realized returns are NaN (unobservable) | 12/12 |

### D) Placebo and lag-decay tests

**Alpha against the weight lag k**

| weight lag | n | alpha%/yr | IR | NW t |
|---|---|---|---|---|
| k=0 CONTEMPORANEOUS (cheating) | 205 | +1.44 | +0.20 | +1.12 |
| k=1 baseline (honest) | 204 | +1.78 | +0.26 | +1.55 |
| k=2 extra lag | 203 | -0.09 | -0.01 | -0.08 |
| k=3 | 202 | +0.42 | +0.07 | +0.45 |
| k=4 (= 1 year) | 201 | +1.83 | +0.29 | +2.00 |

*Baseline standard error is ~0.95%/yr. k=1 is the traded convention.*

**Randomisation placebos (2000 draws, gross of cost)**

| placebo | mean alpha | 95% range | observed | pctile of observed |
|---|---|---|---|---|
| weight permuted in time | +0.02 | [-1.73, +1.82] | +1.78 | 97.3% |
| weight deviation sign-randomised | +0.02 | [-1.85, +1.93] | +1.78 | 96.8% |

*2000 draws, gross of cost. Each placebo breaks the weight-return pairing.*

### E) Out-of-sample bond R2

Campbell-Thompson OOS R2 against an expanding historical-mean forecast, using at each date only returns already realised.

| | |
|---|---|
| evaluation months | 637 |
| span | 1972-05-31 .. 2025-05-31 |
| OOS R2, full evaluation sample | -0.1037 |
| OOS R2, 1990+ | -0.3532 |
| in-sample full-sample R2, for reference | 0.153 |

### F) Burn-in stability: alpha by first traded quarter

| sample | n | alpha%/yr | IR | NW t | meanW |
|---|---|---|---|---|---|
| 1975+ baseline | 204 | +1.78 | +0.26 | +1.55 | 1.003 |
| 1980+ (drop first 5y) | 186 | +1.46 | +0.21 | +1.19 | 0.989 |
| 1985+ (drop first 10y) | 166 | +1.32 | +0.20 | +1.07 | 0.970 |
| 1990+ (drop first 15y) | 146 | +1.23 | +0.18 | +0.88 | 0.975 |

<a id="bond_link"></a>

## `bond_link.py` — Real-time CP against bond returns: level and shape

*out-of-sample R2 on the point forecast, and Cov(tanh(z_CP), 12m bond excess return)*

Two tests of real-time CP against realised 12-month bond excess returns: the out-of-sample R2 on the point forecast, and `Cov(tanh(z_CP), rx)` on the standardised signal the weight actually uses. Block 2 sorts bond returns into CP buckets. Block 3 repeats the covariance test on a December-only panel, which removes the overlap.

Evaluation sample: 602 months, 1975-04-30 -> 2025-05-31 rx = realised average 12m excess return on 2-5y bonds, dated at the month the position is opened.

### 1) Level and shape tests on one sample

| test | what it measures | statistic |
|---|---|---|
| R2_oos, full | level and scale of the point forecast | -0.1037 |
| R2_oos, 1990+ | level and scale, 1990 onward | -0.3532 |
| Cov(tanh z, rx) | standardised shape, the input to the weight | +0.3769 (NW t +1.67) |
| slope of rx on tanh z | the same null in regression form | +1.402 (NW t +1.80) |

### 2) Bond returns by CP bucket

| CP bucket | n months | next-12m bond excess return (%) | mean tanh(z) |
|---|---|---|---|
| deep low CP | 141 | +0.321 | -0.72 |
| low CP | 151 | +0.830 | -0.25 |
| high CP | 187 | +0.482 | +0.24 |
| deep high CP | 123 | +2.949 | +0.68 |

*Overlapping 12m windows, so n is not the effective sample size.*

| | |
|---|---|
| mean 12m bond excess return when CP above its own history | +1.461% |
| mean when CP below | +0.584% |
| spread | +0.876% |
| unconditional mean | +1.036% |
| share of months CP is above | 0.51 |

### 3) Non-overlapping check, December only

**December only: one observation per year, 12m horizon**

| | |
|---|---|
| non-overlapping observations | 50 |
| Cov(tanh z, rx) | +0.3733  (NW t +1.31) |
| slope of rx on tanh z | +1.253  (NW t +1.40) |
| R2 | 0.034 |

<a id="factor_alpha"></a>

## `factor_alpha.py` — Factor-model alpha - strategy return regressed on HML

*r_strategy = alpha + beta * r_HML + e (Newey-West), against the benchmark-difference alpha reported everywhere else*

The strategy return regressed on HML with beta estimated, reported beside the benchmark-difference alpha that forces beta to 1, and the gap between them. Covers the CP overlay by frequency and by sub-period, the seven CP x spread designs on their common sample, and a five-factor specification. Block 2 checks the identity between the timed and active regressions.

alpha_reg is the regression intercept, beta estimated. alpha_diff is mean(timed) - mean(HML), beta forced to 1. Both annualised percent. The last column is their gap, which equals (beta - 1) * E[r_HML].

### 1) CP-only overlay w = 1 + tanh(z_CP), by frequency (1975+, net 10bps)

| label | n | alpha_reg %/yr | NW t(a) | beta | NW t(b-1) | R2 | alpha_diff %/yr | diff - reg |
|---|---|---|---|---|---|---|---|---|
| Monthly | 611 | +1.65 | +1.82 | +0.914 | -0.85 | 0.734 | +1.36 | -0.29 |
| Quarterly | 204 | +2.19 | +2.56 | +0.885 | -1.01 | 0.738 | +1.78 | -0.41 |
| Annual | 51 | +2.04 | +2.66 | +0.965 | -0.37 | 0.778 | +1.91 | -0.13 |

*beta is the beta of the timed portfolio on HML. NW t(b-1) tests beta = 1.*

**E[r_HML] on each sample, the multiplier in the reconciliation**

| | |
|---|---|
| E[HML] on the Monthly sample | +3.39 %/yr |
| E[HML] on the Quarterly sample | +3.58 %/yr |
| E[HML] on the Annual sample | +3.70 %/yr |

### 2) Identity check: the active return regressed on HML

active = timed - hml, so regressing active on hml returns the same intercept with beta lower by exactly 1. The check column reports both differences.

| period | alpha (timed) | alpha (active) | \|d alpha\| | beta (timed) | beta (active) | \|d(beta-1)\| | check |
|---|---|---|---|---|---|---|---|
| Monthly | +1.6466 | +1.6466 | 6.66e-16 | +0.9140 | -0.0860 | 8.33e-16 | PASS |
| Quarterly | +2.1934 | +2.1934 | 2.66e-15 | +0.8850 | -0.1150 | 5.00e-16 | PASS |
| Annual | +2.0354 | +2.0354 | 4.44e-16 | +0.9649 | -0.0351 | 8.33e-17 | PASS |

### 3) CP-only overlay, quarterly sub-periods

| label | n | alpha_reg %/yr | NW t(a) | beta | NW t(b-1) | R2 | alpha_diff %/yr | diff - reg |
|---|---|---|---|---|---|---|---|---|
| 1975-1989 | 58 | +2.40 | +1.93 | +1.116 | +0.65 | 0.782 | +3.18 | +0.77 |
| 1990-2004 | 60 | +2.00 | +1.27 | +0.644 | -1.82 | 0.555 | -0.07 | -2.07 |
| 2005-2019 | 60 | +0.54 | +0.45 | +1.078 | +0.83 | 0.886 | +0.47 | -0.08 |
| 2020-2026 | 26 | +6.27 | +3.61 | +0.854 | -1.72 | 0.895 | +5.98 | -0.29 |

*Quarterly, net 10bps. Sub-periods partition the sample.*

### 4) The seven CP x spread designs, quarterly

All seven designs share one sample: the spread starts later than CP and its rolling percentile needs 40 quarters.

| label | n | alpha_reg %/yr | NW t(a) | beta | NW t(b-1) | R2 | alpha_diff %/yr | diff - reg |
|---|---|---|---|---|---|---|---|---|
| CP-only | 204 | +2.19 | +2.56 | +0.885 | -1.01 | 0.738 | +1.78 | -0.41 |
| Spread-only | 204 | -0.42 | -0.35 | +1.503 | +3.90 | 0.852 | +1.38 | +1.80 |
| Additive | 204 | +0.90 | +1.41 | +1.194 | +3.52 | 0.919 | +1.59 | +0.69 |
| Gated | 204 | +1.52 | +2.25 | +0.875 | -1.17 | 0.794 | +1.07 | -0.45 |
| A monotonic | 204 | +2.70 | +2.28 | +0.781 | -1.19 | 0.513 | +1.92 | -0.78 |
| B U-shaped | 204 | +3.21 | +2.75 | +0.754 | -1.37 | 0.499 | +2.33 | -0.88 |
| S agreement | 204 | +2.15 | +2.61 | +1.055 | +0.76 | 0.820 | +2.34 | +0.20 |

*On the spread's common sample, so the CP-only row differs from section 1's.*

### 4b) S agreement, quarterly sub-periods

| label | n | alpha_reg %/yr | NW t(a) | beta | NW t(b-1) | R2 | alpha_diff %/yr | diff - reg |
|---|---|---|---|---|---|---|---|---|
| 1975-1989 | 58 | +2.91 | +2.21 | +0.967 | -0.17 | 0.680 | +2.69 | -0.22 |
| 1990-2004 | 60 | +2.68 | +2.01 | +1.049 | +0.36 | 0.830 | +2.96 | +0.28 |
| 2005-2019 | 60 | +0.45 | +0.31 | +1.108 | +0.87 | 0.836 | +0.34 | -0.11 |
| 2020-2026 | 26 | +4.61 | +2.15 | +1.074 | +0.75 | 0.914 | +4.76 | +0.15 |

*Quarterly only: the spread percentile uses a 60-quarter rolling window.*

### 5) Five-factor spec (quarterly): r = a + b1 Mkt-RF + b2 SMB + b3 HML + b4 RMW + b5 CMA

Quarterly factors, compounded from daily. NW 4 lags.

| strategy | n | alpha %/yr | NW t(a) | Mkt-RF | SMB | HML | RMW | CMA | R2 |
|---|---|---|---|---|---|---|---|---|---|
| CP-only | 204 | +2.62 | +2.67 | +0.037 | -0.148 | +0.928 | -0.187 | +0.019 | 0.760 |
| S agreement | 204 | +1.71 | +1.72 | +0.043 | -0.071 | +1.009 | -0.022 | +0.145 | 0.825 |

*alpha is annualised percent; loadings are on quarterly factor returns.*

---

# The value spread

<a id="p2_driver"></a>

## `p2_driver.py` — Problem 2 - CP x value-spread

*alignment diagnostic, seven weighting schemes, mechanism, bootstrap on the best design*

Seven ways of combining CP's direction with the value spread's level, all on one common quarterly sample. Blocks are the alignment diagnostic between the two signals, the strategy comparison, next-quarter value returns sorted by CP direction against spread tail, return against risk with drawdown, and a bootstrap on the design with the highest IR.

Common quarterly sample: 206 obs 1975-03-31..2026-06-30

### 1) Alignment diagnostic (CP vs spread)

| | |
|---|---|
| corr(CP signal c, spread pctile p) | -0.018 |
| sign-agreement overall | 0.52 |
| sign-agreement in tails | 0.50 |

**Mean CP signal by spread tercile**

| spread tercile | mean CP signal |
|---|---|
| narrow | -0.040 |
| mid | +0.142 |
| wide | -0.086 |

### 2) Strategy comparison (quarterly, 1975+, net 10bps)

| scheme | n | alpha | IR | NWt | meanW | turn | corr(a,v) |
|---|---|---|---|---|---|---|---|
| CP-only | 204 | +1.78 | 0.26 | 1.55 | 1.003 | 1.22 | -0.21 |
| Spread-only | 204 | +1.38 | 0.14 | 0.83 | 1.188 | 0.34 | +0.63 |
| Additive | 204 | +1.59 | 0.31 | 2.06 | 1.095 | 0.68 | +0.48 |
| Gated | 204 | +1.07 | 0.18 | 1.10 | 0.998 | 0.67 | -0.27 |
| A monotonic | 204 | +1.92 | 0.19 | 1.12 | 0.997 | 1.41 | -0.28 |
| B U-shaped | 204 | +2.33 | 0.23 | 1.36 | 0.956 | 1.49 | -0.31 |
| S agreement | 204 | +2.34 | 0.37 | 2.47 | 1.042 | 1.23 | +0.11 |

### 3) Mechanism: next-qtr value return by CP direction x spread tail (ann.%)

| CP direction | spread tail | next-qtr value (ann.%) | n |
|---|---|---|---|
| CP overweight | narrow | +5.45 | 30 |
| CP overweight | mid | +4.21 | 31 |
| CP overweight | wide | +7.67 | 51 |
| CP underweight | narrow | -2.66 | 27 |
| CP underweight | mid | +3.96 | 16 |
| CP underweight | wide | +1.18 | 51 |

**Incremental active alpha vs CP-only, by CP/spread agreement**

| scheme | agree (ann.%) | conflict (ann.%) |
|---|---|---|
| A monotonic | +0.29 | -0.03 |
| B U-shaped | +0.79 | +0.29 |
| S agreement | +1.32 | -0.24 |

### 3b) Return against risk

**Return and risk, quarterly, 1975+, net 10bps**

| design | alpha %/yr | active vol %/yr | IR | traded vol %/yr | traded SR | max drawdown % |
|---|---|---|---|---|---|---|
| CP-only | +1.78 | 6.80 | 0.26 | 12.98 | 0.41 | -53.7 |
| Spread-only | +1.38 | 10.13 | 0.14 | 20.53 | 0.24 | -78.1 |
| S agreement | +2.34 | 6.26 | 0.37 | 14.68 | 0.40 | -61.0 |
| Static value (benchmark) | 0.00 | 0.00 | n/a | 12.60 | 0.28 | -56.0 |

### 4) Bootstrap on best-IR multiplicative design

| | |
|---|---|
| best-IR multiplicative design | S agreement |
| alpha 95% CI | [+0.47, +4.03] %/yr |
| P(alpha <= 0) | 0.006 |
| CP-only IR vs best IR | 0.26 vs 0.37 |

<a id="spread_checks"></a>

## `spread_checks.py` — S agreement: the same four checks

*truncation, rolling stability, timing versus tilt, and spanning against both the yield curve and its own two ingredients*

The four CP-only checks re-run on the S-agreement design. Truncation at 2000, 2010 and 2018, which cuts the value spread as well as the yield curve; the timing term on a rolling 20-quarter window against CP-only; the timing/tilt split; and spanning against the yield curve and then against the design's own two inputs.

w = 1 + c (1 + 0.75 d sign(c)),  c = tanh(z_CP),  d = 2(p - 0.5) The multiplier 1 + 0.75 d sign(c) lies in [0.25, 1.75] and scales conviction only. Baseline: n=204, 1975-09-30 to 2026-06-30, alpha +2.34%/yr, IR 0.37, NW t 2.47

### 1) Real-time truncation test

Rebuild CP, the spread percentile, the weight and the traded return using only data available up to T. The spread is truncated as well as the yield curve.

| truncated at T | quarters kept | overlap | max \|d weight\| | max \|d active\| | verdict |
|---|---|---|---|---|---|
| 2000-12-31 | 102 | 102 | 0.00e+00 | 0.00e+00 | PASS |
| 2010-12-31 | 142 | 142 | 0.00e+00 | 0.00e+00 | PASS |
| 2018-12-31 | 174 | 174 | 0.00e+00 | 0.00e+00 | PASS |

*Zero difference means no information dated after T influences any value before it.*

### 2) Rolling stability

Cov(w,h) on a moving 20-quarter window, annualised percent, against CP-only.

| design | windows | share positive | worst | best | median |
|---|---|---|---|---|---|
| CP-only | 185 | 69% | -3.80 | +7.26 | +1.35 |
| S agreement | 185 | 78% | -2.90 | +6.51 | +2.25 |

*Windows overlap, so the share is descriptive and not a significance statement.*

### 3) Timing versus tilt

| design | mean w | timing %/yr | t | tilt %/yr | t(w=1) | cost %/yr | total %/yr | timing share | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|
| CP-only | 1.003 | +1.89 | +1.69 | +0.01 | +0.05 | -0.12 | +1.78 | 99% | 3.5e-16 |
| S agreement | 1.042 | +2.32 | +2.47 | +0.15 | +0.63 | -0.12 | +2.34 | 94% | 3.5e-16 |

*All terms annualised percent. timing share = timing / (timing + tilt).*

### 4a) Spanning against the yield curve

Slope, level and curvature timed through identical machinery, then S agreement's active return regressed on theirs.

**Spanning regression against the yield curve**

| | |
|---|---|
| S agreement active return, raw | +2.34 %/yr |
| not explained by slope, level, curvature | +1.88 %/yr |
| NW t on that intercept | +2.92 |
| R2 of S on the three controls | 0.475 |
| n quarters | 204 |

### 4b) Spanning against its own two ingredients

S agreement's active return regressed on the active returns of its two inputs.

**Spanning regression against CP-only and spread-only**

| | |
|---|---|
| not explained by CP-only and spread-only | +0.30 %/yr |
| NW t on that intercept | +0.88 |
| loading on CP-only | +0.893  (t +11.53) |
| loading on spread-only | +0.331  (t +8.09) |
| R2 | 0.885 |
| n quarters | 204 |

---

# International

<a id="country_cp"></a>

## `country_cp.py` — Country CP factors: construction & validation

*full-sample CP R2, forward-rate coefficients and real-time CP span per market*

Each foreign CP factor built from its own curve under the same rules: the span and observation count of the yield data, the full-sample regression R2, the five forward-rate coefficients, and the span of the resulting real-time factor. The US figure is given above the table as a reference.

US anchor: full-sample CP R2 = 0.153

| market | yields span | obs | CP R2 | n | coefs f1..f5 | real-time CP span | CP obs |
|---|---|---|---|---|---|---|---|
| UK (BoE gilt zero-coupon, 1979+) | 1979-01-02..2026-05-29 | 11966 | 0.201 | 557 | +0.30 -6.49 +12.38 -7.14 +1.12 | 1989-12-31..2026-05-31 | 438 |
| Canada (BoC zero-coupon, 1986+) | 1986-01-02..2026-06-03 | 9923 | 0.152 | 474 | -1.01 +1.31 -3.40 +6.99 -3.69 | 1996-12-31..2026-06-30 | 355 |
| Japan (MoF JGB constant-maturity, 1974+; coupon proxy) | 1974-09-24..2026-05-29 | 12578 | 0.235 | 583 | -1.12 -0.12 +0.81 +0.09 +0.57 | 1987-10-31..2026-05-31 | 464 |
| Germany (Bundesbank Svensson zero-coupon, 1972+) | 1972-09-30..2026-05-31 | 645 | 0.126 | 633 | -0.55 +0.84 -2.95 +3.85 -0.95 | 1983-08-31..2026-05-31 | 514 |

<a id="cp_intl"></a>

## `cp_intl.py` — International CP-only timing

*each country's own CP times its own HML (AQR Devil-in-HMLs, HML FF); US shown for reference*

Each market's own CP applied to its own value factor, quarterly and net of 10bps per unit turnover, with the US on two start dates for reference. The foreign value factors are AQR's HML FF series. The second table splits each market at the midpoint of its own sample.

**Own CP -> own HML FF (quarterly, net 10bps)**

| market | n | start | end | staticSR | timedSR | alpha%/yr | IR | NW t | meanW | turn |
|---|---|---|---|---|---|---|---|---|---|---|
| US 1975+ | 204 | 1975-09-30 | 2026-06-30 | +0.28 | +0.41 | +1.78 | +0.26 | +1.55 | 1.00 | 1.22 |
| US 1992+ | 138 | 1992-03-31 | 2026-06-30 | +0.23 | +0.34 | +1.24 | +0.18 | +0.84 | 0.97 | 1.11 |
| UK | 133 | 1993-03-31 | 2026-03-31 | +0.49 | +0.34 | -0.01 | -0.00 | -0.01 | 1.25 | 0.93 |
| Canada | 105 | 2000-03-31 | 2026-03-31 | +0.60 | +0.57 | +3.38 | +0.38 | +1.57 | 1.16 | 0.87 |
| Japan | 141 | 1991-03-31 | 2026-03-31 | +0.55 | +0.57 | +0.03 | +0.01 | +0.03 | 0.86 | 0.95 |
| Germany | 141 | 1991-03-31 | 2026-03-31 | +0.57 | +0.55 | -0.46 | -0.05 | -0.24 | 1.12 | 0.76 |

**First / second half**

| market | n | start | end | staticSR | timedSR | alpha%/yr | IR | NW t | meanW | turn |
|---|---|---|---|---|---|---|---|---|---|---|
| UK H1 | 66 | 1993-03-31 | 2009-06-30 | +0.59 | +0.42 | -0.28 | -0.04 | -0.14 | 1.22 | 1.09 |
| UK H2 | 67 | 2009-09-30 | 2026-03-31 | +0.37 | +0.27 | +0.25 | +0.04 | +0.15 | 1.28 | 0.77 |
| Canada H1 | 52 | 2000-03-31 | 2012-12-31 | +0.73 | +0.64 | +3.47 | +0.35 | +1.11 | 1.23 | 1.05 |
| Canada H2 | 53 | 2013-03-31 | 2026-03-31 | +0.48 | +0.50 | +3.29 | +0.42 | +1.12 | 1.09 | 0.70 |
| Japan H1 | 70 | 1991-03-31 | 2008-06-30 | +0.70 | +0.67 | -0.87 | -0.14 | -0.56 | 1.01 | 1.33 |
| Japan H2 | 71 | 2008-09-30 | 2026-03-31 | +0.38 | +0.48 | +0.92 | +0.18 | +0.71 | 0.72 | 0.58 |
| Germany H1 | 70 | 1991-03-31 | 2008-06-30 | +0.63 | +0.51 | -4.53 | -0.42 | -1.38 | 0.90 | 0.85 |
| Germany H2 | 71 | 2008-09-30 | 2026-03-31 | +0.51 | +0.58 | +3.55 | +0.69 | +2.26 | 1.34 | 0.67 |

| | |
|---|---|
| mean non-US alpha | +0.74%/yr |
| positive markets | 2/4 |
| naive US-CP-on-foreign test (for contrast) | -0.78%/yr, 2/7 positive |

<a id="decomposition_intl"></a>

## `decomposition_intl.py` — Timing vs tilt decomposition - international

*mean weight, the timing/tilt split per market, and a recentred variant*

The timing/tilt split per market. Block 1 reports mean weight with the drift of each market's own z-score, 1b varies the z-score burn-in, and 1c rebuilds the US chain from each foreign start date and compares it against the US on its full history over the same quarters. Block 4 reports a recentred variant of the signal.

### 1) Mean weight by market

The design intends E[w] = 1: an expanding z centres on zero, so E[tanh z] ~ 0.

| market | n CP qtrs | mean z | E[tanh z] | mean z, first third | mean z, last third | CP drift /yr |
|---|---|---|---|---|---|---|
| US | 217 | -0.016 | +0.006 | +0.017 | +0.086 | -0.003 |
| UK | 147 | +0.349 | +0.243 | +0.411 | +0.245 | +0.038 |
| Canada | 119 | +0.205 | +0.166 | +0.207 | -0.078 | +0.020 |
| Japan | 155 | -0.071 | -0.113 | +0.117 | -0.133 | -0.007 |
| Germany | 172 | +0.103 | +0.114 | -0.175 | +0.361 | +0.019 |

*z is the expanding z-score of each market's own CP. CP drift is the OLS slope of CP on time, per year. First and last thirds split the CP quarters evenly.*

### 1b) Mean weight against the z-score burn-in

| market | meanW z12 | alpha z12 | meanW z24 | alpha z24 | meanW z40 | alpha z40 |
|---|---|---|---|---|---|---|
| US | 1.003 | +1.78 | 0.977 | +1.52 | 1.013 | +1.85 |
| UK | 1.252 | -0.01 | 1.214 | -0.07 | 1.252 | +0.69 |
| Canada | 1.156 | +3.38 | 1.123 | +2.68 | 1.165 | +2.85 |
| Japan | 0.865 | +0.03 | 0.870 | +0.27 | 0.772 | +0.09 |
| Germany | 1.125 | -0.46 | 1.125 | -0.46 | 1.172 | -0.33 |

*z12 / z24 / z40 are the z-score burn-in, in quarters.*

### 1c) US curve truncated to each foreign start date

Truncate the US yield curve to each foreign market's start date and rebuild the whole chain from there, then run the full 1961 history over the identical traded window.

| US history matched to | n | traded from | meanW short | meanW full | satur. short | satur. full | timing short | timing full | t short | t full |
|---|---|---|---|---|---|---|---|---|---|---|
| Germany-length (1972+) | 159 | 1986-12-31 | 0.952 | 0.958 | 13.8% | 4.4% | +1.57 | +1.48 | +1.08 | +1.19 |
| Japan-length (1974+) | 151 | 1988-12-31 | 1.181 | 0.969 | 19.2% | 4.6% | +1.29 | +1.43 | +0.76 | +1.09 |
| UK-length (1979+) | 134 | 1993-03-31 | 1.154 | 0.982 | 21.6% | 5.2% | +1.58 | +1.50 | +0.81 | +1.01 |
| Canada-length (1986+) | 106 | 2000-03-31 | 1.362 | 0.969 | 24.5% | 4.7% | -0.12 | +1.10 | -0.06 | +0.62 |

*'short' is the US rebuilt from the foreign start date; 'full' is the US on its 1961 history over the same quarters. Saturation is the share of traded quarters within 0.2 of the tanh bounds.*

| market | n | meanW | saturation | timing %/yr | t |
|---|---|---|---|---|---|
| US (full history) | 204 | 1.003 | 6.4% | +1.89 | +1.69 |
| UK | 133 | 1.252 | 15.8% | -1.57 | -1.51 |
| Canada | 105 | 1.156 | 14.3% | +1.92 | +1.13 |
| Japan | 141 | 0.865 | 11.3% | +0.93 | +1.13 |
| Germany | 141 | 1.125 | 10.6% | -1.41 | -0.74 |

*Saturation is the share of traded quarters within 0.2 of the tanh bounds.*

### 2) Timing and tilt by market

Cov(w,h) is unchanged by adding a constant to w, so timing is invariant to mean weight.

| market | n | meanW | TIMING %/yr | t | t(slope) | tilt %/yr | t(W=1) | cost %/yr | total %/yr | t | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| US | 204 | 1.003 | +1.89 | +1.69 | +1.72 | +0.01 | +0.05 | -0.12 | +1.78 | +1.55 | 3.5e-16 |
| UK | 133 | 1.252 | -1.57 | -1.51 | -1.44 | +1.65 | +3.75 | -0.09 | -0.01 | -0.01 | 2.2e-16 |
| Canada | 105 | 1.156 | +1.92 | +1.13 | +1.22 | +1.55 | +1.84 | -0.09 | +3.38 | +1.57 | 6.9e-16 |
| Japan | 141 | 0.865 | +0.93 | +1.13 | +1.10 | -0.80 | -1.54 | -0.10 | +0.03 | +0.03 | 5.4e-17 |
| Germany | 141 | 1.125 | -1.41 | -0.74 | -0.74 | +1.02 | +1.56 | -0.08 | -0.46 | -0.24 | 3.5e-16 |

*All terms annualised percent. |resid| is total minus the three terms.*

### 2b) First / second half by market, timing term only

| market | n | meanW | TIMING %/yr | t | t(slope) | tilt %/yr | t(W=1) | cost %/yr | total %/yr | t | \|resid\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UK H1 | 66 | 1.221 | -2.03 | -1.14 | -1.06 | +1.85 | +2.22 | -0.11 | -0.28 | -0.14 | 1.3e-16 |
| UK H2 | 67 | 1.283 | -1.00 | -0.93 | -0.96 | +1.33 | +3.24 | -0.08 | +0.25 | +0.15 | 2.2e-16 |
| Canada H1 | 52 | 1.225 | +0.99 | +0.37 | +0.39 | +2.59 | +2.16 | -0.10 | +3.47 | +1.11 | 6.9e-16 |
| Canada H2 | 53 | 1.089 | +2.62 | +1.14 | +1.21 | +0.74 | +0.68 | -0.07 | +3.29 | +1.12 | 1.4e-15 |
| Japan H1 | 70 | 1.008 | -0.80 | -0.65 | -0.67 | +0.07 | +0.06 | -0.13 | -0.87 | -0.56 | 0.0e+00 |
| Japan H2 | 71 | 0.723 | +2.08 | +1.73 | +2.24 | -1.10 | -2.70 | -0.06 | +0.92 | +0.71 | 5.2e-16 |
| Germany H1 | 70 | 0.905 | -3.40 | -1.22 | -1.21 | -1.05 | -0.78 | -0.08 | -4.53 | -1.38 | 0.0e+00 |
| Germany H2 | 71 | 1.341 | +1.77 | +2.66 | +2.81 | +1.84 | +5.67 | -0.07 | +3.55 | +2.26 | 0.0e+00 |

### 3) Summary across the four non-US markets

| | |
|---|---|
| mean TIMING term | -0.03 %/yr |
| positive on timing | 2/4 |
| markets with \|t(timing)\| > 2 | 0/4 |
| mean tilt term | +0.85 %/yr |
| mean total (cp_intl's alpha) | +0.74 %/yr |
| positive on total | 2/4 |
| US timing, for reference | +1.89 %/yr (t +1.69) |

### 4) Recentred signal, for contrast

w = 1 + [tanh(z) - expanding mean of tanh(z)], which forces E[w] toward 1. A change to the strategy, not to the reporting. Not adopted.

| market | meanW base | meanW recentred | timing base | timing recentred | tilt base | tilt recentred | total base | total recentred | t |
|---|---|---|---|---|---|---|---|---|---|
| US | 1.003 | 0.930 | +1.89 | +1.71 | +0.01 | -0.21 | +1.78 | +1.37 | +1.15 |
| UK | 1.252 | 0.946 | -1.57 | -1.29 | +1.65 | -0.34 | -0.01 | -1.73 | -1.71 |
| Canada | 1.156 | 0.899 | +1.92 | +2.04 | +1.55 | -0.73 | +3.38 | +1.23 | +0.99 |
| Japan | 0.865 | 0.897 | +0.93 | +1.18 | -0.80 | -0.58 | +0.03 | +0.51 | +0.47 |
| Germany | 1.125 | 1.138 | -1.41 | -1.21 | +1.02 | +1.13 | -0.46 | -0.15 | -0.08 |

*'base' is the shipped design, 'recentred' the variant above.*

---

# Estimation choices

<a id="r2_investigation"></a>

## `r2_investigation.py` — CP regression R2 by era

*R2 of the forwards-on-12m-bond-return regression, by era and by sampling frequency*

The R2 of the forwards-on-12-month-bond-return regression, by era and by sampling frequency, including the annual December-only specification that removes the overlap.

### Forwards -> 12m avg bond excess return (overlapping monthly), R2 by era

| era | R2 | n |
|---|---|---|
| full 1962-2025 | 0.153 | 768 |
| CP05 window 1964-2003 | 0.243 | 468 |
| 1964-2008 | 0.180 | 528 |
| 1971-2003 | 0.228 | 384 |
| 1990-2026 | 0.236 | 425 |
| post-2003 (ZIRP era) | 0.232 | 257 |
| 2009-2021 ZIRP core | 0.440 | 144 |

### Annual (December-only, non-overlapping) regression

| annual (Dec only) | R2 | n | window |
|---|---|---|---|
| full sample | 0.207 | 64 |  |
| 1964-2003 | 0.334 | 39 | CP05 window |

<a id="rolling_cp"></a>

## `rolling_cp.py` — Rolling-window CP vs expanding baseline (US)

*rolling-window CP re-estimation against the expanding baseline, 120/180/240 months*

Rolling-window CP estimation at 120, 180 and 240 months, then the expanding baseline and the 180-month rolling version compared across frequencies and quarterly sub-periods.

### Rolling-window sensitivity (quarterly, 1975+)

| window | n | alpha%/yr | IR | NW t |
|---|---|---|---|---|
| 120m | 206 | +1.14 | +0.16 | +1.07 |
| 180m | 206 | +0.72 | +0.09 | +0.60 |
| 240m | 206 | +0.63 | +0.10 | +0.65 |

### Expanding CP (baseline)

**Frequency (start 1975)**

| frequency | n | alpha%/yr | IR | NW t | meanW | turn/yr |
|---|---|---|---|---|---|---|
| Monthly | 611 | +1.36 | +0.23 | +1.22 | 0.98 | 2.21 |
| Quarterly | 204 | +1.78 | +0.26 | +1.55 | 1.00 | 1.22 |
| Annual | 51 | +1.91 | +0.26 | +2.32 | 1.03 | 0.55 |

**Quarterly sub-periods**

| sub-period | n | alpha%/yr | IR | NW t |
|---|---|---|---|---|
| 1975-1989 | 58 | +3.18 | +0.50 | +1.62 |
| 1990-2004 | 60 | -0.07 | -0.01 | -0.03 |
| 2005-2019 | 60 | +0.47 | +0.11 | +0.39 |
| 2020-2026 | 26 | +5.98 | +1.03 | +2.99 |

### Rolling CP, 15-year window

**Frequency (start 1975)**

| frequency | n | alpha%/yr | IR | NW t | meanW | turn/yr |
|---|---|---|---|---|---|---|
| Monthly | 616 | +0.67 | +0.11 | +0.63 | 1.02 | 2.16 |
| Quarterly | 206 | +0.72 | +0.09 | +0.60 | 1.05 | 1.23 |
| Annual | 52 | -0.23 | -0.03 | -0.24 | 1.04 | 0.52 |

**Quarterly sub-periods**

| sub-period | n | alpha%/yr | IR | NW t |
|---|---|---|---|---|
| 1975-1989 | 60 | +2.53 | +0.38 | +1.27 |
| 1990-2004 | 60 | +0.35 | +0.04 | +0.11 |
| 2005-2019 | 60 | -0.99 | -0.18 | -0.61 |
| 2020-2026 | 26 | +1.35 | +0.14 | +0.63 |

