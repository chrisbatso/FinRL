# How much do FinRL's state features know? An information-coefficient test on the Dow data

Date: 9 November 2026. Data: the prices committed on the fork branch `eval2/obs-and-reward` under `eval/data/`
(Yahoo adjusted closes, 38 Dow constituents past and present, 2 January 2014 to 20 March 2026, plus VIX).
Code: `ic_test.py`, `null_reconstruction.py`, `timing_overlay.py`, `ex2020_check.py`, `plots.py` in this folder.
Everything ran locally in a few minutes; nothing was posted upstream.

## Question

The two training grids showed that FinRL's agents learn nothing distinguishable from a random allocation,
whatever the algorithm, observation scaling or reward. That leaves two explanations: the learner fails, or the
inputs carry no information. This test isolates the second. It takes the exact features the tutorial puts in
the state (the eight stockstats indicators `macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, close_30_sma,
close_60_sma`, plus the close itself), computes them per ticker as `FeatureEngineer` does, and measures how
much each one knows about forward returns, before any reinforcement learning touches them.

## Design

Universe: FinRL's point-in-time Dow membership re-encoded from `dow.py`; 27 to 30 members per trading day
(median 29; three delisted names have no Yahoo prices). Windows: the agents' training window, 6 January 2014
to 31 December 2021 (2,013 days after indicator warm-up), and the test window, 1 January 2022 to 20 March 2026
(about 1,050 days). Horizons: 1, 5 and 21 trading days, close to close.

Features in three groups:

- raw, as fed to the agent: `close` and the eight indicators; five of them are price levels;
- stationary transforms of the same indicators: MACD / close, Bollinger position inside the band, RSI 30,
  CCI 30, DX 30, close / SMA 30 − 1, close / SMA 60 − 1;
- two reference signals from the literature, not FinRL features: 12-1 momentum and 1-month reversal.

Three tests, each with its own null:

1. **Stock picking.** Cross-sectional Spearman rank correlation between the feature and the forward return
   across the members on each day. Mean over the window, Newey-West t with lag equal to the horizon, hit rate,
   and the smallest IC the window can detect at 80 % power (2.8 standard errors). 45 tests per window;
   Bonferroni |t| = 3.26; expected largest |t| under the null about 2.8.
2. **Timing one name.** Spearman over time between a ticker's feature and its own forward return, averaged
   across tickers. A circular time shift of the return panel is the usual null; it is wrong here, because
   price levels, moving averages, RSI and Bollinger position are built from the very returns they are then
   correlated with, and under a random walk that correlation is biased negative in finite samples (Stambaugh
   1999). So the second null rebuilds everything: 150 stationary block bootstraps (mean block 20 days, the same
   blocks for every ticker so cross-correlation survives) of the daily return panel, from which prices,
   high/low/open, all indicators, the turbulence index and the equal-weight basket are recomputed and every
   statistic rerun.
3. **Market timing.** Spearman over time between a market-level signal (VIX, FinRL's turbulence index, the
   member-mean of each stationary feature) and the forward return of the equal-weight member basket, against
   both nulls.

Two translations into money: a daily rank long-short portfolio per feature (gross leverage 1, sign fixed on
the training window, FinRL's 0.1 % cost per trade), and long-only exposure overlays on the equal-weight
basket, including FinRL's own rule of selling everything when the risk indicator exceeds its threshold.

## Results

### 1. Stock picking: nothing, with enough power to have seen something useful

| window | tests | abs t > 1.96 | abs t > Bonferroni | largest abs t | smallest detectable 1-day IC |
|---|---:|---:|---:|---|---:|
| training | 45 | 2 | 0 | 2.46, Bollinger position, 1 day, IC −0.014 | 0.017 |
| test | 45 | 3 | 0 | 2.50, DX 30, 21 days, IC +0.050 | 0.024 |

Two and three exceedances of 1.96 are what 45 null tests produce (2.25 expected). No FinRL feature keeps its
sign across the windows: Bollinger position goes from −0.014 to +0.005, close / SMA 30 from −0.011 to +0.003.
The raw price-level features (close, both bands, both moving averages) have ICs within ±0.005 of zero and of
each other: ranking stocks by a Bollinger band is ranking them by price. A daily rank IC on 30 names has a
null standard deviation of 0.19, so a mean of 0.017 over 2,000 days is the detection floor; a useful picking
signal on this breadth would need an IC of 0.03 to 0.05, well above it.

The one steady climber in the cumulative plot is the reference 12-1 momentum signal, +0.016 (t 2.1) in
training and +0.019 (t 2.05) in test, the same sign in both windows. It still does not make money here: the
daily rank long-short portfolio on it has a gross Sharpe of −0.06 in both windows, because its hit rate is
below one half (0.46 in training) and it loses on the high-dispersion days that dominate portfolio returns.

### 2. Rank long-short portfolios: negative after FinRL's costs, every one

Gross Sharpe in the test window runs from −0.62 (Bollinger position) to +0.06 (MACD). Net of 0.1 % per trade
every portfolio is negative, between −0.3 and −2.2, because a daily rank portfolio turns over 10 % to 47 % of
its book a day, which costs 3 % to 12 % a year. Any feature-driven daily picking on this universe is eaten
by the tutorial's own transaction cost before it starts.

### 3. Timing one name: the circular-shift null lies, the reconstruction null does not

Under the circular-shift null 29 of 45 timing tests in training and 32 of 45 in test are "significant" at 5 %,
all negative: high price, high RSI, high close / SMA, lower return ahead. Under the reconstruction null they
vanish. Random walks produce the same numbers: for the price level at 21 days the observed mean Spearman is
−0.150 in training against a null mean of −0.170 and a 5th to 95th percentile band of −0.227 to −0.107. Eight
of 45 training tests fall outside the 90 % band (4.5 expected), all oscillators at 21 days with one-sided p
between 0.03 and 0.05; in the test window 2 of 45 do, and none of the training eight repeats. The negative
"mean reversion" that a naive test would report for every FinRL feature is construction bias.

### 4. Market timing: one candidate, which loses money when applied

Training window: 8 of 24 tests fall outside the reconstruction band, all index-level mean reversion. Mean RSI
30 against the basket's 21-day forward return is −0.278 against a band of −0.236 to +0.047; VIX is +0.247
(circular-shift p 0.002, VIX is exogenous so that null is fair for it). It is not a 2020 artefact: without
calendar 2020 the RSI figure is −0.256, and the sign repeats in 12 of 13 calendar years. In the test window
the point estimates keep their signs (RSI −0.206, VIX +0.085) but sit inside the null band (−0.309 to +0.064;
p 0.43 for VIX), and the single exceedance is of the opposite sign.

Turned into prespecified long-only exposure rules (exposure = 1 − trailing-year percentile of mean RSI; or =
trailing-year percentile of VIX; or their average), rebalanced daily at 0.1 % cost, and compared with a static
position at the same mean exposure:

| window | rule | Sharpe | static at same exposure | equal weight |
|---|---|---:|---:|---:|
| training | sell strength (RSI) | 0.54 | 0.81 | 0.84 |
| training | buy fear (VIX) | 0.40 | 0.81 | 0.84 |
| test | sell strength (RSI) | 0.41 | 0.64 | 0.64 |
| test | buy fear (VIX) | 0.31 | 0.64 | 0.64 |
| test | average of both | 0.38 | 0.64 | 0.64 |

The rules lose in the window that motivated them as well as out of sample. Timing raises volatility (0.108
against 0.076 for the static position) without raising return; the bootstrap 90 % interval of the test
Sharpe difference is −0.54 to +0.07 for RSI and −0.68 to −0.02 for VIX. A rank correlation of −0.2 at 21
days, even if real, is not a tradable signal on one index.

FinRL's own risk overlay fares the same. The tutorial's VIX > 70 rule never fires in the test window; in
training it fires on 5 days, sells at the March 2020 bottom, and lowers the Sharpe from 0.84 to 0.82 with a
deeper drawdown. A turbulence threshold at the training 99th percentile lifts training Sharpe to 0.90 (the
threshold was chosen there) and test Sharpe to 0.72 on the strength of 3 days out of the market.

## What this says about "improving the accuracy of the method"

The accuracy ceiling of FinRL's tutorial, as configured, is set by its inputs, and that ceiling is zero to the
precision this data allows. The eight indicators carry no cross-sectional information about which Dow stock
does better next (IC below 0.017 at one day), no timing information beyond what random walks fake, and the
single market-level candidate loses money when traded. An agent fed these features can only learn what the
equal-weight basket already does: stay invested. That is exactly what the grids found it doing, and no
algorithm, reward or normalisation can change it, because the learner is not where the information is missing.

Improvement therefore has to come from outside the learner:

- features with demonstrated predictive content, measured with a test like this one before training, not after;
- breadth: hundreds of names with delistings handled, so that a small IC becomes a detectable Sharpe;
- a method-level caution: any time-series test of indicators built from prices needs a reconstruction null;
  the circular-shift null would have certified 61 of 90 timing effects here.

## Files

`ic_cross_sectional.csv`, `ic_timing_per_ticker.csv`, `ic_market_timing.csv`, `null_reconstruction.csv`
(plus draws and metadata), `rank_long_short.csv`, `risk_overlay.csv`, `timing_overlay.csv`,
`market_timing_ex2020.csv`, `market_timing_by_year.csv`, `summary.json`, `cs_ic_daily_h1.csv`,
`market_series.csv`; figures `ic_bars.png`, `ic_cumulative.png`, `ts_ic_vs_null.png`.
