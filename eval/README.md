# Out-of-sample evaluation of the FinRL stock-trading tutorial

Evaluation branch, never meant for upstream. It answers one question: does the pipeline in
`examples/FinRL_StockTrading_2026_*.py` (Dow 30, five Stable-Baselines3 agents, tutorial hyperparameters)
produce out-of-sample results distinguishable from an equal-weight basket, and how much do they vary with the seed?

## Design

- Universe: the point-in-time Dow 30 of `finrl.meta.preprocessor.dow`, with the tutorial's membership mask.
- Data: `data/raw_prices.csv.gz` is Yahoo Finance OHLCV, 2014-01-02 to 2026-03-20, in the layout of
  `YahooDownloader.fetch_data` (adjusted prices), downloaded once by `download_raw.py` so the CI jobs are offline.
  DWDP, UTX and WBA have no Yahoo history any more; `prepare_data.py` marks them non-member on the days they
  were constituents, so the basket has 29 names on those days (the shipped tutorial raises on this).
- Features: `FeatureEngineer` with the tutorial's indicators, turbulence and VIX (`prepare_data.py`).
- Windows: train 2014-01-06 to 2021-12-31, test 2022-01-01 to 2026-03-20 (about 1,060 trading days).
- Agents: A2C, DDPG, PPO, TD3, SAC with the tutorial's parameters (`train_eval.py`), several seeds, two
  training budgets: the tutorial's 20,000 steps and 100,000 steps. Rollouts are deterministic.
- Baselines: equal-weight point-in-time Dow basket rebalanced daily (the tutorial's own baseline, on the same
  adjusted prices), the DJIA price index, and cash.
- Statistics (`aggregate.py`): per run and for the seed-averaged series, Sharpe difference against equal weight
  with a stationary-bootstrap 95% interval, Newey-West t of the daily excess return, in-sample vs out-of-sample
  Sharpe, trading activity, and the minimum Sharpe improvement the window can detect at 80% power.

## Running

Every push to an `eval/**` branch runs `.github/workflows/evaluate.yml`: `prepare` builds the frame,
`train` is a matrix over `seeds` x `timesteps` from `config.json` (each job trains all `algos`), `aggregate`
writes `report.md` and figures to the `evaluation-report` artifact.

## Results (run 42487102801, 2026-11-08)

`report/report.md` and the figures next to it. Fifty runs: out-of-sample Sharpe 0.57 +/- 0.22 against 0.65 for the
rebalanced equal-weight basket and a random buy-and-hold median of 0.66; all bootstrap intervals for the Sharpe
difference contain zero. The agents become fully invested within a median of six days and barely trade afterwards;
DDPG and TD3 produce identical test series at 20k and 100k steps for four of five seeds because their actions
saturate at the 100-share cap on day one. Details in `report/report.md`.

## Second grid (run 42504959639): observations x reward

`config2.json`, `train_eval2.py`, `.github/workflows/evaluate2.yml`; results in `report2/`. Observations raw or
VecNormalize, reward dollar / log return / excess over equal weight, five algorithms, five seeds, 20k steps: 150
runs, Sharpe 0.52 +/- 0.24 against 0.65 for equal weight; a one-way ANOVA over the 30 cells gives p = 0.74. The
repeated raw+dollar cell reproduced 15 of 25 first-grid runs bit for bit on different runners.
