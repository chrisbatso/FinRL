"""Prespecified long-only timing overlays on the equal-weight Dow basket, judged on the test window.

Two signals the IC tables flagged as candidates, each turned into exposure in [0, 1] with one fixed rule:
  rsi:  exposure = 1 - trailing-252-day percentile rank of the cross-sectional mean RSI 30  (sell strength)
  vix:  exposure = trailing-252-day percentile rank of VIX                                 (buy fear)
  both: average of the two
Exposure set at the close of day t applies to the return from t to t+1; changes cost 0.1 % of the traded exposure.
A 'static' row holds the same average exposure as each rule but constant, so the comparison isolates timing from
simply holding less equity. Reported for the training window (where the rule was motivated) and the test window.
"""
import math, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ic_test import RAW, UNIVERSE, TRAIN, TEST, COST_PER_TRADE, membership, stockstats_features, window
raw = pd.read_csv(sys.argv[1], parse_dates=["date"]); raw = raw[raw.tic.isin(UNIVERSE)]
feats = stockstats_features(raw)
dates = pd.DatetimeIndex(np.sort(raw.date.unique())); tics = sorted(raw.tic.unique())
wide = {c: feats.pivot(index="date", columns="tic", values=c).reindex(index=dates, columns=tics) for c in ["close", "rsi_30"]}
close = wide["close"]; member = membership(dates, tics) & close.notna()
ew1 = (close.shift(-1) / close - 1).where(member).mean(axis=1)
mean_rsi = wide["rsi_30"].where(member).mean(axis=1)
vix = pd.read_csv(sys.argv[2], parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
def trailing_pct(s, n=252):
    return s.rolling(n, min_periods=n).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
expo = {"rsi (sell strength)": 1 - trailing_pct(mean_rsi), "vix (buy fear)": trailing_pct(vix)}
expo["average of both"] = (expo["rsi (sell strength)"] + expo["vix (buy fear)"]) / 2
def sharpe(r): r = r.dropna(); return r.mean() / r.std() * math.sqrt(252)
def maxdd(r): eq = (1 + r.dropna()).cumprod(); return (eq / eq.cummax() - 1).min()
rows = []
for w, (lo, hi) in {"train": TRAIN, "test": TEST}.items():
    base = window(ew1, lo, hi)
    rows.append(dict(window=w, rule="equal weight, exposure 1", sharpe=sharpe(base), ann_return=base.mean() * 252, ann_vol=base.std() * math.sqrt(252), max_dd=maxdd(base), mean_exposure=1.0, ann_cost=0.0))
    for name, e in expo.items():
        ex = window(e, lo, hi); both = ex.notna() & base.notna(); ex, b = ex[both], base[both]
        cost = ex.diff().abs().fillna(0) * COST_PER_TRADE
        r = ex * b - cost
        static = ex.mean() * b
        rows.append(dict(window=w, rule=name, sharpe=sharpe(r), ann_return=r.mean() * 252, ann_vol=r.std() * math.sqrt(252), max_dd=maxdd(r), mean_exposure=ex.mean(), ann_cost=cost.mean() * 252))
        rows.append(dict(window=w, rule=f"  static at same mean exposure", sharpe=sharpe(static), ann_return=static.mean() * 252, ann_vol=static.std() * math.sqrt(252), max_dd=maxdd(static), mean_exposure=ex.mean(), ann_cost=0.0))
df = pd.DataFrame(rows); df.to_csv("out/timing_overlay.csv", index=False)
pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
# how much of the test-window Sharpe difference is noise: bootstrap of daily (rule - static) over the test window
rng = np.random.default_rng(0)
lo, hi = TEST
base = window(ew1, lo, hi)
for name, e in expo.items():
    ex = window(e, lo, hi); both = ex.notna() & base.notna(); ex, b = ex[both], base[both]
    r = (ex * b - ex.diff().abs().fillna(0) * COST_PER_TRADE).to_numpy(); st = (ex.mean() * b).to_numpy(); n = len(r)
    diffs = []
    for _ in range(2000):
        # stationary bootstrap, mean block 20 days
        idx = np.empty(n, int); idx[0] = rng.integers(n)
        for t in range(1, n): idx[t] = rng.integers(n) if rng.random() < 0.05 else (idx[t - 1] + 1) % n
        diffs.append(sharpe(pd.Series(r[idx])) - sharpe(pd.Series(st[idx])))
    d = np.array(diffs); print(f"test Sharpe(rule) - Sharpe(static), {name}: point {sharpe(pd.Series(r)) - sharpe(pd.Series(st)):+.2f}, bootstrap 90% [{np.quantile(d, .05):+.2f}, {np.quantile(d, .95):+.2f}]")
