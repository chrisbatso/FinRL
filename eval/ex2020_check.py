"""Is the training-window market-level mean reversion a single episode? Recompute the h=21 Spearman of
market signals vs the equal-weight basket's forward return, with and without calendar 2020, plus year by year."""
import pathlib, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ic_test import RAW, UNIVERSE, TRAIN, membership, stockstats_features, window
raw = pd.read_csv(sys.argv[1], parse_dates=["date"]); raw = raw[raw.tic.isin(UNIVERSE)]
feats = stockstats_features(raw)
dates = pd.DatetimeIndex(np.sort(raw.date.unique())); tics = sorted(raw.tic.unique())
wide = {c: feats.pivot(index="date", columns="tic", values=c).reindex(index=dates, columns=tics) for c in RAW}
close = wide["close"]; member = membership(dates, tics) & close.notna()
sig = {"mean_rsi_30": wide["rsi_30"].where(member).mean(axis=1),
       "mean_sma60_rel": (close / wide["close_60_sma"] - 1).where(member).mean(axis=1),
       "vix": pd.read_csv(sys.argv[2], parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()}
ew21 = (close.shift(-21) / close - 1).where(member).mean(axis=1)
ew1 = (close.shift(-1) / close - 1).where(member).mean(axis=1)
def rho(s, y, lo, hi, drop_years=()):
    s, y = window(s, lo, hi), window(y, lo, hi)
    keep = ~s.index.year.isin(drop_years)
    s, y = s[keep], y[keep]; both = s.notna() & y.notna()
    return stats.spearmanr(s[both], y[both]).statistic, int(both.sum())
rows = []
for name, s in sig.items():
    full, n = rho(s, ew21, *TRAIN); ex20, n2 = rho(s, ew21, *TRAIN, drop_years=(2020,))
    rows.append(dict(signal=name, train_all=round(full, 3), train_ex2020=round(ex20, 3), days=n, days_ex2020=n2))
print(pd.DataFrame(rows).to_string(index=False))
print("\nyear by year, h=21 Spearman:")
yr = []
for y in range(2014, 2027):
    lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    yr.append({"year": y, **{k: round(rho(s, ew21, lo, hi)[0], 2) for k, s in sig.items()}})
print(pd.DataFrame(yr).to_string(index=False))
pd.DataFrame(rows).to_csv("out/market_timing_ex2020.csv", index=False); pd.DataFrame(yr).to_csv("out/market_timing_by_year.csv", index=False)
