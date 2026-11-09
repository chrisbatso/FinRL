#!/usr/bin/env python
"""Reconstruction null for the time-series tests.

Features such as price level, moving averages, RSI or Bollinger position are built from the very returns
they are then correlated with. Under a random walk, the sample correlation between such a feature and the
stock's own future return is biased negative in finite samples (Stambaugh 1999), so a circular shift of
the returns is not the right null: it breaks the construction link. This script draws stationary block
bootstraps of the daily return panel (same blocks for every ticker, so cross-correlation survives), rebuilds
prices, high/low/open, the stockstats features, the turbulence index and the equal-weight basket, and
recomputes every time-series statistic. The observed statistic is compared with that null's band.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ic_test import (HORIZONS, RAW, REF, STAT, TEST, TRAIN, UNIVERSE, col_corr, membership, stockstats_features,
                     turbulence_index, window)

FEATURES = RAW + [f for f in STAT if f not in RAW] + REF
WINDOWS = {"train": TRAIN, "test": TEST}


def transforms(wide: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = wide["close"]
    wide["macd_rel"] = wide["macd"] / close
    band = (wide["boll_ub"] - wide["boll_lb"]).replace(0, np.nan)
    wide["boll_pos"] = (close - wide["boll_lb"]) / band
    wide["sma30_rel"] = close / wide["close_30_sma"] - 1
    wide["sma60_rel"] = close / wide["close_60_sma"] - 1
    wide["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    wide["rev_1m"] = close / close.shift(21) - 1
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in wide.items()}


def statistics(long: pd.DataFrame, dates, tics, member) -> dict[tuple, float]:
    feats = stockstats_features(long)
    wide = {c: feats.pivot(index="date", columns="tic", values=c).reindex(index=dates, columns=tics) for c in RAW}
    wide = transforms(wide)
    close = wide["close"]
    fwd = {h: close.shift(-h) / close - 1 for h in HORIZONS}
    ew_fwd = {h: fwd[h].where(member).mean(axis=1) for h in HORIZONS}
    turb = turbulence_index(close).replace(0, np.nan)
    out = {}
    for w, (lo, hi) in WINDOWS.items():
        m = window(member, lo, hi)
        for h in HORIZONS:
            yr = window(fwd[h], lo, hi).where(m).rank(axis=0)
            for f in FEATURES:
                xr = window(wide[f], lo, hi).where(m).rank(axis=0)
                out[("ts", w, f, h)] = float(col_corr(xr, yr, 250).mean())
            y = window(ew_fwd[h], lo, hi)
            for name, sig in [("turbulence", turb)] + [(f"mean_{f}", wide[f].where(member).mean(axis=1)) for f in STAT]:
                s = window(sig, lo, hi)
                both = s.notna() & y.notna()
                out[("market", w, name, h)] = float(stats.spearmanr(s[both], y[both]).statistic)
    return out


def stationary_bootstrap_index(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    p = 1.0 / mean_block
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=150)
    ap.add_argument("--block", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw = pd.read_csv(args.prices, parse_dates=["date"])
    raw = raw[raw.tic.isin(UNIVERSE)].copy()
    dates = pd.DatetimeIndex(np.sort(raw.date.unique()))
    piv = {c: raw.pivot(index="date", columns="tic", values=c).reindex(index=dates) for c in ["open", "high", "low", "close", "volume"]}
    full = [t for t in piv["close"].columns if piv["close"][t].notna().all()]
    piv = {c: v[full] for c, v in piv.items()}
    tics = sorted(full)
    member = membership(dates, tics) & piv["close"].notna()

    def to_long(close, high, low, open_, volume):
        frames = []
        for t in tics:
            frames.append(pd.DataFrame({"date": dates, "tic": t, "open": open_[t].values, "high": high[t].values,
                                        "low": low[t].values, "close": close[t].values, "volume": volume[t].values}))
        return pd.concat(frames, ignore_index=True)

    t0 = time.time()
    observed = statistics(to_long(piv["close"], piv["high"], piv["low"], piv["open"], piv["volume"]), dates, tics, member)
    print(f"observed done in {time.time() - t0:.1f}s on {len(tics)} full-history names", flush=True)

    logret = np.log(piv["close"]).diff().iloc[1:]
    ratios = {c: (piv[c] / piv["close"]).iloc[1:] for c in ["high", "low", "open"]}
    vol = piv["volume"].iloc[1:]
    n = len(logret)
    keys = list(observed)
    draws = np.empty((args.draws, len(keys)))
    for b in range(args.draws):
        idx = stationary_bootstrap_index(n, args.block, rng)
        lr = pd.DataFrame(logret.values[idx], index=dates[1:], columns=tics)
        close = pd.concat([piv["close"].iloc[[0]], piv["close"].iloc[0] * np.exp(lr.cumsum())])
        close.index = dates
        def rebuild(c):
            r = pd.DataFrame(ratios[c].values[idx], index=dates[1:], columns=tics)
            r = pd.concat([ratios[c].iloc[[0]], r]); r.index = dates
            return close * r
        high, low, open_ = rebuild("high"), rebuild("low"), rebuild("open")
        volume = pd.concat([vol.iloc[[0]], pd.DataFrame(vol.values[idx], index=dates[1:], columns=tics)]); volume.index = dates
        res = statistics(to_long(close, high, low, open_, volume), dates, tics, member)
        draws[b] = [res[k] for k in keys]
        if b % 10 == 0:
            print(f"draw {b} at {time.time() - t0:.0f}s", flush=True)
    rows = []
    for j, k in enumerate(keys):
        obs = observed[k]
        nd = draws[:, j]
        rows.append(dict(kind=k[0], window=k[1], name=k[2], horizon=k[3], observed=obs, null_mean=float(np.nanmean(nd)),
                         null_sd=float(np.nanstd(nd)), null_q05=float(np.nanquantile(nd, 0.05)), null_q95=float(np.nanquantile(nd, 0.95)),
                         p_lower=float((np.sum(nd <= obs) + 1) / (len(nd) + 1)), p_upper=float((np.sum(nd >= obs) + 1) / (len(nd) + 1)),
                         outside_90=bool(obs < np.nanquantile(nd, 0.05) or obs > np.nanquantile(nd, 0.95))))
    df = pd.DataFrame(rows)
    df.to_csv(out / "null_reconstruction.csv", index=False)
    np.save(out / "null_reconstruction_draws.npy", draws)
    json.dump(dict(draws=args.draws, block=args.block, seed=args.seed, names=tics, keys=[list(map(str, k)) for k in keys]),
              open(out / "null_reconstruction_meta.json", "w"), indent=1)
    print(df[df.kind == "ts"].groupby("window").outside_90.sum().to_dict(), df[df.kind == "market"].groupby("window").outside_90.sum().to_dict())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
