#!/usr/bin/env python
"""Information-coefficient test of FinRL's default state features on the committed Dow data.

Inputs (YahooDownloader layout, as committed under eval/data/ on the fork):
  raw_prices.csv.gz  date,open,high,low,close,volume,tic,day   (adjusted closes)
  vix.csv            date,close

The script re-encodes FinRL's point-in-time Dow membership (finrl/meta/preprocessor/dow.py) and
computes the eight default indicators with stockstats exactly as FeatureEngineer does (per ticker).
It then asks how much each feature knows about forward returns, three ways:

  1. cross-sectional rank IC (stock picking): Spearman between feature and forward return across the
     members on each day; mean, Newey-West t (lag = horizon), hit rate, MDE at 80 % power.
  2. time-series rank IC (timing one name): Spearman over time per ticker, averaged across tickers;
     null by circular time shift of the return panel (preserves autocorrelation and cross-correlation).
  3. market timing: Spearman over time between a market-level signal (VIX, turbulence, member-mean of
     each feature) and the forward return of the equal-weight member basket; same circular-shift null.

Plus two money translations: the Sharpe of a daily rank long-short portfolio per feature (sign fixed
on the training window), and FinRL's own risk overlay (sell everything when the risk indicator exceeds
its threshold) applied to the equal-weight basket.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
from stockstats import StockDataFrame as Sdf

INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
RAW = ["close"] + INDICATORS
STAT = ["macd_rel", "boll_pos", "rsi_30", "cci_30", "dx_30", "sma30_rel", "sma60_rel"]
REF = ["mom_12_1", "rev_1m"]
HORIZONS = (1, 5, 21)
TRAIN = (pd.Timestamp("2014-01-06"), pd.Timestamp("2021-12-31"))
TEST = (pd.Timestamp("2022-01-01"), pd.Timestamp("2026-03-20"))
COST_PER_TRADE = 0.001  # FinRL tutorial: 0.1 % on buys and on sells

# --- point-in-time Dow membership, re-encoded from finrl/meta/preprocessor/dow.py -------------------
INITIAL_2014 = {
    "AXP", "BA", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "KO",
    "MCD", "MMM", "MRK", "MSFT", "NKE", "PFE", "PG", "T", "TRV", "UNH", "UTX", "V", "VZ", "WMT", "XOM",
}
CHANGES = [
    ("2015-03-19", ("AAPL",), ("T",)),
    ("2017-09-01", ("DWDP",), ("DD",)),
    ("2018-06-26", ("WBA",), ("GE",)),
    ("2019-04-02", ("DOW",), ("DWDP",)),
    ("2020-04-06", ("RTX",), ("UTX",)),
    ("2020-08-31", ("AMGN",), ("XOM",)),
    ("2020-08-31", ("HON",), ("PFE",)),
    ("2020-08-31", ("CRM",), ("RTX",)),
    ("2024-02-26", ("AMZN",), ("WBA",)),
    ("2024-11-08", ("NVDA",), ("INTC",)),
    ("2024-11-08", ("SHW",), ("DOW",)),
]
DOW_30_2026 = {
    "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON", "IBM", "JNJ", "KO", "JPM",
    "MCD", "MMM", "MRK", "MSFT", "NVDA", "NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V", "WMT", "DIS", "SHW",
}
UNIVERSE = set(INITIAL_2014) | DOW_30_2026 | {t for _, added, _ in CHANGES for t in added}


def constituents(date: pd.Timestamp) -> set[str]:
    if date >= pd.Timestamp("2026-01-01"):
        return set(DOW_30_2026)
    cons = set(INITIAL_2014)
    for eff, added, removed in CHANGES:
        if date >= pd.Timestamp(eff):
            cons.difference_update(removed)
            cons.update(added)
    assert len(cons) == 30, (date, len(cons))
    return cons


def membership(dates: pd.DatetimeIndex, tics: list[str]) -> pd.DataFrame:
    member = pd.DataFrame(False, index=dates, columns=tics)
    bounds = sorted({pd.Timestamp("2014-01-01"), pd.Timestamp("2026-01-01"), *(pd.Timestamp(e) for e, _, _ in CHANGES)})
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else pd.Timestamp("2100-01-01")
        rows = (member.index >= start) & (member.index < end)
        cols = [t for t in constituents(start) if t in member.columns]
        member.loc[rows, cols] = True
    return member


# --- features -----------------------------------------------------------------------------------------
def stockstats_features(raw: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for tic, g in raw.groupby("tic", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        sdf = Sdf.retype(g[["date", "open", "high", "low", "close", "volume"]].copy())
        out = g[["date", "tic", "close"]].copy()
        for ind in INDICATORS:
            out[ind] = np.asarray(sdf[ind], dtype=float)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def turbulence_index(close: pd.DataFrame, start: int = 252) -> pd.Series:
    """FinRL's calculate_turbulence: expanding-window Mahalanobis distance of today's return vector,
    on the names with a complete history; zero for the first 'start' days and until the third positive value."""
    full = close.columns[close.notna().all(axis=0)]
    ret = close[full].pct_change().iloc[1:]
    values = np.zeros(len(close))
    count = 0
    r = ret.values
    for i in range(start, len(ret)):
        hist = r[:i]
        cur = r[i]
        cov = np.cov(hist, rowvar=False)
        d = cur - hist.mean(axis=0)
        temp = float(d @ np.linalg.pinv(cov) @ d)
        if temp > 0:
            count += 1
            values[i + 1] = temp if count > 2 else 0.0
    return pd.Series(values, index=close.index, name="turbulence")


# --- statistics ---------------------------------------------------------------------------------------
def nw_mean(x: pd.Series, lag: int) -> tuple[float, float, float, int]:
    v = x.dropna().to_numpy(dtype=float)
    n = len(v)
    m = v.mean()
    e = v - m
    var = e @ e / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k] / n)
    se = math.sqrt(max(var, 1e-18) / n)
    return m, se, m / se, n


def cs_rank_ic(x: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame, min_names: int = 10) -> pd.Series:
    both = x.notna() & y.notna() & mask
    xr = x.where(both).rank(axis=1)
    yr = y.where(both).rank(axis=1)
    xr = xr.sub(xr.mean(axis=1), axis=0)
    yr = yr.sub(yr.mean(axis=1), axis=0)
    den = np.sqrt((xr**2).sum(axis=1) * (yr**2).sum(axis=1))
    ic = (xr * yr).sum(axis=1) / den.replace(0, np.nan)
    ic[both.sum(axis=1) < min_names] = np.nan
    return ic


def col_corr(a: pd.DataFrame, b: pd.DataFrame, min_obs: int) -> pd.Series:
    both = a.notna() & b.notna()
    a = a.where(both)
    b = b.where(both)
    am = a - a.mean()
    bm = b - b.mean()
    den = np.sqrt((am**2).sum() * (bm**2).sum())
    c = (am * bm).sum() / den.replace(0, np.nan)
    c[both.sum() < min_obs] = np.nan
    return c


def ts_rank_ic(x: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame, rng: np.random.Generator,
               n_perm: int, min_obs: int = 250) -> tuple[float, float, float, int]:
    xr = x.where(mask).rank(axis=0)
    yr = y.where(mask).rank(axis=0)
    obs = col_corr(xr, yr, min_obs)
    stat = float(obs.mean())
    n_tic = int(obs.notna().sum())
    T = len(yr)
    yv = yr.to_numpy()
    null = np.empty(n_perm)
    for p in range(n_perm):
        k = int(rng.integers(63, T - 63))
        shifted = pd.DataFrame(np.roll(yv, k, axis=0), index=yr.index, columns=yr.columns)
        null[p] = float(col_corr(xr, shifted, min_obs).mean())
    pval = (np.sum(np.abs(null) >= abs(stat)) + 1) / (n_perm + 1)
    return stat, float(np.nanstd(null)), float(pval), n_tic


def ts_spearman_series(sig: pd.Series, y: pd.Series, rng: np.random.Generator, n_perm: int) -> tuple[float, float, float, int]:
    both = sig.notna() & y.notna()
    s = sig[both].to_numpy()
    r = y[both].to_numpy()
    n = len(s)
    rho = stats.spearmanr(s, r).statistic
    rr = stats.rankdata(r)
    sr = stats.rankdata(s)
    null = np.empty(n_perm)
    for p in range(n_perm):
        k = int(rng.integers(63, n - 63))
        null[p] = np.corrcoef(sr, np.roll(rr, k))[0, 1]
    pval = (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)
    return float(rho), float(null.std()), float(pval), n


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else float("nan")


def window(df, lo, hi):
    return df.loc[(df.index >= lo) & (df.index <= hi)]


# --- main ----------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--vix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw = pd.read_csv(args.prices, parse_dates=["date"])
    raw = raw[raw.tic.isin(UNIVERSE)].copy()
    feats = stockstats_features(raw)
    dates = pd.DatetimeIndex(np.sort(raw.date.unique()))
    tics = sorted(raw.tic.unique())

    wide = {c: feats.pivot(index="date", columns="tic", values=c).reindex(index=dates, columns=tics) for c in RAW}
    close = wide["close"]
    member = membership(dates, tics) & close.notna()

    # stationary transforms of the same indicators, and two reference signals from the literature
    wide["macd_rel"] = wide["macd"] / close
    band = (wide["boll_ub"] - wide["boll_lb"]).replace(0, np.nan)
    wide["boll_pos"] = (close - wide["boll_lb"]) / band
    wide["sma30_rel"] = close / wide["close_30_sma"] - 1
    wide["sma60_rel"] = close / wide["close_60_sma"] - 1
    wide["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    wide["rev_1m"] = close / close.shift(21) - 1
    for k, v in wide.items():
        wide[k] = v.replace([np.inf, -np.inf], np.nan)

    fwd = {h: close.shift(-h) / close - 1 for h in HORIZONS}
    ew_fwd = {h: fwd[h].where(member).mean(axis=1) for h in HORIZONS}
    vix = pd.read_csv(args.vix, parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    turb = turbulence_index(close)

    windows = {"train": TRAIN, "test": TEST}
    groups = {**{f: "raw (as fed to the agent)" for f in RAW}, **{f: "stationary transform" for f in STAT},
              **{f: "reference signal" for f in REF}}
    features = RAW + [f for f in STAT if f not in RAW] + REF

    # 1. cross-sectional IC ---------------------------------------------------------------------
    cs_rows, cs_series = [], {}
    for f in features:
        for h in HORIZONS:
            ic = cs_rank_ic(wide[f], fwd[h], member)
            cs_series[(f, h)] = ic
            for w, (lo, hi) in windows.items():
                m, se, t, n = nw_mean(window(ic, lo, hi), lag=h)
                cs_rows.append(dict(window=w, feature=f, group=groups[f], horizon=h, days=n, mean_ic=m, se_nw=se,
                                    t=t, p=2 * stats.norm.sf(abs(t)), hit_rate=float((window(ic, lo, hi) > 0).mean()),
                                    mde_80=2.8 * se))
    cs = pd.DataFrame(cs_rows)
    cs.to_csv(out / "ic_cross_sectional.csv", index=False)

    # 2. time-series IC per ticker, circular-shift null -------------------------------------------
    ts_rows = []
    for f in features:
        for h in HORIZONS:
            for w, (lo, hi) in windows.items():
                stat_, nsd, p, n_tic = ts_rank_ic(window(wide[f], lo, hi), window(fwd[h], lo, hi), window(member, lo, hi),
                                                  rng, args.perms)
                ts_rows.append(dict(window=w, feature=f, group=groups[f], horizon=h, tickers=n_tic, mean_ts_ic=stat_,
                                    null_sd=nsd, p_perm=p))
    ts = pd.DataFrame(ts_rows)
    ts.to_csv(out / "ic_timing_per_ticker.csv", index=False)

    # 3. market timing of the EW basket ---------------------------------------------------------------
    signals = {"vix": vix, "turbulence": turb.replace(0, np.nan)}
    for f in STAT:
        signals[f"mean_{f}"] = wide[f].where(member).mean(axis=1)
    mk_rows = []
    for name, sig in signals.items():
        for h in HORIZONS:
            for w, (lo, hi) in windows.items():
                rho, nsd, p, n = ts_spearman_series(window(sig, lo, hi), window(ew_fwd[h], lo, hi), rng, args.perms)
                mk_rows.append(dict(window=w, signal=name, horizon=h, days=n, spearman=rho, null_sd=nsd, p_perm=p))
    mk = pd.DataFrame(mk_rows)
    mk.to_csv(out / "ic_market_timing.csv", index=False)

    # 4. rank long-short portfolios, daily rebalanced, sign fixed on the training window -----------------
    ls_rows = []
    for f in [x for x in features if x != "close"]:
        x = wide[f].where(member & fwd[1].notna())
        r = x.rank(axis=1)
        r = r.sub(r.mean(axis=1), axis=0)
        wts = r.div(r.abs().sum(axis=1).replace(0, np.nan), axis=0)
        ret = (wts * fwd[1]).sum(axis=1, min_count=1)
        turnover = (wts.fillna(0) - wts.shift(1).fillna(0)).abs().sum(axis=1)
        sign = float(np.sign(cs.query("window=='train' and feature==@f and horizon==1").mean_ic.iloc[0]) or 1.0)
        for w, (lo, hi) in windows.items():
            g = window(ret, lo, hi) * sign
            to = window(turnover, lo, hi)
            ls_rows.append(dict(feature=f, group=groups[f], window=w, sign=sign, sharpe_gross=sharpe(g),
                                sharpe_net=sharpe(g - to * COST_PER_TRADE), ann_return_gross=float(g.mean() * 252),
                                ann_cost=float((to * COST_PER_TRADE).mean() * 252), turnover_per_day=float(to.mean())))
    ls = pd.DataFrame(ls_rows)
    ls.to_csv(out / "rank_long_short.csv", index=False)

    # 5. FinRL's risk overlay on the EW basket --------------------------------------------------------------
    ew1 = ew_fwd[1]
    train_turb = window(turb, *TRAIN)
    rules = {"vix > 70 (tutorial)": vix > 70,
             "turbulence > train q99": turb > train_turb.quantile(0.99),
             "turbulence > train q99.6": turb > train_turb.quantile(0.996)}
    ov_rows = []
    for w, (lo, hi) in windows.items():
        base = window(ew1, lo, hi)
        ov_rows.append(dict(window=w, rule="equal weight, always invested", sharpe=sharpe(base),
                            ann_return=float(base.mean() * 252), days_out=0, max_drawdown=float(((1 + base).cumprod() / (1 + base).cumprod().cummax() - 1).min())))
        for name, flag in rules.items():
            fl = window(flag, lo, hi).astype(float)
            switches = fl.diff().abs().fillna(0)
            r = base * (1 - fl) - switches * COST_PER_TRADE
            eq = (1 + r).cumprod()
            ov_rows.append(dict(window=w, rule=name, sharpe=sharpe(r), ann_return=float(r.mean() * 252), days_out=int(fl.sum()),
                                max_drawdown=float((eq / eq.cummax() - 1).min())))
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(out / "risk_overlay.csv", index=False)

    # 6. power and multiple testing -----------------------------------------------------------------------
    n_members = member.sum(axis=1)
    m_tests = len(features) * len(HORIZONS)
    bonf = float(stats.norm.isf(0.025 / m_tests))
    summary = dict(
        tickers=len(tics), days=len(dates), members_per_day=dict(min=int(n_members.min()), max=int(n_members.max()), median=float(n_members.median())),
        windows={k: [str(v[0].date()), str(v[1].date())] for k, v in windows.items()},
        features=features, horizons=list(HORIZONS), perms=args.perms, seed=args.seed,
        tests_per_window=m_tests, bonferroni_t=bonf, expected_max_abs_t_under_null=math.sqrt(2 * math.log(m_tests)),
        cs_abs_t_over_196={w: int((cs[cs.window == w].t.abs() > 1.96).sum()) for w in windows},
        cs_abs_t_over_bonferroni={w: int((cs[cs.window == w].t.abs() > bonf).sum()) for w in windows},
        cs_largest_abs_t={w: cs[cs.window == w].loc[cs[cs.window == w].t.abs().idxmax(), ["feature", "horizon", "mean_ic", "t"]].to_dict() for w in windows},
        ts_p_under_005={w: int((ts[ts.window == w].p_perm < 0.05).sum()) for w in windows},
        mk_p_under_005={w: int((mk[mk.window == w].p_perm < 0.05).sum()) for w in windows},
        mde_h1_stationary={w: float(cs[(cs.window == w) & (cs.horizon == 1) & (cs.group == "stationary transform")].mde_80.median()) for w in windows},
        null_cs_ic_sd_30_names=1 / math.sqrt(29),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # cumulative IC series for the figure
    pd.DataFrame({f"{f}": cs_series[(f, 1)] for f in STAT + REF}).to_csv(out / "cs_ic_daily_h1.csv")
    pd.DataFrame({"members": n_members, "vix": vix, "turbulence": turb, "ew_fwd1": ew1}).to_csv(out / "market_series.csv")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
