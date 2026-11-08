"""Collect every train_eval run, compare with the baselines and write report.md with figures.

Pure pandas/numpy/matplotlib; does not import finrl, so it can be checked locally on synthetic runs.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRADING_DAYS = 252
INITIAL = 1_000_000.0


def metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    r = values[1:] / values[:-1] - 1
    years = len(r) / TRADING_DAYS
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    return {
        "total_return": values[-1] / values[0] - 1,
        "cagr": (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": std * math.sqrt(TRADING_DAYS),
        "sharpe": (r.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        "max_drawdown": float((values / np.maximum.accumulate(values) - 1).min()),
    }


def sharpe(r: np.ndarray) -> float:
    std = r.std(ddof=1)
    return float(r.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    se = math.sqrt(s / n) if s > 0 else np.nan
    return float(x.mean() / se) if se and se > 0 else np.nan


def stationary_bootstrap_sharpe_diff(ra: np.ndarray, rb: np.ndarray, reps: int = 2000, mean_block: int = 20, seed: int = 0):
    """Politis-Romano stationary bootstrap of SR(ra) - SR(rb) on the paired daily series."""
    rng = np.random.default_rng(seed)
    n = len(ra)
    p = 1.0 / mean_block
    diffs = np.empty(reps)
    for b in range(reps):
        idx = np.empty(n, dtype=int)
        pos = 0
        while pos < n:
            start = rng.integers(n)
            length = min(rng.geometric(p), n - pos)
            idx[pos : pos + length] = (start + np.arange(length)) % n
            pos += length
        diffs[b] = sharpe(ra[idx]) - sharpe(rb[idx])
    return float(diffs.std(ddof=1)), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def load_window(full: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return full[(full["date"] >= start) & (full["date"] < end)].sort_values(["date", "tic"])


def equal_weight_point_in_time(window: pd.DataFrame) -> pd.Series:
    prices = window.pivot(index="date", columns="tic", values="close").sort_index()
    members = window.pivot(index="date", columns="tic", values="is_member").reindex_like(prices).astype(bool)
    daily = prices.pct_change().where(members).mean(axis=1).fillna(0.0)
    return (1 + daily).cumprod() * INITIAL


def activity(actions: pd.DataFrame, window: pd.DataFrame, account: pd.Series) -> dict:
    prices = window.pivot(index="date", columns="tic", values="close").reindex(index=actions.index, columns=actions.columns)
    holdings = actions.cumsum()
    invested = (holdings * prices).sum(axis=1)
    equity = account.reindex(actions.index)
    traded_notional = (actions.abs() * prices).sum(axis=1)
    years = len(actions) / TRADING_DAYS
    return {
        "days_with_trades_frac": float((actions.abs().sum(axis=1) > 0).mean()),
        "positions_held_mean": float((holdings > 0).sum(axis=1).mean()),
        "invested_frac_mean": float((invested / equity).mean()),
        "invested_frac_last": float((invested / equity).iloc[-1]),
        "annual_turnover": float(traded_notional.sum() / equity.mean() / years) if years > 0 else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--dji", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prepare-summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(pathlib.Path(args.config).read_text())
    prep = json.loads(pathlib.Path(args.prepare_summary).read_text())
    full = pd.read_csv(args.processed)
    test = load_window(full, *cfg["test"])
    train = load_window(full, *cfg["train"])

    ew_test = equal_weight_point_in_time(test)
    ew_train = equal_weight_point_in_time(train)
    dji = pd.read_csv(args.dji).set_index("date")["close"]
    dji_test = dji.reindex(ew_test.index).ffill()
    dji_test = dji_test / dji_test.iloc[0] * INITIAL
    baselines = {
        "equal weight, point-in-time Dow": ew_test,
        "DJIA (price index)": dji_test,
        "cash": pd.Series(INITIAL, index=ew_test.index),
    }
    r_ew = ew_test.pct_change().dropna()

    rows = []
    curves: dict[tuple, dict[int, pd.Series]] = {}
    for meta_path in sorted(pathlib.Path(args.results).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        run_dir = meta_path.parent
        acc = pd.read_csv(run_dir / "account_test.csv")
        acc_series = pd.Series(acc["account_value"].to_numpy(dtype=float), index=acc["date"].astype(str))
        acc_train = pd.read_csv(run_dir / "account_train.csv")["account_value"].to_numpy(dtype=float)
        actions = pd.read_csv(run_dir / "actions_test.csv", index_col=0)
        actions.index = actions.index.astype(str)

        r_agent = acc_series.pct_change().dropna()
        common = r_agent.index.intersection(r_ew.index)
        ra, rb = r_agent.loc[common].to_numpy(), r_ew.loc[common].to_numpy()
        excess = ra - rb
        se, ci = stationary_bootstrap_sharpe_diff(ra, rb, reps=args.bootstrap_reps, seed=meta["seed"])

        m_test = metrics(acc_series.to_numpy())
        m_train = metrics(acc_train)
        row = {
            "algo": meta["algo"],
            "seed": meta["seed"],
            "timesteps": meta["timesteps"],
            "train_seconds": meta.get("train_seconds"),
            **{f"test_{k}": v for k, v in m_test.items()},
            **{f"train_{k}": v for k, v in m_train.items()},
            "test_cost_paid": meta["out_of_sample"]["total_cost"],
            "excess_daily_mean_bp": float(excess.mean() * 1e4),
            "excess_nw_t": newey_west_t(excess),
            "sharpe_diff_vs_ew": sharpe(ra) - sharpe(rb),
            "sharpe_diff_se": se,
            "sharpe_diff_ci_low": ci[0],
            "sharpe_diff_ci_high": ci[1],
            **activity(actions, test, acc_series),
        }
        rows.append(row)
        curves.setdefault((meta["algo"], meta["timesteps"]), {})[meta["seed"]] = acc_series

    runs = pd.DataFrame(rows).sort_values(["timesteps", "algo", "seed"])
    runs.to_csv(out / "runs.csv", index=False)

    base_rows = {name: metrics(series.to_numpy()) for name, series in baselines.items()}
    base_df = pd.DataFrame(base_rows).T
    base_df.to_csv(out / "baselines.csv")
    ew_sharpe = base_rows["equal weight, point-in-time Dow"]["sharpe"]

    # pooled: average the seeds' daily returns -> one series per (algo, timesteps)
    pooled = []
    for (algo, steps), by_seed in sorted(curves.items()):
        rets = pd.DataFrame({s: v.pct_change() for s, v in by_seed.items()}).dropna()
        mean_ret = rets.mean(axis=1)
        common = mean_ret.index.intersection(r_ew.index)
        ra, rb = mean_ret.loc[common].to_numpy(), r_ew.loc[common].to_numpy()
        se, ci = stationary_bootstrap_sharpe_diff(ra, rb, reps=args.bootstrap_reps, seed=1)
        pooled.append({"algo": algo, "timesteps": steps, "n_seeds": len(by_seed), "pooled_sharpe": sharpe(ra),
                       "sharpe_diff_vs_ew": sharpe(ra) - sharpe(rb), "ci_low": ci[0], "ci_high": ci[1],
                       "excess_nw_t": newey_west_t(ra - rb)})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out / "pooled.csv", index=False)

    summary = (
        runs.groupby(["timesteps", "algo"])
        .agg(
            n_seeds=("seed", "count"),
            test_sharpe_mean=("test_sharpe", "mean"),
            test_sharpe_std=("test_sharpe", "std"),
            test_sharpe_min=("test_sharpe", "min"),
            test_sharpe_max=("test_sharpe", "max"),
            test_return_mean=("test_total_return", "mean"),
            test_return_std=("test_total_return", "std"),
            test_maxdd_mean=("test_max_drawdown", "mean"),
            train_sharpe_mean=("train_sharpe", "mean"),
            beats_ew_sharpe=("test_sharpe", lambda s: float((s > ew_sharpe).mean())),
            invested_frac=("invested_frac_mean", "mean"),
            trade_days_frac=("days_with_trades_frac", "mean"),
            turnover=("annual_turnover", "mean"),
            sharpe_diff_se_mean=("sharpe_diff_se", "mean"),
            train_minutes=("train_seconds", lambda s: float(s.mean() / 60)),
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)

    # ---- figures ----
    budgets = sorted(runs["timesteps"].unique())
    algos = sorted(runs["algo"].unique())
    for steps in budgets:
        fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 3.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, algo in zip(axes, algos):
            for s, series in sorted(curves.get((algo, steps), {}).items()):
                ax.plot(pd.to_datetime(series.index), series.to_numpy() / INITIAL, lw=0.8, alpha=0.7, label=f"seed {s}")
            ax.plot(pd.to_datetime(ew_test.index), ew_test.to_numpy() / INITIAL, color="black", lw=1.8, label="equal weight")
            ax.plot(pd.to_datetime(dji_test.index), dji_test.to_numpy() / INITIAL, color="grey", lw=1.4, ls="--", label="DJIA")
            ax.set_title(f"{algo.upper()}, {steps:,} steps")
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        axes[0].set_ylabel("value / initial")
        axes[-1].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / f"equity_{steps}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, steps in enumerate(budgets):
        sub = runs[runs["timesteps"] == steps]
        x = np.array([algos.index(a) for a in sub["algo"]], dtype=float) + (i - (len(budgets) - 1) / 2) * 0.18
        ax.scatter(x, sub["test_sharpe"], s=28, alpha=0.8, label=f"{steps:,} steps")
    ax.axhline(ew_sharpe, color="black", lw=1.5, label="equal weight")
    ax.axhline(base_rows["DJIA (price index)"]["sharpe"], color="grey", lw=1.2, ls="--", label="DJIA")
    ax.axhline(0, color="lightgrey", lw=0.8)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel("out-of-sample Sharpe")
    ax.set_title("Test-window Sharpe per seed")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sharpe_by_seed.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for steps in budgets:
        sub = runs[runs["timesteps"] == steps]
        ax.scatter(sub["train_sharpe"], sub["test_sharpe"], s=30, alpha=0.8, label=f"{steps:,} steps")
    lo = min(runs["train_sharpe"].min(), runs["test_sharpe"].min(), 0) - 0.2
    hi = max(runs["train_sharpe"].max(), runs["test_sharpe"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color="lightgrey", lw=1)
    ax.axhline(ew_sharpe, color="black", lw=1, ls=":")
    ax.set_xlabel("in-sample Sharpe (training window)")
    ax.set_ylabel("out-of-sample Sharpe (test window)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "train_vs_test_sharpe.png", dpi=130)
    plt.close(fig)

    # ---- report ----
    def pct(x):
        return f"{100 * x:+.1f}%"

    lines = []
    lines.append("# FinRL stock-trading tutorial: out-of-sample evaluation\n")
    lines.append(f"Universe: point-in-time Dow 30 ({prep['tickers']} tickers over the period, {prep['members_per_day_min']} to {prep['members_per_day_max']} members per day). "
                 f"Training window {cfg['train'][0]} to {cfg['train'][1]} ({train['date'].nunique()} days), test window {cfg['test'][0]} to {cfg['test'][1]} ({test['date'].nunique()} days). "
                 "Environment and hyperparameters as in examples/FinRL_StockTrading_2026_*.py; costs 0.1% per side; trading environment liquidates when VIX >= 70.\n")
    if prep.get("members_without_prices"):
        lines.append("Constituents with no Yahoo prices for part of their membership, marked non-member on those days: "
                     + ", ".join(f"{d['tic']} ({d['min']} to {d['max']})" for d in prep["members_without_prices"]) + ".\n")
    lines.append("## Baselines on the test window\n")
    lines.append(base_df[["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"]].to_markdown(floatfmt=".3f") + "\n")
    lines.append("## Agents: mean over seeds (std in brackets)\n")
    tbl = summary.copy()
    tbl["test Sharpe"] = tbl.apply(lambda r: f"{r.test_sharpe_mean:.2f} ({r.test_sharpe_std:.2f})", axis=1)
    tbl["test return"] = tbl.apply(lambda r: f"{pct(r.test_return_mean)} ({100 * r.test_return_std:.1f})", axis=1)
    tbl["max DD"] = tbl["test_maxdd_mean"].map(pct)
    tbl["train Sharpe"] = tbl["train_sharpe_mean"].map(lambda v: f"{v:.2f}")
    tbl["seeds > EW"] = tbl.apply(lambda r: f"{int(round(r.beats_ew_sharpe * r.n_seeds))}/{int(r.n_seeds)}", axis=1)
    tbl["invested"] = tbl["invested_frac"].map(lambda v: f"{100 * v:.0f}%")
    tbl["turnover/yr"] = tbl["turnover"].map(lambda v: f"{v:.1f}x")
    tbl["min train"] = tbl["train_minutes"].map(lambda v: f"{v:.1f}")
    lines.append(tbl[["timesteps", "algo", "n_seeds", "test Sharpe", "test return", "max DD", "train Sharpe", "seeds > EW", "invested", "turnover/yr", "min train"]]
                 .to_markdown(index=False) + "\n")
    lines.append(f"Equal-weight Sharpe on the same window: {ew_sharpe:.2f}. 'seeds > EW' counts seeds whose test Sharpe exceeds it.\n")
    lines.append("## Seed-averaged series against equal weight\n")
    lines.append("Daily returns averaged across seeds (what an ensemble of the seeds would earn), compared with the equal-weight basket: Sharpe difference with a stationary-bootstrap 95% interval and the Newey-West t-statistic of the daily excess return.\n")
    ptbl = pooled.copy()
    ptbl["Sharpe diff vs EW"] = ptbl.apply(lambda r: f"{r.sharpe_diff_vs_ew:+.2f}  [{r.ci_low:+.2f}, {r.ci_high:+.2f}]", axis=1)
    ptbl["NW t"] = ptbl["excess_nw_t"].map(lambda v: f"{v:+.2f}")
    ptbl["pooled Sharpe"] = ptbl["pooled_sharpe"].map(lambda v: f"{v:.2f}")
    lines.append(ptbl[["timesteps", "algo", "n_seeds", "pooled Sharpe", "Sharpe diff vs EW", "NW t"]].to_markdown(index=False) + "\n")
    se_mean = runs["sharpe_diff_se"].mean()
    lines.append("## What this test window can detect\n")
    lines.append(f"The bootstrap standard error of a single run's Sharpe difference against equal weight averages {se_mean:.2f}. "
                 f"At 80% power and a two-sided 5% level, the smallest Sharpe improvement this window can establish is about {2.8 * se_mean:.2f}; "
                 f"anything smaller than that is indistinguishable from noise with {test['date'].nunique()} test days, whatever the point estimate says.\n")
    lines.append("## Figures\n")
    for steps in budgets:
        lines.append(f"![equity {steps}](equity_{steps}.png)\n")
    lines.append("![sharpe by seed](sharpe_by_seed.png)\n")
    lines.append("![train vs test](train_vs_test_sharpe.png)\n")
    lines.append("## Per-run table\n")
    cols = ["timesteps", "algo", "seed", "test_sharpe", "test_total_return", "test_max_drawdown", "train_sharpe", "sharpe_diff_vs_ew", "sharpe_diff_ci_low", "sharpe_diff_ci_high", "excess_nw_t", "invested_frac_mean", "days_with_trades_frac", "annual_turnover", "test_cost_paid"]
    lines.append(runs[cols].to_markdown(index=False, floatfmt=".3f") + "\n")
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
