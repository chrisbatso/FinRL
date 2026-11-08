"""Train one tutorial agent and roll it out, deterministically, on the training and the test window.

Environment and hyperparameters are those of examples/FinRL_StockTrading_2026_2_train.py and
..._3_Backtest.py (hmax 100, 1,000,000 initial cash, 0.1% costs, reward scaling 1e-4, membership mask,
turbulence_threshold=70 on the VIX column for the trading environment). One deviation: an off-policy
replay buffer larger than the number of training steps is capped at that number, which cannot change
behaviour (the buffer never fills) but avoids allocating 3 GB for TD3's 1,000,000 buffer on a CI runner.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

TUTORIAL_PARAMS = {
    "a2c": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
    "ddpg": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
    "ppo": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128},
    "td3": {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
    "sac": {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
}


def make_env(df: pd.DataFrame, turbulence_threshold=None) -> StockTradingEnv:
    stock_dim = len(df.tic.unique())
    state_space = 1 + 2 * stock_dim + (len(INDICATORS) + 1) * stock_dim
    return StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        reward_scaling=1e-4,
        member_col="is_member",
        turbulence_threshold=turbulence_threshold,
        risk_indicator_col="vix",
        print_verbosity=10**9,
    )


def rollout(model, df: pd.DataFrame, turbulence_threshold, out: pathlib.Path, split: str) -> dict:
    env = make_env(df, turbulence_threshold)
    account, actions = DRLAgent.DRL_prediction(model=model, environment=env)
    account.to_csv(out / f"account_{split}.csv", index=False)
    actions.to_csv(out / f"actions_{split}.csv")
    values = account["account_value"].to_numpy(dtype=float)
    return {
        "days": int(len(values)),
        "final_value": float(values[-1]),
        "total_return": float(values[-1] / values[0] - 1),
        "total_cost": float(env.cost),
        "trades": int(env.trades),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--algo", required=True, choices=sorted(TUTORIAL_PARAMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, required=True)
    ap.add_argument("--train", nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--test", nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full = pd.read_csv(args.processed)
    train = data_split(full, *args.train)
    test = data_split(full, *args.test)
    out = pathlib.Path(args.out) / f"{args.algo}_seed{args.seed}_steps{args.timesteps}"
    out.mkdir(parents=True, exist_ok=True)

    params = dict(TUTORIAL_PARAMS[args.algo])
    if "buffer_size" in params:
        params["buffer_size"] = min(params["buffer_size"], args.timesteps)

    env_train, _ = make_env(train).get_sb_env()
    agent = DRLAgent(env=env_train)
    model = agent.get_model(args.algo, model_kwargs=params, seed=args.seed, verbose=0)
    t0 = time.perf_counter()
    model = agent.train_model(model=model, tb_log_name=args.algo, total_timesteps=args.timesteps)
    train_seconds = time.perf_counter() - t0

    meta = {
        "algo": args.algo,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "params": params,
        "train_window": args.train,
        "test_window": args.test,
        "train_days": int(train.index.nunique()),
        "test_days": int(test.index.nunique()),
        "stock_dim": int(len(train.tic.unique())),
        "train_seconds": round(train_seconds, 1),
        "in_sample": rollout(model, train, None, out, "train"),
        "out_of_sample": rollout(model, test, 70, out, "test"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "params"}, indent=2))


if __name__ == "__main__":
    main()
