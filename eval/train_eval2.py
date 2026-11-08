"""Second grid: the tutorial pipeline with the two changes the first grid pointed at.

obs     raw     the environment's observation as is (cash near 1e6, prices in hundreds, indicators unscaled)
        norm    Stable-Baselines3 VecNormalize on observations (running mean and std, clipped at 10 std),
                statistics frozen for the rollouts
reward  dollar  the tutorial's reward: daily change in total asset value times 1e-4
        logret  100 times the daily log return of total asset value
        excess  100 times (daily log return of the agent minus daily log return of the equal-weight
                point-in-time basket), so holding the basket scores zero and sitting in cash in a rising
                market scores negative

Everything else (universe, features, costs, hmax, hyperparameters) is train_eval.py.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.preprocessors import data_split
from train_eval import TUTORIAL_PARAMS
from train_eval import make_env

OBS = ("raw", "norm")
REWARDS = ("dollar", "logret", "excess")


def ew_log_return(df: pd.DataFrame) -> np.ndarray:
    """Daily log return of the equal-weight point-in-time basket, indexed by the frame's day index."""
    prices = df.pivot(index="date", columns="tic", values="close").sort_index()
    members = (
        df.pivot(index="date", columns="tic", values="is_member")
        .reindex_like(prices)
        .astype(bool)
    )
    daily = prices.pct_change().where(members).mean(axis=1).fillna(0.0)
    return np.log1p(daily.to_numpy())


class ReturnReward(gym.Wrapper):
    """Replace the dollar reward with a (relative) log-return reward, computed from the env's own bookkeeping."""

    def __init__(self, env, kind: str, ew_log_ret: np.ndarray | None = None):
        super().__init__(env)
        self.kind = kind
        self.ew_log_ret = ew_log_ret

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.kind == "dollar" or terminated:
            return obs, reward, terminated, truncated, info
        inner = self.env.unwrapped
        memory = inner.asset_memory
        if len(memory) >= 2 and memory[-2] > 0 and memory[-1] > 0:
            r = math.log(memory[-1] / memory[-2])
        else:
            r = 0.0
        if self.kind == "excess":
            r -= float(self.ew_log_ret[inner.day])
        return obs, 100.0 * r, terminated, truncated, info


def rollout(
    model,
    df: pd.DataFrame,
    turbulence_threshold,
    vecnorm,
    out: pathlib.Path,
    split: str,
) -> dict:
    """Deterministic rollout on the plain environment, normalising observations with the frozen statistics."""
    env = make_env(df, turbulence_threshold)
    obs, _ = env.reset()
    done = False
    while not done:
        x = np.asarray(obs, dtype=np.float32)
        if vecnorm is not None:
            x = vecnorm.normalize_obs(x)
        action, _ = model.predict(x, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    account = env.save_asset_memory()
    account.to_csv(out / f"account_{split}.csv", index=False)
    env.save_action_memory().to_csv(out / f"actions_{split}.csv")
    values = account["account_value"].to_numpy(dtype=float)
    return {
        "days": int(len(values)),
        "final_value": float(values[-1]),
        "total_return": float(values[-1] / values[0] - 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--algo", required=True, choices=sorted(TUTORIAL_PARAMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, required=True)
    ap.add_argument("--obs", required=True, choices=OBS)
    ap.add_argument("--reward", required=True, choices=REWARDS)
    ap.add_argument("--train", nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--test", nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full = pd.read_csv(args.processed)
    train = data_split(full, *args.train)
    test = data_split(full, *args.test)
    variant = f"{args.obs}+{args.reward}"
    out = (
        pathlib.Path(args.out) / f"{args.algo}_seed{args.seed}_{args.obs}_{args.reward}"
    )
    out.mkdir(parents=True, exist_ok=True)

    params = dict(TUTORIAL_PARAMS[args.algo])
    if "buffer_size" in params:
        params["buffer_size"] = min(params["buffer_size"], args.timesteps)

    ew = ew_log_return(train) if args.reward == "excess" else None
    venv = DummyVecEnv([lambda: ReturnReward(make_env(train), args.reward, ew)])
    vecnorm = None
    if args.obs == "norm":
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vecnorm = venv

    agent = DRLAgent(env=venv)
    model = agent.get_model(args.algo, model_kwargs=params, seed=args.seed, verbose=0)
    t0 = time.perf_counter()
    model = agent.train_model(
        model=model,
        tb_log_name=f"{args.algo}_{variant}",
        total_timesteps=args.timesteps,
    )
    train_seconds = time.perf_counter() - t0
    if vecnorm is not None:
        vecnorm.training = False

    meta = {
        "algo": args.algo,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "obs": args.obs,
        "reward": args.reward,
        "variant": variant,
        "params": params,
        "train_window": args.train,
        "test_window": args.test,
        "train_days": int(train.index.nunique()),
        "test_days": int(test.index.nunique()),
        "stock_dim": int(len(train.tic.unique())),
        "train_seconds": round(train_seconds, 1),
        "in_sample": rollout(model, train, None, vecnorm, out, "train"),
        "out_of_sample": rollout(model, test, 70, vecnorm, out, "test"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "params"}, indent=2))


if __name__ == "__main__":
    main()
