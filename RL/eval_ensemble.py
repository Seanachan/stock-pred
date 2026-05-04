"""Ensemble eval: 3 deploy models, majority-vote action per stock per step.

Cheap test (no retrain). Compare ensemble alpha vs single-seed mean.
If ensemble alpha > best single seed (+1.62%), keep ensemble for deploy.
If ~ mean (-11%), confirms approach has real ceiling.
"""

import json

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import RL.env  # noqa: F401
from RL.constant import stock_ids
from RL.train_deploy import (
    SPLIT_VAL_END,
    SPLIT_VAL_START,
    ew_basket_return,
    load_data,
    make_env,
    max_drawdown,
)

SEED_TAGS = [
    ("deploy", "vec_normalize_deploy.pkl", "ppo_deploy"),
    ("deploy_seed1", "vec_normalize_deploy_seed1.pkl", "ppo_deploy_seed1"),
    ("deploy_seed2", "vec_normalize_deploy_seed2.pkl", "ppo_deploy_seed2"),
]


def majority_vote(action_arrays):
    """Per-stock majority vote across N (n_stocks,) discrete actions. Ties -> lowest value."""
    stacked = np.stack(action_arrays)
    voted = np.zeros(stacked.shape[1], dtype=stacked.dtype)
    for col in range(stacked.shape[1]):
        vals, counts = np.unique(stacked[:, col], return_counts=True)
        voted[col] = vals[counts.argmax()]
    return voted


def main():
    val_data = load_data(SPLIT_VAL_START, SPLIT_VAL_END)
    ew = ew_basket_return(val_data)

    env = DummyVecEnv([make_env(val_data, eval_mode=True)])

    models = []
    normalizers = []
    for tag, norm_path, model_path in SEED_TAGS:
        norm_env = DummyVecEnv([make_env(val_data, eval_mode=True)])
        vn = VecNormalize.load(norm_path, norm_env)
        vn.training = False
        vn.norm_reward = False
        m = PPO.load(model_path, device="cpu")
        models.append(m)
        normalizers.append(vn)
        print(f"loaded {tag}")

    obs = env.reset()
    done = False
    final = {}
    step = 0
    while not done:
        per_seed_actions = []
        for m, vn in zip(models, normalizers):
            normed = vn.normalize_obs(obs)
            a, _ = m.predict(normed, deterministic=True)
            per_seed_actions.append(a[0])
        voted = majority_vote(per_seed_actions)
        obs, _r, done, info = env.step(np.array([voted]))
        final = info[0]
        step += 1

    ret = final["return_rate"]
    alpha = ret - ew
    trades = final["total_trades"]
    mdd = max_drawdown(final["asset_history"])

    print(f"\n{'=' * 70}\nENSEMBLE VAL RESULT (majority vote, 3 seeds)\n{'=' * 70}")
    print(f"agent return : {ret * 100:+7.2f}%")
    print(f"EW baseline  : {ew * 100:+7.2f}%")
    print(f"alpha        : {alpha * 100:+7.2f}%")
    print(f"max drawdown : {mdd * 100:7.2f}%")
    print(f"trades       : {trades}")
    print(f"steps        : {step}")
    print()
    print("vs singles:")
    print(f"  seed0 alpha: +1.62%")
    print(f"  seed1 alpha: -8.11%")
    print(f"  seed2 alpha: -27.10%")
    print(f"  mean alpha : -11.20%")

    with open("deploy_val_ensemble.json", "w") as f:
        json.dump(
            {
                "method": "majority_vote_3seeds",
                "val": (SPLIT_VAL_START, SPLIT_VAL_END),
                "return": ret,
                "ew": ew,
                "alpha": alpha,
                "max_drawdown": mdd,
                "trades": trades,
            },
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
