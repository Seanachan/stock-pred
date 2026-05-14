"""v3 smoke retrain: validates log-utility reward + cash cap kills cash-parking.

500k timesteps, single seed, 8 parallel envs. ~45-90 min on CPU.
Gate: if val alpha > 0 vs EW basket, scale to 5 seeds x 2M steps.
"""

import os
import sys

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

import RL.train as t


def main():
    seed = int(os.environ.get("SMOKE_SEED", 42))
    total_timesteps = int(os.environ.get("SMOKE_STEPS", 500_000))
    n_envs = int(os.environ.get("SMOKE_NENVS", 8))
    tag = os.environ.get("SMOKE_TAG", "v3_smoke")

    print(
        f"v3 smoke: seed={seed} steps={total_timesteps:,} "
        f"n_envs={n_envs} tag={tag}"
    )

    data = t.load_data()
    if not data:
        print("ERROR: no stock CSVs loaded from RL/data/")
        sys.exit(1)

    env = SubprocVecEnv([t.make_env(data) for _ in range(n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    try:
        # train_agent saves to PPO_v8_seed{seed} by hardcoded naming.
        # We rename outputs post-train.
        t.train_agent(env, total_timesteps=total_timesteps, seed=seed)
    finally:
        env.close()

    # rename outputs so the smoke artifact is unambiguous
    src_model = f"ppo_trading_agent_v8_seed{seed}.zip"
    src_norm = f"vec_normalize_v8_seed{seed}.pkl"
    dst_model = f"ppo_{tag}_seed{seed}.zip"
    dst_norm = f"vec_normalize_{tag}_seed{seed}.pkl"
    for src, dst in [(src_model, dst_model), (src_norm, dst_norm)]:
        if os.path.exists(src):
            os.rename(src, dst)
            print(f"  renamed {src} -> {dst}")

    print(f"\nSMOKE DONE: model={dst_model} norm={dst_norm}")
    print("Next: walk_forward eval on val split to check alpha sign.")


if __name__ == "__main__":
    main()
