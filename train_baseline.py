from typing import Optional

import ale_py
import gymnasium as gym
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack
from wandb.integration.sb3 import WandbCallback

from util import seed_everything


def build_env(mode: Optional[str] = None) -> VecEnv:
    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=42)
    env = VecFrameStack(env, n_stack=4)

    return env


def main():
    seed_everything(42)

    wandb.init(
        project="decision-transformer-dqn-baseline",
        sync_tensorboard=True,
        monitor_gym=True,
    )

    gym.register_envs(ale_py)

    env = build_env()

    model = DQN("CnnPolicy", env, verbose=1, buffer_size=10_000)
    model.learn(
        total_timesteps=1_000_000,
        callback=WandbCallback(
            verbose=2,
        ),
        log_interval=1000,
    )

    model.save("artifacts/dqn_breakout")

    env.close()


if __name__ == "__main__":
    main()
