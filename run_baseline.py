from typing import Optional

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from util import seed_everything


def build_env(mode: Optional[str] = None) -> VecEnv:
    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=42)
    env = VecFrameStack(env, n_stack=4)
    return env


def main():
    seed_everything(42)
    gym.register_envs(ale_py)
    env = build_env()
    model = DQN.load("artifacts/dqn_breakout")
    obs = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode="human")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
