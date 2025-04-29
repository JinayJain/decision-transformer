import random

import ale_py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack


def seed_everything(seed: int) -> None:
    """
    Seed all random number generators for reproducibility.

    Args:
        seed: The seed value to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_env(n_envs: int) -> VecEnv:
    gym.register_envs(ale_py)

    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=n_envs, seed=42)
    env = VecFrameStack(env, n_stack=4)

    return env
