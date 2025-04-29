import os
import pickle

import ale_py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EveryNTimesteps,
)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from util import seed_everything


def build_env(n_envs: int) -> VecEnv:
    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=n_envs, seed=42)
    env = VecFrameStack(env, n_stack=4)

    return env


class SaveRolloutsCallback(BaseCallback):
    def __init__(
        self, env: VecEnv, save_dir="artifacts/rollouts", n_episodes=50, verbose=0
    ):
        super().__init__(verbose)

        assert env.num_envs == 1

        self.test_env = env
        self.save_dir = save_dir
        self.n_episodes = n_episodes
        self.episode_count = 0

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        print(f"Saving rollouts, episode count: {self.episode_count}")

        # Each episode will be a dictionary with observations, actions, rewards, dones
        trajectories = []

        for _ in range(self.n_episodes):
            obs = self.test_env.reset()  # Get initial observation
            done = False

            # Initialize lists to store trajectory data
            observations = []
            actions = []
            rewards = []
            dones = []
            infos = []

            # Run one episode
            while not done:
                observations.append(obs.copy())

                # Predict action
                action, _ = self.model.predict(obs, deterministic=True)

                # Step environment
                next_obs, reward, done, info = self.test_env.step(action)

                # Store transition
                actions.append(action.copy())
                rewards.append(reward.copy())
                dones.append(done)
                infos.append(info)

                # Update observation
                obs = next_obs

            # Create episode trajectory
            trajectory = {
                "observations": np.array(observations),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "dones": np.array(dones),
                "infos": infos,
            }

            trajectories.append(trajectory)

            self.episode_count += 1

        # Save the trajectories
        timestamp = self.num_timesteps
        save_path = os.path.join(self.save_dir, f"rollouts_{timestamp}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(trajectories, f)

        print(f"Saved {len(trajectories)} trajectories to {save_path}")

        return True


def main():
    seed_everything(42)

    gym.register_envs(ale_py)

    env = build_env(n_envs=4)
    rollout_env = build_env(n_envs=1)

    model = DQN("CnnPolicy", env, verbose=1, buffer_size=10_000)
    model.learn(
        total_timesteps=1_000_000,
        callback=CallbackList(
            [
                EveryNTimesteps(
                    n_steps=10_000, callback=SaveRolloutsCallback(rollout_env)
                ),
            ]
        ),
        log_interval=1000,
    )

    # model.save("artifacts/dqn_breakout")

    env.close()


if __name__ == "__main__":
    main()
