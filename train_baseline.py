import json
import os
import pickle
import time

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
from stable_baselines3.common.vec_env import VecEnv

from util import build_env, seed_everything


class SaveRolloutsCallback(BaseCallback):
    def __init__(
        self, env: VecEnv, save_dir="artifacts/rollouts", n_episodes=50, verbose=0
    ):
        super().__init__(verbose)

        assert env.num_envs == 1

        self.test_env = env
        self.save_dir = save_dir
        self.n_episodes = n_episodes

        self.metadata_path = os.path.join(save_dir, "metadata.jsonl")

        os.makedirs(save_dir, exist_ok=False)

    def _run_episode(self):
        obs = self.test_env.reset()  # Get initial observation
        done = False

        # Initialize lists to store trajectory data
        observations = []
        actions = []
        rewards = []
        dones = []

        while not done:
            observations.append(obs[0].copy())

            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)

            next_obs, reward, done, _ = self.test_env.step(action)

            assert (
                obs.shape[0] == action.shape[0] == reward.shape[0] == done.shape[0] == 1
            )

            actions.append(action[0].copy())
            rewards.append(reward[0].copy())
            dones.append(done[0])

            obs = next_obs

        return observations, actions, rewards, dones

    def _compute_rewards_to_go(self, rewards: np.ndarray) -> np.ndarray:
        flipped_rewards = np.flip(rewards, axis=0)
        cumsum_rewards = np.cumsum(flipped_rewards)
        rewards_to_go = np.flip(cumsum_rewards, axis=0)

        return rewards_to_go

    def _on_step(self) -> bool:
        print(f"Saving rollouts at t={self.num_timesteps}")

        with open(self.metadata_path, "a") as metadata_file:
            for i in range(self.n_episodes):
                observations, actions, rewards, dones = self._run_episode()

                rewards_to_go = self._compute_rewards_to_go(rewards)

                save_path = os.path.join(
                    self.save_dir, f"rollout_t-{self.num_timesteps}_{i}.npz"
                )

                np.savez_compressed(
                    save_path,
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    rewards_to_go=rewards_to_go,
                )

                total_reward = sum(rewards).item()
                metadata = {
                    "timestep": self.num_timesteps,
                    "episode": i,
                    "path": save_path,
                    "length": len(rewards),
                    "total_reward": total_reward,
                }

                metadata_file.write(json.dumps(metadata) + "\n")

        return True


def main():
    seed_everything(42)

    gym.register_envs(ale_py)

    env = build_env(n_envs=4)
    test_env = build_env(n_envs=1, is_eval=True)

    model = DQN("CnnPolicy", env, verbose=1, buffer_size=30_000)
    model.learn(
        total_timesteps=1_000_000,
        callback=CallbackList(
            [
                EveryNTimesteps(
                    n_steps=100_000,
                    callback=SaveRolloutsCallback(test_env, n_episodes=1_000),
                ),
            ]
        ),
        log_interval=4,
    )

    model.save("artifacts/dqn_breakout")

    env.close()
    test_env.close()


if __name__ == "__main__":
    main()
