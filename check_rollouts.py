import argparse
import os
import pickle
from glob import glob

import numpy as np


def load_rollouts(path):
    """Load rollouts from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def print_trajectory_stats(trajectory):
    """Print statistics for a single trajectory."""
    print(f"Observations shape: {trajectory['observations'].shape}")
    print(f"Actions shape: {trajectory['actions'].shape}")
    print(f"Rewards shape: {trajectory['rewards'].shape}")
    print(f"Returns-to-go shape: {trajectory['returns_to_go'].shape}")
    print(f"Total reward: {trajectory['rewards'].sum()}")
    print(f"Initial return-to-go: {trajectory['returns_to_go'][0]}")
    print(f"Episode length: {len(trajectory['rewards'])}")


def print_dataset_stats(trajectories):
    """Print statistics for a dataset of trajectories."""
    n_trajectories = len(trajectories)
    episode_lengths = [len(traj["rewards"]) for traj in trajectories]
    total_rewards = [traj["rewards"].sum() for traj in trajectories]

    print(f"Number of trajectories: {n_trajectories}")
    print(
        f"Average episode length: {np.mean(episode_lengths):.2f} (min: {min(episode_lengths)}, max: {max(episode_lengths)})"
    )
    print(
        f"Average total reward: {np.mean(total_rewards):.2f} (min: {min(total_rewards):.2f}, max: {max(total_rewards):.2f})"
    )

    # Calculate the total dataset size
    total_transitions = sum(episode_lengths)
    print(f"Total transitions: {total_transitions}")

    # Check observation and action dimensions
    obs_dim = (
        trajectories[0]["observations"].shape[-1]
        if len(trajectories[0]["observations"].shape) > 1
        else 1
    )
    act_dim = (
        trajectories[0]["actions"].shape[-1]
        if len(trajectories[0]["actions"].shape) > 1
        else 1
    )
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")


def main():
    parser = argparse.ArgumentParser(
        description="Check saved rollouts for Decision Transformer training."
    )
    parser.add_argument(
        "--rollout_dir",
        type=str,
        default="artifacts/rollouts",
        help="Directory containing rollout files",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific rollout file to check (optional)",
    )
    parser.add_argument(
        "--sample", action="store_true", help="Print details of a sample trajectory"
    )
    args = parser.parse_args()

    if args.file:
        # Check a specific file
        path = args.file
        print(f"Checking rollout file: {path}")
        trajectories = load_rollouts(path)
    else:
        # Get the most recent rollout file
        files = sorted(glob(os.path.join(args.rollout_dir, "*.pkl")))
        if not files:
            print(f"No rollout files found in {args.rollout_dir}")
            return

        path = files[-1]  # Most recent file
        print(f"Checking most recent rollout file: {path}")
        trajectories = load_rollouts(path)

    # Print dataset statistics
    print("\n===== Dataset Statistics =====")
    print_dataset_stats(trajectories)

    # Print a sample trajectory if requested
    if args.sample and trajectories:
        print("\n===== Sample Trajectory =====")
        # Choose the trajectory with the highest return
        best_traj_idx = np.argmax([traj["rewards"].sum() for traj in trajectories])
        print(f"Showing trajectory {best_traj_idx} (highest return)")
        print_trajectory_stats(trajectories[best_traj_idx])


if __name__ == "__main__":
    main()
