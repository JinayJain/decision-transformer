import argparse
import json
import os
import pickle
import random
import shutil
import sys

import numpy as np
import torch
from tqdm import tqdm

from dataset import RolloutDataset
from util import seed_everything


def split_and_pad(x, chunk_size):
    splits = np.split(x, np.arange(chunk_size, len(x), chunk_size))
    masks = []

    for i in range(len(splits)):
        mask = np.full(chunk_size, False)
        mask[: len(splits[i])] = 1
        masks.append(mask)

        if len(splits[i]) < chunk_size:
            pad_axes = [(0, 0) for _ in splits[i].shape]
            pad_axes[0] = (0, chunk_size - len(splits[i]))

            splits[i] = np.pad(
                splits[i],
                pad_axes,
                mode="constant",
            )

    return splits, masks


def save_shard(shard, shard_idx, shard_metadata, output_dir):
    shard_path = os.path.join(output_dir, f"shard_{shard_idx}.npz")

    for key in shard.keys():
        shard[key] = np.array(shard[key])

    np.savez_compressed(shard_path, **shard)

    shard_metadata.append(
        {
            "shard_path": shard_path,
            "num_samples": len(shard["observations"]),
        }
    )

    del shard


def main():
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--shard_size", type=int, default=1000)
    args = parser.parse_args()

    dataset = RolloutDataset(args.data_dir)

    current_shard = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "rewards_to_go": [],
        "masks": [],
    }
    shard_idx = 0

    if os.path.exists(args.output_dir):
        if (
            input(
                "The output directory already exists. Do you want to overwrite it? (y/n): "
            )
            != "y"
        ):
            sys.exit(1)
        else:
            shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)

    shard_metadata = []

    for rollout in tqdm(dataset):
        observations = rollout["observations"]
        actions = rollout["actions"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        rewards_to_go = rollout["rewards_to_go"]

        observations, masks = split_and_pad(observations, args.chunk_size)
        actions, _ = split_and_pad(actions, args.chunk_size)
        rewards, _ = split_and_pad(rewards, args.chunk_size)
        dones, _ = split_and_pad(dones, args.chunk_size)
        rewards_to_go, _ = split_and_pad(rewards_to_go, args.chunk_size)

        for i in range(len(observations)):
            current_shard["observations"].append(observations[i])
            current_shard["actions"].append(actions[i])
            current_shard["rewards"].append(rewards[i])
            current_shard["dones"].append(dones[i])
            current_shard["rewards_to_go"].append(rewards_to_go[i])
            current_shard["masks"].append(masks[i])

        if len(current_shard["observations"]) >= args.shard_size:
            save_shard(current_shard, shard_idx, shard_metadata, args.output_dir)

            current_shard = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": [],
                "rewards_to_go": [],
                "masks": [],
            }
            shard_idx += 1

    if len(current_shard) > 0:
        save_shard(current_shard, shard_idx, shard_metadata, args.output_dir)

    # shuffle the shard metadata
    random.shuffle(shard_metadata)

    with open(os.path.join(args.output_dir, "shard_metadata.json"), "w") as f:
        json.dump(shard_metadata, f)


if __name__ == "__main__":
    main()
