import argparse

import h5py
import numpy as np
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


def main():
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    args = parser.parse_args()

    dataset = RolloutDataset(args.data_dir)

    with h5py.File(args.output_file, "w") as f:
        f.create_dataset(
            "observations",
            shape=(0, args.chunk_size, 84, 84, 4),
            maxshape=(None, args.chunk_size, 84, 84, 4),
            chunks=(1, args.chunk_size, 84, 84, 4),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=2,
        )
        f.create_dataset(
            "actions",
            shape=(0, args.chunk_size),
            maxshape=(None, args.chunk_size),
            chunks=(1, args.chunk_size),
            dtype=np.int64,
        )
        f.create_dataset(
            "rewards",
            shape=(0, args.chunk_size),
            maxshape=(None, args.chunk_size),
            chunks=(1, args.chunk_size),
            dtype=np.float32,
        )
        f.create_dataset(
            "dones",
            shape=(0, args.chunk_size),
            maxshape=(None, args.chunk_size),
            chunks=(1, args.chunk_size),
            dtype="bool",
        )
        f.create_dataset(
            "rewards_to_go",
            shape=(0, args.chunk_size),
            maxshape=(None, args.chunk_size),
            chunks=(1, args.chunk_size),
            dtype=np.float32,
        )
        f.create_dataset(
            "masks",
            shape=(0, args.chunk_size),
            maxshape=(None, args.chunk_size),
            chunks=(1, args.chunk_size),
            dtype="bool",
        )

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

            f["observations"].resize(
                f["observations"].shape[0] + len(observations), axis=0
            )
            f["actions"].resize(f["actions"].shape[0] + len(actions), axis=0)
            f["rewards"].resize(f["rewards"].shape[0] + len(rewards), axis=0)
            f["dones"].resize(f["dones"].shape[0] + len(dones), axis=0)
            f["rewards_to_go"].resize(
                f["rewards_to_go"].shape[0] + len(rewards_to_go), axis=0
            )
            f["masks"].resize(f["masks"].shape[0] + len(masks), axis=0)

            f["observations"][-len(observations) :] = observations
            f["actions"][-len(actions) :] = actions
            f["rewards"][-len(rewards) :] = rewards
            f["dones"][-len(dones) :] = dones
            f["rewards_to_go"][-len(rewards_to_go) :] = rewards_to_go
            f["masks"][-len(masks) :] = masks


if __name__ == "__main__":
    main()
