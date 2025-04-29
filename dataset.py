import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class RolloutDataset(Dataset):
    def __init__(self, data_dir="artifacts/rollouts"):
        """
        Dataset for loading rollout data saved by SaveRolloutsCallback.

        Args:
            data_dir: Directory containing .npz rollout files and metadata.jsonl
        """
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, "metadata.jsonl")

        # Load metadata to get all file paths
        self.metadata = []
        with open(self.metadata_path, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        # Sort by timestep then episode number
        self.metadata.sort(key=lambda x: (x["timestep"], x["episode"]))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        data = np.load(meta["path"])

        # Convert numpy arrays to torch tensors
        observations = torch.from_numpy(np.array(data["observations"])).float()
        actions = torch.from_numpy(np.array(data["actions"])).long()
        rewards = torch.from_numpy(np.array(data["rewards"])).float()
        dones = torch.from_numpy(np.array(data["dones"])).float()
        rewards_to_go = torch.from_numpy(np.array(data["rewards_to_go"])).float()

        # Return as dictionary
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "rewards_to_go": rewards_to_go,
            "timestep": meta["timestep"],
            "episode": meta["episode"],
            "total_reward": meta["total_reward"],
            "length": meta["length"],
        }
