import json
import os

import h5py
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
        actions = torch.from_numpy(np.array(data["actions"])).float()
        rewards = torch.from_numpy(np.array(data["rewards"])).float()
        dones = torch.from_numpy(np.array(data["dones"])).bool()
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


class ChunkedRolloutDataset(Dataset):
    def __init__(self, data_file):
        """
        Dataset for loading chunked rollout data stored in an HDF5 file.

        Args:
            data_file: Path to the HDF5 file containing the chunked data.
        """
        self.data_file = data_file
        self.h5_file = None  # Initialize lazily or handle in __getitem__ if multiprocessing issues arise

        # Open the HDF5 file in read mode
        try:
            self.h5_file = h5py.File(self.data_file, "r")
        except Exception as e:
            print(f"Error opening HDF5 file {self.data_file}: {e}")
            raise

        # Get total number of samples from one of the datasets
        try:
            self.total_samples = self.h5_file["observations"].shape[0]
        except Exception as e:
            print(f"Error accessing datasets in HDF5 file {self.data_file}: {e}")
            if self.h5_file:
                self.h5_file.close()
            raise

    def __del__(self):
        """Close the HDF5 file when the dataset object is destroyed."""
        if self.h5_file:
            self.h5_file.close()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset size {self.total_samples}"
            )

        if not self.h5_file:
            try:
                self.h5_file = h5py.File(self.data_file, "r")
            except Exception as e:
                print(f"Error reopening HDF5 file in getitem: {e}")
                raise  # Or return an error state

        try:
            # Read the data for the specified index directly from HDF5
            observations = self.h5_file["observations"][idx]
            actions = self.h5_file["actions"][idx]
            rewards = self.h5_file["rewards"][idx]
            dones = self.h5_file["dones"][idx]
            rewards_to_go = self.h5_file["rewards_to_go"][idx]
            masks = self.h5_file["masks"][idx]
        except Exception as e:
            print(f"Error reading data at index {idx} from {self.data_file}: {e}")
            # Consider how to handle read errors - skip, raise, etc.
            raise

        # Convert numpy arrays to torch tensors
        return {
            "observations": torch.from_numpy(observations).float(),
            "actions": torch.from_numpy(actions).long(),
            "rewards": torch.from_numpy(rewards).float(),
            "dones": torch.from_numpy(dones).bool(),
            "rewards_to_go": torch.from_numpy(rewards_to_go).float(),
            "masks": torch.from_numpy(masks).bool(),
        }
