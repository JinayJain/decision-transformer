import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChunkedRolloutDataset
from model import DecisionTransformer
from util import seed_everything


def main():
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = sys.argv[1]

    dataset = ChunkedRolloutDataset(data_path)

    model = DecisionTransformer()
    model.load_state_dict(torch.load("artifacts/model_2100.pt"))
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    n_correct = 0
    n_total = 0

    pbar = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for batch in pbar:
        batch_size, seq_len, *_ = batch["observations"].shape

        for t in range(seq_len):
            obs = batch["observations"][:, : t + 1].to(device)
            action = batch["actions"][:, : t + 1].to(device)
            rtg = batch["rewards_to_go"][:, : t + 1].to(device)

            print(rtg)
            logits = model(obs, action[:, :t], rtg)
            print(logits.shape)

            # loss for only the last action
            if t > 0:
                loss = criterion(logits[:, -1, :], action[:, -1])

                print(loss.item(), action[:, -1], logits[:, -1, :].argmax(dim=-1))

    print(f"Final Accuracy: {n_correct / n_total:.2%}")


if __name__ == "__main__":
    main()
