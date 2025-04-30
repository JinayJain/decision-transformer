import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ChunkedRolloutDataset
from model import DecisionTransformer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = sys.argv[1]

    dataset = ChunkedRolloutDataset(data_path)

    model = DecisionTransformer()
    model.load_state_dict(torch.load("artifacts/model_3150.pt"))
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    n_correct = 0
    n_total = 0

    pbar = tqdm(loader)
    for batch in pbar:
        obs = batch["observations"].to(device)
        action = batch["actions"].to(device)
        rtg = batch["rewards_to_go"].to(device)
        done = batch["dones"].to(device)

        logits = model(obs, action, rtg)
        logits = logits[:, -1, :]

        pred_action = logits.argmax(dim=-1).item()
        target_action = action[:, -1].item()

        n_correct += pred_action == target_action
        n_total += 1

        current_acc = n_correct / n_total
        pbar.set_postfix({"acc": f"{current_acc:.2%}"})

    print(f"Final Accuracy: {n_correct / n_total:.2%}")


if __name__ == "__main__":
    main()
