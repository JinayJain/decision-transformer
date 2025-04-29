import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import wandb
from dataset import RolloutDataset
from model import DecisionTransformer


def collate_fn(batch):
    # TODO: stack and mask unequal length sequences
    pass


def main():
    wandb.init(project="decision-transformer-breakout")

    dataset = RolloutDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = DecisionTransformer()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch in train_loader:
            observations = batch["observations"]
            actions = batch["actions"]
            rewards_to_go = batch["rewards_to_go"]

            pred_action_logits = model(observations, actions, rewards_to_go)
            targets = actions[:, 1:]
            loss = criterion(pred_action_logits[:, :-1].view(-1, 4), targets.view(-1))

            wandb.log({"loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
