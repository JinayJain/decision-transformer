import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from dataset import RolloutDataset
from model import DecisionTransformer
from util import seed_everything


def collate_fn(batch):
    # TODO: stack and mask unequal length sequences
    pass


def main():
    seed_everything(42)

    wandb.init(project="decision-transformer-breakout")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RolloutDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = DecisionTransformer().to(device)

    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    val_every = 1000
    for epoch in range(epochs):
        for batch in tqdm(train_loader):
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            rewards_to_go = batch["rewards_to_go"].to(device)

            pred_action_logits = model(observations, actions, rewards_to_go)
            loss = criterion(pred_action_logits.view(-1, 4), actions.view(-1))

            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            global_step += 1

            if global_step % val_every == 0:
                model.eval()

                with torch.no_grad():
                    val_loss = 0
                    for batch in val_loader:
                        observations = batch["observations"].to(device)
                        actions = batch["actions"].to(device)
                        rewards_to_go = batch["rewards_to_go"].to(device)

                        pred_action_logits = model(observations, actions, rewards_to_go)
                        loss = criterion(
                            pred_action_logits.view(-1, 4), actions.view(-1)
                        )
                        val_loss += loss.item()

                    val_loss /= len(val_loader)

                    wandb.log({"val_loss": val_loss}, step=global_step)

                    torch.save(model.state_dict(), f"artifacts/model_{global_step}.pt")

                model.train()


if __name__ == "__main__":
    main()
