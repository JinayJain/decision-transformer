import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from dataset import ChunkedRolloutDataset
from model import DecisionTransformer
from util import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to HDF5 file containing chunked data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--val_every", type=int, default=1000, help="Validate every N steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="artifacts", help="Directory to save models"
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.init(project="decision-transformer-breakout", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChunkedRolloutDataset(data_file=args.data_file)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = DecisionTransformer().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            observations = batch["observations"].to(device)
            actions = batch["actions"].long().to(device)
            rewards_to_go = batch["rewards_to_go"].to(device)
            masks = batch["masks"].to(device)

            pred_action_logits = model(
                observations, actions, rewards_to_go, masks=masks
            )

            pred_action_logits = pred_action_logits.view(
                -1, pred_action_logits.size(-1)
            )
            actions = actions.view(-1)
            masked_actions = torch.where(
                masks.view(-1), actions, torch.tensor(-100, device=device)
            )

            loss = criterion(pred_action_logits, masked_actions)

            wandb.log(
                {"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]},
                step=global_step,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % args.val_every == 0:
                model.eval()
                val_loss_accum = 0.0
                val_steps = 0
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc="Validation", leave=False):
                        observations = val_batch["observations"].to(device)
                        actions = val_batch["actions"].long().to(device)
                        rewards_to_go = val_batch["rewards_to_go"].to(device)
                        masks = val_batch["masks"].to(device)

                        pred_action_logits = model(
                            observations, actions, rewards_to_go, masks=masks
                        )

                        pred_action_logits = pred_action_logits.view(
                            -1, pred_action_logits.size(-1)
                        )
                        actions = actions.view(-1)
                        masked_actions = torch.where(
                            masks.view(-1), actions, torch.tensor(-100, device=device)
                        )

                        loss = criterion(pred_action_logits, masked_actions)

                        if not torch.isnan(loss):
                            val_loss_accum += loss.item()
                            val_steps += 1

                if val_steps > 0:
                    avg_val_loss = val_loss_accum / val_steps
                    wandb.log({"val/loss": avg_val_loss}, step=global_step)
                    print(f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}")

                    os.makedirs(args.output_dir, exist_ok=True)
                    torch.save(
                        model.state_dict(), f"{args.output_dir}/model_{global_step}.pt"
                    )
                else:
                    print(f"Step: {global_step}, Validation skipped (no valid steps)")

                model.train()


if __name__ == "__main__":
    main()
