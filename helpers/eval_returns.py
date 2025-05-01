import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import DecisionTransformer
from util import build_env


def evaluate(model, device, env, target_return, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        observations = [torch.from_numpy(obs).to(device)]
        actions = []
        returns_to_go = [target_return]
        total_reward = 0.0
        done = [False]
        while not all(done):
            obs_tensor = torch.stack(
                observations[-model.max_window_size :], dim=1
            ).float()
            action_tensor = (
                torch.tensor(actions[-model.max_window_size :], device=device)
                .unsqueeze(0)
                .long()
                if len(actions) > 0
                else torch.empty(device=device, size=(1, 0), dtype=torch.long)
            )
            rtg_tensor = (
                torch.tensor(returns_to_go[-model.max_window_size :], device=device)
                .unsqueeze(0)
                .float()
            )
            with torch.no_grad():
                pred_action_logits = model(obs_tensor, action_tensor, rtg_tensor)
                action = torch.argmax(pred_action_logits, dim=-1)[:, -1]
            obs, reward, done, info = env.step(action)
            env.render(mode="human")
            obs = torch.from_numpy(obs).to(device)
            rtg = max(0, returns_to_go[-1] - reward.item())
            observations.append(obs)
            actions.append(action.item())
            returns_to_go.append(rtg)
            total_reward += reward.item()
        rewards.append(total_reward)
    return np.mean(rewards), np.median(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--returns",
        type=float,
        nargs="*",
        default=None,
        help="List of target returns to evaluate. If not set, uses linspace.",
    )
    parser.add_argument("--output", type=str, default="eval_returns.png")
    args = parser.parse_args()

    model = DecisionTransformer()
    model.load_state_dict(torch.load(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    env = build_env(1, is_eval=True)

    if args.returns is not None and len(args.returns) > 0:
        target_returns = np.array(args.returns)
    else:
        target_returns = np.linspace(0.0, 200.0, 21)

    means = []
    medians = []
    stds = []
    for tr in target_returns:
        mean, median, std = evaluate(model, device, env, tr, n_episodes=args.episodes)
        means.append(mean)
        medians.append(median)
        stds.append(std)
        print(f"Target return: {tr:.2f} | Avg reward: {mean:.2f} ± {std:.2f}")

    # save statistics to a json file
    with open(args.output.replace(".png", ".json"), "w") as f:
        json.dump({"means": means, "medians": medians, "stds": stds}, f)


if __name__ == "__main__":
    main()
