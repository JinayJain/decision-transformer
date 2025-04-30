import time

import torch

from model import DecisionTransformer
from util import build_env


def main():
    model = DecisionTransformer()
    model.load_state_dict(torch.load("artifacts/model_3500.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    env = build_env(1, is_eval=True)

    obs = env.reset()

    observations = [torch.from_numpy(obs).to(device)]  # [Tensor(1, 84, 84, 4)]

    actions = []  # [int]

    target_return = 90.0

    returns_to_go = [target_return]  # [float]

    max_window_size = 60

    while True:
        obs_tensor = torch.stack(
            observations[-max_window_size:], dim=1
        ).float()  # [1, T, 84, 84, 4]
        action_tensor = (
            torch.tensor(actions[-max_window_size:], device=device).unsqueeze(0).long()
            if len(actions) > 0
            else torch.empty(device=device, size=(1, 0), dtype=torch.long)
        )  # [1, T]
        rtg_tensor = (
            torch.tensor(returns_to_go[-max_window_size:], device=device)
            .unsqueeze(0)
            .float()
        )  # [1, T]

        with torch.no_grad():
            pred_action_logits = model(obs_tensor, action_tensor, rtg_tensor)
            action = torch.argmax(pred_action_logits, dim=-1)[:, -1]

        obs, reward, done, info = env.step(action)
        env.render(mode="human")

        obs = torch.from_numpy(obs).to(device)
        action = torch.tensor(action, device=device)
        rtg = max(0, returns_to_go[-1] - reward.item())

        observations.append(obs)
        actions.append(action.item())
        returns_to_go.append(rtg)

        print(rtg)

        if done.all():
            obs = env.reset()
            observations = [torch.from_numpy(obs).to(device)]
            actions = []
            returns_to_go = [target_return]


if __name__ == "__main__":
    main()
