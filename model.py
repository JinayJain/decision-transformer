import torch
from einops import rearrange
from torch import nn
from x_transformers import Decoder


class ObservationEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, d_model),
        )

    def forward(self, x) -> torch.Tensor:
        batch_size, seq_len, *_ = x.shape

        x = rearrange(x, "b t h w c -> (b t) c h w")
        x = self.net(x)
        x = rearrange(x, "(b t) d -> b t d", b=batch_size, t=seq_len)

        return x


class ActionEncoder(nn.Module):
    def __init__(self, d_model: int, n_actions: int):
        super().__init__()

        self.d_model = d_model
        self.n_actions = n_actions

        self.embed = nn.Embedding(n_actions, d_model)

    def forward(self, x) -> torch.Tensor:
        batch_size, seq_len, *_ = x.shape

        x = rearrange(x, "b t -> (b t)")
        x = self.embed(x)
        x = rearrange(x, "(b t) d -> b t d", b=batch_size, t=seq_len)

        return x


class ReturnsToGoEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        self.proj = nn.Linear(1, d_model)

    def forward(self, x) -> torch.Tensor:
        batch_size, seq_len, *_ = x.shape

        x = rearrange(x, "b t -> (b t) 1")
        x = self.proj(x)
        x = rearrange(x, "(b t) d -> b t d", b=batch_size, t=seq_len)

        return x


class DecisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.d_model = 128

        self.obs_enc = ObservationEncoder(self.d_model)
        self.action_enc = ActionEncoder(self.d_model, 4)
        self.rtg_enc = ReturnsToGoEncoder(self.d_model)

        self.transformer = Decoder(
            dim=self.d_model,
            depth=6,
            heads=8,
            attn_flash=True,
        )

        self.action_proj = nn.Linear(self.d_model, 4, bias=False)
        self.action_proj.weight = self.action_enc.embed.weight

    def forward(self, observations, actions, returns_to_go):
        """
        observations: (batch_size, n_obs, obs_dim)
        actions: (batch_size, n_actions)
        returns_to_go: (batch_size, n_returns)
        """

        batch_size, n_obs, *_ = observations.shape

        _, n_actions = actions.shape
        _, n_returns = returns_to_go.shape

        # embed observations, actions, returns_to_go into a shared embedding space
        obs_embed = self.obs_enc(observations)  # (batch_size, n_obs, d_model)
        action_embed = self.action_enc(actions)  # (batch_size, n_actions, d_model)
        rtg_embed = self.rtg_enc(returns_to_go)  # (batch_size, n_returns, d_model)

        # TODO: add positional encodings, by timestep

        # interleave returns_to_go, observations, actions
        x = torch.empty(
            batch_size,
            n_obs + n_returns + n_actions,
            self.d_model,
            device=observations.device,
        )

        x[:, ::3, :] = rtg_embed
        x[:, 1::3, :] = obs_embed
        x[:, 2::3, :] = action_embed

        # pass through transformer model
        x = self.transformer(x)  # (batch_size, 3 * seq_len, d_model)

        x = x[
            :, 1::3, :
        ]  # get the hidden states after the observations, for predicting the actions

        # predict next actions
        action_logits = self.action_proj(x)  # (batch_size, seq_len, n_actions)

        return action_logits
