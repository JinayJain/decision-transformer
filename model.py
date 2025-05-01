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
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, d_model),
        )

    def forward(self, x) -> torch.Tensor:
        batch_size, seq_len, *_ = x.shape

        x = x / 255.0

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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x) -> torch.Tensor:
        batch_size, seq_len, *_ = x.shape

        x = rearrange(x, "b t -> (b t)")
        x = self.dropout(self.embed(x))
        x = rearrange(x, "(b t) d -> b t d", b=batch_size, t=seq_len)

        return x


class ReturnsToGoEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        self.proj = nn.Linear(1, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x) -> torch.Tensor:
        batch_size, seq_len, *_ = x.shape

        x = rearrange(x, "b t -> (b t) 1")
        x = self.dropout(self.proj(x))
        x = rearrange(x, "(b t) d -> b t d", b=batch_size, t=seq_len)

        return x


class DecisionTransformer(nn.Module):
    def __init__(self, d_model: int = 128, max_window_size: int = 30):
        super().__init__()

        self.d_model = d_model
        self.max_window_size = max_window_size

        self.obs_enc = ObservationEncoder(self.d_model)
        self.action_enc = ActionEncoder(self.d_model, 4)
        self.rtg_enc = ReturnsToGoEncoder(self.d_model)

        self.transformer = Decoder(
            dim=self.d_model,
            depth=6,
            heads=8,
            attn_flash=True,
            layer_dropout=0.1,
        )

        self.action_proj = nn.Linear(self.d_model, 4)

        self.pos_emb = nn.Embedding(max_window_size, d_model)

    def forward(self, observations, actions, returns_to_go, masks=None):
        """
        observations: (batch_size, n_obs, obs_dim)
        actions: (batch_size, n_actions)
        returns_to_go: (batch_size, n_returns)
        """

        batch_size, n_obs, *_ = observations.shape

        n_actions = actions.shape[1]
        n_returns = returns_to_go.shape[1]

        pos_emb = self.pos_emb(
            torch.arange(n_obs, device=observations.device, dtype=torch.long)
        )[None, :, :]

        # embed observations, actions, returns_to_go into a shared embedding space
        obs_embed = (
            self.obs_enc(observations) + pos_emb[:, :n_obs, :]
        )  # (batch_size, n_obs, d_model)
        action_embed = (
            self.action_enc(actions) + pos_emb[:, :n_actions, :]
        )  # (batch_size, n_actions, d_model)
        rtg_embed = (
            self.rtg_enc(returns_to_go) + pos_emb[:, :n_returns, :]
        )  # (batch_size, n_returns, d_model)

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

        if masks is not None:
            transformer_mask = masks.repeat_interleave(3, dim=1)
        else:
            transformer_mask = None

        # pass through transformer model
        x = self.transformer(
            x, mask=transformer_mask
        )  # (batch_size, 3 * seq_len, d_model)

        # Get the hidden states corresponding to observations to predict the next actions
        # Slicing retrieves the embeddings after the observation embeddings: R S A R S A ... -> _ S _ _ S _ ...
        x = x[:, 1::3, :]  # (batch_size, seq_len, d_model)

        # predict next actions
        action_logits = self.action_proj(x)  # (batch_size, seq_len, n_actions)

        return action_logits
