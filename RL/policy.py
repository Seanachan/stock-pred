"""Per-asset shared-weight encoder for permutation-aware portfolio policies.

Each stock's (12 feats + inv_norm) goes through the *same* small MLP, so the
network sees stocks as instances of one shared concept rather than 46 separate
slots. Cash is concatenated as a single scalar at the end.

Output shape: (B, num_stocks * emb_dim + 1). PPO's pi/vf MLPs read this flat
vector; permutation-equivariance is preserved at the encoder level (the
strongest part of the network) even though the final action head still maps
position → action.
"""

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PerAssetEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        num_stocks: int,
        feat_per_stock: int = 12,
        emb_dim: int = 32,
        hidden: int = 64,
    ):
        per_stock_in = feat_per_stock + 1
        features_dim = num_stocks * emb_dim + 1
        super().__init__(observation_space, features_dim=features_dim)

        self.num_stocks = num_stocks
        self.feat_per_stock = feat_per_stock
        self.emb_dim = emb_dim

        self.shared = nn.Sequential(
            nn.Linear(per_stock_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        N = self.num_stocks
        F = self.feat_per_stock

        feats = obs[:, : N * F].reshape(B, N, F)
        cash = obs[:, N * F : N * F + 1]
        inv = obs[:, N * F + 1 :].unsqueeze(-1)

        x = torch.cat([feats, inv], dim=-1)
        emb = self.shared(x)
        flat = emb.reshape(B, N * self.emb_dim)
        return torch.cat([flat, cash], dim=-1)
