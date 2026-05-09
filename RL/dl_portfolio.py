"""Supervised portfolio optimization (Zhang/Zohren/Roberts 2020).

Per-asset shared MLP → score → softmax over (N stocks + cash) → portfolio
weights. Differentiable Sharpe ratio over the realized portfolio-return
series is the training loss; gradient flows directly through the trades.

No RL, no environment, no exploration. The network outputs the next-day
target weight w_t conditioned on features observed at t-1; loss is
computed end-to-end on the resulting portfolio P&L series.

Tensors:
  feats:  (T, N, F)         daily per-stock features (12-d)
  rets:   (T, N)            daily simple returns r_{i,t} = (p_t/p_{t-1})-1
  mask:   (T, N) bool       True where the asset has a valid bar that day
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def build_tensors(stock_data, feature_extractor, stock_ids, device="cpu"):
    """Convert {sid: DataFrame} → aligned (T, N, F) feats and (T, N) returns.

    Drops rows with too little history. Returns numpy arrays plus the date
    index used; caller wraps to torch on the right device.
    """
    market_dfs = feature_extractor.extract_features(stock_data)
    common_dates = sorted(
        set.intersection(
            *[set(df.index) for df in market_dfs.values() if df is not None and len(df) > 0]
        )
    )
    if not common_dates:
        raise RuntimeError("no common date across stocks")
    feat_cols = [
        "return", "bias_5", "bias_20", "macd_h", "rsi_14", "bb_pos", "atr",
        "capacity_change", "return_rank", "rsi_14_rank", "bias_20_rank",
        "capacity_change_rank",
    ]
    F = len(feat_cols)
    N = len(stock_ids)
    T = len(common_dates)
    feats = np.zeros((T, N, F), dtype=np.float32)
    rets = np.zeros((T, N), dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)
    closes = np.zeros((T, N), dtype=np.float32)
    for j, sid in enumerate(stock_ids):
        df = market_dfs.get(sid)
        if df is None or df.empty:
            continue
        sub = df.reindex(common_dates)
        for k, c in enumerate(feat_cols):
            if c in sub.columns:
                feats[:, j, k] = sub[c].fillna(0.0).to_numpy(dtype=np.float32)
        if "close" in sub.columns:
            close = sub["close"].to_numpy(dtype=np.float32)
            closes[:, j] = np.nan_to_num(close, nan=0.0)
            r = sub["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
            rets[:, j] = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            mask[:, j] = (~sub["close"].isna()) & (close > 0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    return {
        "feats": torch.from_numpy(feats).to(device),
        "rets": torch.from_numpy(rets).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "closes": torch.from_numpy(closes).to(device),
        "dates": common_dates,
        "feat_cols": feat_cols,
    }


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Closed-form sparse projection onto the simplex (Martins & Astudillo 2016).

    Same shape as input, sums to 1 along `dim`, with explicit zeros where
    softmax would have left tiny tails. Differentiable subgradient.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    sorted_z, _ = torch.sort(z, dim=dim, descending=True)
    K = z.shape[dim]
    rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    rng_shape = [1] * z.ndim
    rng_shape[dim] = K
    rng = rng.view(rng_shape)
    cumsum = torch.cumsum(sorted_z, dim=dim)
    cond = 1 + rng * sorted_z > cumsum
    k = cond.to(z.dtype).sum(dim=dim, keepdim=True)
    cumsum_k = cumsum.gather(dim, k.long() - 1)
    tau = (cumsum_k - 1) / k
    return torch.clamp(z - tau, min=0)


class PortfolioNet(nn.Module):
    """Per-asset shared encoder → score → softmax/sparsemax over (N+1) weights.

    The shared encoder is permutation-equivariant: any stock with the same
    feature vector gets the same score, which is what we want for portfolio
    construction — the model must learn from features, not slot identity.
    A learned scalar `cash_logit` competes with stock scores.

    Set `use_sparsemax=True` for explicit zero weights on most stocks (v3.1
    sparsity path); softmax (default) gives a smooth distribution.
    """

    def __init__(self, num_stocks: int, feat_per_stock: int = 12, hidden: int = 64,
                 emb: int = 32, max_weight: float = 0.10, use_sparsemax: bool = False):
        super().__init__()
        self.N = num_stocks
        self.F = feat_per_stock
        self.max_weight = max_weight
        self.use_sparsemax = use_sparsemax

        self.shared = nn.Sequential(
            nn.Linear(feat_per_stock, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb),
            nn.ReLU(),
        )
        self.head = nn.Linear(emb, 1)
        self.cash_logit = nn.Parameter(torch.zeros(1))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """feats: (T, N, F) → weights (T, N+1) summing to 1.

        Stocks with mask=False get a very negative logit so the projection
        kills them (no allocation to delisted / pre-IPO stocks).
        """
        emb = self.shared(feats)
        scores = self.head(emb).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e6)
        T = feats.shape[0]
        cash = self.cash_logit.expand(T, 1)
        logits = torch.cat([scores, cash], dim=1)
        weights = sparsemax(logits, dim=1) if self.use_sparsemax else torch.softmax(logits, dim=1)
        if self.max_weight is not None and self.max_weight < 1.0:
            weights = self._cap_renorm(weights)
        return weights

    def _cap_renorm(self, w: torch.Tensor) -> torch.Tensor:
        """Cap stock weights at max_weight; excess goes to cash slot."""
        N = self.N
        stock = w[:, :N]
        cash = w[:, N:]
        capped = torch.clamp(stock, max=self.max_weight)
        excess = (stock - capped).sum(dim=1, keepdim=True)
        return torch.cat([capped, cash + excess], dim=1)


def realized_returns(weights: torch.Tensor, asset_rets: torch.Tensor,
                     tx_cost: float = 0.0042) -> torch.Tensor:
    """Sequential portfolio return series with L1 turnover cost.

    weights[t] is the target *after* observing t-1, applied to receive
    asset_rets[t] (i.e., we buy at close of t-1, hold through t).
    Transaction cost is C·Σ|w_t − w_{t-1}^drift|, where w_{t-1}^drift is
    last period's weight after asset drift over t-1 → t.
    """
    T = weights.shape[0]
    cash_zero = torch.zeros(T, 1, device=asset_rets.device, dtype=asset_rets.dtype)
    rets_full = torch.cat([asset_rets, cash_zero], dim=1)
    pnl = (weights * rets_full).sum(dim=1)
    if T < 2:
        return pnl
    drifted = weights[:-1] * (1.0 + rets_full[1:])
    drifted = drifted / (drifted.sum(dim=1, keepdim=True) + 1e-9)
    delta = (weights[1:] - drifted).abs().sum(dim=1)
    pnl = pnl.clone()
    pnl[1:] = pnl[1:] - tx_cost * delta
    return pnl


def sharpe_loss(pnl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Negative Sharpe ratio — minimise."""
    return -(pnl.mean() / (pnl.std() + eps))


def entropy_penalty(weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Average entropy of the weight distribution. Add to loss to push sparser.

    H(p) = -Σ p log p. EW over 47 ≈ log(47) ≈ 3.85. Concentrated 1-hot ≈ 0.
    """
    return -(weights * (weights + eps).log()).sum(dim=-1).mean()


def train_one_fold(
    train_data,
    val_data,
    feature_extractor,
    stock_ids,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden: int = 64,
    emb: int = 32,
    max_weight: float = 0.10,
    tx_cost: float = 0.0042,
    use_sparsemax: bool = False,
    entropy_lambda: float = 0.0,
    device: str = "cpu",
    log_every: int = 25,
    seed: int = 0,
) -> dict:
    """Fit on train_data, evaluate on val_data, return summary dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = build_tensors(train_data, feature_extractor, stock_ids, device=device)
    va = build_tensors(val_data, feature_extractor, stock_ids, device=device)

    net = PortfolioNet(
        num_stocks=len(stock_ids),
        feat_per_stock=tr["feats"].shape[-1],
        hidden=hidden,
        emb=emb,
        max_weight=max_weight,
        use_sparsemax=use_sparsemax,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    feats_x = tr["feats"][:-1]
    rets_y = tr["rets"][1:]
    mask_x = tr["mask"][:-1]

    best_sharpe = -math.inf
    history = []
    for epoch in range(epochs):
        net.train()
        weights = net(feats_x, mask_x)
        pnl = realized_returns(weights, rets_y, tx_cost=tx_cost)
        loss = sharpe_loss(pnl)
        if entropy_lambda > 0:
            loss = loss + entropy_lambda * entropy_penalty(weights)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            sharpe = -loss.item()
            mean_ret = pnl.mean().item()
            std_ret = pnl.std().item()
            best_sharpe = max(best_sharpe, sharpe)
            if (epoch + 1) % log_every == 0:
                print(
                    f"  epoch {epoch + 1:>4}  sharpe={sharpe:+.3f}  "
                    f"μ={mean_ret * 100:+.3f}%  σ={std_ret * 100:.3f}%"
                )
        history.append((epoch, sharpe))

    net.eval()
    with torch.no_grad():
        feats_v = va["feats"][:-1]
        rets_v = va["rets"][1:]
        mask_v = va["mask"][:-1]
        weights_v = net(feats_v, mask_v)
        pnl_v = realized_returns(weights_v, rets_v, tx_cost=tx_cost)
        cum = (1 + pnl_v).cumprod(0)
        ret = float(cum[-1].item() - 1)
        sharpe_v = float(pnl_v.mean() / (pnl_v.std() + 1e-9))
        avg_w = weights_v.mean(dim=0).cpu().numpy()
        active = (weights_v[:, : len(stock_ids)] > 0.005).sum(dim=1).float().mean().item()

    ew_ret = sum(
        (df["close"].iloc[-1] / df["close"].iloc[0] - 1)
        for df in val_data.values()
        if len(df) > 1
    ) / max(1, len(val_data))

    return {
        "return": ret,
        "ew": ew_ret,
        "alpha": ret - ew_ret,
        "val_sharpe": sharpe_v,
        "best_train_sharpe": best_sharpe,
        "active_stocks_avg": active,
        "cash_weight_avg": float(avg_w[-1]),
        "stock_weights_avg": [float(x) for x in avg_w[:-1]],
    }
