"""Train the v4 LSTM portfolio network on all available data → deploy artifact.

Reads RL/data/*.csv, fits with the same recent-500-day single-pass Sharpe-loss
recipe that won the walk-forward (v4_w50h32: mean alpha +4.09%), saves a
self-contained checkpoint that deploy_rl.py can load without any RL env.

Saved fields:
  state_dict       — PortfolioNetLSTM weights
  config           — {num_stocks, feat_per_stock, window_len, hidden, max_weight,
                      use_sparsemax}
  stock_ids        — fixed order used at training time; deploy must align
  feat_cols        — feature column names in the order the encoder consumes
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from RL.constant import stock_ids
from RL.dl_portfolio import (
    PortfolioNetLSTM,
    build_tensors,
    realized_returns,
    sharpe_loss,
    windowize,
)
from RL.feature import FeatureExtractor


def load_data(start: str, end: str) -> dict:
    out = {}
    for sid in stock_ids:
        fp = f"RL/data/{sid}.csv"
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[start:end].dropna()
        if len(df) > 30:
            out[sid] = df
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default=None,
                        help="default = today")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--train-recent", type=int, default=500)
    parser.add_argument("--out", default="models/dl_v4_deploy.pt")
    parser.add_argument("--seed", type=int, default=0,
                        help="single seed (used when --seeds omitted)")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seed list, e.g. '0,1,2,3,4'. "
                             "When set, trains one model per seed and saves "
                             "as models/dl_v4_seed{N}.pt (overrides --out).")
    args = parser.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y%m%d")
    print(f"window: {args.start}..{end}  recent: {args.train_recent}d  "
          f"window_len: {args.window}  hidden: {args.hidden}")

    data = load_data(args.start, end)
    print(f"loaded {len(data)}/{len(stock_ids)} stocks")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    fe = FeatureExtractor(stock_ids)
    tr = build_tensors(data, fe, stock_ids, device=device)
    feat_per_stock = tr["feats"].shape[-1]
    feats_x, rets_y, mask_x = windowize(
        tr["feats"], tr["rets"], tr["mask"], args.window
    )
    if args.train_recent and feats_x.shape[0] > args.train_recent:
        feats_x = feats_x[-args.train_recent:]
        rets_y = rets_y[-args.train_recent:]
        mask_x = mask_x[-args.train_recent:]
    print(f"train tensor: {feats_x.shape}")

    seed_list = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    )

    for seed in seed_list:
        print(f"\n=== training seed {seed} ===")
        torch.manual_seed(seed)
        net = PortfolioNetLSTM(
            num_stocks=len(stock_ids),
            feat_per_stock=feat_per_stock,
            window_len=args.window,
            hidden=args.hidden,
            max_weight=args.max_weight,
            use_sparsemax=False,
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

        best = -1e9
        for epoch in range(args.epochs):
            net.train()
            weights = net(feats_x, mask_x)
            pnl = realized_returns(weights, rets_y, tx_cost=0.0042)
            loss = sharpe_loss(pnl)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sharpe = -loss.item()
            best = max(best, sharpe)
            if (epoch + 1) % 50 == 0:
                print(f"  epoch {epoch + 1:>4}  sharpe={sharpe:+.3f}  "
                      f"best={best:+.3f}  μ={pnl.mean().item() * 100:+.3f}%")

        out_path = (
            f"models/dl_v4_seed{seed}.pt" if args.seeds else args.out
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": net.state_dict(),
            "config": {
                "num_stocks": len(stock_ids),
                "feat_per_stock": feat_per_stock,
                "window_len": args.window,
                "hidden": args.hidden,
                "max_weight": args.max_weight,
                "use_sparsemax": False,
            },
            "stock_ids": stock_ids,
            "feat_cols": tr["feat_cols"],
            "trained_through": end,
            "seed": seed,
        }, out_path)
        print(f"saved {out_path}  (best train sharpe {best:+.3f})")


if __name__ == "__main__":
    main()
