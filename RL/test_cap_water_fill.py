"""Standalone tests for cap_water_fill (no pytest). Run:
    CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill
"""
import numpy as np
import torch

from RL.dl_portfolio import cap_water_fill_np, cap_water_fill_torch, ensemble_topn_truncate


def _softmax_rows(shape, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=shape)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=0):
    """Feasible (n_stocks + cash) simplex with one stock above a 0.10 cap.
    n_stocks must be large enough that n_stocks*cap >> invested mass, so
    water-fill has headroom to absorb overflow without falling back to cash."""
    rng = np.random.default_rng(seed)
    rest = rng.dirichlet(np.ones(n_stocks - 1)) * (1.0 - cash - spike)
    stocks = np.concatenate([[spike], rest])
    return np.concatenate([stocks, [cash]])[None, :]  # (1, n_stocks+1)


def test_simplex_and_cap():
    w = _softmax_rows((7, 47), 0)            # 46 stocks + cash
    out = cap_water_fill_np(w, 0.10)
    n = w.shape[-1] - 1
    assert np.allclose(out.sum(-1), 1.0, atol=1e-9), out.sum(-1)
    assert (out[..., :n] <= 0.10 + 1e-9).all(), out[..., :n].max()
    # deliberate cash preserved or grown (never shrunk)
    assert (out[..., -1] + 1e-12 >= w[..., -1]).all()


def test_no_overflow_unchanged():
    # 10 stocks + cash, each 1/11 < cap -> nothing to redistribute
    w = np.full((1, 11), 1.0 / 11.0)
    out = cap_water_fill_np(w, 0.10)
    assert np.allclose(out, w, atol=1e-9), out


def test_overflow_goes_to_stocks_not_cash():
    # one stock at 0.30 over a 0.10 cap, 20 stocks -> ample headroom
    w = _concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=5)
    out = cap_water_fill_np(w, 0.10)
    assert (out[0, :20] <= 0.10 + 1e-9).all(), out[0, :20].max()
    assert abs(out[0, -1] - 0.10) < 1e-9, out[0, -1]   # deliberate cash preserved
    assert abs(out.sum() - 1.0) < 1e-9


def test_torch_numpy_parity():
    w = _softmax_rows((5, 47), 1)
    a = cap_water_fill_np(w, 0.10)
    b = cap_water_fill_torch(torch.from_numpy(w), 0.10).numpy()
    assert np.abs(a - b).max() < 1e-6, np.abs(a - b).max()


def test_torch_differentiable():
    w = torch.tensor(_softmax_rows((3, 47), 2), requires_grad=True)
    out = cap_water_fill_torch(w, 0.10)
    out.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_net_cap_renorm_branch():
    from RL.dl_portfolio import PortfolioNetLSTM
    # 20 stocks + cash; one stock at 0.30 (over cap), deliberate cash = 0.10
    w = torch.tensor(_concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=6))
    net_cash = PortfolioNetLSTM(num_stocks=20, feat_per_stock=14,
                                max_weight=0.10, cap_overflow="cash")
    net_wf = PortfolioNetLSTM(num_stocks=20, feat_per_stock=14,
                              max_weight=0.10, cap_overflow="waterfill")
    out_cash = net_cash._cap_renorm(w)
    out_wf = net_wf._cap_renorm(w)
    # legacy dumps the ~0.20 overflow to cash -> cash > 0.10
    assert out_cash[0, -1].item() > 0.10 + 1e-6, out_cash[0, -1].item()
    # water-fill keeps deliberate cash ~0.10
    assert abs(out_wf[0, -1].item() - 0.10) < 1e-6, out_wf[0, -1].item()
    assert (out_wf[0, :20] <= 0.10 + 1e-9).all()


def test_aggregate_weights_branch():
    from RL.walk_forward_dl_ensemble import aggregate_weights
    # 2 seeds, 1 timestep, 20 stocks + cash. Both spike stock 0 over cap.
    s0 = _concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=7)  # (1, 21)
    s1 = _concentrated(n_stocks=20, cash=0.10, spike=0.28, seed=8)
    cash_out = aggregate_weights([s0, s1], cap=0.10, cap_overflow="cash")
    wf_out = aggregate_weights([s0, s1], cap=0.10, cap_overflow="waterfill")
    assert (wf_out[:, :20] <= 0.10 + 1e-9).all()
    assert np.allclose(wf_out.sum(-1), 1.0)
    # water-fill parks less in cash than legacy (averaged spike ~0.29 > cap)
    assert wf_out[0, -1] < cash_out[0, -1] - 1e-6, (wf_out[0, -1], cash_out[0, -1])


def test_deploy_walkforward_parity():
    from RL.walk_forward_dl_ensemble import aggregate_weights
    rng = np.random.default_rng(3)
    seeds = [rng.dirichlet(np.ones(47))[None, :] for _ in range(5)]  # (1, N+1)
    scores = [0.21, 0.34, 0.29, 0.30, 0.25]
    wf = aggregate_weights(seeds, per_seed_scores=scores, cap=0.10,
                           cap_overflow="waterfill")[0]
    # deploy-side math: same sharpe-weight, clip, renorm, water-fill
    stacked = np.stack([s[0] for s in seeds])
    sc = np.asarray(scores)
    ws = np.exp(sc - sc.max())
    ws /= ws.sum()
    agg = (stacked * ws[:, None]).sum(0)
    agg = np.clip(agg, 0.0, None)
    agg /= agg.sum()
    deploy = cap_water_fill_np(agg, 0.10)
    assert np.abs(wf - deploy).max() < 1e-9, np.abs(wf - deploy).max()


def test_residual_to_cash():
    # 5 stocks each 0.19 (all > cap 0.10), cash 0.05. Max investable = 5*0.10 =
    # 0.50 < 0.95 stock mass, so 0.45 must spill to cash -> cash 0.50.
    w = np.array([[0.19, 0.19, 0.19, 0.19, 0.19, 0.05]])
    out = cap_water_fill_np(w, 0.10)
    assert (out[0, :5] <= 0.10 + 1e-9).all(), out[0, :5]
    assert abs(out.sum() - 1.0) < 1e-9
    assert abs(out[0, -1] - 0.50) < 1e-9, out[0, -1]
    b = cap_water_fill_torch(torch.from_numpy(w), 0.10).numpy()
    assert np.abs(out - b).max() < 1e-6, np.abs(out - b).max()


def test_top_k_concentration():
    rng = np.random.default_rng(11)
    # 30 stocks: one spike over cap + diffuse tail; cash 0.05
    tail = rng.dirichlet(np.ones(29)) * (1.0 - 0.05 - 0.40)
    w = np.concatenate([[0.40], tail, [0.05]])[None, :]  # (1, 31): 30 stocks + cash
    out_k = cap_water_fill_np(w, 0.10, top_k=5)
    out_all = cap_water_fill_np(w, 0.10, top_k=None)
    # cap + simplex invariants hold under top_k
    assert (out_k[0, :30] <= 0.10 + 1e-9).all(), out_k[0, :30].max()
    assert abs(out_k.sum() - 1.0) < 1e-9
    # at most top_k stocks gained weight; all-recipients spreads strictly wider
    gained_k = int(((out_k[0, :30] - w[0, :30]) > 1e-9).sum())
    gained_all = int(((out_all[0, :30] - w[0, :30]) > 1e-9).sum())
    assert gained_k <= 5, gained_k
    assert gained_all > gained_k, (gained_all, gained_k)
    # torch/numpy parity with top_k
    b = cap_water_fill_torch(torch.from_numpy(w), 0.10, top_k=5).numpy()
    assert np.abs(out_k - b).max() < 1e-6, np.abs(out_k - b).max()


def test_ensemble_topn_truncate():
    rng = np.random.default_rng(21)
    w = rng.dirichlet(np.ones(47), size=(6,))  # (6, 47): 46 stocks + cash, rows sum 1
    out = ensemble_topn_truncate(w, top_n=15)
    n = w.shape[-1] - 1
    assert np.allclose(out.sum(-1), 1.0, atol=1e-9)
    # at most 15 stocks nonzero per row
    assert (np.count_nonzero(out[:, :n], axis=1) <= 15).all(), np.count_nonzero(out[:, :n], axis=1)
    # the kept stocks are exactly the top-15 of the input
    for i in range(w.shape[0]):
        top15 = set(np.argsort(-w[i, :n])[:15])
        kept = set(np.nonzero(out[i, :n])[0])
        assert kept <= top15, (i, kept - top15)
    # None and >= n_stocks are no-ops
    assert np.array_equal(ensemble_topn_truncate(w, None), w)
    assert np.array_equal(ensemble_topn_truncate(w, 999), w)


def test_n90_cap_converges():
    # 90 stocks, concentrated softmax -> overflow must redistribute and the
    # cap must hold tightly with the default n_iters at this universe size.
    rng = np.random.default_rng(90)
    logits = rng.normal(size=(8, 91)) * 3.0   # 90 stocks + cash, peaky
    w = np.exp(logits - logits.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    out = cap_water_fill_np(w, 0.10)           # uses default n_iters
    assert (out[:, :90] <= 0.10 + 1e-9).all(), out[:, :90].max()
    assert np.allclose(out.sum(-1), 1.0, atol=1e-9)
    # residual driven to cash should be tiny when headroom is ample (90*0.10=9.0)
    leaked = out[:, -1].mean() - w[:, -1].mean()
    assert leaked < 0.05, leaked   # < 5% unintended cash from non-convergence


if __name__ == "__main__":
    test_simplex_and_cap()
    test_no_overflow_unchanged()
    test_overflow_goes_to_stocks_not_cash()
    test_torch_numpy_parity()
    test_torch_differentiable()
    test_net_cap_renorm_branch()
    test_aggregate_weights_branch()
    test_deploy_walkforward_parity()
    test_residual_to_cash()
    test_top_k_concentration()
    test_ensemble_topn_truncate()
    test_n90_cap_converges()
    print("PASS cap_water_fill")
