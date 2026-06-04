"""Standalone tests for cap_water_fill (no pytest). Run:
    CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill
"""
import numpy as np
import torch

from RL.dl_portfolio import cap_water_fill_np, cap_water_fill_torch


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


if __name__ == "__main__":
    test_simplex_and_cap()
    test_no_overflow_unchanged()
    test_overflow_goes_to_stocks_not_cash()
    test_torch_numpy_parity()
    test_torch_differentiable()
    print("PASS cap_water_fill")
