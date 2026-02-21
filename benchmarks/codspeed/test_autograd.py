"""CodSpeed benchmarks for PyTorch autograd operations.

These benchmarks measure the performance of automatic differentiation,
covering forward pass, backward pass, and gradient computation.
"""

import pytest
import torch


# --- Forward + Backward ---


def _run_forward_backward(a, b):
    c = torch.matmul(a, b)
    loss = c.sum()
    loss.backward()


@pytest.mark.benchmark
def test_autograd_matmul_backward():
    a = torch.rand(128, 128, requires_grad=True)
    b = torch.rand(128, 128, requires_grad=True)
    _run_forward_backward(a, b)


@pytest.mark.benchmark
def test_autograd_chain_ops():
    x = torch.rand(256, 256, requires_grad=True)
    y = torch.relu(x)
    y = torch.sigmoid(y)
    y = y.sum()
    y.backward()


# --- Gradient Computation ---


@pytest.mark.benchmark
def test_autograd_grad():
    x = torch.rand(512, 512, requires_grad=True)
    y = (x**2).sum()
    torch.autograd.grad(y, x)


# --- No-grad Context ---


@pytest.mark.benchmark
def test_no_grad_matmul():
    a = torch.rand(256, 256)
    b = torch.rand(256, 256)
    with torch.no_grad():
        torch.matmul(a, b)
