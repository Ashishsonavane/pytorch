"""CodSpeed benchmarks for PyTorch core tensor operations.

These benchmarks measure the performance of fundamental tensor operations
on CPU, covering creation, arithmetic, reductions, and linear algebra.
"""

import pytest
import torch


# --- Tensor Creation ---


@pytest.mark.benchmark
def test_tensor_creation_zeros():
    torch.zeros(1000, 1000)


@pytest.mark.benchmark
def test_tensor_creation_rand():
    torch.rand(1000, 1000)


@pytest.mark.benchmark
def test_tensor_creation_arange():
    torch.arange(0, 1_000_000, dtype=torch.float32)


# --- Element-wise Arithmetic ---


class TestElementwiseOps:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.a = torch.rand(1000, 1000)
        self.b = torch.rand(1000, 1000)

    @pytest.mark.benchmark
    def test_add(self):
        torch.add(self.a, self.b)

    @pytest.mark.benchmark
    def test_mul(self):
        torch.mul(self.a, self.b)

    @pytest.mark.benchmark
    def test_div(self):
        torch.div(self.a, self.b)

    @pytest.mark.benchmark
    def test_exp(self):
        torch.exp(self.a)

    @pytest.mark.benchmark
    def test_sqrt(self):
        torch.sqrt(self.a)


# --- Reduction Operations ---


class TestReductionOps:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = torch.rand(1000, 1000)

    @pytest.mark.benchmark
    def test_sum(self):
        torch.sum(self.x)

    @pytest.mark.benchmark
    def test_mean(self):
        torch.mean(self.x)

    @pytest.mark.benchmark
    def test_max(self):
        torch.max(self.x)

    @pytest.mark.benchmark
    def test_argmax(self):
        torch.argmax(self.x)

    @pytest.mark.benchmark
    def test_sum_dim(self):
        torch.sum(self.x, dim=1)


# --- Matrix Operations ---


class TestMatrixOps:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.a = torch.rand(256, 256)
        self.b = torch.rand(256, 256)

    @pytest.mark.benchmark
    def test_matmul(self):
        torch.matmul(self.a, self.b)

    @pytest.mark.benchmark
    def test_transpose(self):
        self.a.t()

    @pytest.mark.benchmark
    def test_reshape(self):
        self.a.reshape(512, 128)

    @pytest.mark.benchmark
    def test_cat(self):
        torch.cat([self.a, self.b], dim=0)

    @pytest.mark.benchmark
    def test_stack(self):
        torch.stack([self.a, self.b])
