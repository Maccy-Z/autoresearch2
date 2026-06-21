import torch
import torch.nn as nn
import time
import math
import gc
import torch._logging
from torch.autograd import Function

from forward_relu2 import FFNSparseRelu2, FFNSparseRelu2_3, RELU2_SCALE


# Benchmark config: set to `2` or `3` for the inner FFN block depth.
FFN_BLOCK_LAYERS = 2
USE_COMPILED_DENSE = True
LAYERS = 12
BATCH_SIZE = 10000
DIM = 4096
EVAL_STEPS = 1
SPARSE_STEPS = 3


class FFNRelu2(Function):
    """Dense baseline autograd FFN with ReLU-squared activation.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = k * relu(x @ W1.T)^2`` and ``output = z @ W2.T``.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        """Run the dense ReLU-squared FFN forward pass."""
        preact = x @ W1.T
        r = preact.relu()
        z = r.square()
        z.mul_(RELU2_SCALE)
        ctx.save_for_backward(x, W1, W2, r)
        return z @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients using ``d k*relu(a)^2 / da = 2*k*relu(a)``."""
        x, W1, W2, r = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        grad_z = grad_output @ W2
        z = r.square()
        z.mul_(RELU2_SCALE)
        grad_W2 = grad_output.T @ z

        grad_preact = grad_z * (2.0 * RELU2_SCALE * r)
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()
        del r, z

        grad_x = None
        if needs_x:
            grad_x = grad_preact @ W1

        grad_W1 = grad_preact.T @ x
        return grad_x, grad_W1, grad_W2, None, None


class FFNRelu2_3(Function):
    """Dense baseline FFN with two hidden ReLU-squared layers."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T
        r1 = z1.relu()
        z1 = r1.square()
        z1.mul_(RELU2_SCALE)

        z2 = z1 @ W2.T
        r2 = z2.relu()
        z2 = r2.square()
        z2.mul_(RELU2_SCALE)

        ctx.save_for_backward(x, W1, W2, W3)
        return z2 @ W3.T

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        z1 = x @ W1.T
        r1 = z1.relu()
        z1 = r1.square()
        z1.mul_(RELU2_SCALE)

        z2 = z1 @ W2.T
        r2 = z2.relu()
        z2 = r2.square()
        z2.mul_(RELU2_SCALE)

        grad_z2 = grad_output @ W3
        grad_W3 = grad_output.T @ z2
        grad_preact2 = grad_z2 * (2.0 * RELU2_SCALE * r2)

        grad_W2 = grad_preact2.T @ z1
        grad_z1 = grad_preact2 @ W2
        grad_preact1 = grad_z1 * (2.0 * RELU2_SCALE * r1)

        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()
        del r1, r2, z2

        grad_x = grad_preact1 @ W1 if needs_x else None
        grad_W1 = grad_preact1.T @ x
        return grad_x, grad_W1, grad_W2, grad_W3


def generate_parameters(dim, G, dtype, expansion=5.25, device="cuda"):
    """Create FFN weights ``W1[H, dim]`` and ``W2[dim, H]`` with ``H=floor(dim*expansion)``."""
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)
    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)
    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01 * shift * W1.std()
    W2 = W2 - 0.01 * shift * W2.std()
    return W1, W2


def generate_parameters_3(dim, G, dtype, expansion=5.25, device="cuda"):
    """Create 3-layer FFN weights ``W1[H, D]``, ``W2[H, H]``, ``W3[D, H]``."""
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    W2 = torch.empty(hdim, hdim, device=device, dtype=dtype)
    W3 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)
    torch.nn.init.xavier_uniform_(W2, generator=G)
    torch.nn.init.xavier_uniform_(W3, generator=G)
    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01 * shift * W1.std()
    W2 = W2 + 0.01 * shift * W2.std()
    W3 = W3 - 0.01 * shift * W3.std()
    return W1, W2, W3


def ffn_relu2(x, W1, W2):
    """Plain 2-layer FFN: ``k*relu(x @ W1.T)^2 @ W2.T``."""
    preact = x @ W1.T
    preact = preact.relu()
    preact = preact.square()
    preact = preact * RELU2_SCALE
    return preact @ W2.T


def ffn_relu2_3(x, W1, W2, W3):
    """Plain 3-layer FFN: two ReLU-squared hidden layers."""
    z1 = x @ W1.T
    z1 = z1.relu()
    z1 = z1.square()
    z1 = z1 * RELU2_SCALE
    z2 = z1 @ W2.T
    z2 = z2.relu()
    z2 = z2.square()
    z2 = z2 * RELU2_SCALE
    return z2 @ W3.T


class DeepFFNRelu2(nn.Module):
    """Stack of residual FFN layers ``x <- x + FFN_relu2(x)`` for benchmarking."""
    def __init__(self, dtype, layers=12, hidm=4096):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.block_layers = FFN_BLOCK_LAYERS
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for _ in range(layers):
            if self.block_layers == 2:
                W1, W2 = generate_parameters(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
            else:
                W1, W2, W3 = generate_parameters_3(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
                self.W3s.append(nn.Parameter(W3))
        if self.block_layers == 3 and USE_COMPILED_DENSE:
            self.block_forward = torch.compile(ffn_relu2_3)
        elif self.block_layers == 2 and USE_COMPILED_DENSE:
            self.block_forward = torch.compile(ffn_relu2)
        else:
            self.block_forward = None

    def forward_base(self, x):
        """Run the dense ReLU-squared baseline on ``x[B, D]``."""
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
                if self.block_forward is None:
                    x = x + FFNRelu2.apply(x_inner, W1, W2)
                else:
                    x = x + self.block_forward(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + self.block_forward(x_inner, W1, W2, W3)
        return x

    def forward(self, x):
        """Run the sparse ReLU-squared FFN on ``x[B, D]``."""
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
                x = x + FFNSparseRelu2.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + FFNSparseRelu2_3.apply(x_inner, W1, W2, W3)
        return x


def run_step(x, model, sparse=False, steps=1):
    """Benchmark ``steps`` train iterations and return tracking stats, VRAM, and time."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")

    start = time.perf_counter()
    for _ in range(steps):
        model.zero_grad()
        if sparse:
            y = model.forward(x)
        else:
            y = model.forward_base(x)
        loss = (y - x).pow(2).mean()
        loss.backward()

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps

    tracking = [loss.detach().cpu()]
    for n, p in model.named_parameters():
        if p.grad is not None:
            tracking.append(p.grad.std().cpu())
    tracking = torch.stack(tracking) * 1e3
    return tracking, allocated, avg_time


def evaluate():
    """Compare dense and sparse ReLU-squared FFN training."""
    print(f"Using ReLU-squared normalisation k={RELU2_SCALE:.6f}")
    hdim = DIM
    bs = 2048 if FFN_BLOCK_LAYERS == 3 else BATCH_SIZE
    layers = LAYERS
    atol = 1e-2
    rtol = 1e-2
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    model = DeepFFNRelu2(layers=layers, hidm=hdim, dtype=dtype)

    run_step(x, model, sparse=False, steps=EVAL_STEPS)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=EVAL_STEPS)
    print(f"Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms")
    print("-" * 50)

    run_step(x, model, sparse=True, steps=EVAL_STEPS)
    tracking, vram, avg_time = run_step(x, model, sparse=True, steps=SPARSE_STEPS)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f"Total time: {avg_time:.2f} ms")

    if not torch.allclose(tracking, tracking_dn, atol=atol, rtol=rtol):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=atol, rtol=rtol)

    if FFN_BLOCK_LAYERS == 2:
        assert vram < vram_dn * 0.88


def run_base():
    """Configure PyTorch matmul/logging settings and run the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
