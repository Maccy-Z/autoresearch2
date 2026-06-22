import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import gc
import torch._logging
from torch.autograd import Function

from forward_relu2 import FFNSparseRelu2, FFNSparseRelu2_3, RELU2_SCALE


# Benchmark config: set to `2` or `3` for the inner FFN block depth.
FFN_BLOCK_LAYERS = 2
LAYERS = 3
BATCH_SIZE = 10000
DIM = 4096

def print_memory(msg):
    memory = torch.cuda.memory_allocated("cuda")/1024**2
    print(f'{msg}: {memory:.2f} MB')


class FFNRelu2_2(Function):
    """Dense 2-layer FFN with ReLU-squared activation.

    For ``x[B, D]``, ``W1[H, D]``, ``W2[D, H]`` computes
    ``z = k * relu(x @ W1.T)^2`` and ``output = z @ W2.T``.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        preact = x @ W1.T
        r = preact.relu_()         # save the relu output r
        z = r.square()
        z.mul_(RELU2_SCALE)        # z = k * r^2
        ctx.save_for_backward(x, W1, W2, r)
        return z @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, r = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        grad_z = grad_output @ W2
        z = r.square().mul_(RELU2_SCALE)
        grad_W2 = grad_output.T @ z

        grad_preact = grad_z * (2.0 * RELU2_SCALE * r)
        del grad_z
        grad_W1 = grad_preact.T @ x

        del r, z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_x = None
        if needs_x:
            grad_x = grad_preact @ W1

        return grad_x, grad_W1, grad_W2, None, None


class FFNRelu2_3(Function):
    """Dense 3-layer FFN with ReLU-squared activations — in-place forward, saved relu masks."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T              # pre-activation 1  [B, H]
        r1 = z1.relu_()            # r1 = relu(z1), in-place
        z1 = r1.square()
        z1.mul_(RELU2_SCALE)       # z1 = k * r1^2

        z2 = z1 @ W2.T             # pre-activation 2  [B, H]
        r2 = z2.relu_()            # r2 = relu(z2), in-place
        z2 = r2.square()
        z2.mul_(RELU2_SCALE)       # z2 = k * r2^2

        ctx.save_for_backward(x, r1, r2, W1, W2, W3)
        return z2 @ W3.T            # output [B, D]

    @staticmethod
    def backward(ctx, grad_output):
        x, r1, r2, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        # layer 3: linear
        grad_a2 = grad_output @ W3                                # [B, H]
        grad_W3 = grad_output.T @ (r2.square().mul_(RELU2_SCALE)) # [D, H]
        # relu2 backward: d/d(r2) of k*r2^2 = 2*k*r2
        grad_z2 = grad_a2 * (2.0 * RELU2_SCALE * r2)

        # layer 2: linear
        grad_a1 = grad_z2 @ W2                                    # [B, H]
        grad_W2 = grad_z2.T @ (r1.square().mul_(RELU2_SCALE))    # [H, H]
        del grad_z2, grad_a2

        # relu1 backward: d/d(r1) of k*r1^2 = 2*k*r1
        grad_z1 = grad_a1 * (2.0 * RELU2_SCALE * r1)
        del grad_a1, r1, r2
        ctx.maybe_clear_saved_tensors()

        # layer 1: linear
        grad_x = grad_z1 @ W1 if needs_x else None
        grad_W1 = grad_z1.T @ x                                   # [H, D]
        del grad_z1

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
    print(f'{W1.nbytes/1024**2 = }, {W2.nbytes/1024**2 = }, {W3.nbytes/1024**2 = }')
    return W1, W2, W3


class DeepFFN(nn.Module):
    """Stack of residual FFN layers ``x <- x + FFN(x)`` for benchmarking."""
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
        if self.block_layers == 3:
            self.block_forward = FFNRelu2_3.apply
        elif self.block_layers == 2:
            self.block_forward = FFNRelu2_2.apply # ffn_relu2#  #
        else:
            raise NotImplementedError
    #     self.setup_hooks()
    #
    # @staticmethod
    # def hook(w):
    #     w.grad = None
    #     return
    #
    # def setup_hooks(self):
    #     for n, p in self.named_parameters():
    #         p.register_post_accumulate_grad_hook(self.hook)

    # @torch.compile
    def forward_base(self, x):
        """Run the dense baseline on ``x[B, D]`` through all residual layers."""
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + self.block_forward(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = self.block_forward(x_inner, W1, W2, W3)
        return x

    def forward(self, x):
        """Run the sparse-activation FFN on ``x[B, D]`` through all residual layers."""
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNSparseRelu2.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = FFNSparseRelu2_3.apply(x_inner, W1, W2, W3)
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
        loss = y.sum()
        loss.backward()

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps

    tracking = [loss.detach().cpu()]
    for n, p in model.named_parameters():
        if p.grad is not None:
            tracking.append(p.grad.std().cpu())
    tracking = torch.stack(tracking) #* 1e3
    return tracking, allocated, avg_time


def evaluate():
    """Compare dense and sparse FFN training for correctness, memory, and speed."""
    hdim = DIM
    bs = BATCH_SIZE
    layers = LAYERS
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    model = DeepFFN(layers=layers, hidm=hdim, dtype=dtype)

    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f"Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms")
    print("-" * 50)

    run_step(x, model, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, sparse=True, steps=1)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f"Total time: {avg_time:.2f} ms")
    print()

    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-3):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-3)

        assert vram < vram_dn * 0.88


def run_base():
    """Configure PyTorch matmul/logging settings and run the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(precision=8)
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
