import torch
import torch.nn as nn
import time
import math
import gc
import torch._logging
from torch.autograd import Function

from forward_methods import FFNSparse, FFNSparse3


# Benchmark config: set to `2` or `3` for the inner FFN block depth.
FFN_BLOCK_LAYERS = 3
LAYERS = 4
BATCH_SIZE = 10000
DIM = 4096

def print_memory(msg):
    memory = torch.cuda.memory_allocated("cuda")/1024**2
    print(f'{msg}: {memory:.2f} MB')

class FFNv3(Function):
    """Dense baseline autograd FFN for comparison.

    For ``x[B, D]``, ``W1[H, D]``, and ``W2[D, H]`` computes
    ``z = relu(x @ W1.T)`` and ``output = z @ W2.T``.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        """Run the dense FFN forward pass and save tensors for backward."""
        z = x @ W1.T
        z.relu_()
        output = z @ W2.T
        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute dense FFN gradients from ``grad_output[B, D]``."""
        x, W1, W2, z = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        grad_z = grad_output @ W2
        grad_W2 = grad_output.T @ z

        grad_preact = torch.ops.aten.threshold_backward.grad_input(
            grad_z, z, 0, grad_input=grad_z
        )
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()
        del z

        grad_x = None
        if needs_x:
            grad_x = grad_preact @ W1

        grad_W1 = grad_preact.T @ x
        return grad_x, grad_W1, grad_W2, None, None


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


class FFNv4(Function):
    """Dense 3-layer FFN — normal backward (saved intermediates, no checkpointing)."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T          # pre-activation 1  [B, H]
        z1.relu_()              # in-place ReLU → z1 is now post-activation
        z2 = z1 @ W2.T          # pre-activation 2  [B, H]
        z2.relu_()              # in-place ReLU → z2 is now post-activation
        output = z2 @ W3.T      # output           [B, D]
        ctx.save_for_backward(x, z1, z2, W1, W2, W3)
        # print(f'{z1.nbytes / 1024**2 :.2f}, {z2.nbytes / 1024**2 :.2f}')
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, a1, a2, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]
        # a1, a2 are already post-activation (relu_ was in-place)

        # layer 3: linear
        grad_a2 = grad_output @ W3          # [B, H]
        grad_W3 = grad_output.T @ a2        # [D, H]
        # relu2 backward — fused CUDA kernel, faster than torch.where
        grad_z2 = torch.ops.aten.threshold_backward.grad_input(
            grad_a2, a2, 0, grad_input=grad_a2
        )

        # layer 2: linear
        grad_a1 = grad_z2 @ W2              # [B, H]
        grad_W2 = grad_z2.T @ a1            # [H, H]

        del grad_z2, grad_a2
        # relu1 backward
        grad_z1 = torch.ops.aten.threshold_backward.grad_input(
            grad_a1, a1, 0, grad_input=grad_a1
        )
        del grad_a1, a1, a2
        ctx.maybe_clear_saved_tensors()

        # layer 1: linear
        grad_x = grad_z1 @ W1 if needs_x else None

        grad_W1 = grad_z1.T @ x             # [H, D]
        del grad_z1

        return grad_x, grad_W1, grad_W2, grad_W3


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
            self.block_forward = FFNv4.apply
        elif self.block_layers == 2:
            self.block_forward = FFNv3.apply
        else:
            raise NotImplementedError
        # Simulate hook-based efficient optimiser, that applies gradient update as soon as possible and clears grads.
        self.setup_hooks()

    @staticmethod
    def hook(w):
        w.grad = None
        return

    def setup_hooks(self):
        for n, p in self.named_parameters():
            p.register_post_accumulate_grad_hook(self.hook)

    # @torch.compile
    def forward_base(self, x):
        """Run the dense baseline on ``x[B, D]`` through all residual layers."""
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
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
                x_inner = x
                x = x + FFNSparse.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = FFNSparse3.apply(x_inner, W1, W2, W3)
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
    tracking = torch.stack(tracking) * 1e3
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

    # run_step(x, model, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, sparse=True, steps=1)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f"Total time: {avg_time:.2f} ms")

    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)

        assert vram < vram_dn * 0.88


def run_base():
    """Configure PyTorch matmul/logging settings and run the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
