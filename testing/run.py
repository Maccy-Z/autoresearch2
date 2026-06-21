import torch
import math
import time
from torch.autograd import Function


def generate_parameters_3(dim, G, dtype, expansion=5.25, device="cuda"):
    """Create 3-layer FFN weights ``W1[H, D]``, ``W2[H, H]``, ``W3[D, H]``."""
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, requires_grad=True, dtype=dtype)
    W2 = torch.empty(hdim, hdim, device=device, requires_grad=True, dtype=dtype)
    W3 = torch.empty(dim, hdim, device=device, requires_grad=True, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)
    torch.nn.init.xavier_uniform_(W2, generator=G)
    torch.nn.init.xavier_uniform_(W3, generator=G)
    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = (W1 + 0.01 * shift * W1.std()).detach().requires_grad_(True)
    W2 = (W2 + 0.01 * shift * W2.std()).detach().requires_grad_(True)
    W3 = (W3 - 0.01 * shift * W3.std()).detach().requires_grad_(True)
    return W1, W2, W3


def ffn_relu3(x, W1, W2, W3):
    """Plain 3-layer FFN: ``relu(relu(x @ W1.T) @ W2.T) @ W3.T``."""
    z1 = x @ W1.T
    z1.relu_()
    z2 = z1 @ W2.T
    z2.relu_()
    return z2 @ W3.T


class FFN_manual(Function):
    """Dense 3-layer FFN — normal backward (saved intermediates, no checkpointing)."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T          # pre-activation 1  [B, H]
        z1.relu_()              # in-place ReLU → z1 is now post-activation
        z2 = z1 @ W2.T          # pre-activation 2  [B, H]
        z2.relu_()              # in-place ReLU → z2 is now post-activation
        output = z2 @ W3.T      # output           [B, D]
        ctx.save_for_backward(x, z1, z2, W1, W2, W3)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, a1, a2, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]
        torch.cuda.empty_cache()
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
        return grad_x, grad_W1, grad_W2, grad_W3


def profile_both(dim=4096, batch=256, dtype=torch.bfloat16, device="cuda",
                 warmup_iters=3, bench_iters=10):
    """Profile ``FFN_manual`` and ``ffn_relu3`` for time and peak VRAM.

    Returns
    -------
    dict
        Keys: ``"manual"``, ``"plain"`` — each maps to a dict with
        ``"forward_time_ms"``, ``"backward_time_ms"``, ``"peak_vram_gb"``.
    """
    G = torch.Generator(device=device).manual_seed(42)
    W1, W2, W3 = generate_parameters_3(dim, G, dtype, device=device)
    print(f'{W1.shape = }, {W2.shape = }, {W3.shape = }')

    def zero_grads():
        W1.grad = None
        W2.grad = None
        W3.grad = None
        x.grad = None

    def warmup(fn, *args):
        for _ in range(warmup_iters):
            zero_grads()
            out = fn(*args)
            loss = out.sum()
            loss.backward()
            del out, loss

    results = {}
    for label, fn in [("manual", FFN_manual.apply), ("plain", ffn_relu3)]:
        # ---- warmup ----
        x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
        torch.cuda.reset_peak_memory_stats(device)   # reset BEFORE warmup so peak includes it
        warmup(fn, x, W1, W2, W3)

        # ---- benchmark ----
        x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)

        # time forward only (averaged) — no autograd needed
        torch.cuda.synchronize(device)
        fwd_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                out = fn(x, W1, W2, W3)
        torch.cuda.synchronize(device)
        fwd_ms = (time.perf_counter() - fwd_start) * 1000 / bench_iters

        # time forward+backward together (averaged), subtract fwd to get bwd
        torch.cuda.synchronize(device)
        total_start = time.perf_counter()
        for _ in range(bench_iters):
            zero_grads()
            out = fn(x, W1, W2, W3)
            loss = out.sum()
            loss.backward()
            del out, loss
        torch.cuda.synchronize(device)
        total_ms = (time.perf_counter() - total_start) * 1000 / bench_iters
        bwd_ms = total_ms - fwd_ms

        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        results[label] = {
            "forward_time_ms": fwd_ms,
            "backward_time_ms": bwd_ms,
            "peak_vram_mb": peak,
        }
    return results


def run_model():
    """Run the profiling comparison and print results."""
    torch.set_float32_matmul_precision("high")
    results = profile_both()
    for label, r in results.items():
        print(f"--- {label} ---")
        print(f"  forward  : {r['forward_time_ms']:.3f} ms")
        print(f"  backward : {r['backward_time_ms']:.3f} ms")
        print(f"  peak VRAM: {r['peak_vram_mb']:.3f} MB")
        print()


if __name__ == "__main__":
    run_model()
