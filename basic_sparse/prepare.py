import torch
import torch.nn as nn
import time
import math
import gc
import torch._logging
from torch.autograd import Function

from forward_methods import FFNSparse


class FFNv3(Function):
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        z = x @ W1.T
        z.relu_()
        output = z @ W2.T
        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
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
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)
    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)
    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01 * shift * W1.std()
    W2 = W2 - 0.01 * shift * W2.std()
    return W1, W2


class DeepFFN(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            W1, W2 = generate_parameters(hidm, G, dtype=dtype)
            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))
        self.setup_hooks()

    @staticmethod
    def hook(w):
        w.grad = None

    def setup_hooks(self):
        for n, p in self.named_parameters():
            p.register_post_accumulate_grad_hook(self.hook)

    def forward_base(self, x):
        for W1, W2 in zip(self.W1s, self.W2s):
            x_inner = x
            x = x + FFNv3.apply(x_inner, W1, W2)
        return x

    @torch.compile
    def forward(self, x):
        for W1, W2 in zip(self.W1s, self.W2s):
            x_inner = x
            x = x + FFNSparse.apply(x_inner, W1, W2)
        return x


def run_step(x, model, sparse=False, steps=1):
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
    hdim = 4096
    bs = 10000
    layers = 12
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    model = DeepFFN(layers=layers, hidm=hdim, dtype=dtype)

    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f"Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms")
    print("-" * 50)

    run_step(x, model, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, sparse=True, steps=3)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f"Total time: {avg_time:.2f} ms")

    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):
        print("Predicted values are different.")
        print(f"{tracking_dn = }")
        print(f"{tracking = }")
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)

    assert vram < vram_dn * 0.88


def run_base():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
