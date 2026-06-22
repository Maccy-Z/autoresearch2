import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import gc
import torch._logging
from torch.autograd import Function

from forward_methods import ValueBuffer
from forward_relu2 import FFNSparseRelu2, FFNSparseRelu2_3


RELU2_SCALE = 3 ** -0.5


class FFNRelu2(Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        z = x @ W1.T
        r = z.relu_()
        z = r.square()
        z.mul_(RELU2_SCALE)
        ctx.save_for_backward(x, W1, W2, r)
        return z @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, r = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        z = r.square().mul_(RELU2_SCALE)
        grad_W2 = grad_output.T @ z
        del z
        grad_z = grad_output @ W2
        grad_preact = grad_z * (2.0 * RELU2_SCALE * r)
        del grad_z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_x = grad_preact @ W1 if needs_x else None
        grad_W1 = grad_preact.T @ x
        return grad_x, grad_W1, grad_W2


class FFNRelu2_3(Function):
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T
        r1 = z1.relu_()
        z1 = r1.square()
        z1.mul_(RELU2_SCALE)

        z2 = z1 @ W2.T
        r2 = z2.relu_()
        z2 = r2.square()
        z2.mul_(RELU2_SCALE)

        ctx.save_for_backward(x, r1, r2, W1, W2, W3)
        return z2 @ W3.T

    @staticmethod
    def backward(ctx, grad_output):
        x, r1, r2, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        grad_a2 = grad_output @ W3
        grad_W3 = grad_output.T @ (r2.square().mul_(RELU2_SCALE))
        grad_z2 = grad_a2 * (2.0 * RELU2_SCALE * r2)

        grad_a1 = grad_z2 @ W2
        grad_W2 = grad_z2.T @ (r1.square().mul_(RELU2_SCALE))
        del grad_z2, grad_a2

        grad_z1 = grad_a1 * (2.0 * RELU2_SCALE * r1)
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_x = grad_z1 @ W1 if needs_x else None
        grad_W1 = grad_z1.T @ x
        return grad_x, grad_W1, grad_W2, grad_W3


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


def generate_parameters_3(dim, G, dtype, expansion=5.25, device="cuda"):
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


class DeepFFN(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096, block_layers=2):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.block_layers = block_layers
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
            self.block_forward = FFNRelu2.apply
        else:
            raise NotImplementedError
        self.setup_hooks()

    @staticmethod
    def hook(w):
        w.grad = None
        return

    def setup_hooks(self):
        for _, p in self.named_parameters():
            p.register_post_accumulate_grad_hook(self.hook)

    def forward_base(self, x):
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + self.block_forward(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = self.block_forward(x_inner, W1, W2, W3)
        return x

    def forward(self, x, buffer):
        buffer.ready_buffer()
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNSparseRelu2.apply(x_inner, W1, W2, buffer)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = FFNSparseRelu2_3.apply(x_inner, W1, W2, W3, buffer)
        return x


def run_step(x, model, buffer: ValueBuffer = None, sparse=False, steps=1):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")

    if buffer is not None:
        buffer.init_buffer()
    start = time.perf_counter()

    for _ in range(steps):
        torch.cuda.reset_peak_memory_stats("cuda")
        model.zero_grad()
        if sparse:
            y = model.forward(x, buffer)
        else:
            y = model.forward_base(x)
        loss = (y - x).pow(2).mean()
        loss.backward()

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps
    tracking = [loss.detach().cpu()]
    for _, p in model.named_parameters():
        if p.grad is not None:
            tracking.append(p.grad.std().cpu())
    tracking = torch.stack(tracking) * 1e3
    return tracking, allocated, avg_time


def make_sparse_buffer(bs, hdim, layers, block_layers):
    factor = 0.55 * (2 if block_layers == 3 else 1)
    return ValueBuffer(int(bs * math.floor(hdim * 5.25) * layers * factor), dtype=torch.bfloat16, device="cuda")


def evaluate():
    hdim = 4096
    bs = 10_000
    layers = 3
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    model = DeepFFN(layers=layers, hidm=hdim, dtype=dtype, block_layers=2)
    model._sparse_data = make_sparse_buffer(bs, hdim, layers, 3)

    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f'Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms')
    print("-" * 50)

    # run_step(x, model, model._sparse_data, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, model._sparse_data, sparse=True, steps=1)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f'Total time: {avg_time:.2f} ms')

    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):
        print(f'Predicted values are different.')
        print(f'{tracking_dn = }')
        print(f'{tracking = }')
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)

    assert vram < vram_dn * 1.1


def run_base():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
