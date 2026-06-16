import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import gc
import torch._logging
import logging

from prepare_layer import FFNv3, FFNv2, FFNv1, FFNckpt
from sparse import FFNSparse


def generate_parameters(dim, G, dtype, expansion=5.25, device="cuda"):
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01*shift*W1.std()
    W2 = W2 - 0.01*shift*W2.std()
    return W1, W2


class DeepFFN(nn.Module):
    def __init__(self, layer, dtype, layers=12, hidm=4096):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.layer = layer

        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            W1, W2 = generate_parameters(hidm, G, dtype=dtype)

            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))

    @torch.compile(dynamic=True)
    def forward(self, x):
        """ x.shape = [BS, dim] """
        for W1, W2 in zip(self.W1s, self.W2s):
            x_inner = F.rms_norm(x, x.shape[-1:])
            x = x + self.layer.apply(x_inner, W1, W2)
        return x


def run_step(x, model, steps=1):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(steps):
        model.zero_grad()
        y = model(x)
        loss = (y - x).pow(2).mean()
        # VRAM usage
        allocated = torch.cuda.memory_allocated("cuda") / 1024**2
        loss.backward()

    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps

    # Track gradient to ensure correctness
    grad_stds = []
    for i, p in enumerate(model.parameters()):
        if i > 5:
            break
        grad_stds.append(p.grad.std().cpu())
    grad_stds = torch.stack(grad_stds) * 1e3
    return loss.cpu().detach(), grad_stds.cpu().detach(), allocated, avg_time


def evaluate(x, dtype):
    # Dense exact solution
    model = DeepFFN(FFNSparse, dtype=dtype)
    loss_dn, grad_stds_dn, vram_dn, _ = run_step(x, model, steps=1)
    del model
    print(f'{vram_dn = :.2f} MB')

    # # Our model
    # model = DeepFFN(FFNSparse, dtype=dtype)
    # # Warmup
    # run_step(x, model, steps=2)
    #
    # # Main run
    # loss, grad_stds, vram, avg_time = run_step(x, model, steps=5)
    #
    # # Make sure we are close
    # torch.testing.assert_close(loss_dn, loss)
    # torch.testing.assert_close(grad_stds_dn, grad_stds)
    # # make sure vram has been reduced
    # # assert vram < vram_dn*0.9, f"VRAM usage not reduced enough: {vram:.2f} MB >= {vram_dn:.2f} MB"
    #
    # print(f"VRAM allocated by tensors: {vram:.2f} MB")
    # print(f'{avg_time = :.2f} ms')


def run_base():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    # import torch


    # torch._logging.set_logs(
    #     dynamo=logging.INFO,
    #     dynamic=logging.INFO,
    #     graph_breaks=True,
    #     recompiles=True,
    # )

    torch._functorch.config.activation_memory_budget = 0.5

    hdim = 4096
    bs = 10_000
    dtype = torch.bfloat16

    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G)

    evaluate(x, dtype=dtype)


if __name__ == "__main__":
    run_base()
