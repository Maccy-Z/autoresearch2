import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc

from layer_versions import FFNv3, FFNSparse


def generate_parameters(dim, G, dtype, expansion=4, device="cuda"):
    hdim = dim * expansion
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    W1 = W1 + 0.01*W1.std()
    W2 = W2 #+ W2.std()
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

    @torch.compile()
    def forward(self, x):
        """ x.shape = [BS, dim] """
        for W1, W2 in zip(self.W1s, self.W2s):
            x_inner = F.layer_norm(x, x.shape[-1:])
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
    return loss.cpu(), grad_stds, allocated, avg_time



def evaluate(x, dtype):
    # Dense exact solution
    model = DeepFFN(FFNv3, dtype=dtype)
    loss_dn, grad_stds_dn, vram_dn, _ = run_step(x, model, steps=1)

    print(f'{vram_dn = :.2f} MB')

    model = DeepFFN(FFNSparse, dtype=dtype)
    # Warmup
    run_step(x, model, steps=5)

    # Main run
    loss, grad_stds, vram, avg_time = run_step(x, model, steps=5)

    # Make sure we are close
    torch.testing.assert_close(loss_dn, loss)
    torch.testing.assert_close(grad_stds_dn, grad_stds)

    print(f'Loss = {loss.detach().item()}')
    print(f'{grad_stds = }')
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f'{avg_time = :.2f} ms')


def run_base():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    hdim = 4096
    dtype = torch.bfloat16

    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(10_000, hdim, dtype=dtype, device="cuda", generator=G)

    evaluate(x, dtype=dtype)


if __name__ == "__main__":
    run_base()
