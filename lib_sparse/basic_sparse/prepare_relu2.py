import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._logging

from forward_relu2 import FFNSparseRelu2, FFNSparseRelu2_3
from shared.experiment import FFNRelu2_2, FFNRelu2_3, gen_params, gen_params_3, run_step

# Benchmark config: set to `2` or `3` for the inner FFN block depth.
FFN_BLOCK_LAYERS = 3
LAYERS = 2
BATCH_SIZE = 10000
DIM = 4096


class DeepFFN(nn.Module):
    """Stack of residual FFN layers ``x <- x + FFN(x)`` for benchmarking."""
    def __init__(self, dtype, layers=12, hidm=4096):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.block_layers = FFN_BLOCK_LAYERS
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for _ in range(layers):
            if self.block_layers == 2:
                W1, W2 = gen_params(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
            else:
                W1, W2, W3 = gen_params_3(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
                self.W3s.append(nn.Parameter(W3))
        if self.block_layers == 3:
            self.block_forward = FFNRelu2_3.apply
        elif self.block_layers == 2:
            self.block_forward = FFNRelu2_2.apply
        else:
            raise NotImplementedError
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
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + self.block_forward(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = self.block_forward(x_inner, W1, W2, W3)
        return x

    def forward(self, x, _=None):
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

    tracking = tracking * 1e3
    tracking_dn = tracking_dn * 1e3
    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-3):
        print("Predicted values are different.")
    print(f"{tracking_dn = }")
    print(f"{tracking = }")
    torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-3)

    assert vram < vram_dn * 0.88


def run_base():
    """Configure PyTorch matmul/logging settings and run the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(linewidth=1000)
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
