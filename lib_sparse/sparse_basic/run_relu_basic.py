import torch
import torch._logging

from forward_methods import FFNSparse, FFNSparse3
from shared.experiment import run_step, DeepFFN_abc

# Benchmark config: set to `2` or `3` for the inner FFN block depth.
FFN_BLOCK_LAYERS = 3
LAYERS = 3
BATCH_SIZE = 10000
DIM = 4096

class DeepFFN(DeepFFN_abc):
    """Stack of residual FFN layers ``x <- x + FFN(x)`` for benchmarking."""
    def __init__(self, dtype):
        super().__init__(dtype, LAYERS, DIM, FFN_BLOCK_LAYERS)

    # @torch.compile
    def forward(self, x, _=None):
        """Run the sparse-activation FFN on ``x[B, D]`` through all residual layers."""
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
                x = x + FFNSparse.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + FFNSparse3.apply(x_inner, W1, W2, W3)
        return x


def evaluate():
    """Compare dense and sparse FFN training for correctness, memory, and speed."""
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(BATCH_SIZE, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    model = DeepFFN(dtype=dtype)

    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f"Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms")
    print("-" * 50)

    run_step(x, model, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, sparse=True, steps=1)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f"Total time: {avg_time:.2f} ms")

    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):
        print("Predicted values are different.")

        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)

        assert vram < vram_dn * 0.88
    print(f"{tracking_dn = }")
    print(f"{tracking = }")


def run_base():
    """Configure PyTorch matmul/logging settings and run the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
