import torch
import torch._logging
import math

from forward_methods import FFNSparse, FFNSparse3, TensorBuffer, FFNSparseCustomOp
from shared.experiment import run_step, DeepFFN_abc
from shared.utils import setup_hooks, remove_hooks

FFN_BLOCK_LAYERS = 2
LAYERS = 12
BATCH_SIZE = 10000
DIM = 4096

class DeepFFN(DeepFFN_abc):
    handles: list

    def __init__(self, dtype):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__(dtype, LAYERS, DIM, FFN_BLOCK_LAYERS)

    # @torch.compile
    def forward(self, x, buffer: TensorBuffer):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        buffer.ready_buffer()
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
                x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + FFNSparse3.apply(x_inner, W1, W2, W3, buffer)
        return x


def evaluate():
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(BATCH_SIZE, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = DeepFFN(dtype=dtype)
    setup_hooks(model)

    # Run baseline
    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f'Baseline: {vram_dn = :.0f} MB, {avg_time=:.2f} ms')
    print("-"*50)

    # Setup sparse buffer and run model
    hdim_expanded = math.floor(DIM * 5.25)
    buffer_scale = 0.55 * (2 if FFN_BLOCK_LAYERS == 3 else 1)
    buffer_size = int(BATCH_SIZE * hdim_expanded * LAYERS * buffer_scale)
    buffer = TensorBuffer(buffer_size, dtype=dtype, device="cuda")

    run_step(x, model, buffer, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=2)
    print(f"VRAM allocated by tensors: {vram:.0f} MB")
    print(f'Total time: {avg_time:.2f} ms')

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):
        print(f'Predicted values are different.')
        print(f'{tracking_dn = }')
        print(f'{tracking = }')
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)

        # Make sure vram usage is low enough
        assert vram < vram_dn * 0.95


def run_base():
    """Configure deterministic/debug settings and launch the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)

    evaluate()


if __name__ == "__main__":
    run_base()
