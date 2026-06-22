import torch
import torch.nn as nn
import torch._logging
import math

from forward_methods import FFNSparse, FFNSparse3, TensorBuffer, FFNSparseCustomOp
from shared.experiment import generate_parameters, generate_parameters_3, FFN, FFN_3, run_step

FFN_BLOCK_LAYERS = 3
LAYERS = 4
BATCH_SIZE = 10000
DIM = 4096

class DeepFFN(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.block_layers = FFN_BLOCK_LAYERS
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            if self.block_layers == 2:
                W1, W2 = generate_parameters(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
            else:
                W1, W2, W3 = generate_parameters_3(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
                self.W3s.append(nn.Parameter(W3))

        if self.block_layers == 2:
            self.block_forward = FFN.apply
        elif self.block_layers == 3:
            self.block_forward = FFN_3.apply
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
        """ x.shape = [BS, dim] """

        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
                x = x + FFN.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + self.block_forward(x_inner, W1, W2, W3)
        return x

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
    model = DeepFFN(layers=LAYERS, hidm=DIM, dtype=dtype)

    # Run baseline
    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f'Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms')
    print("-"*50)

    # Setup sparse buffer and run model
    hdim_expanded = math.floor(DIM * 5.25)
    buffer_scale = 0.53 * (2 if FFN_BLOCK_LAYERS == 3 else 1)
    buffer_size = int(BATCH_SIZE * hdim_expanded * LAYERS * buffer_scale)
    buffer = TensorBuffer(buffer_size, dtype=dtype, device="cuda")

    run_step(x, model, buffer, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=1)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
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
    torch._logging.set_logs(
        graph_breaks=True,
    )

    evaluate()


if __name__ == "__main__":
    run_base()
