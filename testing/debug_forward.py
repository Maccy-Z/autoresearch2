from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import time
import math
import gc
import torch._logging
import logging
from cprint import c_print

from torch.utils.checkpoint import create_selective_checkpoint_contexts
from forward_methods import FFNSparseDirect, FFNSparseCustomFFN
if TYPE_CHECKING:
    from torch import Tensor


# ------------------- Global value buffer ------------------------------------
_global_vals: torch.Tensor = None
_global_offset: torch.Tensor = None

def init_sparse_buffer(size: int, device, dtype):
    """Initialise global sparse-buffer state used by the debug benchmark."""
    global _global_vals, _global_offset, _global_counter
    # _global_vals = torch.empty(size, device=device, dtype=dtype)
    _global_offset = torch.zeros(1, device=device, dtype=torch.int32)
    #
    # c_print(f'Global buffer: {_global_vals.nbytes/(1024**2)}MB', color='green')
    # c_print(f'Maximum number of elements: {_global_vals.numel()}', color='green')


def reset_sparse_globals():
    global _global_offset
    if _global_offset is not None:
        _global_offset.zero_()


def get_globals():
    return [_global_vals, _global_offset]


def generate_parameters(dim, G, dtype, expansion=5.25, device="cuda"):
    """Create one FFN layer's W1 and W2 parameters with deterministic initialisation."""
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01*shift*W1.std()
    W2 = W2 - 0.01*shift*W2.std()
    return W1, W2


# The selective save list does not materially change peak VRAM for the current
# autograd.Function boundary, so keep the outer checkpoint simple.
OPS_TO_SAVE = []
USE_SELECTIVE_CHECKPOINT = False
FORWARD_IMPL = FFNSparseCustomFFN


def checkpoint_context_fn():
    return create_selective_checkpoint_contexts(OPS_TO_SAVE)


class DeepFFN(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            W1, W2 = generate_parameters(hidm, G, dtype=dtype)

            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))

        if USE_SELECTIVE_CHECKPOINT:
            self.forward = lambda *args: torch.utils.checkpoint.checkpoint(
                            self.forward_base, *args,
                            use_reentrant=False,
                            context_fn=checkpoint_context_fn,
                        )
        else:
            self.forward = self.forward_base

    # @torch.no_grad()
    def forward_base(self, x, sparse_data, buffer_size):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        sparse_data = [None, None]
        sparse_data[0] = torch.empty(buffer_size, device="cuda", dtype=torch.bfloat16)
        sparse_data[1] = torch.zeros(1, device="cuda", dtype=torch.int32)
        for W1, W2 in zip(self.W1s, self.W2s):
            x = x + FORWARD_IMPL.apply(x, W1, W2, sparse_data)
        return x


def run_step(x, model, buffer_size, steps=1):
    """Run forward/backward steps and return peak allocated VRAM plus average step time."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats("cuda")
    torch.cuda.synchronize()
    sparse_data = get_globals()
    start = time.perf_counter()

    for _ in range(steps):
        torch.cuda.reset_peak_memory_stats("cuda")
        model.zero_grad()
        reset_sparse_globals()
        y = model.forward(x, sparse_data, buffer_size)
        loss = (y - x).pow(2).mean()
        loss.backward()
        # VRAM usage
        allocated = torch.cuda.max_memory_allocated("cuda") / 1024**2
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps

    return allocated, avg_time


def evaluate():
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    hdim = 4096
    bs = 10_000
    layers = 4
    dtype = torch.bfloat16

    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G)
    # Our model
    model = DeepFFN(layers=layers, dtype=dtype)
    # model = torch.compile(model)
    # torch._functorch.config.activation_memory_budget = 0.8

    # Setup sparse buffer
    hdim_expanded = math.floor(hdim * 5.25)
    buffer_size = int(bs * hdim_expanded * layers * 0.55)
    init_sparse_buffer(
        buffer_size, device="cuda", dtype=dtype,
    )
    run_step(x, model, buffer_size, steps=2)
    # Main run
    vram, avg_time = run_step(x, model, buffer_size, steps=5)

    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f'{avg_time = :.2f} ms')
    return vram, avg_time



def run_base():
    """Configure deterministic/debug settings and launch the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    torch._logging.set_logs(
        # dynamo=logging.INFO,
        # dynamic=logging.INFO,
        graph_breaks=True,
        # recompiles=True,
    )


    evaluate()


if __name__ == "__main__":
    run_base()
