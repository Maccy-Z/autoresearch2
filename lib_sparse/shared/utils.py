import torch
import torch.nn as nn
from torch import Tensor
from cprint import c_print

# Constant for RELU^2 scaling
RELU2_SCALE = 3 ** -0.5
BLOCK_M = 128
BLOCK_N = 128


class BitsparseTensor:
    """Tile-wise bitmask sparse tensor for a dense matrix of shape ``shape``.

    ``vals`` stores positive entries in row-major tile order, ``bitmask`` marks
    nonzero locations with one bit per element, and ``prefix[t]`` gives the
    starting offset of tile ``t`` in ``vals``.

    ``vals_offset`` is an optional int32 tensor giving the starting offset of
    this layer's values inside a shared ``vals`` buffer.  When ``None`` (the
    default), a zero tensor is created so the tensor is self-contained.
    """
    vals: Tensor
    bitmask: Tensor
    prefix: Tensor
    vals_offset: Tensor
    BLOCK_M: int
    BLOCK_N: int
    grid_m: int
    grid_n: int

    def __init__(self, vals, bitmask, prefix,
                 grid_m, grid_n, BLOCK_M, BLOCK_N, shape,
                 vals_offset=None):
        """Store compressed values and tile metadata for later unpack/masking."""
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape
        if vals_offset is None:
            vals_offset = torch.tensor(0, device=vals.device, dtype=torch.int32)
        self.vals_offset = vals_offset

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, "
                f"nnz={self.prefix[-1]}, sparsity={self.sparsity_ratio():.2f})")

    def vram_size(self):
        val_size = self.vals.element_size() * self.prefix[-1]
        bitmask_size = self.bitmask.element_size() * self.bitmask.nelement()
        prefix_size = self.prefix.element_size() * self.prefix.nelement()
        return (val_size + bitmask_size + prefix_size) / 1024 ** 2

    def sparsity_ratio(self):
        return 1 - self.prefix[-1] / (self.shape[0] * self.shape[1])


class TensorBuffer:
    vals: Tensor = None
    offset: Tensor = None

    def __init__(self, size, device, dtype):
        self.size = size
        self.device = device
        self.dtype = dtype

    def init_buffer(self):
        if self.vals is None:
            self.vals = torch.empty(self.size, device=self.device, dtype=self.dtype)

            c_print(f'Global buffer: {self.vals.nbytes / (1024 ** 2)}MB', color='green')
            c_print(f'Maximum number of elements: {self.vals.numel()}', color='green')

    def ready_buffer(self):
        """ Set offset tensor inside main training loop, since this needs to be consistent. """
        self.offset = torch.zeros(1, device=self.device, dtype=torch.int32)


def tile_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> tuple[int, int, int, int, int]:
    """Return tile-grid dimensions and tile storage sizes for a dense matrix shape."""
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n
    return grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES


@torch.no_grad()
def inplace_mm_(A, W, B=2048):
    """ A <- AW inplace operation. Done with batches. """
    m, n = A.shape

    x = torch.empty((B, n), device=A.device, dtype=A.dtype)
    y = torch.empty_like(x)

    for i in range(0, m, B):
        b = min(B, m - i)
        x[:b].copy_(A[i:i+b])
        torch.mm(x[:b], W, out=y[:b])
        A[i:i+b].copy_(y[:b])
    return A


def print_memory(msg):
    memory = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    c_print(f'{msg}: {memory:.2f} MB', color="bright_cyan")


def setup_hooks(model: nn.Module):
    """ Simulate hook optimiser that applies update + clears grads immediately."""
    def hook(w):
        w.grad = None
        return

    model.handles = []
    for n, p in model.named_parameters():
        handle = p.register_post_accumulate_grad_hook(hook)
        model.handles.append(handle)


def remove_hooks(model):
    for handle in model.handles:
        handle.remove()
