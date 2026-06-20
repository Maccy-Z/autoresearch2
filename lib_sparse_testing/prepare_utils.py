import torch
from cprint import c_print
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor


class BitsparseTensor:
    """Bitmask sparse tensor."""
    vals: Tensor            # Nonzero values
    bitmask: Tensor         # Bitmask of nonzero values.
    prefix: Tensor          # Int32 tensor of where each block starts in the vals array.
    vals_offset: Tensor     # Starting offset in vals for each tile.
    BLOCK_M: int            # Size of each tile [M, N]
    BLOCK_N: int
    grid_m: int             # Number of tiles in [M, N] dimensions. grid_m = ceil[M/BLOCK_M]
    grid_n: int

    def __init__(self, vals: Tensor, bitmask: Tensor, prefix: Tensor,
                 vals_offset: Tensor,
                 grid_m: int, grid_n: int, BLOCK_M: int, BLOCK_N: int,
                 shape):
        super().__init__()
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        self.vals_offset = vals_offset + 1 - 1
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, "
                f"nnz={self.vals.numel()})\n")

    def vram_size(self):
        val_size = self.vals.element_size() * self.vals.nelement()
        bitmask_size = self.bitmask.element_size() * self.bitmask.nelement()
        prefix_size = self.prefix.element_size() * self.prefix.nelement()
        return (val_size + bitmask_size + prefix_size)/1024**2

    def sparsity_ratio(self):
        return self.vals.numel() / (self.shape[0] * self.shape[1])


class ValueBuffer:
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