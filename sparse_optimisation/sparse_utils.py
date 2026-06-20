import torch
from cprint import c_print
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor


class BitsparseTensor:
    vals: Tensor
    bitmask: Tensor
    scales: Tensor
    prefix: Tensor
    vals_offset: Tensor
    BLOCK_M: int
    BLOCK_N: int
    grid_m: int
    grid_n: int

    def __init__(self, vals: Tensor, bitmask: Tensor, scales: Tensor,
                 grid_m: int, grid_n: int, BLOCK_M: int, BLOCK_N: int,
                 shape):
        self.vals = vals
        self.bitmask = bitmask
        self.scales = scales
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)})")

    def vram_size(self):
        val_size = self.vals.element_size() * self.vals.nelement()
        bitmask_size = self.bitmask.element_size() * self.bitmask.nelement()
        return (val_size + bitmask_size)/1024**2

    def sparsity_ratio(self):
        return 0.5


class ValueBuffer:
    vals: Tensor = None
    scales: Tensor = None
    offset: Tensor = None

    def __init__(self, size, device, dtype):
        self.byte_size = size * dtype.itemsize
        self.device = device

    def init_buffer(self):
        if self.vals is None:
            self.vals = torch.empty(self.byte_size, device=self.device, dtype=torch.int8)
            c_print(f'Global buffer: {self.vals.nbytes / (1024 ** 2)}MB', color='green')
        if self.scales is None:
            max_tiles = 200000
            self.scales = torch.empty(max_tiles, device=self.device, dtype=torch.float32)
            c_print(f'Scale buffer: {self.scales.nbytes / 1024}KB', color='green')

    def ready_buffer(self):
        self.offset = torch.zeros(1, device=self.device, dtype=torch.int32)