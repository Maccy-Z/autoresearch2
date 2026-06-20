import torch
from cprint import c_print
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor


class RowSparseTensor:
    """Row-wise sparse tensor with int8 quantization."""
    vals: Tensor
    row_bitmask: Tensor
    row_offsets: Tensor
    scales: Tensor
    shape: tuple

    def __init__(self, vals, row_bitmask, row_offsets, scales, shape):
        self.vals = vals
        self.row_bitmask = row_bitmask
        self.row_offsets = row_offsets
        self.scales = scales
        self.shape = shape

    def __repr__(self):
        return f"RowSparseTensor(shape={list(self.shape)})"


class ValueBuffer:
    vals: Tensor = None
    row_bitmask: Tensor = None
    row_scales: Tensor = None

    def __init__(self, size, device, dtype):
        self.byte_size = size * dtype.itemsize
        self.device = device

    def init_buffer(self):
        if self.vals is None:
            self.vals = torch.empty(self.byte_size, device=self.device, dtype=torch.int8)
            c_print(f'Values buffer: {self.vals.nbytes / (1024 ** 2):.1f}MB', color='green')
        if self.row_bitmask is None:
            max_rows = 200000
            max_row_bytes = 5000
            self.row_bitmask = torch.empty(max_rows * max_row_bytes, device=self.device, dtype=torch.uint8)
            self.row_scales = torch.empty(max_rows, device=self.device, dtype=torch.float32)
            c_print(f'Bitmask buffer: {self.row_bitmask.nbytes / (1024 ** 2):.1f}MB', color='green')

    def ready_buffer(self):
        self._offset = 0
