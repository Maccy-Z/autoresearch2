import torch
from torch import Tensor


class BitsparseTensor:
    """Tile-wise bitmask sparse tensor for a dense matrix of shape ``shape``.

    ``vals`` stores positive entries in row-major tile order, ``bitmask`` marks
    nonzero locations with one bit per element, and ``prefix[t]`` gives the
    starting offset of tile ``t`` in ``vals``.
    """
    vals: Tensor
    bitmask: Tensor
    prefix: Tensor
    BLOCK_M: int
    BLOCK_N: int
    grid_m: int
    grid_n: int

    def __init__(self, vals, bitmask, prefix,
                 grid_m, grid_n, BLOCK_M, BLOCK_N, shape):
        """Store compressed values and tile metadata for later unpack/masking."""
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.shape = shape
