import torch
from torch import Tensor
from torch.autograd import Function

from backward_method import FFN_backward
from sparse_kernels import _compact_vals_kernel, _tile_pack_kernel
from sparse_utils import BitsparseTensor


BLOCK_M = 128
BLOCK_N = 128


def _tile_grid(M: int, N: int) -> tuple[int, int, int, int, int]:
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n
    return grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES


def dense_to_tilesparse(dense: Tensor) -> BitsparseTensor:
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = _tile_grid(M, N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    _tile_pack_kernel[(grid_m, grid_n)](
        dense, tile_counts, tile_bitmasks,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8, num_stages=2,
    )

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    total_nnz = tile_prefix[-1].item()
    vals = torch.empty(total_nnz, device=dense.device, dtype=dense.dtype)

    _compact_vals_kernel[(num_tiles,)](
        dense, tile_prefix, vals,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=16, num_stages=2,
    )

    return BitsparseTensor(vals, tile_bitmasks, tile_prefix,
                           grid_m, grid_n, BLOCK_M, BLOCK_N, dense.shape)


class FFNSparse(Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        ctx.save_for_backward(x, W1, W2)
        preact = x @ W1.T
        preact.relu_()
        ctx.z_sparse = dense_to_tilesparse(preact)
        return preact @ W2.T

    backward = staticmethod(FFN_backward)
