import torch
from torch import Tensor
from torch.autograd import Function

from backward_method import FFN_backward, FFN3_backward
from sparse_kernels import _compact_vals_kernel, _tile_pack_kernel
from sparse_utils import BitsparseTensor
from shared.utils import _tile_grid


BLOCK_M = 128
BLOCK_N = 128


def dense_to_tilesparse(dense: Tensor) -> BitsparseTensor:
    """Compress positive entries of ``dense[M, N]`` into tile bitmasks and values.

    This stores the ReLU activation sparsely: ``mask = dense > 0`` and
    ``vals = dense[mask]`` in row-major tile order.
    """
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = _tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    _tile_pack_kernel[(grid_m, grid_n)](
        dense, tile_counts, tile_bitmasks,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
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
    """Autograd FFN using sparse storage for the hidden ReLU activation.

    Forward formula for ``x[B, D]``, ``W1[H, D]``, ``W2[D, H]``:
    ``z = relu(x @ W1.T)`` and ``y = z @ W2.T``.
    """
    @staticmethod
    def forward(ctx, x, W1, W2):
        """Compute FFN output and save ``z`` as a ``BitsparseTensor`` for backward."""
        ctx.save_for_backward(x, W1, W2)
        preact = x @ W1.T
        preact.relu_()
        ctx.z_sparse = dense_to_tilesparse(preact)
        return preact @ W2.T

    backward = staticmethod(FFN_backward)


class FFNSparse3(Function):
    """Autograd FFN block with two hidden ReLU layers."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        ctx.save_for_backward(x, W1, W2, W3)
        z1 = x @ W1.T
        z1.relu_()
        ctx.z1_sparse = dense_to_tilesparse(z1)
        z2 = z1 @ W2.T
        del z1
        z2.relu_()
        ctx.z2_sparse = dense_to_tilesparse(z2)
        return z2 @ W3.T

    backward = staticmethod(FFN3_backward)
