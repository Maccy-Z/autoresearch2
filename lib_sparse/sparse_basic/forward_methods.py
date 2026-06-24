import torch
from torch import Tensor
from torch.autograd import Function

from shared.functions import FFN3_backward, FFN_backward, BLOCK_M, BLOCK_N
from shared.triton_operators import tile_pack, compact_vals
from shared.utils import tile_grid, BitsparseTensor


def dense_to_tilesparse(dense: Tensor) -> BitsparseTensor:
    """Compress positive entries of ``dense[M, N]`` into tile bitmasks and values.

    This stores the ReLU activation sparsely: ``mask = dense > 0`` and
    ``vals = dense[mask]`` in row-major tile order.
    """
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    tile_pack(dense, tile_counts, tile_bitmasks,
              M, N, grid_m, grid_n, BLOCK_M, BLOCK_N, TILE_NUMEL, TILE_BYTES)

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    total_nnz = tile_prefix[-1].item()
    vals = torch.empty(total_nnz, device=dense.device, dtype=dense.dtype)

    compact_vals(dense, tile_prefix, vals, torch.tensor(0, device=dense.device, dtype=torch.int32),
                 M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL)

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
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h)
        return h @ W2.T

    backward = staticmethod(FFN_backward)


class FFNSparse3(Function):
    """Autograd FFN block with two hidden ReLU layers."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        ctx.save_for_backward(x, W1, W2, W3)
        z1 = x @ W1.T
        h1 = z1.relu_()
        ctx.h1_sparse = dense_to_tilesparse(h1)
        z2 = h1 @ W2.T
        del z1, h1
        h2 = z2.relu_()
        ctx.h2_sparse = dense_to_tilesparse(h2)
        return h2 @ W3.T

    backward = staticmethod(FFN3_backward)
