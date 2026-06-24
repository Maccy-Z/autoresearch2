import torch
from torch import Tensor
from torch.autograd import Function
from torch.library import custom_op

from shared.triton_operators import tile_pack, compact_vals
from shared.utils import tile_grid, BitsparseTensor, TensorBuffer
from shared.functions import FFN3_backward, FFN_backward, BLOCK_M, BLOCK_N
from backward_method import FFN_backward_sparse


BACKWARD_IMPL = FFN_backward
# BACKWARD_IMPL = FFN_backward_sparse


def _make_bitsparse(
    vals: Tensor, bitmask: Tensor, prefix: Tensor,
    vals_offset: Tensor,
    shape: tuple[int, int]
) -> BitsparseTensor:
    """Build a BitsparseTensor wrapper around packed values, bitmasks, and prefixes."""
    grid_m = (shape[0] + BLOCK_M - 1) // BLOCK_M
    grid_n = (shape[1] + BLOCK_N - 1) // BLOCK_N
    return BitsparseTensor(
        vals, bitmask, prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, shape,
        vals_offset=vals_offset,
    )


def _dense_to_tilesparse_pack_impl(
    dense: Tensor, vals: Tensor, offset: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack a dense matrix into tile-sparse metadata and append values into the shared buffer."""
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    tile_pack(dense, tile_counts, tile_bitmasks,
              M, N, grid_m, grid_n, BLOCK_M, BLOCK_N, TILE_NUMEL, TILE_BYTES)

    new_offset = offset.clone()

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    compact_vals(dense, tile_prefix, vals, new_offset,
                 M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL)

    offset.add_(tile_prefix[-1])
    return tile_bitmasks, tile_prefix, new_offset


def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: TensorBuffer,
) -> BitsparseTensor:
    """Convert a dense activation matrix into a BitsparseTensor backed by sparse_data."""
    vals, offset = sparse_data.vals, sparse_data.offset
    bitmask, prefix, vals_offset = _dense_to_tilesparse_pack_impl(
        dense, vals, offset
    )
    return _make_bitsparse(vals, bitmask, prefix, vals_offset, dense.shape)


@custom_op("bitsparse_forward_methods::ffn_sparse_forward", mutates_args={"vals", "offset"})
def ffn_sparse_forward_op(
    x: Tensor, W1: Tensor, W2: Tensor, vals: Tensor,
    offset: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run FFN forward and pack the ReLU activation into the sparse value buffer."""
    preact = x @ W1.T
    preact.relu_()
    bitmask, prefix, vals_offset = _dense_to_tilesparse_pack_impl(
        preact, vals, offset
    )
    output = preact @ W2.T
    return output, bitmask, prefix, vals_offset


@ffn_sparse_forward_op.register_fake
def _(
    x: Tensor, W1: Tensor, W2: Tensor, vals: Tensor,
    offset: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return fake tensor outputs for tracing ffn_sparse_forward_op."""
    M = x.shape[0]
    N = W1.shape[0]
    _, _, num_tiles, _, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)
    return (
        torch.empty((M, W2.shape[0]), device=x.device, dtype=x.dtype),
        torch.empty(num_tiles * TILE_BYTES, device=x.device, dtype=torch.uint8),
        torch.empty(num_tiles + 1, device=x.device, dtype=torch.int32),
        torch.empty_like(offset),
    )


class FFNSparse(Function):
    """Forward of FFN."""

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data):
        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        h_sparse = dense_to_tilesparse(h, sparse_data)
        ctx.h_sparse = h_sparse
        return h @ W2.T

    backward = staticmethod(BACKWARD_IMPL)


class FFNSparse3(Function):
    """Autograd FFN block with two hidden ReLU layers."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3, sparse_data):
        ctx.save_for_backward(x, W1, W2, W3)
        z1 = x @ W1.T
        h1 = z1.relu_()
        ctx.h1_sparse = dense_to_tilesparse(h1, sparse_data)
        z2 = h1 @ W2.T
        del z1, h1
        h2 = z2.relu_()
        ctx.h2_sparse = dense_to_tilesparse(h2, sparse_data)

        return h2 @ W3.T

    backward = staticmethod(FFN3_backward)


class FFNSparseCustomOp(Function):
    """Forward with matmul, pack, and second matmul hidden behind one custom op.
    Useful for compiling
    """

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data: TensorBuffer):
        vals, offset = sparse_data.vals, sparse_data.offset
        output, bitmask, prefix, vals_offset = ffn_sparse_forward_op(
            x, W1, W2, vals, offset
        )
        h_sparse = _make_bitsparse(
            vals, bitmask, prefix, vals_offset,
            (x.shape[0], W1.shape[0]),
        )
        ctx.h_sparse = h_sparse
        ctx.save_for_backward(x, W1, W2)
        return output

    backward = staticmethod(BACKWARD_IMPL)
