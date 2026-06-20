import torch
from torch import Tensor
from torch.autograd import Function
from torch.library import custom_op

from backward_method import FFN_backward_sparse, FFN_backward
from sparse_kernels import _tile_pack_int8_kernel
from sparse_utils import BitsparseTensor, ValueBuffer


DEFAULT_BLOCK_M = 128
DEFAULT_BLOCK_N = 128
BACKWARD_IMPL = FFN_backward


def _tile_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> tuple[int, int, int, int, int]:
    """Return tile-grid dimensions and tile storage sizes for a dense matrix shape."""
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n
    return grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES


def _make_bitsparse(
    vals: Tensor, bitmask: Tensor, scales: Tensor,
    shape: tuple[int, int], BLOCK_M: int, BLOCK_N: int,
) -> BitsparseTensor:
    grid_m = (shape[0] + BLOCK_M - 1) // BLOCK_M
    grid_n = (shape[1] + BLOCK_N - 1) // BLOCK_N
    return BitsparseTensor(
        vals, bitmask, scales,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        shape,
    )


def _dense_to_tilesparse_pack_impl(
    dense: Tensor, vals: Tensor, scales: Tensor,
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = DEFAULT_BLOCK_N,
) -> tuple[Tensor, Tensor, Tensor]:
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = _tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)
    layer_vals = vals[:num_tiles * TILE_NUMEL]
    layer_scales = scales[:num_tiles]

    _tile_pack_int8_kernel[(grid_m, grid_n)](
        dense, tile_bitmasks, layer_vals, layer_scales,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=16, num_stages=2,
    )

    return tile_bitmasks, layer_vals, layer_scales


@torch.compile
def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: ValueBuffer,
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = DEFAULT_BLOCK_N,
) -> BitsparseTensor:
    vals = sparse_data.vals
    scales = sparse_data.scales
    bitmask, tile_vals, tile_scales = _dense_to_tilesparse_pack_impl(
        dense, vals, scales, BLOCK_M, BLOCK_N
    )
    return _make_bitsparse(tile_vals, bitmask, tile_scales, dense.shape, BLOCK_M, BLOCK_N)


@custom_op("bitsparse_forward_methods::ffn_sparse_forward", mutates_args={"vals", "offset"})
def ffn_sparse_forward_op(
    x: Tensor, W1: Tensor, W2: Tensor, vals: Tensor,
    offset: Tensor, BLOCK_M: int, BLOCK_N: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run FFN forward and pack the ReLU activation into the sparse value buffer."""
    preact = x @ W1.T
    preact.relu_()
    bitmask, prefix, vals_offset = _dense_to_tilesparse_pack_impl(
        preact, vals, offset, BLOCK_M, BLOCK_N
    )
    output = preact @ W2.T
    return output, bitmask, prefix, vals_offset


@ffn_sparse_forward_op.register_fake
def _(
    x: Tensor, W1: Tensor, W2: Tensor, vals: Tensor,
    offset: Tensor, BLOCK_M: int,  BLOCK_N: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return fake tensor outputs for tracing ffn_sparse_forward_op."""
    M = x.shape[0]
    N = W1.shape[0]
    _, _, num_tiles, _, TILE_BYTES = _tile_grid(M, N, BLOCK_M, BLOCK_N)
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
        preact = x @ W1.T
        preact.relu_()
        z_sparse = dense_to_tilesparse(preact, sparse_data)
        ctx.z_sparse = z_sparse
        return preact @ W2.T

    backward = staticmethod(BACKWARD_IMPL)


class FFNSparseCustomOp(Function):
    """Forward with matmul, pack, and second matmul hidden behind one custom op."""

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data: ValueBuffer):
        vals, offset = sparse_data.vals, sparse_data.offset
        output, bitmask, prefix, vals_offset = ffn_sparse_forward_op(
            x, W1, W2, vals, offset, DEFAULT_BLOCK_M, DEFAULT_BLOCK_N
        )
        z_sparse = _make_bitsparse(
            vals, bitmask, prefix, vals_offset,
            (x.shape[0], W1.shape[0]),
            DEFAULT_BLOCK_M, DEFAULT_BLOCK_N,
        )
        ctx.z_sparse = z_sparse
        ctx.save_for_backward(x, W1, W2)
        return output

    backward = staticmethod(BACKWARD_IMPL)
