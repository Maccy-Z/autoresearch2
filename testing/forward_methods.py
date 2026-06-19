import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.library import custom_op

from backward_method import ffn_backward_sparse_grad_z
from test_sparse import (
    BitsparseTensor,
    _compact_vals_kernel,
    _tile_pack_kernel,
)


DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
BACKWARD_IMPL = ffn_backward_sparse_grad_z


def _tile_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> tuple[int, int, int, int, int]:
    """Return tile-grid dimensions and tile storage sizes for a dense matrix shape."""
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n
    return grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES


def _make_bitsparse(
    vals: torch.Tensor,
    bitmask: torch.Tensor,
    prefix: torch.Tensor,
    vals_offset: torch.Tensor,
    shape: tuple[int, int],
    BLOCK_M: int,
    BLOCK_N: int,
) -> BitsparseTensor:
    """Build a BitsparseTensor wrapper around packed values, bitmasks, and prefixes."""
    grid_m = (shape[0] + BLOCK_M - 1) // BLOCK_M
    grid_n = (shape[1] + BLOCK_N - 1) // BLOCK_N
    return BitsparseTensor(
        vals, bitmask, prefix, vals_offset,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        shape,
    )


def _dense_to_tilesparse_pack_impl(
    dense: torch.Tensor,
    vals: torch.Tensor,
    offset: torch.Tensor,
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = DEFAULT_BLOCK_N,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a dense matrix into tile-sparse metadata and append values into the shared buffer."""
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

    total_offset = offset.clone()

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    _compact_vals_kernel[(num_tiles,)](
        dense, tile_prefix, vals,
        total_offset,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=4, num_stages=2,
    )

    offset.add_(tile_prefix[-1])
    return tile_bitmasks, tile_prefix, total_offset


def dense_to_tilesparse(
    dense: torch.Tensor,
    sparse_data: tuple[torch.Tensor, torch.Tensor],
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = DEFAULT_BLOCK_N,
) -> BitsparseTensor:
    """Convert a dense activation matrix into a BitsparseTensor backed by sparse_data."""
    vals, offset = sparse_data
    bitmask, prefix, vals_offset = _dense_to_tilesparse_pack_impl(
        dense, vals, offset, BLOCK_M, BLOCK_N
    )
    return _make_bitsparse(vals, bitmask, prefix, vals_offset, dense.shape, BLOCK_M, BLOCK_N)


@custom_op("bitsparse_forward_methods::ffn_sparse_forward", mutates_args={"vals", "offset"})
def ffn_sparse_forward_op(
    x: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    vals: torch.Tensor,
    offset: torch.Tensor,
    BLOCK_M: int,
    BLOCK_N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    x: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    vals: torch.Tensor,
    offset: torch.Tensor,
    BLOCK_M: int,
    BLOCK_N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _backward(ctx, grad_output: torch.Tensor):
    """Dispatch the custom autograd backward through the configured sparse backward implementation."""
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None

    grad_x, grad_W1, grad_W2 = BACKWARD_IMPL(grad_output, z_sparse, x, W1, W2)
    return grad_x, grad_W1, grad_W2, None


class FFNSparseDirect(Function):
    """Forward with direct Python/Triton dense_to_tilesparse visible to compile."""

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data):
        preact = x @ W1.T
        preact = F.relu(preact)
        z_sparse = dense_to_tilesparse(preact, sparse_data)
        ctx.z_sparse = z_sparse
        ctx.save_for_backward(x, W1, W2)
        return preact @ W2.T

    backward = staticmethod(_backward)


class FFNSparseCustomFFN(Function):
    """Forward with matmul, pack, and second matmul hidden behind one custom op."""

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data):
        vals, offset = sparse_data
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

    backward = staticmethod(_backward)
