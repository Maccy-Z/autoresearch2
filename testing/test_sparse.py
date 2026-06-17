from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from torch.autograd import Function
from cprint import c_print
import triton
from torch.library import triton_op, wrap_triton

from sparse_pack import _tile_pack_kernel, _compact_vals_kernel
from sparse_unpack import _unpack_batch_kernel, _mask_with_bitmask_kernel
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


class FFNSparse(Function):
    """ Sparse feedforward layer """
    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [exp_fact*in_dim, in_dim]
        W2.shape = [dim, exp_fact*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, exp_fact*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        z_sparse = dense_to_tilesparse(z, sparse_data)
        ctx.save_for_backward(x, W1, W2)
        ctx.z_sparse = z_sparse
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2 = ctx.saved_tensors
        z_sparse = ctx.z_sparse
        ctx.z_sparse = None

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2                   # [BS, exp_fact*in_dim]
        grad_W2 = spAx(z_sparse, grad_output.T)      # [dim, exp_fact*in_dim]

        # z = relu(preact) — apply mask via bitmask kernel (in-place on grad_z)
        _mask_with_bitmask_kernel[(z_sparse.grid_m, z_sparse.grid_n)](
            grad_z, z_sparse.bitmask,
            z_sparse.shape[0], z_sparse.shape[1],
            BLOCK_M=z_sparse.BLOCK_M, BLOCK_N=z_sparse.BLOCK_N,
            TILE_BYTES=z_sparse.BLOCK_M * z_sparse.BLOCK_N // 8,
            num_warps=4, num_stages=2,
        )
        grad_preact = grad_z

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x.contiguous()        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2, None, None

# # @torch.compiler.disable()
# def dense_to_tilesparse(dense: torch.Tensor, sparse_data, BLOCK_M=64, BLOCK_N=64) -> BitsparseTensor:
#     """Pack a dense 2D tensor into the per-tile compressed sparse format.
#
#     Returns a BitsparseTensor.
#     """
#     vals, offset = sparse_data
#     M, N = dense.shape
#
#     TILE_NUMEL = BLOCK_M * BLOCK_N
#     TILE_BYTES = TILE_NUMEL // 8
#
#     grid_m = (M + BLOCK_M - 1) // BLOCK_M
#     grid_n = (N + BLOCK_N - 1) // BLOCK_N
#     num_tiles = grid_m * grid_n
#
#     # --- launch: tile pack (bitmask + counts) ---
#     tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
#     tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)
#
#     _tile_pack_kernel[(grid_m, grid_n)](
#         dense, tile_counts, tile_bitmasks,
#         M, N,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
#         TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
#         num_warps=4, num_stages=2,
#     )
#
#     # --- Manage offsetts ---
#     total_offset = offset.sum()
#
#     # --- device-side prefix sum (local, no global offset baked in) ---
#     tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
#     torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
#     tile_prefix[0] = 0
#     # --- record moment offset, launch compact with offset ---
#     _compact_vals_kernel[(num_tiles,)](
#         dense, tile_prefix, vals,
#         total_offset,
#         M, N, grid_n,
#         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
#         TILE_NUMEL=TILE_NUMEL,
#         num_warps=4, num_stages=2,
#     )
#
#     # --- advance global offset for the next layer ---
#     offset.index_add_(
#         0,
#         torch.tensor([0], device="cuda"),#_global_counter.reshape(1),
#         tile_prefix[-1].reshape(1),
#     )
#     return BitsparseTensor(
#         vals, tile_bitmasks, tile_prefix,
#         total_offset,
#         grid_m, grid_n, BLOCK_M, BLOCK_N,
#         dense.shape
#     )

# @torch.compiler.disable()
def dense_to_tilesparse(
    dense: torch.Tensor,
    sparse_data: tuple[torch.Tensor, ...],
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
) -> BitsparseTensor:
    vals, offset = sparse_data
    total_offset = offset[0].clone()

    tile_bitmasks, tile_prefix, my_offset, grid_m, grid_n, offset_inc = dense_to_tilesparse_op(
        dense,
        vals,
        total_offset,
        BLOCK_M,
        BLOCK_N,
    )
    offset.index_add_(
            0,
            torch.tensor([0], device="cuda"),
            offset_inc.reshape(1),
        )

    return BitsparseTensor(
        vals,
        tile_bitmasks,
        tile_prefix,
        my_offset,
        grid_m,
        grid_n,
        BLOCK_M,
        BLOCK_N,
        dense.shape,
    )
#
# @triton_op("bitsparse::dense_to_tilesparse", mutates_args={"vals"})
def dense_to_tilesparse_op(
    dense: torch.Tensor,
    vals: torch.Tensor,
    offset: torch.Tensor,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor]:

    M, N = dense.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n

    tile_counts = torch.empty(
        (num_tiles,),
        device=dense.device, dtype=torch.int32,
    )

    tile_bitmasks = torch.empty(
        (num_tiles * TILE_BYTES,),
        device=dense.device, dtype=torch.uint8,
    )

    wrap_triton(_tile_pack_kernel)[(grid_m, grid_n)](
        dense,  tile_counts, tile_bitmasks,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
    )

    tile_prefix = torch.empty(
        (num_tiles + 1,),
        device=dense.device, dtype=torch.int32,
    )

    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    my_offset = offset.clone()

    wrap_triton(_compact_vals_kernel)[(num_tiles,)](
        dense, tile_prefix, vals, my_offset,
        M, N,
        grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=4, num_stages=2,
    )

    return tile_bitmasks, tile_prefix, my_offset, grid_m, grid_n, tile_prefix[-1:].clone()


# @torch.compiler.disable()
def spAx(x_sparse: BitsparseTensor, W: Tensor, square: bool = False) -> Tensor:
    """
    y = W @ sparse_x.  If square=True, uses sparse_x^2 instead.
    x.shape = [M, N]
    W.shape = [K, M]
    """
    vals = x_sparse.vals
    bitmask = x_sparse.bitmask
    prefix = x_sparse.prefix
    BLOCK_M, BLOCK_N = x_sparse.BLOCK_M, x_sparse.BLOCK_N
    grid_m, grid_n = x_sparse.grid_m, x_sparse.grid_n
    M, N = x_sparse.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    num_tiles = grid_m * grid_n
    dense = torch.empty(M, N, device=W.device, dtype=vals.dtype)

    _unpack_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix,
        x_sparse.vals_offset,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
    )
    return W @ dense

# # @torch.compiler.disable()
# def spAx(x_sparse: BitsparseTensor, W: Tensor) -> Tensor:
#     """
#     y = W @ sparse_x.
#     x.shape = [M, N]
#     W.shape = [K, M]
#
#     Unpacks the sparse representation into dense row-batches, then uses dense matmul for each W column slice.
#     """
#     #
#     vals = x_sparse.vals
#     bitmask = x_sparse.bitmask
#     prefix = x_sparse.prefix
#     BLOCK_M, BLOCK_N = x_sparse.BLOCK_M, x_sparse.BLOCK_N
#     _, grid_n = x_sparse.grid_m, x_sparse.grid_n
#     M, N = x_sparse.shape
#     if W.shape[1] != M:
#         raise ValueError(f"W.shape must be [K, {M}] for W @ sparse_x, got {tuple(W.shape)}")
#     K = W.shape[0]
#
#     TILE_NUMEL = BLOCK_M * BLOCK_N
#     TILE_BYTES = TILE_NUMEL // 8
#
#     ROW_BATCH = 2048
#
#     out = torch.zeros(K, N, device=W.device, dtype=W.dtype)
#
#     for m_start in range(0, M, ROW_BATCH):
#         m_end = min(m_start + ROW_BATCH, M)
#         batch_rows = m_end - m_start
#         first_m_tile = m_start // BLOCK_M
#
#         dense_batch = torch.empty(batch_rows, N, device=W.device, dtype=vals.dtype)
#
#         num_row_tiles_in_batch = (batch_rows + BLOCK_M - 1) // BLOCK_M
#         num_tiles_in_batch = num_row_tiles_in_batch * grid_n
#
#         _unpack_batch_kernel[(num_tiles_in_batch,)](
#             vals, bitmask, prefix,
#             x_sparse.vals_offset,
#             dense_batch,
#             first_m_tile, grid_n, N, batch_rows,
#             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
#             TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
#             num_warps=8, num_stages=2,
#         )
#
#         out.add_(W[:, m_start:m_end] @ dense_batch)
#
#     return out