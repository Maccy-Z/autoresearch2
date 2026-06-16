from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from torch.autograd import Function

from sparse_pack import _tile_pack_kernel, _compact_vals_kernel
from sparse_unpack import _unpack_batch_kernel, _unpack_batch_squared_kernel, _mask_with_bitmask_kernel, _grad_relu2_kernel
if TYPE_CHECKING:
    from torch import Tensor

# ------------------- Global value buffer ------------------------------------
_global_vals: torch.Tensor = None
_global_offset: torch.Tensor = None


def init_sparse_buffer(size: int, device, dtype):
    global _global_vals, _global_offset
    _global_vals = torch.empty(size, device=device, dtype=dtype)
    _global_offset = torch.zeros(1, device=device, dtype=torch.int32)


def reset_sparse_globals():
    global _global_offset
    if _global_offset is not None:
        _global_offset.zero_()


class BitsparseTensor:
    """Bitmask sparse tensor."""
    vals: Tensor            # Nonzero values
    bitmask: Tensor         # Bitmask of nonzero values.
    prefix: Tensor          # Int32 tensor of where each block starts in the vals array.
    vals_offset: Tensor
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
        self.vals_offset = vals_offset
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
        return 1 - self.vals.numel() / (self.shape[0] * self.shape[1])


class FFNSparse(Function):
    """ Sparse feedforward layer """
    @staticmethod
    def forward(ctx, x, W1, W2):
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

        z_sparse = dense_to_tilesparse(z)

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
        grad_W1 = grad_preact.T @ x        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2


class FFNSpRelu2(Function):
    """ Sparse feedforward layer with relu² activation """
    @staticmethod
    def forward(ctx, x, W1, W2):
        preact = x @ W1.T
        z = F.relu(preact)
        output = (z ** 2) @ W2.T

        z_sparse = dense_to_tilesparse(z)

        ctx.save_for_backward(x, W1, W2)
        ctx.z_sparse = z_sparse
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2 = ctx.saved_tensors
        z_sparse = ctx.z_sparse
        ctx.z_sparse = None

        grad_z = grad_output @ W2

        # grad_preact = grad_z * (2 * z) — mask only, squaring done by spAx_squared
        _grad_relu2_kernel[(z_sparse.grid_m, z_sparse.grid_n)](
            grad_z, z_sparse.vals, z_sparse.bitmask, z_sparse.prefix,
            z_sparse.vals_offset,
            z_sparse.shape[0], z_sparse.shape[1],
            BLOCK_M=z_sparse.BLOCK_M, BLOCK_N=z_sparse.BLOCK_N,
            TILE_NUMEL=z_sparse.BLOCK_M * z_sparse.BLOCK_N,
            TILE_BYTES=z_sparse.BLOCK_M * z_sparse.BLOCK_N // 8,
            num_warps=4, num_stages=2,
        )
        grad_preact = grad_z

        grad_W2 = spAx_squared(z_sparse, grad_output.T)

        grad_x = grad_preact @ W1
        grad_W1 = grad_preact.T @ x

        return grad_x, grad_W1, grad_W2


def dense_to_tilesparse(dense: torch.Tensor, BLOCK_M=64, BLOCK_N=64) -> BitsparseTensor:
    """Pack a dense 2D tensor into the per-tile compressed sparse format.

    Returns a BitsparseTensor.
    """
    global _global_vals, _global_offset

    M, N = dense.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n

    # --- launch: tile pack (bitmask + counts) ---
    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    _tile_pack_kernel[(grid_m, grid_n)](
        dense, tile_counts, tile_bitmasks,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
    )

    # --- device-side prefix sum (local, no global offset baked in) ---
    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    # --- record moment offset, launch compact with offset ---
    my_offset = _global_offset.clone()
    _compact_vals_kernel[(num_tiles,)](
        dense, tile_prefix, _global_vals,
        my_offset,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=4, num_stages=2,
    )

    # --- advance global offset for the next layer ---
    _global_offset = _global_offset + tile_prefix[-1]

    return BitsparseTensor(
        _global_vals, tile_bitmasks, tile_prefix,
        my_offset,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        dense.shape
    )


def spAx(x_sparse: BitsparseTensor, W: Tensor) -> Tensor:
    """
    y = W @ sparse_x.
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


def spAx_squared(x_sparse: BitsparseTensor, W: Tensor) -> Tensor:
    """y = W @ (sparse_x)^2. Squares during unpack."""
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
    _unpack_batch_squared_kernel[(num_tiles,)](
        vals, bitmask, prefix, x_sparse.vals_offset,
        dense, 0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
    )
    return W @ dense

