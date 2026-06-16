from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from torch.autograd import Function

from sparse_pack import _tile_pack_kernel, _compact_vals_kernel
from sparse_unpack import _unpack_batch_kernel
if TYPE_CHECKING:
    from torch import Tensor


class BitsparseTensor(torch.Tensor):
    """Bitmask sparse tensor."""
    vals: Tensor
    bitmask: Tensor
    prefix: Tensor
    BLOCK_M: int
    BLOCK_N: int
    grid_m: int
    grid_n: int

    @staticmethod
    @torch.compiler.disable
    def __new__(cls, vals: Tensor, bitmask: Tensor, prefix: Tensor,
                grid_m: int, grid_n: int, BLOCK_M: int, BLOCK_N: int,
                shape, dtype, device):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype, device=device, requires_grad=True,
        )

    def __init__(self, vals: Tensor, bitmask: Tensor, prefix: Tensor,
                 grid_m: int, grid_n: int, BLOCK_M: int, BLOCK_N: int,
                 shape, dtype, device):
        super().__init__()
        self.vals = vals
        self.bitmask = bitmask
        self.prefix = prefix
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, device={self.device}, "
                f"nnz={self.vals.numel()}, requires_grad={self.requires_grad})\n")

    @classmethod
    def _from_parts(cls, sparse):
        # Create clone of tensor, used for .detach(), .clone() etc. Bypasses normal user-facing construction logic.
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            sparse.shape,
            strides=sparse.stride(),
            storage_offset=sparse.storage_offset(),
            dtype=sparse.dtype,
            layout=sparse.layout,
            device=sparse.device,
            requires_grad=sparse.requires_grad,
        )
        obj.vals = sparse.vals
        obj.bitmask = sparse.bitmask
        obj.prefix = sparse.prefix
        obj.grid_m = sparse.grid_m
        obj.grid_n = sparse.grid_n
        obj.BLOCK_M = sparse.BLOCK_M
        obj.BLOCK_N = sparse.BLOCK_N
        return obj

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.detach.default:
            self = args[0]
            return BitsparseTensor._from_parts(self)

        raise NotImplementedError

    def vram_size(self):
        val_size = self.vals.element_size() * self.vals.nelement()
        bitmask_size = self.bitmask.element_size() * self.bitmask.nelement()
        prefix_size = self.prefix.element_size() * self.prefix.nelement()
        return (val_size + bitmask_size + prefix_size)/1024**2

    def sparsity_ratio(self):
        return 1 - self.vals.numel() / self.numel()


@torch.compiler.disable
def dense_to_tilesparse(dense: torch.Tensor, BLOCK_M=64, BLOCK_N=128) -> BitsparseTensor:
    """Pack a dense 2D tensor into the per-tile compressed sparse format.

    Returns a BitsparseTensor.
    """
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

    # --- host: exclusive prefix sum over per-tile counts ---
    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    total_nnz = tile_prefix[-1].item()

    # --- launch: compact nonzeros into contiguous vals ---
    vals = torch.empty(total_nnz, device=dense.device, dtype=dense.dtype)
    _compact_vals_kernel[(num_tiles,)](
        dense, tile_prefix, vals,
        M, N, grid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=8, num_stages=2,
    )

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        dense.shape, dense.dtype, dense.device,
    )


def unpack_bitmask_to_bool(sparse: BitsparseTensor) -> torch.Tensor:
    """
    Unpack a packed tile bitmask into a dense [M, N] boolean tensor.
    """
    bitmask = sparse.bitmask
    M, N = sparse.shape
    grid_m, grid_n = sparse.grid_m, sparse.grid_n
    BLOCK_M, BLOCK_N = sparse.BLOCK_M, sparse.BLOCK_N
    tile_numel = BLOCK_M * BLOCK_N
    tile_bytes = tile_numel // 8
    num_tiles = grid_m * grid_n

    packed = bitmask.reshape(num_tiles, tile_bytes).to(torch.int16)
    bit_pos = torch.arange(8, device=bitmask.device, dtype=torch.int16)
    bits = ((packed.unsqueeze(-1) >> bit_pos) & 1).to(torch.bool)

    tiles = bits.reshape(grid_m, grid_n, BLOCK_M, BLOCK_N)
    dense_mask = tiles.permute(0, 2, 1, 3).reshape(grid_m * BLOCK_M, grid_n * BLOCK_N)
    return dense_mask[:M, :N]


def sp_relu_Ax(W: Tensor, x: Tensor, BLOCK_M=64, BLOCK_N=128) -> BitsparseTensor:
    """
    y = relu(x @ W.T), then pack into a compact per-tile
    sparse representation.

    Returns a BitsparseTensor:
    """

    # Do matmul as normal
    y1 = F.relu(F.linear(x, W))

    return dense_to_tilesparse(y1, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


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
    if W.shape[1] != M:
        raise ValueError(f"W.shape must be [K, {M}] for W @ sparse_x, got {tuple(W.shape)}")
    K = W.shape[0]

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    num_tiles = grid_m * grid_n
    dense = torch.empty(M, N, device=W.device, dtype=vals.dtype)

    _unpack_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8, num_stages=2,
    )

    return W @ dense


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

        ctx.save_for_backward(x, W1, W2, z_sparse)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, z_sparse = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2                   # [BS, exp_fact*in_dim]
        grad_W2 = spAx(z_sparse, grad_output.T)      # [dim, exp_fact*in_dim]

        # z = relu(preact)
        relu_grad = unpack_bitmask_to_bool(z_sparse)
        grad_preact = grad_z * relu_grad

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2


def main():
    W = torch.randn(1024, 1024, device="cuda")
    x = torch.randn(100, 1024, device="cuda")

    sp = sp_relu_Ax(W, x)
    print(sp)
    out = spAx(sp, W)
    print(out)

if __name__ == "__main__":
    main()
