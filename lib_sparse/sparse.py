from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F


from sparse_pack import _compact_vals_kernel, _tile_pack_kernel
from sparse_unpack import _unpack_batch_kernel
if TYPE_CHECKING:
    from torch import Tensor


class BitsparseTensor(torch.Tensor):
    """ Bitmask sparse tensor. """
    vals: Tensor            # Nonzero values
    bitmask: Tensor         # Bitmask of nonzero values.
    prefix: Tensor          # Int32 tensor of where each block starts in the vals array.
    BLOCK_M: int            # Size of each tile [M, N]
    BLOCK_N: int
    grid_m: int             # Number of tiles in [M, N] dimensions. grid_m = ceil[M/BLOCK_M]
    grid_n: int

    @staticmethod
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

# ---------------------------------------------------------------------------
# Layer 1: x @ W1.T, then ReLU, then sparsify into a compact per-tile
# bitmask format for low-memory storage between layers.
#
# The matmul is done with torch.nn.functional.linear so it matches the
# reference precision regardless of dtype (bfloat16, float32, etc.).
# Triton kernels handle only the tile-based sparsification and compaction.
# ---------------------------------------------------------------------------

def sp_relu_Ax(W: Tensor, x: Tensor, BLOCK_M=128, BLOCK_N=128) -> BitsparseTensor:
    """
    y = relu(x @ W.T), then pack into a compact per-tile
    sparse representation.

    Returns a BitsparseTensor containing:
    """

    M, K = x.shape
    N = W.shape[0]

    # Do matmul as normal
    y1 = F.relu(F.linear(x, W))

    # Pack into sparse tensor
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n

    tile_counts = torch.empty(num_tiles, device=x.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=x.device, dtype=torch.uint8)
    tile_scratch = torch.empty(num_tiles * TILE_NUMEL, device=x.device, dtype=x.dtype)

    _tile_pack_kernel[(grid_m, grid_n)](
        y1, tile_counts, tile_bitmasks, tile_scratch,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
    )

    tile_prefix = torch.empty(num_tiles + 1, device=x.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    # This requires a GPU-CPU sync to know the size to allocate.
    total_nnz = tile_prefix[-1].item()
    vals = torch.empty(total_nnz, device=x.device, dtype=x.dtype)

    _compact_vals_kernel[(num_tiles,)](
        tile_scratch, tile_prefix, vals,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=8, num_stages=2,
    )

    return BitsparseTensor(
        vals, tile_bitmasks, tile_prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N,
        (M, N), x.dtype, x.device,
    )


# ---------------------------------------------------------------------------
# Layer 2: relu(sparse_A @ W2)
#
# Unpacks the compact sparse representation into batched dense row-blocks,
# then uses torch.nn.functional.linear for the matmul + ReLU.  ROW_BATCH
# controls the peak temporary memory (ROW_BATCH * K elements).
# ---------------------------------------------------------------------------

def spAx(x_sparse: BitsparseTensor, W: Tensor):
    """
    y = relu(sparse_x @ W2).

    Unpacks the sparse representation into dense row-batches, then uses
    F.linear for the matmul and ReLU.
    """
    #
    vals = x_sparse.vals
    bitmask = x_sparse.bitmask
    prefix = x_sparse.prefix
    BLOCK_M, BLOCK_N = x_sparse.BLOCK_M, x_sparse.BLOCK_N
    grid_m, grid_n = x_sparse.grid_m, x_sparse.grid_n
    M, N = x_sparse.shape
    N = W.shape[0]

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    ROW_BATCH = 2048

    out = torch.empty(M, N, device=W.device, dtype=W.dtype)

    for m_start in range(0, M, ROW_BATCH):
        m_end = min(m_start + ROW_BATCH, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M

        dense_batch = torch.empty(batch_rows, N, device=W.device, dtype=vals.dtype)

        num_row_tiles_in_batch = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles_in_batch * grid_n

        _unpack_batch_kernel[(num_tiles_in_batch,)](
            vals, bitmask, prefix,
            dense_batch,
            first_m_tile, grid_n, N, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=8, num_stages=2,
        )

        batch_out = F.linear(dense_batch, W)
        out[m_start:m_end].copy_(batch_out)

    return out



def main():
    W = torch.randn(1024, 1024, device="cuda")
    x = torch.randn(100, 1024, device="cuda")

    sp = sp_relu_Ax(W, x)
    print(sp)
    out = spAx(sp, W)
    print(out)

if __name__ == "__main__":
    main()
