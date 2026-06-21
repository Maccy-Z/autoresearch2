import torch
from torch import Tensor

from sparse_kernels import _unpack_batch_kernel, _mask_with_bitmask_kernel
from sparse_utils import BitsparseTensor


def AspB(A: Tensor, B_sparse: BitsparseTensor) -> Tensor:
    vals = B_sparse.vals
    bitmask = B_sparse.bitmask
    prefix = B_sparse.prefix
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_m, grid_n = B_sparse.grid_m, B_sparse.grid_n
    M, N = B_sparse.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    num_tiles = grid_m * grid_n

    dense = torch.empty(M, N, device=A.device, dtype=vals.dtype)
    _unpack_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8, num_stages=2,
    )
    return A @ dense


def FFN_backward(ctx, grad_output: Tensor):
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, z)

    grad_z = grad_output @ W2
    _mask_with_bitmask_kernel[(z.grid_m, z.grid_n)](
        grad_z, z.bitmask,
        z.shape[0], z.shape[1],
        BLOCK_M=z.BLOCK_M, BLOCK_N=z.BLOCK_N,
        TILE_BYTES=z.BLOCK_M * z.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )
    del z

    if needs_x:
        grad_x = grad_z @ W1
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None
