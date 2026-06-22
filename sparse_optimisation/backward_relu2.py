import torch
from torch import Tensor

from forward_relu2 import RELU2_SCALE
from backward_method import AspB_block, spAB, grad_z_sparse_inplace
from sparse_kernels import (
    _relu2_grad_sparse_values_kernel,
    _relu2_grad_with_sparse_kernel,
    _unpack_batch_kernel,
    _unpack_relu2_batch_kernel,
)


def AspRelu2B(A: Tensor, B_sparse) -> Tensor:
    vals = B_sparse.vals
    bitmask = B_sparse.bitmask
    prefix = B_sparse.prefix
    vals_offset = B_sparse.vals_offset
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_m, grid_n = B_sparse.grid_m, B_sparse.grid_n
    M, N = B_sparse.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    num_tiles = grid_m * grid_n

    dense = torch.empty(M, N, device=A.device, dtype=vals.dtype)
    _unpack_relu2_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix, vals_offset,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=3,
    )
    return A @ dense


def FFN_relu2_backward(ctx, grad_output: Tensor):
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, z)

    grad_z = grad_output @ W2
    _relu2_grad_with_sparse_kernel[(z.grid_m, z.grid_n)](
        grad_z, z.vals, z.bitmask, z.prefix, z.vals_offset,
        z.shape[0], z.shape[1],
        BLOCK_M=z.BLOCK_M, BLOCK_N=z.BLOCK_N,
        TILE_NUMEL=z.BLOCK_M * z.BLOCK_N,
        TILE_BYTES=z.BLOCK_M * z.BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )
    del z

    grad_x = grad_z @ W1 if needs_x else None
    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, z_sparse)
    grad_z_sparse = grad_z_sparse_inplace(grad_output, W2, z_sparse)
    del z_sparse
    grad_W1 = AspB_block(x.T, grad_z_sparse).T

    grad_x = spAB(grad_z_sparse, W1) if needs_x else None
    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_3_backward(ctx, grad_output: Tensor):
    x, W1, W2, W3 = ctx.saved_tensors
    z1 = ctx.z1_sparse
    z2 = ctx.z2_sparse
    ctx.z1_sparse = None
    ctx.z2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = AspRelu2B(grad_output.T, z2)

    grad_z2 = grad_output @ W3
    _relu2_grad_with_sparse_kernel[(z2.grid_m, z2.grid_n)](
        grad_z2, z2.vals, z2.bitmask, z2.prefix, z2.vals_offset,
        z2.shape[0], z2.shape[1],
        BLOCK_M=z2.BLOCK_M, BLOCK_N=z2.BLOCK_N,
        TILE_NUMEL=z2.BLOCK_M * z2.BLOCK_N,
        TILE_BYTES=z2.BLOCK_M * z2.BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )
    grad_W2 = AspRelu2B(grad_z2.T, z1)
    del z2

    grad_z1 = grad_z2 @ W2
    del grad_z2, W3
    _relu2_grad_with_sparse_kernel[(z1.grid_m, z1.grid_n)](
        grad_z1, z1.vals, z1.bitmask, z1.prefix, z1.vals_offset,
        z1.shape[0], z1.shape[1],
        BLOCK_M=z1.BLOCK_M, BLOCK_N=z1.BLOCK_N,
        TILE_NUMEL=z1.BLOCK_M * z1.BLOCK_N,
        TILE_BYTES=z1.BLOCK_M * z1.BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )
    del z1
    grad_x = grad_z1 @ W1 if needs_x else None
    grad_W1 = grad_z1.T @ x
    del grad_z1, x, W1, W2
    return grad_x, grad_W1, grad_W2, grad_W3, None
