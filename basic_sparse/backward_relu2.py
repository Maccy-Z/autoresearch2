import torch
from torch import Tensor

from forward_relu2 import RELU2_SCALE
from sparse_kernels import (
    _relu2_grad_sparse_values_kernel,
    _relu2_grad_with_sparse_kernel,
    _unpack_batch_kernel,
    _unpack_relu2_batch_kernel,
)


def AspRelu2B(A: Tensor, B_sparse) -> Tensor:
    """Compute ``A @ (k * B^2)`` where sparse ``B`` stores ``relu(preact)``.

    Shapes: ``A[P, M]`` and sparse ``B[M, N]`` produce ``out[P, N]``.
    The current implementation unpacks ``k * B^2`` to dense before matmul.
    """
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
    _unpack_relu2_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=3,
    )
    return A @ dense


def spRelu2GradB(grad_output: Tensor, W2: Tensor, r_sparse, BLOCK_K: int = 64):
    """Overwrite sparse ``r`` values with ``dpreact = grad_output @ W2 * 2*k*r``."""
    M, N = r_sparse.shape
    TILE_NUMEL = r_sparse.BLOCK_M * r_sparse.BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    _relu2_grad_sparse_values_kernel[(r_sparse.grid_m, r_sparse.grid_n)](
        grad_output, W2,
        r_sparse.vals, r_sparse.bitmask, r_sparse.prefix,
        r_sparse.vals,
        M, N, r_sparse.grid_n,
        D=grad_output.shape[1],
        BLOCK_M=r_sparse.BLOCK_M, BLOCK_N=r_sparse.BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )
    return r_sparse


def spAB(A_sparse, B: Tensor, row_batch: int = 1536) -> Tensor:
    """Compute ``A_sparse @ B`` by unpacking row batches of sparse ``A``."""
    vals = A_sparse.vals
    bitmask = A_sparse.bitmask
    prefix = A_sparse.prefix
    BLOCK_M, BLOCK_N = A_sparse.BLOCK_M, A_sparse.BLOCK_N
    grid_n = A_sparse.grid_n
    M, N = A_sparse.shape
    K = B.shape[1]

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    out = torch.empty(M, K, device=B.device, dtype=B.dtype)

    for m_start in range(0, M, row_batch):
        m_end = min(m_start + row_batch, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        dense_batch = torch.empty(batch_rows, N, device=B.device, dtype=vals.dtype)
        _unpack_batch_kernel[(num_row_tiles * grid_n,)](
            vals, bitmask, prefix,
            dense_batch,
            first_m_tile, grid_n, N, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=8, num_stages=2,
        )
        torch.mm(dense_batch, B, out=out[m_start:m_end])
    return out


def AspB_block(A: Tensor, B_sparse, row_batch: int = 1536) -> Tensor:
    """Compute ``A @ B_sparse`` by unpacking row batches of sparse ``B``."""
    vals = B_sparse.vals
    bitmask = B_sparse.bitmask
    prefix = B_sparse.prefix
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_n = B_sparse.grid_n
    M, N = B_sparse.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    out = torch.zeros(A.shape[0], N, device=A.device, dtype=A.dtype)

    for m_start in range(0, M, row_batch):
        m_end = min(m_start + row_batch, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        dense_batch = torch.empty(batch_rows, N, device=A.device, dtype=vals.dtype)
        _unpack_batch_kernel[(num_row_tiles * grid_n,)](
            vals, bitmask, prefix,
            dense_batch,
            first_m_tile, grid_n, N, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=8, num_stages=2,
        )
        out.add_(A[:, m_start:m_end] @ dense_batch)
    return out


def FFN_relu2_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``.

    Shapes: ``grad_output[B, D]``, ``x[B, D]``, ``W1[H, D]``, ``W2[D, H]``.
    With sparse ``r = relu(preact)``, gradients are
    ``dW2 = grad_output.T @ (k * r^2)``, ``dz = grad_output @ W2``,
    ``dpreact = dz * 2 * k * r`` for active
    entries, ``dx = dpreact @ W1``, and ``dW1 = dpreact.T @ x``.
    """
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, z)

    grad_z = grad_output @ W2
    _relu2_grad_with_sparse_kernel[(z.grid_m, z.grid_n)](
        grad_z, z.vals, z.bitmask, z.prefix,
        z.shape[0], z.shape[1],
        BLOCK_M=z.BLOCK_M, BLOCK_N=z.BLOCK_N,
        TILE_NUMEL=z.BLOCK_M * z.BLOCK_N,
        TILE_BYTES=z.BLOCK_M * z.BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )
    del z

    if needs_x:
        grad_x = grad_z @ W1
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2


def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
    """Backward keeping ``dpreact`` in sparse storage to reduce peak memory."""
    x, W1, W2 = ctx.saved_tensors
    r_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, r_sparse)
    grad_sparse = spRelu2GradB(grad_output, W2, r_sparse)
    grad_W1 = AspB_block(x.T, grad_sparse).T

    if needs_x:
        grad_x = spAB(grad_sparse, W1)
    else:
        grad_x = None
    return grad_x, grad_W1, grad_W2


def FFN_relu2_3_backward(ctx, grad_output: Tensor):
    """Backward for ``z1 = k*relu(a1)^2``, ``z2 = k*relu(a2)^2`` using sparse caches."""
    x, W1, W2, W3 = ctx.saved_tensors
    z1 = ctx.z1_sparse
    z2 = ctx.z2_sparse
    ctx.z1_sparse = None
    ctx.z2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = AspRelu2B(grad_output.T, z2)

    grad_z2 = grad_output @ W3
    _relu2_grad_with_sparse_kernel[(z2.grid_m, z2.grid_n)](
        grad_z2, z2.vals, z2.bitmask, z2.prefix,
        z2.shape[0], z2.shape[1],
        BLOCK_M=z2.BLOCK_M, BLOCK_N=z2.BLOCK_N,
        TILE_NUMEL=z2.BLOCK_M * z2.BLOCK_N,
        TILE_BYTES=z2.BLOCK_M * z2.BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )

    grad_preact2 = grad_z2
    grad_W2 = AspRelu2B(grad_preact2.T, z1)

    grad_z1 = grad_preact2 @ W2
    _relu2_grad_with_sparse_kernel[(z1.grid_m, z1.grid_n)](
        grad_z1, z1.vals, z1.bitmask, z1.prefix,
        z1.shape[0], z1.shape[1],
        BLOCK_M=z1.BLOCK_M, BLOCK_N=z1.BLOCK_N,
        TILE_NUMEL=z1.BLOCK_M * z1.BLOCK_N,
        TILE_BYTES=z1.BLOCK_M * z1.BLOCK_N // 8,
        RELU2_SCALE=RELU2_SCALE,
        num_warps=8, num_stages=2,
    )

    if needs_x:
        grad_x = grad_z1 @ W1
    else:
        grad_x = None

    grad_W1 = grad_z1.T @ x
    return grad_x, grad_W1, grad_W2, grad_W3
