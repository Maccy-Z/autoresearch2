import torch
from torch import Tensor

from sparse_kernels import _unpack_batch_kernel, _mask_with_bitmask_kernel, \
    _grad_z_sparse_values_kernel
from sparse_utils import BitsparseTensorBuffer


def AspB_block(A: Tensor, B_sparse: BitsparseTensorBuffer, row_batch=20000) -> Tensor:
    """ y = A @ B_sparse. Done blockwise to reduce peak vram usage.
        A.shape = [K, M]
        B.shape = [M, N]
    """
    vals = B_sparse.vals
    bitmask = B_sparse.bitmask
    prefix = B_sparse.prefix
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_n = B_sparse.grid_n
    M, N = B_sparse.shape
    K = A.shape[0]

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    out = torch.zeros(K, N, device=A.device, dtype=A.dtype)

    row_tiles_per_batch = max(1, row_batch // BLOCK_M)
    for first_m_tile in range(0, B_sparse.grid_m, row_tiles_per_batch):
        m_start = first_m_tile * BLOCK_M
        m_end = min(m_start + row_tiles_per_batch * BLOCK_M, M)
        batch_rows = m_end - m_start
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=A.device, dtype=vals.dtype)
        _unpack_batch_kernel[(num_tiles_in_batch,)](
            vals, bitmask, prefix,
            B_sparse.vals_offset,
            dense_batch,
            first_m_tile, grid_n, N, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=8, num_stages=2,
        )
        A_batch = A[:, m_start:m_end]
        out.add_(A_batch @ dense_batch)

    return out


def AspB(A: Tensor, B_sparse: BitsparseTensorBuffer) -> Tensor:
    """
    y = A @ B_sparse.
    A.shape = [K, M]
    B.shape = [M, N]
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

    _unpack_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix,
        B_sparse.vals_offset,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8, num_stages=2,
    )

    return A @ dense


def spAB(A_sparse: BitsparseTensorBuffer, B: Tensor, row_batch: int = 2048) -> Tensor:
    """ y = A_sparse @ B
        A.shape = [M, N]
        B.shape = [N, K]
    """

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
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=B.device, dtype=vals.dtype)
        _unpack_batch_kernel[(num_tiles_in_batch,)](
            vals, bitmask, prefix,
            A_sparse.vals_offset,
            dense_batch,
            first_m_tile, grid_n, N, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=8, num_stages=2,
        )
        torch.mm(dense_batch, B, out=out[m_start:m_end])
    return out


def grad_z_sparse_inplace(
    grad_output: Tensor, W2: Tensor, z_sparse: BitsparseTensorBuffer,
    BLOCK_K: int = 32,
) -> BitsparseTensorBuffer:
    """ Combine grad_z = grad_output @ W2,
                grad_z = grad_z * (z>0)
                grad_z = sparse(grad_z)
        Overwrite z_sparse values.
    """

    M, N = z_sparse.shape

    TILE_NUMEL = z_sparse.BLOCK_M * z_sparse.BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    _grad_z_sparse_values_kernel[(z_sparse.grid_m, z_sparse.grid_n)](
        grad_output, W2,
        z_sparse.bitmask, z_sparse.prefix, z_sparse.vals_offset,
        z_sparse.vals,
        M, N, z_sparse.grid_n,
        D=grad_output.shape[1],
        BLOCK_M=z_sparse.BLOCK_M, BLOCK_N=z_sparse.BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8, num_stages=3,
    )
    return z_sparse


def FFN_backward(ctx, grad_output: Tensor):
    """Compute FFN gradients."""
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, z_sparse)

    grad_z = grad_output @ W2
    # grad_z <- grad_z * relu_grad
    _mask_with_bitmask_kernel[(z_sparse.grid_m, z_sparse.grid_n)](
        grad_z, z_sparse.bitmask,
        z_sparse.shape[0], z_sparse.shape[1],
        BLOCK_M=z_sparse.BLOCK_M, BLOCK_N=z_sparse.BLOCK_N,
        TILE_BYTES=z_sparse.BLOCK_M * z_sparse.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )

    del z_sparse
    if needs_x:
        grad_x = grad_z @ W1
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_backward_sparse(ctx, grad_output: Tensor):
    """Compute FFN gradients while keeping grad_z in the existing bit-sparse storage."""
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, z_sparse)
    grad_z_sparse = grad_z_sparse_inplace(grad_output, W2, z_sparse)
    del z_sparse
    grad_W1 = AspB_block(x.T, grad_z_sparse).T

    if needs_x:
        grad_x = spAB(grad_z_sparse, W1)
    else:
        grad_x = None
    return grad_x, grad_W1, grad_W2, None


def FFN3_backward(ctx, grad_output: Tensor):
    """Compute 3-layer ReLU FFN gradients using sparse saved activations."""
    x, W1, W2, W3 = ctx.saved_tensors
    z1 = ctx.z1_sparse
    z2 = ctx.z2_sparse
    ctx.z1_sparse = None
    ctx.z2_sparse = None
    needs_x = ctx.needs_input_grad[0]


    grad_z2 = grad_output @ W3
    grad_W3 = AspB(grad_output.T, z2)

    _mask_with_bitmask_kernel[(z2.grid_m, z2.grid_n)](
        grad_z2, z2.bitmask,
        z2.shape[0], z2.shape[1],
        BLOCK_M=z2.BLOCK_M, BLOCK_N=z2.BLOCK_N,
        TILE_BYTES=z2.BLOCK_M * z2.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )
    del z2

    grad_W2 = AspB(grad_z2.T, z1)
    grad_z1 = grad_z2 @ W2

    del grad_z2
    _mask_with_bitmask_kernel[(z1.grid_m, z1.grid_n)](
        grad_z1, z1.bitmask,
        z1.shape[0], z1.shape[1],
        BLOCK_M=z1.BLOCK_M, BLOCK_N=z1.BLOCK_N,
        TILE_BYTES=z1.BLOCK_M * z1.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )
    del z1

    grad_x = grad_z1 @ W1 if needs_x else None
    grad_W1 = grad_z1.T @ x
    del grad_z1

    return grad_x, grad_W1, grad_W2, grad_W3, None
