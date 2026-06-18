import torch
import triton
import triton.language as tl
from torch.library import custom_op

from test_sparse import BitsparseTensor, _mask_with_bitmask_kernel, _unpack_batch_kernel


@triton.jit
def _grad_z_sparse_values_kernel(
    grad_output_ptr,
    W2_ptr,
    bitmask_ptr,
    prefix_ptr,
    vals_out_ptr,
    M, N, D, grid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_id = pid_m * grid_n + pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, D, BLOCK_K):
        k = k_start + offs_k
        go = tl.load(
            grad_output_ptr + offs_m[:, None] * D + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < D),
            other=0.0,
        )
        w2 = tl.load(
            W2_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < D) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(go, w2)

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    mask_bits = tl.reshape((bytes_2d >> bit_pos) & 1, (TILE_NUMEL,))

    ranks = tl.cumsum(mask_bits, 0) - 1
    vals = tl.reshape(acc, (TILE_NUMEL,))
    base = tl.load(prefix_ptr + tile_id)
    tl.store(vals_out_ptr + base + ranks, vals, mask=(mask_bits == 1))


def spax_direct(x_sparse: BitsparseTensor, W: torch.Tensor) -> torch.Tensor:
    vals = x_sparse.vals
    bitmask = x_sparse.bitmask
    prefix = x_sparse.prefix
    BLOCK_M, BLOCK_N = x_sparse.BLOCK_M, x_sparse.BLOCK_N
    grid_n = x_sparse.grid_n
    M, N = x_sparse.shape
    if W.shape[1] != M:
        raise ValueError(f"W.shape must be [K, {M}] for W @ sparse_x, got {tuple(W.shape)}")
    K = W.shape[0]

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    row_batch = 2048
    out = torch.zeros(K, N, device=W.device, dtype=W.dtype)

    for m_start in range(0, M, row_batch):
        m_end = min(m_start + row_batch, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=W.device, dtype=vals.dtype)
        _unpack_batch_kernel[(num_tiles_in_batch,)](
            vals, bitmask, prefix,
            x_sparse.vals_offset,
            dense_batch,
            first_m_tile, grid_n, N, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=4, num_stages=2,
        )
        out.add_(W[:, m_start:m_end] @ dense_batch)

    return out


def grad_z_sparse_direct(
    grad_output: torch.Tensor,
    W2: torch.Tensor,
    z_sparse: BitsparseTensor,
    BLOCK_K: int = 32,
) -> BitsparseTensor:
    M, N = z_sparse.shape
    if grad_output.shape[0] != M:
        raise ValueError(f"grad_output.shape must start with {M}, got {tuple(grad_output.shape)}")
    if W2.shape[1] != N:
        raise ValueError(f"W2.shape must be [D, {N}], got {tuple(W2.shape)}")
    if grad_output.shape[1] != W2.shape[0]:
        raise ValueError(
            f"grad_output.shape[1] must equal W2.shape[0], got "
            f"{grad_output.shape[1]} and {W2.shape[0]}"
        )

    nnz = int(z_sparse.prefix[-1].item())
    vals = torch.empty(nnz, device=grad_output.device, dtype=grad_output.dtype)
    vals_offset = torch.zeros(1, device=grad_output.device, dtype=torch.int32)

    TILE_NUMEL = z_sparse.BLOCK_M * z_sparse.BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    _grad_z_sparse_values_kernel[(z_sparse.grid_m, z_sparse.grid_n)](
        grad_output, W2,
        z_sparse.bitmask, z_sparse.prefix,
        vals,
        M, N, grad_output.shape[1], z_sparse.grid_n,
        BLOCK_M=z_sparse.BLOCK_M, BLOCK_N=z_sparse.BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=3,
    )

    return BitsparseTensor(
        vals, z_sparse.bitmask, z_sparse.prefix, vals_offset,
        z_sparse.grid_m, z_sparse.grid_n,
        z_sparse.BLOCK_M, z_sparse.BLOCK_N,
        z_sparse.shape,
    )


def sparse_dense_right_direct(
    x_sparse: BitsparseTensor,
    W: torch.Tensor,
    row_batch: int = 2048,
) -> torch.Tensor:
    vals = x_sparse.vals
    bitmask = x_sparse.bitmask
    prefix = x_sparse.prefix
    BLOCK_M, BLOCK_N = x_sparse.BLOCK_M, x_sparse.BLOCK_N
    grid_n = x_sparse.grid_n
    M, N = x_sparse.shape
    if W.shape[0] != N:
        raise ValueError(f"W.shape must be [{N}, K] for sparse_x @ W, got {tuple(W.shape)}")
    K = W.shape[1]

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    out = torch.empty(M, K, device=W.device, dtype=W.dtype)

    for m_start in range(0, M, row_batch):
        m_end = min(m_start + row_batch, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M
        num_row_tiles = (batch_rows + BLOCK_M - 1) // BLOCK_M
        num_tiles_in_batch = num_row_tiles * grid_n

        dense_batch = torch.empty(batch_rows, N, device=W.device, dtype=vals.dtype)
        _unpack_batch_kernel[(num_tiles_in_batch,)](
            vals, bitmask, prefix,
            x_sparse.vals_offset,
            dense_batch,
            first_m_tile, grid_n, N, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=4, num_stages=2,
        )
        torch.mm(dense_batch, W, out=out[m_start:m_end])

    return out


def ffn_backward_direct(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    z_sparse: BitsparseTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    grad_W2 = spax_direct(z_sparse, grad_output.T)

    grad_z = grad_output @ W2

    _mask_with_bitmask_kernel[(z_sparse.grid_m, z_sparse.grid_n)](
        grad_z, z_sparse.bitmask,
        z_sparse.shape[0], z_sparse.shape[1],
        BLOCK_M=z_sparse.BLOCK_M, BLOCK_N=z_sparse.BLOCK_N,
        TILE_BYTES=z_sparse.BLOCK_M * z_sparse.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )

    grad_x = grad_z @ W1
    grad_W1 = grad_z.T @ x#.contiguous()
    return grad_x, grad_W1, grad_W2


def ffn_backward_sparse_grad_z(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    z_sparse: BitsparseTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_W2 = spax_direct(z_sparse, grad_output.T)

    grad_z_sparse = grad_z_sparse_direct(grad_output, W2, z_sparse)

    grad_x = sparse_dense_right_direct(grad_z_sparse, W1)
    grad_W1 = spax_direct(grad_z_sparse, x.T).T
    return grad_x, grad_W1, grad_W2



BACKWARD_METHODS = {
    "direct": ffn_backward_direct,
    "sparse_grad_z": ffn_backward_sparse_grad_z,
}


SPAX_METHODS = {
    "direct": spax_direct,
    "sparse_dense_right": sparse_dense_right_direct,
}
