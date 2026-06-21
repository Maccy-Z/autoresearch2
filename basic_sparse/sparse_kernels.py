"""
Per-tile compressed sparse format — simplified per-layer allocation.

A dense 2D tensor X in R^{MxN} is partitioned into tiles [BLOCK_M x BLOCK_N].
Each tile is compressed into:
  bitmask — uint8 packed, TILE_BYTES bytes/tile
  vals    — compact nonzero values, stored contiguously per tile
  prefix  — int32 exclusive prefix sum: prefix[t] = start of tile t in vals
"""

import triton
import triton.language as tl


@triton.jit
def _tile_pack_kernel(
    dense_ptr,
    tile_counts_ptr,
    tile_bitmasks_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """Pack each ``BLOCK_M x BLOCK_N`` dense tile into bitmasks and nnz counts."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.num_programs(1) + pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    tile = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_flat = tl.reshape(tile, (TILE_NUMEL,))
    nz = (tile_flat > 0.0)
    nz_int = nz.to(tl.int32)

    nz_2d = tl.reshape(nz_int, (TILE_BYTES, 8))
    weights = tl.arange(0, 8)[None, :]
    packed = tl.sum(nz_2d << weights, 1).to(tl.uint8)
    tl.store(tile_bitmasks_ptr + pid * TILE_BYTES + tl.arange(0, TILE_BYTES), packed)

    nnz = tl.sum(nz_int)
    tl.store(tile_counts_ptr + pid, nnz)


@triton.jit
def _compact_vals_kernel(
    dense_ptr,
    tile_prefix_ptr,
    vals_out_ptr,
    M, N, grid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    """Write positive dense values into ``vals_out`` using per-tile prefix offsets."""
    pid = tl.program_id(0)
    base = tl.load(tile_prefix_ptr + pid)

    tile_m = pid // grid_n
    tile_n = pid % grid_n

    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    v = tl.reshape(v, (TILE_NUMEL,))
    nz = (v > 0.0).to(tl.int32)

    ranks = tl.cumsum(nz, 0) - 1
    tl.store(vals_out_ptr + base + ranks, v, mask=(nz == 1))


@triton.jit
def _unpack_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """Unpack sparse tiles back into a dense ``batch_rows x K`` matrix slice."""
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bm = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bm_2d = tl.reshape(bm, (TILE_BYTES, 1))
    bits = (bm_2d >> tl.arange(0, 8)[None, :]) & 1
    mask_bits = tl.reshape(bits, (TILE_NUMEL,))

    base = tl.load(prefix_ptr + tile_id)
    ranks = tl.cumsum(mask_bits, 0) - 1
    v = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)
    v_2d = tl.reshape(v, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, v_2d, mask=(offs_m < batch_rows) & (offs_k < K))


@triton.jit
def _unpack_relu2_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
):
    """Unpack stored ``r = relu(a)`` tiles as ``k * r^2`` into dense output."""
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bm = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bm_2d = tl.reshape(bm, (TILE_BYTES, 1))
    bits = (bm_2d >> tl.arange(0, 8)[None, :]) & 1
    mask_bits = tl.reshape(bits, (TILE_NUMEL,))

    base = tl.load(prefix_ptr + tile_id)
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)
    z = RELU2_SCALE * r * r
    z_2d = tl.reshape(z, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, z_2d, mask=(offs_m < batch_rows) & (offs_k < K))


@triton.jit
def _mask_with_bitmask_kernel(
    grad_ptr, bitmask_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    """Apply the saved ReLU mask in-place: ``grad = grad * bitmask``."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    gz = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bm = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bm_2d = tl.reshape(bm, (TILE_BYTES, 1))
    bits = tl.reshape((bm_2d >> tl.arange(0, 8)[None, :]) & 1, (BLOCK_M, BLOCK_N))

    masked = tl.where(bits != 0, gz, 0.0)
    tl.store(grad_ptr + offs, masked, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _relu2_grad_with_sparse_kernel(
    grad_ptr, vals_ptr, bitmask_ptr, prefix_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
):
    """Apply derivative for ``z = k * r^2`` from stored ``r = relu(a)``."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    grad = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bm = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bm_2d = tl.reshape(bm, (TILE_BYTES, 1))
    mask_bits = tl.reshape((bm_2d >> tl.arange(0, 8)[None, :]) & 1, (TILE_NUMEL,))

    base = tl.load(prefix_ptr + tile_id)
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0).to(tl.float32)
    scale = 2.0 * RELU2_SCALE * r
    scale_2d = tl.reshape(scale, (BLOCK_M, BLOCK_N))
    bits_2d = tl.reshape(mask_bits, (BLOCK_M, BLOCK_N))

    grad_preact = tl.where(bits_2d != 0, grad * scale_2d, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _relu2_grad_sparse_values_kernel(
    grad_output_ptr,
    W2_ptr,
    vals_ptr,
    bitmask_ptr,
    prefix_ptr,
    vals_out_ptr,
    M, N, grid_n,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
):
    """Compute sparse ``dpreact = (grad_output @ W2) * 2*k*r`` values."""
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
    bm = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bm_2d = tl.reshape(bm, (TILE_BYTES, 1))
    mask_bits = tl.reshape((bm_2d >> tl.arange(0, 8)[None, :]) & 1, (TILE_NUMEL,))

    base = tl.load(prefix_ptr + tile_id)
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0).to(tl.float32)
    grad_flat = tl.reshape(acc, (TILE_NUMEL,)) * (2.0 * RELU2_SCALE * r)
    tl.store(vals_out_ptr + base + ranks, grad_flat, mask=(mask_bits == 1))
