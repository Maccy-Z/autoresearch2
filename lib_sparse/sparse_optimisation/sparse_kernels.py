"""
Per-tile compressed sparse format and packing routines.

A dense 2D tensor X ∈ R^{M×N} is partitioned into a grid of tiles, each
of shape [BLOCK_M × BLOCK_N].  Every tile is independently compressed:

  bitmask  — uint8 packed bitmask (8 bits per byte), TILE_BYTES bytes/tile.
             Row-major within the tile, so bit at flat offset f lives in
             byte f//8 at bit position f%8.  A set bit (1) means the
             element is nonzero after ReLU: X[i,j] > 0.

  vals     — a single compact 1D array containing all nonzero values
             across all tiles, concatenated in grid-major order.

  prefix   — int32 prefix sum of per-tile nonzero counts: prefix[t] is
             the starting offset of tile t's values inside vals.
             prefix[num_tiles] equals the total number of nonzero values.

Forward packing pipeline: _tile_pack_kernel → torch.cumsum → _compact_vals_kernel
Backward unpack:          _unpack_batch_kernel  (reconstructs dense from sparse)
Backward gradient mask:   _mask_with_bitmask_kernel  (∂L/∂z  ⊙  (z > 0))
Backward sparse grad_z:   _grad_z_sparse_values_kernel  (∂L/∂z stored sparsely)
"""

import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 1 — _tile_pack_kernel
#   Computes:  bitmask[t] = pack(X_tile > 0)    ∀ tile t
#              counts[t]  = ||X_tile > 0||₀     (number of positive entries)
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _tile_pack_kernel(
    dense_ptr,          # pointer to dense input X ∈ R^{M×N}
    tile_counts_ptr,    # output: int32[n_tiles] nonzero counts per tile
    tile_bitmasks_ptr,  # output: uint8[n_tiles × TILE_BYTES] packed bitmasks
    M, N,               # dimensions of X
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,    # = BLOCK_M × BLOCK_N
    TILE_BYTES: tl.constexpr,    # = TILE_NUMEL // 8
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.num_programs(1) + pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    tile = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_flat = tl.reshape(tile, (TILE_NUMEL,))
    nz = (tile_flat > 0.0)                      # boolean: ReLU mask for this tile

    # Pack 8 bools → 1 uint8: reshape to [TILE_BYTES, 8], shift each bit
    # position by j ∈ {0..7}, sum across bit axis.
    #   bytes_val[b] = Σ_{j=0}^{7} nz[8b + j] · 2^j
    nz_reshaped = tl.reshape(nz, (TILE_BYTES, 8))
    bit_weights = tl.arange(0, 8)[None, :]
    bytes_val = tl.sum(nz_reshaped.to(tl.int32) << bit_weights, 1).to(tl.uint8)
    tl.store(tile_bitmasks_ptr + pid * TILE_BYTES + tl.arange(0, TILE_BYTES), bytes_val)

    nnz = tl.sum(nz.to(tl.int32))
    tl.store(tile_counts_ptr + pid, nnz)


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 2 — _compact_vals_kernel
#   Given prefix[t] = Σ_{i=0}^{t-1} count[i] (exclusive prefix sum),
#   scatters tile t's nonzero values into a global compact buffer:
#     vals[prefix[t] : prefix[t+1]] = {X[p,q] : (p,q) ∈ tile t, X[p,q] > 0}
#   Values within each tile are stored in row-major order.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _compact_vals_kernel(
    dense_ptr,          # input:  dense X ∈ R^{M×N}
    tile_prefix_ptr,    # input:  int32[n_tiles+1] exclusive prefix sum of counts
    vals_out_ptr,       # output: compact bf16 buffer for nonzero values
    layer_offset_ptr,   # input:  int32[1] global offset where this layer starts
    M, N, grid_n,       # dimensions and tile grid
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.load(layer_offset_ptr)
    base = tl.load(tile_prefix_ptr + pid) + offset   # absolute position in vals

    tile_m = pid // grid_n
    tile_n = pid % grid_n

    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v_2d = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    v = tl.reshape(v_2d, (TILE_NUMEL,))

    nz = (v > 0.0).to(tl.int32)

    # rank[i] = number of nonzero entries before position i within this tile.
    # Used as the offset from 'base' to write the i-th nonzero value.
    ranks = tl.cumsum(nz, 0) - 1
    tl.store(vals_out_ptr + base + ranks, v, mask=(nz == 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 3 — _unpack_batch_kernel
#   Reconstructs dense tiles from the sparse representation.
#   For each tile t in a batch of rows:
#     D_tile = 0
#     for each nonzero position i in tile t (from bitmask[t]):
#         D_tile[i] = vals[prefix[t] + rank[i]]
#   This computes: D_rowslice = gather(vals, bitmask, prefix)
#   where D_rowslice ∈ R^{batch_rows × K} is written into dense_ptr.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _unpack_batch_kernel(
    vals_ptr,           # input:  compact nonzero values (bf16)
    bitmask_ptr,        # input:  uint8 packed bitmasks
    prefix_ptr,         # input:  int32[n_tiles+1] exclusive prefix sum
    layer_offset_ptr,   # input:  int32[1] global offset for this layer
    dense_ptr,          # output: dense bf16 buffer of shape [batch_rows, K]
    first_m_tile,       # first row-tile in this batch
    grid_n_sparse, K,   # tile grid width and dense row stride
    batch_rows,         # number of rows in this output batch
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse       # which row-tile within batch
    k_tile = pid % grid_n_sparse                   # which column-tile

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    # Unpack uint8 bitmask → bool mask of length TILE_NUMEL.
    #   mask[i] = (bitmask[i//8] >> (i%8)) & 1
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    # rank[i] = cumulative count of set bits before position i
    # The nonzero values for this tile occupy vals[base : base + count[tile]],
    # and the i-th nonzero belongs at vals[base + rank[i]].
    offset = tl.load(layer_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
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
    vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    offset = tl.load(vals_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)
    rdtype = r.dtype
    r = r.to(tl.float32)
    z = RELU2_SCALE * r * r
    z = z.to(rdtype)
    z_2d = tl.reshape(z, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, z_2d, mask=(offs_m < batch_rows) & (offs_k < K))


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 4 — _mask_with_bitmask_kernel
#   In-place masks a dense gradient matrix ∂L/∂Z using the stored bitmask.
#   Computes:  ∂L/∂Z  ←  ∂L/∂Z  ⊙  (Z > 0)
#   where ⊙ is element-wise multiplication.
#   This applies the ReLU backward pass: grad is zero where the original
#   pre-activation was ≤ 0, unchanged otherwise.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _mask_with_bitmask_kernel(
    grad_ptr,           # input/output: dense gradient ∂L/∂Z ∈ R^{M×N} (in-place)
    bitmask_ptr,        # input:  uint8 packed bitmasks
    M, N,               # dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    gz = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = tl.reshape((bytes_2d >> bit_pos) & 1, (BLOCK_M, BLOCK_N))

    # Element-wise:  gz[p,q] = 0 if Z[p,q] ≤ 0, else gz[p,q]
    masked = tl.where(bits != 0, gz, 0.0)
    tl.store(grad_ptr + offs, masked, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _relu2_grad_sparse_values_kernel(
    grad_output_ptr,
    W2_ptr,
    vals_ptr,
    bitmask_ptr,
    prefix_ptr,
    vals_offset_ptr,
    vals_out_ptr,
    M, N, grid_n,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
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
    mask_bits = tl.reshape((bytes_2d >> tl.arange(0, 8)[None, :]) & 1, (TILE_NUMEL,))

    offset = tl.load(vals_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0).to(tl.float32)
    grad_flat = tl.reshape(acc, (TILE_NUMEL,)) * (2.0 * RELU2_SCALE * r)
    tl.store(vals_out_ptr + base + ranks, grad_flat, mask=(mask_bits == 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel 5 — _grad_z_sparse_values_kernel
#   Computes sparse grad_z = (∂L/∂Y) @ W₂, masked by the ReLU activation
#   pattern, and writes the result into the existing sparse vals buffer.
#
#   For each tile t:
#     acc = 0
#     for k_block in 0..D step BLOCK_K:
#         acc += ∂L/∂Y[row_slice, k_slice] @ W₂[k_slice, col_slice]
#     for each nonzero position i in tile t:
#         vals[tile_start + rank[i]] = acc[i]
#
#   This fuses three operations into one kernel:
#     grad_z = (∂L/∂Y @ W₂) ⊙ (Z > 0)   with output kept sparse.
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _grad_z_sparse_values_kernel(
    grad_output_ptr,    # input:  ∂L/∂Y ∈ R^{M×D}
    W2_ptr,             # input:  W₂ ∈ R^{D×N}
    bitmask_ptr,        # input:  uint8 packed bitmasks
    prefix_ptr,         # input:  int32[n_tiles+1] exclusive prefix sum
    vals_offset_ptr,    # input:  int32[1] global offset for this layer
    vals_out_ptr,       # output: compact sparse buffer (overwrites forward vals)
    M, N, grid_n,
    D: tl.constexpr,                                # inner dimension of matmul
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,                          # inner-loop tile size
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    tile_id = pid_m * grid_n + pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Blocked matrix multiplication:  acc += ∂L/∂Y[BM,D] @ W₂[D,BN]
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

    # Mask acc by bitmask and scatter into compact buffer.
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    mask_bits = tl.reshape((bytes_2d >> bit_pos) & 1, (TILE_NUMEL,))

    ranks = tl.cumsum(mask_bits, 0) - 1
    vals = tl.reshape(acc, (TILE_NUMEL,))
    base = tl.load(vals_offset_ptr) + tl.load(prefix_ptr + tile_id)
    tl.store(vals_out_ptr + base + ranks, vals, mask=(mask_bits == 1))

