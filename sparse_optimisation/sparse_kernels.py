"""
Per-tile compressed sparse format and packing routines.

A dense 2D tensor [M, N] is partitioned into a grid of tiles, each
[BLOCK_M × BLOCK_N].  Every tile is independently compressed:

  bitmask  — uint8 packed bitmask (8 bits per byte), TILE_BYTES bytes/tile.
             Row-major within the tile, so bit at flat offset f lives in
             byte f//8 at bit position f%8.  A set bit (1) means the
             element is nonzero.

  vals     — a single compact 1D array containing all nonzero values
             across all tiles, concatenated in grid-major order.

  prefix   — int32 prefix sum of per-tile nonzero counts: prefix[i] is
             the starting offset of tile i's values inside vals.
             prefix[num_tiles] equals len(vals).
"""

import triton
import triton.language as tl
from triton import language as tl


# ---------------------------------------------------------------------------
# Kernel 1 — split dense into tiles and produce per-tile bitmask + counts
# ---------------------------------------------------------------------------
@triton.jit
def _tile_pack_kernel(
    dense_ptr,
    tile_counts_ptr,
    tile_bitmasks_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.num_programs(1) + pid_n

    # load one [BLOCK_M × BLOCK_N] tile from the dense matrix
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    tile = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    # flatten to 1D (row-major) for bitmask packing
    tile_flat = tl.reshape(tile, (TILE_NUMEL,))
    nz = (tile_flat > 0.0)

    # pack bits → bytes: 8 bools per uint8, little-endian within each byte
    nz_reshaped = tl.reshape(nz, (TILE_BYTES, 8))
    bit_weights = tl.arange(0, 8)[None, :]
    bytes_val = tl.sum(nz_reshaped.to(tl.int32) << bit_weights, 1).to(tl.uint8)
    tl.store(tile_bitmasks_ptr + pid * TILE_BYTES + tl.arange(0, TILE_BYTES), bytes_val)

    # per-tile nonzero count (used for prefix sum in the next stage)
    nnz = tl.sum(nz.to(tl.int32))
    tl.store(tile_counts_ptr + pid, nnz)


@triton.jit
def _tile_pack_int8_kernel(
    dense_ptr,
    tile_bitmasks_ptr,
    tile_vals_ptr,
    tile_scales_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.num_programs(1) + pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    tile = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_flat = tl.reshape(tile, (TILE_NUMEL,))
    nz = (tile_flat > 0.0)

    nz_reshaped = tl.reshape(nz, (TILE_BYTES, 8))
    bit_weights = tl.arange(0, 8)[None, :]
    bytes_val = tl.sum(nz_reshaped.to(tl.int32) << bit_weights, 1).to(tl.uint8)
    tl.store(tile_bitmasks_ptr + pid * TILE_BYTES + tl.arange(0, TILE_BYTES), bytes_val)

    tile_max = tl.max(tl.abs(tile_flat))
    scale = tile_max / 127.0
    scale = tl.maximum(scale, 1e-6)
    quantized = (tile_flat / scale).to(tl.int32)
    quantized = tl.minimum(tl.maximum(quantized, -128), 127)
    tl.store(tile_vals_ptr + pid * TILE_NUMEL + tl.arange(0, TILE_NUMEL), quantized.to(tl.int8))
    tl.store(tile_scales_ptr + pid, scale)


# ---------------------------------------------------------------------------
# Kernel 2 — compact: scatter each tile's nonzeros into a single contiguous
#            vals array using the precomputed prefix offsets
# ---------------------------------------------------------------------------
@triton.jit
def _compact_vals_kernel(
    dense_ptr,
    tile_prefix_ptr,
    vals_out_ptr,
    layer_offset_ptr,
    M, N, grid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = tl.load(layer_offset_ptr)
    base = tl.load(tile_prefix_ptr + pid) + offset

    tile_m = pid // grid_n
    tile_n = pid % grid_n

    rm = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    v_2d = tl.load(dense_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)
    v = tl.reshape(v_2d, (TILE_NUMEL,))

    nz = (v > 0.0).to(tl.int32)

    # local rank of each nonzero within the tile → global position in vals_out
    ranks = tl.cumsum(nz, 0) - 1
    tl.store(vals_out_ptr + base + ranks, v, mask=(nz == 1))


@triton.jit
def _unpack_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr,
    layer_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """
    Each program unpacks ONE sparse tile [BLOCK_M x BLOCK_N] from the
    compact representation and scatters it into the dense output buffer.

    Grid layout: (num_row_tiles_in_batch * grid_n_sparse,) flat,
    where pid // grid_n_sparse picks the row-tile within the batch
    and  pid % grid_n_sparse picks the K-tile.
    """
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    # Read the packed uint8 bitmask and unpack into a [TILE_NUMEL] bool mask.
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)

    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    # The compact vals array stores this tile's nonzeros contiguously,
    # starting at prefix[tile_id].  Use a local prefix-sum (cumsum) over the
    # bitmask to map each nonzero to its rank within the tile.
    offset = tl.load(layer_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset

    ranks = tl.cumsum(mask_bits, 0) - 1
    v = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)

    # Reshape the flat result back to [BLOCK_M, BLOCK_N] and write into the
    # dense batch buffer at the correct (row, col) position.
    v_2d = tl.reshape(v, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, v_2d, mask=(offs_m < batch_rows) & (offs_k < K))


@triton.jit
def _unpack_batch_int8_kernel(
    vals_ptr, scales_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr,
    TILES_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base_tile = pid * TILES_PER_BLOCK

    for t in range(TILES_PER_BLOCK):
        tile_id = base_tile + t
        row_tile = tile_id // grid_n_sparse
        k_tile = tile_id % grid_n_sparse

        scale = tl.load(scales_ptr + tile_id)
        vals8 = tl.load(vals_ptr + tile_id * TILE_NUMEL + tl.arange(0, TILE_NUMEL))
        v_flat = vals8.to(tl.float32) * scale
        v_2d = tl.reshape(v_flat, (BLOCK_M, BLOCK_N))

        row_base = (row_tile - first_m_tile) * BLOCK_M
        offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
        offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        offs = offs_m * K + offs_k
        tl.store(dense_ptr + offs, v_2d, mask=(offs_m < batch_rows) & (offs_k < K))


@triton.jit
def _mask_with_bitmask_kernel(
    grad_ptr, bitmask_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    """
    Mask a dense gradient matrix (in-place) using the stored bitmask,
    zeroing out elements that were originally zero in the sparse matrix.

    Each program covers one tile [BLOCK_M x BLOCK_N].  The 2-D grid
    iterates over all tiles of the M×N matrix.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    # Row/column indices for this tile.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    gz = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    # Read the packed uint8 bitmask for this tile and unpack into bools.
    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)

    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = tl.reshape((bytes_2d >> bit_pos) & 1, (BLOCK_M, BLOCK_N))

    # Zero out gradient elements where the bitmask is 0 (in-place store).
    masked = tl.where(bits != 0, gz, 0.0)
    tl.store(grad_ptr + offs, masked, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _grad_z_sparse_values_kernel(
    grad_output_ptr, W2_ptr, bitmask_ptr,
    prefix_ptr, vals_offset_ptr, vals_out_ptr,
    M, N, grid_n,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """Compute sparse grad_z values for active bitmask entries and write them into vals_out."""
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
    base = tl.load(vals_offset_ptr) + tl.load(prefix_ptr + tile_id)
    tl.store(vals_out_ptr + base + ranks, vals, mask=(mask_bits == 1))
