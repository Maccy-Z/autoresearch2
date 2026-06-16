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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

