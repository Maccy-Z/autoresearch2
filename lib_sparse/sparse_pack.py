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

The (vals, meta) pair returned by dense_to_tilesparse is consumed by
bitsparse_unpack in matmul_bitsparse/sparse_unpack.py.
"""
import torch
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
    tile_scratch_ptr,
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

    # keep a dense copy of the tile for the compaction pass below
    tl.store(tile_scratch_ptr + pid * TILE_NUMEL + tl.arange(0, TILE_NUMEL), tile_flat)


# ---------------------------------------------------------------------------
# Kernel 2 — compact: scatter each tile's nonzeros into a single contiguous
#            vals array using the precomputed prefix offsets
# ---------------------------------------------------------------------------
@triton.jit
def _compact_vals_kernel(
    tile_scratch_ptr,
    tile_prefix_ptr,
    vals_out_ptr,
    TILE_NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    base = tl.load(tile_prefix_ptr + pid)          # offset of this tile in vals_out

    # reload the dense tile from scratch and re-derive the nonzero mask
    offs = tl.arange(0, TILE_NUMEL)
    v = tl.load(tile_scratch_ptr + pid * TILE_NUMEL + offs)
    nz = (v > 0.0).to(tl.int32)

    # local rank of each nonzero within the tile → global position in vals_out
    ranks = tl.cumsum(nz, 0) - 1
    tl.store(vals_out_ptr + base + ranks, v, mask=(nz == 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def dense_to_tilesparse(dense: torch.Tensor, BLOCK_M=128, BLOCK_N=128):
    """Pack a dense 2D tensor into the per-tile compressed sparse format.

    Returns (vals, meta) where meta is a dict with keys:
      bitmask, prefix, grid_m, grid_n, BLOCK_M, BLOCK_N
    """
    M, N = dense.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n

    # --- launch: tile pack (bitmask + counts + dense scratch) ---
    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)
    tile_scratch = torch.empty(num_tiles, TILE_NUMEL, device=dense.device, dtype=dense.dtype)

    _tile_pack_kernel[(grid_m, grid_n)](
        dense, tile_counts, tile_bitmasks, tile_scratch,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
    )

    # --- host: exclusive prefix sum over per-tile counts ---
    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    total_nnz = tile_prefix[-1].item()

    # --- launch: compact nonzeros into contiguous vals ---
    vals = torch.empty(total_nnz, device=dense.device, dtype=dense.dtype)
    _compact_vals_kernel[(num_tiles,)](
        tile_scratch, tile_prefix, vals,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=8, num_stages=2,
    )

    meta = {
        'bitmask': tile_bitmasks,
        'prefix': tile_prefix,
        'grid_m': grid_m,
        'grid_n': grid_n,
        'BLOCK_M': BLOCK_M,
        'BLOCK_N': BLOCK_N,
    }
    return vals, meta
