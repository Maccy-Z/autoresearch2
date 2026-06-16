import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'full_matmul')

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sparse_pack import _compact_vals_kernel


# ---------------------------------------------------------------------------
# Layer 1: x @ W1.T, then ReLU, then sparsify into a compact per-tile
# bitmask format for low-memory storage between layers.
#
# The matmul is done with torch.nn.functional.linear so it matches the
# reference precision regardless of dtype (bfloat16, float32, etc.).
# Triton kernels handle only the tile-based sparsification and compaction.
# ---------------------------------------------------------------------------

def sp_relu_Ax(W1, x, BLOCK_M=128, BLOCK_N=128):
    """
    Layer 1: compute y = relu(x @ W1.T), then pack into a compact per-tile
    sparse representation.

    Returns (vals, meta) where:
      vals    — 1D tensor of nonzero elements, dtype matches x.
      meta    — dict with bitmask, prefix offsets, and shape metadata.
    """
    from sparse_pack import _tile_pack_kernel

    M, K = x.shape
    N = W1.shape[0]

    y1 = torch.nn.functional.relu(torch.nn.functional.linear(x, W1))

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n

    tile_counts = torch.empty(num_tiles, device=x.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=x.device, dtype=torch.uint8)
    tile_scratch = torch.empty(num_tiles * TILE_NUMEL, device=x.device, dtype=x.dtype)

    _tile_pack_kernel[(grid_m, grid_n)](
        y1, tile_counts, tile_bitmasks, tile_scratch,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=4, num_stages=2,
    )

    tile_prefix = torch.empty(num_tiles + 1, device=x.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    total_nnz = tile_prefix[-1].item()
    vals = torch.empty(total_nnz, device=x.device, dtype=x.dtype)

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
        'shape': (M, N),
    }
    return vals, meta


# ---------------------------------------------------------------------------
# Layer 2: relu(sparse_A @ W2)
#
# Unpacks the compact sparse representation into batched dense row-blocks,
# then uses torch.nn.functional.linear for the matmul + ReLU.  ROW_BATCH
# controls the peak temporary memory (ROW_BATCH * K elements).
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """
    Each program unpacks ONE sparse tile [BLOCK_M x BLOCK_K] from the
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
    base = tl.load(prefix_ptr + tile_id)

    ranks = tl.cumsum(mask_bits, 0) - 1
    v = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)

    # Reshape the flat result back to [BLOCK_M, BLOCK_K] and write into the
    # dense batch buffer at the correct (row, col) position.
    v_2d = tl.reshape(v, (BLOCK_M, BLOCK_K))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_K + tl.arange(0, BLOCK_K))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, v_2d, mask=offs_m < batch_rows)


def relu_spAx(vals, meta, W2):
    """
    Layer 2: compute y = relu(unpack_sparse(vals, meta) @ W2).

    Unpacks the sparse representation into dense row-batches, then uses
    torch.nn.functional.linear for the matmul and ReLU.
    """
    M, K = meta['shape']
    N = W2.shape[0]

    BLOCK_M = meta['BLOCK_M']
    BLOCK_K = meta['BLOCK_N']
    TILE_NUMEL = BLOCK_M * BLOCK_K
    TILE_BYTES = TILE_NUMEL // 8

    grid_n_sparse = meta['grid_n']
    ROW_BATCH = 2048

    out = torch.empty(M, N, device=W2.device, dtype=W2.dtype)

    for m_start in range(0, M, ROW_BATCH):
        m_end = min(m_start + ROW_BATCH, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M

        dense_batch = torch.empty(batch_rows, K, device=W2.device, dtype=vals.dtype)

        num_row_tiles_in_batch = triton.cdiv(batch_rows, BLOCK_M)
        num_tiles_in_batch = num_row_tiles_in_batch * grid_n_sparse

        _unpack_batch_kernel[(num_tiles_in_batch,)](
            vals, meta['bitmask'], meta['prefix'],
            dense_batch,
            first_m_tile, grid_n_sparse, K, batch_rows,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=8, num_stages=2,
        )

        batch_out = F.relu(
            F.linear(dense_batch, W2)
        )
        out[m_start:m_end].copy_(batch_out)

    return out


if __name__ == "__main__":
    from prepare import run_base
    torch.set_float32_matmul_precision("high")
    run_base()
