import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'full_matmul')

import torch
import triton
import triton.language as tl
from sparse_pack import _compact_vals_kernel


# ---------------------------------------------------------------------------
# Layer 1: dense matmul x @ W1.T, then ReLU, then sparsify into a compact
# per-tile bitmask format for low-memory storage between layers.
#
# Output: (vals, meta) where vals holds only the nonzero elements and meta
# contains per-tile bitmasks, prefix-sum offsets, and shape metadata.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_K': 128}, num_warps=8, num_stages=1),
    ],
    key=['M', 'N', 'K', 'INPUT_PRECISION'],
)
@triton.jit
def _matmul_sparse_kernel(
    A_ptr, B_ptr,
    tile_counts_ptr,
    tile_bitmasks_ptr,
    tile_scratch_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """
    Each program computes one output tile [BLOCK_M x BLOCK_N] = C[i:j, p:q]
    via a standard tiled dense matmul over the K dimension.
    Then it applies ReLU, packs a uint8 bitmask for the nonzero pattern,
    counts the nonzeros, and writes the full dense tile to a scratch buffer
    for the downstream compaction pass.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.num_programs(1) + pid_n

    # Block pointers for the input operands.
    # A: [M, K] row-major  —  B: [K, N] column-major (i.e. W1 stored as [N, K]).
    a_base = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(0, 1),
    )
    b_base = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(1, K),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )

    # Tiled matrix multiply + ReLU
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_base, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_base, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        a_base = tl.advance(a_base, (0, BLOCK_K))
        b_base = tl.advance(b_base, (BLOCK_K, 0))

    acc = tl.maximum(acc, 0)

    # Pack the nonzero pattern into a uint8 bitmask (1 bit per element).
    # Flatten the tile row-major, split into bytes (8 bits each), pack with
    # little-endian bit positions.
    acc_flat = tl.reshape(acc, (TILE_NUMEL,))
    nz = (acc_flat > 0.0)

    nz_reshaped = tl.reshape(nz, (TILE_BYTES, 8))
    bit_weights = tl.arange(0, 8)[None, :]
    bytes_val = tl.sum(nz_reshaped.to(tl.int32) << bit_weights, 1).to(tl.uint8)

    tl.store(tile_bitmasks_ptr + pid * TILE_BYTES + tl.arange(0, TILE_BYTES), bytes_val)

    # Per-tile nonzero count (used for prefix-sum on the host).
    nnz = tl.sum(nz.to(tl.int32))
    tl.store(tile_counts_ptr + pid, nnz)

    # Write the full dense tile to scratch for the compaction kernel.
    offs = tl.arange(0, TILE_NUMEL)
    tl.store(tile_scratch_ptr + pid * TILE_NUMEL + offs, acc_flat)


def sp_relu_Ax(W1, x, BLOCK_M=128, BLOCK_N=128, input_precision="tf32"):
    """
    Layer 1: compute y = relu(x @ W1.T), then pack into a compact per-tile
    sparse representation.

    input_precision controls the tensor-core precision ("tf32", "tf32x3",
    or "ieee") for the Triton dense matmul.  When the input dtype is not
    float32 the matmul falls back to torch to match the reference path;
    Triton is still used for the sparsification step itself.

    Returns (vals, meta) where:
      vals    — 1D tensor of nonzero elements, dtype matches x.
      meta    — dict with bitmask, prefix offsets, and shape metadata.
    """
    M, K = x.shape
    N = W1.shape[0]

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n

    tile_counts = torch.empty(num_tiles, device=x.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=x.device, dtype=torch.uint8)
    tile_scratch = torch.empty(num_tiles * TILE_NUMEL, device=x.device, dtype=x.dtype)

    if x.dtype != torch.float32:
        y1 = torch.nn.functional.relu(torch.nn.functional.linear(x, W1))
        from sparse_pack import _tile_pack_kernel

        _tile_pack_kernel[(grid_m, grid_n)](
            y1, tile_counts, tile_bitmasks, tile_scratch,
            M, N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            num_warps=4, num_stages=2,
        )
    else:
        x_f32 = x.to(torch.float32)
        W1_f32 = W1.to(torch.float32)

        grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_sparse_kernel[grid](
            x_f32, W1_f32, tile_counts, tile_bitmasks, tile_scratch,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
            INPUT_PRECISION=input_precision,
        )

    # Host-side exclusive prefix sum → start offset of each tile's nonzeros.
    tile_prefix = torch.empty(num_tiles + 1, device=x.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    # Compact: gather only nonzero values into a single contiguous 1D array.
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
# Consumes the compact (vals, meta) from Layer 1.  Instead of unpacking the
# entire sparse matrix to dense (which would use M*K floats), we process
# the rows in batched chunks of ROW_BATCH rows.
#
# For each batch:
#   1. _unpack_batch_kernel  — reconstructs a small dense [batch_rows, K] block
#      from the per-tile bitmask and compact vals.
#   2. _dense_matmul_relu_kernel — standard autotuned dense matmul + ReLU
#      on that block against W2.T.
#
# Peak temporary memory is ROW_BATCH * K floats instead of M * K floats.
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=1),
    ],
    key=['M', 'N', 'K', 'INPUT_PRECISION'],
)
@triton.jit
def _dense_matmul_relu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """
    Standard tiled dense matmul C = relu(A @ B).
    A is [M, K] row-major, B is [K, N] row-major (W2.T, contiguous).
    Each program computes one output tile [BLOCK_M x BLOCK_N] and applies ReLU.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_base = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(0, 1),
    )
    b_base = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )
    c_base = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_base, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_base, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        a_base = tl.advance(a_base, (0, BLOCK_K))
        b_base = tl.advance(b_base, (BLOCK_K, 0))

    acc = tl.maximum(acc, 0)
    tl.store(c_base, acc, boundary_check=(0, 1))


def sp_relu_spAx(vals, meta, W2, input_precision="tf32"):
    """
    Layer 2: compute y = relu(unpack_sparse(vals, meta) @ W2).

    Processes rows in batches of ROW_BATCH to avoid materializing the
    full M×K dense intermediate.  When W2 is float32 the matmul uses the
    autotuned Triton kernel; for other dtypes it falls back to torch to
    match the reference computation path.

    input_precision controls the tensor-core precision ("tf32", "tf32x3",
    or "ieee") for the Triton dense matmul (float32 path only).
    """
    M, K = meta['shape']
    N = W2.shape[0]

    BLOCK_M = meta['BLOCK_M']
    BLOCK_K = meta['BLOCK_N']
    TILE_NUMEL = BLOCK_M * BLOCK_K
    TILE_BYTES = TILE_NUMEL // 8

    grid_n_sparse = meta['grid_n']
    ROW_BATCH = 2048

    use_triton = W2.dtype == torch.float32
    out = torch.empty(M, N, device=W2.device, dtype=torch.float32 if use_triton else W2.dtype)
    B = W2.T.contiguous()

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

        if use_triton:
            grid = lambda m: (triton.cdiv(batch_rows, m['BLOCK_M']), triton.cdiv(N, m['BLOCK_N']))
            _dense_matmul_relu_kernel[grid](
                dense_batch, B, out[m_start:m_end],
                batch_rows, N, K,
                BLOCK_M=BLOCK_M,
                INPUT_PRECISION=input_precision,
            )
        else:
            batch_out = torch.nn.functional.relu(
                torch.nn.functional.linear(dense_batch, W2)
            )
            out[m_start:m_end].copy_(batch_out)

    return out


if __name__ == "__main__":
    from prepare import run_base
    torch.set_float32_matmul_precision("high")
    run_base()
