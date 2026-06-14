import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'full_matmul')

import torch
import triton
import triton.language as tl
from sparse_pack import _compact_vals_kernel


# ---------------------------------------------------------------------------
# Layer 1: matmul(x, W1.T) + ReLU + sparsify  (from matmul_bitsparse)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_K': 128}, num_warps=8, num_stages=1),
    ],
    key=['M', 'N', 'K'],
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * tl.num_programs(1) + pid_n

    a_base = tl.make_block_ptr(
        base=A_ptr, shape=(M, K), strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(0, 1),
    )
    b_base = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(1, K),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_base, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_base, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        a_base = tl.advance(a_base, (0, BLOCK_K))
        b_base = tl.advance(b_base, (BLOCK_K, 0))

    acc = tl.maximum(acc, 0)

    acc_flat = tl.reshape(acc, (TILE_NUMEL,))
    nz = (acc_flat > 0.0)

    nz_reshaped = tl.reshape(nz, (TILE_BYTES, 8))
    bit_weights = tl.arange(0, 8)[None, :]
    bytes_val = tl.sum(nz_reshaped.to(tl.int32) << bit_weights, 1).to(tl.uint8)

    tl.store(tile_bitmasks_ptr + pid * TILE_BYTES + tl.arange(0, TILE_BYTES), bytes_val)

    nnz = tl.sum(nz.to(tl.int32))
    tl.store(tile_counts_ptr + pid, nnz)

    offs = tl.arange(0, TILE_NUMEL)
    tl.store(tile_scratch_ptr + pid * TILE_NUMEL + offs, acc_flat)


def sp_relu_Ax(W1, x, BLOCK_M=128, BLOCK_N=128, input_precision="tf32"):
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

    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_sparse_kernel[grid](
        x, W1, tile_counts, tile_bitmasks, tile_scratch,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        INPUT_PRECISION=input_precision,
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
# Layer 2: relu(sparse_A @ W2) — batched row-unpack + dense matmul
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
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

    base = tl.load(prefix_ptr + tile_id)

    ranks = tl.cumsum(mask_bits, 0) - 1
    v = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)

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
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _dense_matmul_relu_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
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


def sp_relu_spAx(vals, meta, W2):
    M, K = meta['shape']
    N = W2.shape[0]
    B = W2.T.contiguous()

    BLOCK_M = meta['BLOCK_M']
    BLOCK_K = meta['BLOCK_N']
    TILE_NUMEL = BLOCK_M * BLOCK_K
    TILE_BYTES = TILE_NUMEL // 8

    grid_n_sparse = meta['grid_n']
    ROW_BATCH = 1024

    out = torch.empty(M, N, device=B.device, dtype=torch.float32)

    for m_start in range(0, M, ROW_BATCH):
        m_end = min(m_start + ROW_BATCH, M)
        batch_rows = m_end - m_start
        first_m_tile = m_start // BLOCK_M

        dense_batch = torch.empty(batch_rows, K, device=B.device, dtype=vals.dtype)

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

        grid = lambda m: (triton.cdiv(batch_rows, m['BLOCK_M']), triton.cdiv(N, m['BLOCK_N']))
        _dense_matmul_relu_kernel[grid](
            dense_batch, B, out[m_start:m_end],
            batch_rows, N, K,
            BLOCK_M=BLOCK_M,
            INPUT_PRECISION="tf32",
        )

    return out


if __name__ == "__main__":
    from prepare import run_base
    torch.set_float32_matmul_precision("high")
    run_base()
