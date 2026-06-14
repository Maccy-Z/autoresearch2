import sys
sys.path.insert(0, '.')

import torch
import triton
import triton.language as tl
from prepare import evaluate_kernel
from sparse_pack import _compact_vals_kernel


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


def sparse_relu_Ax(W1, x, BLOCK_M=128, BLOCK_N=128, input_precision="tf32"):
    """Compute ReLU(W1 @ x.T) and return the non-zero values with sparse metadata."""
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
    }
    return vals, meta


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    evaluate_kernel(sparse_relu_Ax)
