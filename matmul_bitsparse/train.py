import torch
import triton
import triton.language as tl
from prepare import evaluate_kernel

BLOCK_M = 128
BLOCK_N = 128
TILE_NUMEL = BLOCK_M * BLOCK_N
TILE_BYTES = TILE_NUMEL // 8


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_sparse_kernel(
    A_ptr, B_ptr,
    tile_counts_ptr,
    tile_bitmasks_ptr,
    tile_vals_scratch_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)
    pid = pid_m * grid_n + pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = B_ptr + rn[:, None] * K + rk[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (k + rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rn[:, None] < N) & (k + rk[None, :] < K), other=0.0)
        acc += tl.dot(a, tl.trans(b), input_precision="tf32")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    valid = (rm[:, None] < M) & (rn[None, :] < N)
    acc = tl.where(valid, acc, 0.0)
    acc = tl.maximum(acc, 0)

    acc_flat = tl.reshape(acc, (TILE_NUMEL,))
    nz = (acc_flat > 0.0)

    nz_reshaped = tl.reshape(nz, (TILE_BYTES, 8))
    bit_select = tl.arange(0, 8)[None, :]
    bytes_val = tl.sum(nz_reshaped.to(tl.int32) << bit_select, 1).to(tl.uint8)

    mask_offs = pid * TILE_BYTES + tl.arange(0, TILE_BYTES)
    tl.store(tile_bitmasks_ptr + mask_offs, bytes_val)

    nnz = tl.sum(nz.to(tl.int32))
    tl.store(tile_counts_ptr + pid, nnz)

    nz_i32 = nz.to(tl.int32)
    ranks = tl.cumsum(nz_i32, 0) - 1
    scratch_base = tile_vals_scratch_ptr + pid * TILE_NUMEL
    tl.store(scratch_base + ranks, acc_flat, mask=nz)


@triton.jit
def _compact_vals_kernel(
    tile_vals_scratch_ptr,
    tile_prefix_ptr,
    vals_out_ptr,
    TILE_NUMEL: tl.constexpr,
):
    pid = tl.program_id(0)
    base = tl.load(tile_prefix_ptr + pid)
    nnz = tl.load(tile_prefix_ptr + pid + 1) - base

    offs = tl.arange(0, TILE_NUMEL)
    v = tl.load(tile_vals_scratch_ptr + pid * TILE_NUMEL + offs, mask=offs < nnz, other=0.0)
    tl.store(vals_out_ptr + base + offs, v, mask=offs < nnz)


def sparse_relu_Ax(W1, x):
    M, K = x.shape
    N = W1.shape[0]

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n

    tile_counts = torch.empty(num_tiles, device=x.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=x.device, dtype=torch.uint8)
    tile_vals_scratch = torch.empty(num_tiles, TILE_NUMEL, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_sparse_kernel[grid](
        x, W1, tile_counts, tile_bitmasks, tile_vals_scratch,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
    )

    tile_prefix = torch.empty(num_tiles + 1, device=x.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    total_nnz = tile_prefix[-1].item()

    vals = torch.empty(total_nnz, device=x.device, dtype=x.dtype)

    _compact_vals_kernel[(num_tiles,)](
        tile_vals_scratch, tile_prefix, vals,
        TILE_NUMEL=TILE_NUMEL,
        num_warps=4, num_stages=2,
    )

    return vals, tile_bitmasks, tile_prefix, grid_m, grid_n


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    evaluate_kernel(sparse_relu_Ax)
