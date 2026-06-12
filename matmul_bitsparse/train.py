import torch
import triton
import triton.language as tl
from prepare import evaluate_kernel

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr,
                   M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Compute C = A @ B^T where A is (M,K) row-major and B is (N,K) row-major."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

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

    acc = tl.maximum(acc, 0)
    c_ptrs = C_ptr + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def _count_pack_kernel(dense_ptr, block_counts_ptr, packed_mask_out,
                       N: tl.constexpr, BYTE_BLOCK: tl.constexpr):
    """Count nonzeros per block and simultaneously pack the per-byte bitmask,
    storing per-block popcounts into block_counts_ptr."""
    pid = tl.program_id(0)
    byte_offs = pid * BYTE_BLOCK + tl.arange(0, BYTE_BLOCK)
    byte_valid = byte_offs * 8 < N

    byte_offs_2d = byte_offs[:, None]
    bit_offs_2d = tl.arange(0, 8)[None, :]

    elem_offs_2d = byte_offs_2d * 8 + bit_offs_2d
    in_bounds_2d = elem_offs_2d < N
    x_2d = tl.load(dense_ptr + elem_offs_2d, mask=in_bounds_2d, other=0.0,
                   eviction_policy="evict_first")
    nz_2d = (x_2d != 0).to(tl.int32)

    byte_nz = tl.sum(nz_2d, 1)
    byte_val = tl.sum(nz_2d << bit_offs_2d, 1)

    block_count = tl.sum(byte_nz, 0)
    tl.store(block_counts_ptr + pid, block_count)
    tl.store(packed_mask_out + byte_offs, byte_val.to(tl.uint8),
             mask=byte_valid)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def _vals_kernel(dense_ptr, block_prefix_ptr, vals_out,
                 N: tl.constexpr, BLOCK: tl.constexpr):
    """Extract nonzero values from each block and compact them into vals_out
    using the precomputed prefix offsets."""
    pid = tl.program_id(0)
    elem_offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem_valid = elem_offs < N

    x = tl.load(dense_ptr + elem_offs, mask=elem_valid, other=0.0,
                eviction_policy="evict_first")
    nz = (x > 0).to(tl.int32)

    byte_pos = tl.cumsum(nz, 0) - nz
    global_base = tl.load(block_prefix_ptr + pid)
    val_idx = global_base + byte_pos

    tl.store(vals_out + val_idx, x, mask=elem_valid & (nz == 1))


def bitsparse_pack(dense: torch.Tensor, block=4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a dense tensor into a compressed sparse representation.

    Given a 2D dense CUDA tensor, extract nonzero values in row-major order
    (vals) and produce a uint8 bitmask (packed_mask, 8 bits per byte).
    Processed in blocks for parallelism using Triton kernels."""
    device = dense.device

    N = dense.numel()                             # total number of elements
    flat = dense.flatten()                        # flatten to 1D for processing
    n_bytes = triton.cdiv(N, 8)                   # bytes needed for the bitmask
    n_blocks = triton.cdiv(N, block)              # number of element blocks

    packed_mask = torch.empty(n_bytes, device=device, dtype=torch.uint8)
    block_counts = torch.empty(n_blocks, device=device, dtype=torch.int32)
    block_prefix = torch.empty(n_blocks, device=device, dtype=torch.int32)

    # Step 1: count nonzeros per block and pack the bitmask
    _count_pack_kernel[(n_blocks,)](
        flat, block_counts, packed_mask, N=N, BYTE_BLOCK=block // 8,
    )

    # Step 2: exclusive prefix sum over counts → per-block offsets + total NZ count
    torch.cumsum(block_counts, 0, out=block_prefix)
    total_count = block_prefix[-1].clone()
    block_prefix.sub_(block_counts)

    # Step 3: compact nonzero values into vals using the prefix offsets
    vals = torch.empty(total_count, device=device, dtype=dense.dtype)

    _vals_kernel[(n_blocks,)](
        flat, block_prefix, vals,
        N=N, BLOCK=block,
    )

    return vals, packed_mask


def sparse_relu_Ax(W1, x):
    M, K = x.shape
    N = W1.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _matmul_kernel[grid](x, W1, out, M, N, K)

    vals, mask = bitsparse_pack(out)
    return vals, mask


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")

    evaluate_kernel(sparse_relu_Ax)
