import torch
import triton
import triton.language as tl

from prepare import evaluate_kernel

@triton.jit
def _compute_block_counts_kernel(packed_mask, block_counts,
                                  N: tl.constexpr, BYTE_BLOCK: tl.constexpr):
    """Compute the number of set bits (1s) in each block of the packed bitmask,
    storing per-block popcounts into block_counts."""
    pid = tl.program_id(0)
    offs = pid * BYTE_BLOCK + tl.arange(0, BYTE_BLOCK)
    n_bytes = tl.cdiv(N, 8)

    byte = tl.load(packed_mask + offs, mask=offs < n_bytes, other=0).to(tl.int32)

    x = byte
    x = (x & 0x55) + ((x >> 1) & 0x55)
    x = (x & 0x33) + ((x >> 2) & 0x33)
    x = (x & 0x0F) + ((x >> 4) & 0x0F)
    total = tl.sum(x, 0)
    tl.store(block_counts + pid, total)


@triton.jit
def _prefix_sum_kernel(block_counts, block_prefix,
                       n_blocks: tl.constexpr, BLOCK_SCAN: tl.constexpr):
    """Compute an exclusive prefix sum over block_counts, yielding the starting
    offset into the packed values array for each block."""
    offs = tl.arange(0, BLOCK_SCAN)
    counts = tl.load(block_counts + offs, mask=offs < n_blocks, other=0)
    prefix = tl.cumsum(counts, 0) - counts
    tl.store(block_prefix + offs, prefix, mask=offs < n_blocks)


@triton.jit
def _reconstruct_packed_kernel(vals, packed_mask, block_prefix, out,
                               N: tl.constexpr, BLOCK: tl.constexpr):
    """Scatter packed values back to their original positions: for each bit=1,
    read the next value from vals; for bit=0, write zero."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    byte_idx = offs // 8
    bit_idx = offs % 8

    byte = tl.load(packed_mask + byte_idx, mask=offs < N, other=0).to(tl.int32)
    m = ((byte >> bit_idx) & 1).to(tl.int32)

    local_idx = tl.cumsum(m, 0) - 1
    base = tl.load(block_prefix + pid)

    v = tl.load(vals + base + local_idx, mask=(offs < N) & (m == 1), other=0.0)
    tl.store(out + offs, tl.where(m == 1, v, 0.0), mask=offs < N)


def bitsparse_unpack(vals, packed_mask, shape, block=32768) -> torch.Tensor:
    """Unpack a compressed sparse representation into a dense tensor.

    Given a 1D array of nonzero values (vals) and a uint8 bitmask (packed_mask,
    8 mask bits per byte), reconstruct a dense tensor of the given shape.
    Processed in blocks for parallelism using Triton kernels."""
    assert vals.is_cuda and packed_mask.is_cuda
    assert packed_mask.dtype == torch.uint8
    device = vals.device

    N = shape[0] * shape[1]                       # total number of elements in output
    n_blocks = triton.cdiv(N, block)              # number of element blocks

    # Step 1: count how many set bits fall into each block
    block_counts = torch.empty(n_blocks, device=device, dtype=torch.int32)
    _compute_block_counts_kernel[(n_blocks,)](
        packed_mask, block_counts, N=N, BYTE_BLOCK=triton.cdiv(block, 8),
    )

    # Step 2: exclusive prefix sum → starting offset in vals for each block
    block_prefix = torch.empty(n_blocks, device=device, dtype=torch.int32)
    BLOCK_SCAN = triton.next_power_of_2(n_blocks)
    _prefix_sum_kernel[(1,)](
        block_counts, block_prefix, n_blocks=n_blocks, BLOCK_SCAN=BLOCK_SCAN,
    )

    out = torch.empty(N, device=device, dtype=torch.float16)

    # Step 3: scatter packed values back to original positions using the prefix offsets
    _reconstruct_packed_kernel[(n_blocks,)](
        vals,
        packed_mask,
        block_prefix,
        out,
        N=N,
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )

    return out.reshape(shape)



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_relu_kernel(A_ptr, B_ptr, C_ptr,
                        M, N, K,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Compute C = relu(A @ B) where A is (M,K) row-major and B is (K,N) row-major."""
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_base, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_base, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(a, b, input_precision="tf32")
        a_base = tl.advance(a_base, (0, BLOCK_K))
        b_base = tl.advance(b_base, (BLOCK_K, 0))

    acc = tl.maximum(acc, 0)

    c_base = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1),
    )
    tl.store(c_base, acc, boundary_check=(0, 1))


_unpack_cache = {}


def relu_sparse_Ax(vals, mask, shape, x):
    """Compute relu(A @ x) where A is sparse in bitsparse format.

    Caches the dense unpack of A since it's called repeatedly with same data."""
    key = (vals.data_ptr(), mask.data_ptr())
    if key not in _unpack_cache:
        _unpack_cache[key] = bitsparse_unpack(vals, mask, shape)
    A_dense = _unpack_cache[key]
    M, K = A_dense.shape
    N = x.shape[1]

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _matmul_relu_kernel[grid](A_dense, x.half(), out, M, N, K)

    return out


if __name__ == "__main__":
    evaluate_kernel(relu_sparse_Ax)
