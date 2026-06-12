import torch
import triton
import triton.language as tl

from prepare import evaluate_kernel

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


def bitsparse_unpack(vals, packed_mask, row_offsets, shape, block=65536) -> torch.Tensor:
    """Unpack a compressed sparse representation into a dense tensor.

    Uses precomputed row_offsets to skip per-block bit-counting during unpack."""
    assert vals.is_cuda and packed_mask.is_cuda
    assert packed_mask.dtype == torch.uint8
    device = vals.device
    M, K = shape

    N = M * K
    n_blocks = triton.cdiv(N, block)

    rows_per_block = block // K
    block_prefix = row_offsets[torch.arange(0, n_blocks, device=device) * rows_per_block]

    out = torch.empty(N, device=device, dtype=torch.bfloat16)

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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=1),
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
        acc += tl.dot(a, b)
        a_base = tl.advance(a_base, (0, BLOCK_K))
        b_base = tl.advance(b_base, (BLOCK_K, 0))

    acc = tl.maximum(acc, 0)

    c_base = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1),
    )
    tl.store(c_base, acc, boundary_check=(0, 1))


def relu_sparse_Ax(vals, mask, row_offsets, shape, x):
    """Compute relu(A @ x) where A is sparse in bitsparse format."""
    A_dense = bitsparse_unpack(vals, mask, row_offsets, shape)
    M, K = A_dense.shape
    N = x.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _matmul_relu_kernel[grid](A_dense, x.to(torch.bfloat16), out, M, N, K)

    return out


if __name__ == "__main__":
    evaluate_kernel(relu_sparse_Ax)
