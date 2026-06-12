import torch
import triton
import triton.language as tl
from sparse_unpack import bitsparse_unpack
from prepare import evaluate_kernel


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
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

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = B_ptr + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (k + rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32")
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    acc = tl.maximum(acc, 0)
    c_ptrs = C_ptr + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def relu_sparse_Ax(vals, mask, shape, x):
    """Compute relu(A @ x) where A is sparse in bitsparse format.

    Baseline: unpack A to dense, then dense matmul + relu."""
    A_dense = bitsparse_unpack(vals, mask, shape)
    M, K = A_dense.shape
    N = x.shape[1]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    _matmul_relu_kernel[grid](A_dense, x, out, M, N, K)

    return out


if __name__ == "__main__":
    evaluate_kernel(relu_sparse_Ax)
