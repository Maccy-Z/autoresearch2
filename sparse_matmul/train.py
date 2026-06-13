import sys
import torch
import triton
import triton.language as tl

sys.path.insert(0, '.')
from matmul_bitsparse.sparse_unpack import bitsparse_unpack
from prepare import evaluate_kernel


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
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        INPUT_PRECISION: tl.constexpr):
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
        acc += tl.dot(a, b, input_precision=INPUT_PRECISION)
        a_base = tl.advance(a_base, (0, BLOCK_K))
        b_base = tl.advance(b_base, (BLOCK_K, 0))

    acc = tl.maximum(acc, 0)

    c_base = tl.make_block_ptr(
        base=C_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1),
    )
    tl.store(c_base, acc, boundary_check=(0, 1))


def relu_sparse_Ax(vals, meta, x, *, dtype=torch.bfloat16, tf32=False):
    """Compute relu(A @ x) where A is sparse in per-tile bitsparse format.

    Args:
        vals, meta: per-tile sparse representation of A
        x: dense input tensor [K, N]
        dtype: compute dtype for A and x (torch.bfloat16, torch.float16, torch.float32)
        tf32: whether to use TensorFloat32 for fp32 dot products
    """
    A_dense = bitsparse_unpack(vals, meta, meta['shape'])
    M, K = A_dense.shape
    N = x.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda m: (triton.cdiv(M, m['BLOCK_M']), triton.cdiv(N, m['BLOCK_N']))
    _matmul_relu_kernel[grid](
        A_dense.to(dtype), x.to(dtype), out, M, N, K,
        INPUT_PRECISION="tf32" if tf32 else "ieee",
    )

    return out


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    evaluate_kernel(relu_sparse_Ax)
