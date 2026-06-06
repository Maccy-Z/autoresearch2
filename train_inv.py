import torch
import triton
import triton.language as tl

from prepare_inv import evaluate_kernel


@triton.jit
def _count_nonzero_kernel(dense_ptr, block_counts_ptr,
                          N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(dense_ptr + offs, mask=offs < N, other=0.0)
    count = tl.sum((x != 0).to(tl.int64), 0)
    tl.store(block_counts_ptr + pid, count)


@triton.jit
def _compress_kernel(dense_ptr, block_prefix_ptr, vals_out, packed_int32,
                     N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < N

    x = tl.load(dense_ptr + offs, mask=valid, other=0.0)
    is_nz = (x != 0)

    local_count = tl.cumsum(is_nz.to(tl.int64), 0) - 1
    base = tl.load(block_prefix_ptr + pid)

    write_vals = valid & is_nz
    tl.store(vals_out + base + local_count, x, mask=write_vals)

    byte_idx = offs // 8
    bit_pos = offs % 8
    bit_val = is_nz.to(tl.int32) << bit_pos
    tl.atomic_or(packed_int32 + byte_idx, bit_val, mask=valid)


def compress_dense(dense, shape, block=8192):
    """
    dense: 2D dense CUDA tensor
    shape: (rows, cols)
    Returns (vals, packed_mask) where vals are non-zero values in row-major
    order and packed_mask is uint8 bitmask.
    """
    assert dense.is_cuda

    rows, cols = shape
    N = rows * cols
    flat = dense.flatten()
    n_bytes = triton.cdiv(N, 8)
    n_blocks = triton.cdiv(N, block)

    block_counts = torch.empty(n_blocks, device=dense.device, dtype=torch.int64)
    _count_nonzero_kernel[(n_blocks,)](
        flat, block_counts, N=N, BLOCK=block,
    )
    block_prefix = torch.cat([
        torch.zeros(1, device=dense.device, dtype=torch.int64),
        torch.cumsum(block_counts, dim=0)[:-1].to(torch.int64),
    ])

    total_count = block_prefix[-1] + block_counts[-1]
    vals = torch.empty(total_count.item(), device=dense.device, dtype=dense.dtype)

    packed_int32 = torch.zeros(n_bytes, device=dense.device, dtype=torch.int32)

    _compress_kernel[(n_blocks,)](
        flat, block_prefix, vals, packed_int32, N=N, BLOCK=block, num_warps=8,
    )

    packed_mask = packed_int32.view(torch.uint8)[::4].clone()
    return vals, packed_mask


if __name__ == "__main__":
    evaluate_kernel(compress_dense)
