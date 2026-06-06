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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['N'],
    warmup=25,
    rep=3,
    cache_results=True,
)
@triton.autotune(
    configs=[
        triton.Config({}),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ],
    key=['N'],
    warmup=25,
    rep=3,
)
@triton.jit
def _compress_kernel(dense_ptr, block_prefix_ptr, vals_out, mask_dense,
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
    tl.store(mask_dense + offs, is_nz.to(tl.uint8), mask=valid)


@triton.jit
def _pack_mask_kernel(mask_dense, packed_mask_out,
                      N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_bytes = tl.cdiv(N, 8)
    byte_mask = offs < n_bytes

    byte_val = tl.zeros([BLOCK], dtype=tl.int32)
    for b in tl.static_range(8):
        elem_offs = offs * 8 + b
        in_bounds = (elem_offs < N) & byte_mask
        m = tl.load(mask_dense + elem_offs, mask=in_bounds, other=0).to(tl.int32)
        byte_val += m << b

    tl.store(packed_mask_out + offs, byte_val.to(tl.uint8), mask=byte_mask)


def compress_dense(dense, shape, block=512):
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
    mask_dense = torch.empty(N, device=dense.device, dtype=torch.uint8)

    _compress_kernel[(n_blocks,)](
        flat, block_prefix, vals, mask_dense, N=N, BLOCK=block,
    )

    packed_mask = torch.empty(n_bytes, device=dense.device, dtype=torch.uint8)
    n_pack_blocks = triton.cdiv(n_bytes, block // 8)
    _pack_mask_kernel[(n_pack_blocks,)](
        mask_dense, packed_mask, N=N, BLOCK=block // 8,
    )

    return vals, packed_mask


if __name__ == "__main__":
    evaluate_kernel(compress_dense)
