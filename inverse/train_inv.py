import torch
import triton
import triton.language as tl

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from prepare_inv import evaluate_kernel


@triton.jit
def _count_pack_kernel(dense_ptr, block_counts_ptr, packed_mask_out,
                       N: tl.constexpr, BYTE_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    byte_offs = pid * BYTE_BLOCK + tl.arange(0, BYTE_BLOCK)
    byte_valid = byte_offs * 8 < N

    byte_offs_2d = byte_offs[:, None]
    bit_offs_2d = tl.arange(0, 8)[None, :]

    elem_offs_2d = byte_offs_2d * 8 + bit_offs_2d
    in_bounds_2d = elem_offs_2d < N
    x_2d = tl.load(dense_ptr + elem_offs_2d, mask=in_bounds_2d, other=0.0)
    nz_2d = (x_2d != 0).to(tl.int32)

    byte_nz = tl.sum(nz_2d, 1)
    byte_val = tl.sum(nz_2d << bit_offs_2d, 1)

    block_count = tl.sum(byte_nz, 0)
    tl.store(block_counts_ptr + pid, block_count)
    tl.store(packed_mask_out + byte_offs, byte_val.to(tl.uint8),
             mask=byte_valid)


@triton.jit
def _vals_kernel(dense_ptr, block_prefix_ptr, vals_out,
                 N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    elem_offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem_valid = elem_offs < N

    x = tl.load(dense_ptr + elem_offs, mask=elem_valid, other=0.0)
    nz = (x != 0).to(tl.int32)

    byte_pos = tl.cumsum(nz, 0) - nz
    global_base = tl.load(block_prefix_ptr + pid)
    val_idx = global_base + byte_pos

    tl.store(vals_out + val_idx, x, mask=elem_valid & (nz == 1))


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

    packed_mask = torch.empty(n_bytes, device=dense.device, dtype=torch.uint8)
    _count_pack_kernel[(n_blocks,)](
        flat, block_counts, packed_mask, N=N, BYTE_BLOCK=block // 8,
        num_warps=4,
    )

    block_prefix = torch.zeros(n_blocks, device=dense.device, dtype=torch.int64)
    torch.cumsum(block_counts[:-1], dim=0, out=block_prefix[1:])

    total_count = block_prefix[-1] + block_counts[-1]
    vals = torch.empty(total_count.item(), device=dense.device, dtype=dense.dtype)

    _vals_kernel[(n_blocks,)](
        flat, block_prefix, vals,
        N=N, BLOCK=block,
    )

    return vals, packed_mask


if __name__ == "__main__":
    evaluate_kernel(compress_dense)
