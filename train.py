import torch
import triton
import triton.language as tl

from prepare import evaluate_kernel

@triton.jit
def _compute_block_counts_kernel(packed_mask, block_counts,
                                  N: tl.constexpr, BYTE_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BYTE_BLOCK + tl.arange(0, BYTE_BLOCK)
    n_bytes = tl.cdiv(N, 8)

    byte = tl.load(packed_mask + offs, mask=offs < n_bytes, other=0).to(tl.int32)

    x = byte
    x = (x & 0x55) + ((x >> 1) & 0x55)
    x = (x & 0x33) + ((x >> 2) & 0x33)
    x = (x & 0x0F) + ((x >> 4) & 0x0F)
    total = tl.sum(x, 0).to(tl.int32)
    tl.store(block_counts + pid, total)


@triton.jit
def _reconstruct_packed_kernel(vals, packed_mask, block_prefix, out,
                               N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    byte_idx = offs // 8
    bit_idx = offs % 8

    byte = tl.load(packed_mask + byte_idx, mask=offs < N, other=0).to(tl.int32)
    m = ((byte >> bit_idx) & 1).to(tl.int32)

    local_idx = tl.cumsum(m, 0) - 1
    base = tl.load(block_prefix + pid).to(tl.int32)

    v = tl.load(vals + base + local_idx, mask=(offs < N) & (m == 1), other=0.0)
    tl.store(out + offs, tl.where(m == 1, v, 0.0), mask=offs < N)


def reconstruct_bitmask(vals, packed_mask, shape, block=8192):
    """
    vals: 1D CUDA tensor containing nonzero / filled values
    packed_mask: 1D CUDA uint8 tensor, 8 mask bits per byte
    shape: output shape, e.g. (rows, cols)
    """
    assert vals.is_cuda and packed_mask.is_cuda
    assert packed_mask.dtype == torch.uint8

    N = shape[0] * shape[1]
    n_blocks = triton.cdiv(N, block)

    block_counts = torch.empty(n_blocks, device=vals.device, dtype=torch.int32)
    _compute_block_counts_kernel[(n_blocks,)](
        packed_mask, block_counts, N=N, BYTE_BLOCK=triton.cdiv(block, 8),
    )
    block_prefix = torch.cat([
        torch.zeros(1, device=vals.device, dtype=torch.int32),
        torch.cumsum(block_counts, dim=0)[:-1],
    ])

    out = torch.empty(N, device=vals.device, dtype=vals.dtype)

    _reconstruct_packed_kernel[(n_blocks,)](
        vals,
        packed_mask,
        block_prefix,
        out,
        N=N,
        BLOCK=block,
        num_warps=8,
    )

    return out.reshape(shape)


if __name__ == "__main__":
    evaluate_kernel(reconstruct_bitmask)
