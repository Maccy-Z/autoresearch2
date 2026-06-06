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
    total = tl.sum(x, 0)
    tl.store(block_counts + pid, total)


@triton.jit
def _reconstruct_dense_kernel(vals, dense_mask, block_prefix, out,
                              N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    m = tl.load(dense_mask + offs, mask=offs < N, other=0).to(tl.int32)
    local_idx = tl.cumsum(m, 0) - 1
    base = tl.load(block_prefix + pid)

    v = tl.load(vals + base + local_idx, mask=(offs < N) & (m == 1), other=0.0)
    tl.store(out + offs, tl.where(m == 1, v, 0.0), mask=offs < N)


def reconstruct_bitmask(vals, packed_mask, shape, block=8192):
    assert vals.is_cuda and packed_mask.is_cuda
    assert packed_mask.dtype == torch.uint8

    N = shape[0] * shape[1]
    n_blocks = triton.cdiv(N, block)

    block_counts = torch.empty(n_blocks, device=vals.device, dtype=torch.int64)
    _compute_block_counts_kernel[(n_blocks,)](
        packed_mask, block_counts, N=N, BYTE_BLOCK=triton.cdiv(block, 8),
    )
    block_prefix = torch.cat([
        torch.zeros(1, device=vals.device, dtype=torch.int64),
        torch.cumsum(block_counts, dim=0)[:-1].to(torch.int64),
    ])

    byte_ids = torch.arange(N, device=vals.device) // 8
    bit_ids = torch.arange(N, device=vals.device) % 8
    dense_mask = ((packed_mask[byte_ids] >> bit_ids) & 1).to(torch.int32)

    out = torch.empty(N, device=vals.device, dtype=vals.dtype)

    _reconstruct_dense_kernel[(n_blocks,)](
        vals,
        dense_mask,
        block_prefix,
        out,
        N=N,
        BLOCK=block,
        num_warps=8,
    )

    return out.reshape(shape)


if __name__ == "__main__":
    evaluate_kernel(reconstruct_bitmask)
