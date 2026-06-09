import torch
import triton
import triton.language as tl


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
def _prefix_sum_kernel(block_counts, block_prefix,
                       n_blocks: tl.constexpr, BLOCK_SCAN: tl.constexpr):
    offs = tl.arange(0, BLOCK_SCAN)
    counts = tl.load(block_counts + offs, mask=offs < n_blocks, other=0)
    prefix = tl.cumsum(counts, 0) - counts
    tl.store(block_prefix + offs, prefix, mask=offs < n_blocks)


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
    base = tl.load(block_prefix + pid)

    v = tl.load(vals + base + local_idx, mask=(offs < N) & (m == 1), other=0.0)
    tl.store(out + offs, tl.where(m == 1, v, 0.0), mask=offs < N)


def bitsparse_unpack(vals, packed_mask, shape, block=8192) -> torch.Tensor:
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
    block_prefix = torch.empty(n_blocks, device=vals.device, dtype=torch.int32)
    BLOCK_SCAN = triton.next_power_of_2(n_blocks)
    _prefix_sum_kernel[(1,)](
        block_counts, block_prefix, n_blocks=n_blocks, BLOCK_SCAN=BLOCK_SCAN,
    )

    out = torch.empty(N, device=vals.device, dtype=vals.dtype)

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


