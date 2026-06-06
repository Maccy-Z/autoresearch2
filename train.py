import torch
import triton
import triton.language as tl

from prepare import evaluate_kernel

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


def reconstruct_bitmask(vals, packed_mask, shape, block=256):
    """
    vals: 1D CUDA tensor containing nonzero / filled values
    packed_mask: 1D CUDA uint8 tensor, 8 mask bits per byte
    shape: output shape, e.g. (rows, cols)
    """
    assert vals.is_cuda and packed_mask.is_cuda
    assert packed_mask.dtype == torch.uint8

    N = shape[0] * shape[1]
    n_blocks = triton.cdiv(N, block)

    # For testing/simple use: unpack bits in PyTorch only to compute block prefixes.
    # For production, you may want a Triton prefix-count kernel too.
    byte_ids = torch.arange(N, device=packed_mask.device) // 8
    bit_ids = torch.arange(N, device=packed_mask.device) % 8
    dense_mask = ((packed_mask[byte_ids] >> bit_ids) & 1).to(torch.int32)

    padded = torch.nn.functional.pad(
        dense_mask,
        (0, n_blocks * block - N),
    )

    block_counts = padded.view(n_blocks, block).sum(dim=1)
    block_prefix = torch.cat([
        torch.zeros(1, device=vals.device, dtype=torch.int64),
        torch.cumsum(block_counts, dim=0)[:-1].to(torch.int64),
    ])

    out = torch.empty(N, device=vals.device, dtype=vals.dtype)

    _reconstruct_packed_kernel[(n_blocks,)](
        vals,
        packed_mask,
        block_prefix,
        out,
        N=N,
        BLOCK=block,
    )

    return out.reshape(shape)


if __name__ == "__main__":
    evaluate_kernel(reconstruct_bitmask)
