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
def _reconstruct_packed_kernel(
    vals, packed_mask, block_prefix, out,
    N: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)

    byte_offsets = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte = tl.load(packed_mask + byte_offsets, mask=byte_offsets * 8 < N, other=0).to(tl.int32)

    pop = byte
    pop = (pop & 0x55) + ((pop >> 1) & 0x55)
    pop = (pop & 0x33) + ((pop >> 2) & 0x33)
    pop = (pop & 0x0F) + ((pop >> 4) & 0x0F)

    pop_prefix = tl.cumsum(pop, 0)
    base = tl.load(block_prefix + pid)
    vals_off = base + pop_prefix - pop

    bit_offsets = tl.arange(0, 8)
    bits = ((byte[:, None] >> bit_offsets[None, :]) & 1).to(tl.int32)
    elem_offsets = byte_offsets[:, None] * 8 + bit_offsets[None, :]

    bit_prefix = tl.cumsum(bits, 1) - bits
    vals_idx = vals_off[:, None] + bit_prefix

    mask = (elem_offsets < N) & (bits == 1)
    v = tl.load(vals + vals_idx, mask=mask, other=0.0)
    tl.store(out + elem_offsets, v, mask=elem_offsets < N)


def reconstruct_bitmask(vals, packed_mask, shape, block=4096):
    """
    vals: 1D CUDA tensor containing nonzero / filled values
    packed_mask: 1D CUDA uint8 tensor, 8 mask bits per byte
    shape: output shape, e.g. (rows, cols)
    """
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

    out = torch.empty(N, device=vals.device, dtype=vals.dtype)

    _reconstruct_packed_kernel[(n_blocks,)](
        vals,
        packed_mask,
        block_prefix,
        out,

        N=N,
        BLOCK_BYTES=triton.cdiv(block, 8),
    )

    return out.reshape(shape)


if __name__ == "__main__":
    evaluate_kernel(reconstruct_bitmask)
