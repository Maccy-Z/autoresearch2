import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from prepare import evaluate_kernel

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def _count_pack_kernel(dense_ptr, block_counts_ptr, packed_mask_out,
                       N: tl.constexpr, BYTE_BLOCK: tl.constexpr):
    """Count nonzeros per block and simultaneously pack the per-byte bitmask,
    storing per-block popcounts into block_counts_ptr."""
    pid = tl.program_id(0)
    byte_offs = pid * BYTE_BLOCK + tl.arange(0, BYTE_BLOCK)
    byte_valid = byte_offs * 8 < N

    byte_offs_2d = byte_offs[:, None]
    bit_offs_2d = tl.arange(0, 8)[None, :]

    elem_offs_2d = byte_offs_2d * 8 + bit_offs_2d
    in_bounds_2d = elem_offs_2d < N
    x_2d = tl.load(dense_ptr + elem_offs_2d, mask=in_bounds_2d, other=0.0)
    nz_2d = (x_2d > 0).to(tl.int32)

    byte_nz = tl.sum(nz_2d, 1)
    byte_val = tl.sum(nz_2d << bit_offs_2d, 1)

    block_count = tl.sum(byte_nz, 0)
    tl.store(block_counts_ptr + pid, block_count)
    tl.store(packed_mask_out + byte_offs, byte_val.to(tl.uint8),
             mask=byte_valid)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def _vals_kernel(dense_ptr, block_prefix_ptr, vals_out,
                 N: tl.constexpr, BLOCK: tl.constexpr):
    """Extract nonzero values from each block and compact them into vals_out
    using the precomputed prefix offsets."""
    pid = tl.program_id(0)
    elem_offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem_offs = tl.max_contiguous(tl.multiple_of(elem_offs, BLOCK), BLOCK)
    elem_valid = elem_offs < N

    x = tl.load(dense_ptr + elem_offs, mask=elem_valid, other=0.0,
                eviction_policy="evict_first")
    nz = (x > 0).to(tl.int32)

    byte_pos = tl.cumsum(nz, 0) - nz
    global_base = tl.load(block_prefix_ptr + pid)
    val_idx = global_base + byte_pos

    tl.store(vals_out + val_idx, x, mask=elem_valid & (nz == 1))


def bitsparse_pack(dense: torch.Tensor, block=1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a dense tensor into a compressed sparse representation.

    Given a 2D dense CUDA tensor, extract nonzero values in row-major order
    (vals) and produce a uint8 bitmask (packed_mask, 8 bits per byte).
    Processed in blocks for parallelism using Triton kernels."""
    device = dense.device

    N = dense.numel()                             # total number of elements
    flat = dense.flatten()                        # flatten to 1D for processing
    n_bytes = triton.cdiv(N, 8)                   # bytes needed for the bitmask
    n_blocks = triton.cdiv(N, block)              # number of element blocks

    packed_mask = torch.empty(n_bytes, device=device, dtype=torch.uint8)
    block_counts = torch.empty(n_blocks, device=device, dtype=torch.int32)
    block_prefix = torch.empty(n_blocks, device=device, dtype=torch.int32)

    # Step 1: count nonzeros per block and pack the bitmask
    _count_pack_kernel[(n_blocks,)](
        flat, block_counts, packed_mask, N=N, BYTE_BLOCK=block // 8,
    )

    # Step 2: exclusive prefix sum over counts → per-block offsets
    torch.cumsum(block_counts, 0, out=block_prefix)
    block_prefix.sub_(block_counts)

    # Step 3: compact nonzero values into vals using the prefix offsets
    # Preallocate N to avoid CPU sync for total_count; only first total_count entries are written
    vals = torch.empty(N, device=device, dtype=dense.dtype)

    _vals_kernel[(n_blocks,)](
        flat, block_prefix, vals,
        N=N, BLOCK=block,
    )

    return vals, packed_mask


def sparse_relu_Ax(W1, x):
    x = F.linear(x, W1)
    vals, mask = bitsparse_pack(x)
    return vals, mask


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    evaluate_kernel(sparse_relu_Ax)

