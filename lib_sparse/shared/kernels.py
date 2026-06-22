import triton
import triton.language as tl


@triton.jit
def _relu2_grad_with_sparse_kernel(
    grad_ptr, vals_ptr, bitmask_ptr, prefix_ptr, vals_offset_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
    RELU2_SCALE: tl.constexpr,
):
    """Apply derivative for ``z = k * r^2`` from stored ``r = relu(a)``."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    grad = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)
    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    mask_bits = tl.reshape((bytes_2d >> tl.arange(0, 8)[None, :]) & 1, (TILE_NUMEL,))

    offset = tl.load(vals_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset
    ranks = tl.cumsum(mask_bits, 0) - 1
    r = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0).to(tl.float32)
    scale = 2.0 * RELU2_SCALE * r
    scale_2d = tl.reshape(scale, (BLOCK_M, BLOCK_N))
    bits_2d = tl.reshape(mask_bits, (BLOCK_M, BLOCK_N))

    grad_preact = tl.where(bits_2d != 0, grad * scale_2d, 0.0)
    tl.store(grad_ptr + offs, grad_preact, mask=(rm[:, None] < M) & (rn[None, :] < N))

