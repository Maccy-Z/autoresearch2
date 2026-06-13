import torch
import triton
import triton.language as tl

BLOCK_M = tl.constexpr(128)
BLOCK_N = tl.constexpr(128)
TILE_NUMEL = BLOCK_M.value * BLOCK_N.value
TILE_BYTES = TILE_NUMEL // 8


@triton.jit
def _unpack_tiles_kernel(
    tile_bitmasks_ptr,
    vals_ptr,
    tile_prefix_ptr,
    out_ptr,
    grid_n, out_M, out_N,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid = pid_m * grid_n + pid_n

    byte_offs = pid * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(tile_bitmasks_ptr + byte_offs).to(tl.int32)

    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    base = tl.load(tile_prefix_ptr + pid)

    local_ranks = tl.cumsum(mask_bits, 0) - 1
    v = tl.load(vals_ptr + base + local_ranks, mask=(mask_bits == 1), other=0.0)

    tile_offs = tl.arange(0, TILE_NUMEL)
    tile_rows = tile_offs // BLOCK_N
    tile_cols = tile_offs % BLOCK_N

    global_r = pid_m * BLOCK_M + tile_rows
    global_c = pid_n * BLOCK_N + tile_cols

    out_offs = global_r * out_N + global_c
    out_mask = (global_r < out_M) & (global_c < out_N)

    tl.store(out_ptr + out_offs, v, mask=out_mask)


def bitsparse_unpack(
    vals: torch.Tensor,
    tile_bitmasks: torch.Tensor,
    tile_prefix: torch.Tensor,
    grid_m: int,
    grid_n: int,
    shape: list[int],
) -> torch.Tensor:
    assert vals.is_cuda and tile_bitmasks.is_cuda and tile_prefix.is_cuda
    assert tile_bitmasks.dtype == torch.uint8
    device = vals.device

    out_M, out_N = shape[0], shape[1]

    out = torch.empty(out_M, out_N, device=device, dtype=torch.float32)

    _unpack_tiles_kernel[(grid_m, grid_n)](
        tile_bitmasks, vals, tile_prefix,
        out,
        grid_n, out_M, out_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8,
        num_stages=2,
    )

    return out
