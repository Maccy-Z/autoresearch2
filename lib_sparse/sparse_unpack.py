import triton
import triton.language as tl


@triton.jit
def _unpack_batch_kernel(
    vals_ptr, bitmask_ptr, prefix_ptr,
    layer_offset_ptr,
    dense_ptr,
    first_m_tile, grid_n_sparse, K, batch_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    """
    Each program unpacks ONE sparse tile [BLOCK_M x BLOCK_N] from the
    compact representation and scatters it into the dense output buffer.

    Grid layout: (num_row_tiles_in_batch * grid_n_sparse,) flat,
    where pid // grid_n_sparse picks the row-tile within the batch
    and  pid % grid_n_sparse picks the K-tile.
    """
    pid = tl.program_id(0)
    row_tile_in_batch = pid // grid_n_sparse
    k_tile = pid % grid_n_sparse

    orig_row_tile = first_m_tile + row_tile_in_batch
    tile_id = orig_row_tile * grid_n_sparse + k_tile

    # Read the packed uint8 bitmask and unpack into a [TILE_NUMEL] bool mask.
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)

    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    # The compact vals array stores this tile's nonzeros contiguously,
    # starting at prefix[tile_id].  Use a local prefix-sum (cumsum) over the
    # bitmask to map each nonzero to its rank within the tile.
    offset = tl.load(layer_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset

    ranks = tl.cumsum(mask_bits, 0) - 1
    v = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)

    # Reshape the flat result back to [BLOCK_M, BLOCK_N] and write into the
    # dense batch buffer at the correct (row, col) position.
    v_2d = tl.reshape(v, (BLOCK_M, BLOCK_N))

    row_base = row_tile_in_batch * BLOCK_M
    offs_m = (row_base + tl.arange(0, BLOCK_M))[:, None]
    offs_k = (k_tile * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs = offs_m * K + offs_k
    tl.store(dense_ptr + offs, v_2d, mask=offs_m < batch_rows)


@triton.jit
def _mask_with_bitmask_kernel(
    grad_ptr, bitmask_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    """
    Mask a dense gradient matrix (in-place) using the stored bitmask,
    zeroing out elements that were originally zero in the sparse matrix.

    Each program covers one tile [BLOCK_M x BLOCK_N].  The 2-D grid
    iterates over all tiles of the M×N matrix.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)

    # Row/column indices for this tile.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    gz = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    # Read the packed uint8 bitmask for this tile and unpack into bools.
    tile_id = pid_m * grid_n + pid_n
    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)

    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = tl.reshape((bytes_2d >> bit_pos) & 1, (BLOCK_M, BLOCK_N))

    # Zero out gradient elements where the bitmask is 0 (in-place store).
    masked = tl.where(bits != 0, gz, 0.0)
    tl.store(grad_ptr + offs, masked, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _grad_relu2_kernel(
    grad_ptr, vals_ptr, bitmask_ptr, prefix_ptr,
    layer_offset_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    TILE_NUMEL: tl.constexpr, TILE_BYTES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grid_n = tl.num_programs(1)
    tile_id = pid_m * grid_n + pid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs = rm[:, None] * N + rn[None, :]
    gz = tl.load(grad_ptr + offs, mask=(rm[:, None] < M) & (rn[None, :] < N), other=0.0)

    byte_offs = tile_id * TILE_BYTES + tl.arange(0, TILE_BYTES)
    bytes_val = tl.load(bitmask_ptr + byte_offs).to(tl.int32)

    bytes_2d = tl.reshape(bytes_val, (TILE_BYTES, 1))
    bit_pos = tl.arange(0, 8)[None, :]
    bits = (bytes_2d >> bit_pos) & 1
    mask_bits = tl.reshape(bits.to(tl.int32), (TILE_NUMEL,))

    offset = tl.load(layer_offset_ptr)
    base = tl.load(prefix_ptr + tile_id) + offset

    ranks = tl.cumsum(mask_bits, 0) - 1
    z_vals = tl.load(vals_ptr + base + ranks, mask=(mask_bits == 1), other=0.0)
    z_2d = tl.reshape(z_vals, (BLOCK_M, BLOCK_N))

    masked = gz * 2.0 * z_2d
    tl.store(grad_ptr + offs, masked, mask=(rm[:, None] < M) & (rn[None, :] < N))

    # Square vals in-place with fp32 intermediate (matching PyTorch's f32 accumulate)
    z_f32 = z_vals.to(tl.float32)
    tl.store(vals_ptr + base + ranks, z_f32 * z_f32, mask=(mask_bits == 1))
