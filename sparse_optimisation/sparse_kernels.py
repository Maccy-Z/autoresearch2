"""
Per-row sparse format with int8 quantization.

A dense 2D tensor [M, N] is stored row-wise:
  row_bitmask — uint8 packed bitmask per row, ROW_BYTES bytes/row.
  vals        — compact int8 nonzero values, row-major order.
  row_offsets — int32 CSR offsets: row_offsets[i] = start of row i in vals.
  scales      — float32 per-row quantization scale.
"""

import triton
import triton.language as tl


@triton.jit
def _row_pack_count_kernel(
    dense_ptr,
    row_bitmask_ptr,
    row_counts_ptr,
    M, N, stride_n,
    ROW_BYTES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_ptr = dense_ptr + pid * stride_n
    nnz = 0

    for col_block in range(0, N, BLOCK_COLS):
        cols = col_block + tl.arange(0, BLOCK_COLS)
        vals = tl.load(row_ptr + cols, mask=cols < N, other=0.0)
        bits = (vals > 0.0).to(tl.int32)

        byte_base = col_block // 8
        byte_idx = byte_base + tl.arange(0, BLOCK_COLS // 8)

        bits_2d = tl.reshape(bits, (BLOCK_COLS // 8, 8))
        shift_weights = tl.arange(0, 8)[None, :]
        bytes_val = tl.sum(bits_2d << shift_weights, 1).to(tl.uint8)

        tl.store(row_bitmask_ptr + pid * ROW_BYTES + byte_idx,
                 bytes_val, mask=byte_idx < ROW_BYTES)

        nnz += tl.sum(bits)

    tl.store(row_counts_ptr + pid, nnz)


@triton.jit
def _row_pack_vals_kernel(
    dense_ptr,
    row_offsets_ptr,
    vals_out_ptr,
    scales_out_ptr,
    M, N, stride_n,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_ptr = dense_ptr + pid * stride_n
    offset = tl.load(row_offsets_ptr + pid)

    row_max = 0.0
    for col_block in range(0, N, BLOCK_COLS):
        cols = col_block + tl.arange(0, BLOCK_COLS)
        vals = tl.load(row_ptr + cols, mask=cols < N, other=0.0)
        row_max = tl.maximum(row_max, tl.max(tl.abs(vals)))

    scale = row_max / 127.0
    scale = tl.maximum(scale, 1e-6)
    tl.store(scales_out_ptr + pid, scale)

    base = offset
    for col_block in range(0, N, BLOCK_COLS):
        cols = col_block + tl.arange(0, BLOCK_COLS)
        vals = tl.load(row_ptr + cols, mask=cols < N, other=0.0)
        q = tl.minimum(tl.maximum((vals / scale).to(tl.int32), 0), 127)
        nz = vals > 0.0
        tl.store(vals_out_ptr + base + tl.arange(0, BLOCK_COLS),
                 q.to(tl.int8), mask=nz & (cols < N))
        base += tl.sum(nz.to(tl.int32))


@triton.jit
def _row_unpack_kernel(
    vals_ptr, row_bitmask_ptr, row_offsets_ptr, scales_ptr,
    dense_ptr,
    M, N, stride_n,
    ROW_BYTES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_ptr = dense_ptr + pid * stride_n
    offset = tl.load(row_offsets_ptr + pid)
    scale = tl.load(scales_ptr + pid)
    pos = offset

    for col_block in range(0, N, BLOCK_COLS):
        cols = col_block + tl.arange(0, BLOCK_COLS)
        byte_base = col_block // 8

        bm = tl.load(row_bitmask_ptr + pid * ROW_BYTES + byte_base +
                     tl.arange(0, BLOCK_COLS // 8),
                     mask=(byte_base + tl.arange(0, BLOCK_COLS // 8)) < ROW_BYTES,
                     other=0).to(tl.int32)

        bm_2d = tl.reshape(bm, (BLOCK_COLS // 8, 1))
        bit_pos = tl.arange(0, 8)[None, :]
        bits = tl.reshape((bm_2d >> bit_pos) & 1, (BLOCK_COLS,)).to(tl.int32)
        nnz_chunk = tl.sum(bits)
        ranks = tl.cumsum(bits, 0) - 1
        vals8 = tl.load(vals_ptr + pos + ranks, mask=(bits == 1), other=0)
        out = vals8.to(tl.float32) * scale
        tl.store(row_ptr + cols, out, mask=(cols < N) & (bits == 1))
        pos += nnz_chunk
