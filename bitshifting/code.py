from __future__ import annotations

import math

import torch
from torch import Tensor
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _compress_15bit_kernel(
    input_ptr,          # uint16 input
    output_ptr,         # uint8 output
    numel: tl.constexpr,
    compressed_numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Treat the input as a continuous stream of 15-bit little-endian values and
    emit that stream as bytes.
    """
    output_offsets = (
        tl.program_id(axis=0) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    output_mask = output_offsets < compressed_numel

    # First bit represented by each output byte.
    bit_positions = output_offsets * 8
    input_indices = bit_positions // 15
    bit_offsets = bit_positions % 15

    value0 = tl.load(
        input_ptr + input_indices,
        mask=output_mask & (input_indices < numel),
        other=0,
    ).to(tl.uint32)

    value1 = tl.load(
        input_ptr + input_indices + 1,
        mask=output_mask & ((input_indices + 1) < numel),
        other=0,
    ).to(tl.uint32)

    # The requested byte may cross a 15-bit value boundary.
    packed = (value0 >> bit_offsets) | (
        value1 << (15 - bit_offsets)
    )

    tl.store(
        output_ptr + output_offsets,
        packed & 0xFF,
        mask=output_mask,
    )


@triton.jit
def _uncompress_15bit_kernel(
    input_ptr,          # uint8 input
    output_ptr,         # uint16 output
    numel: tl.constexpr,
    compressed_numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_offsets = (
        tl.program_id(axis=0) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    output_mask = output_offsets < numel

    bit_positions = output_offsets * 15
    byte_indices = bit_positions // 8
    bit_offsets = bit_positions % 8

    # A 15-bit value can span at most three bytes.
    byte0 = tl.load(
        input_ptr + byte_indices,
        mask=output_mask & (byte_indices < compressed_numel),
        other=0,
    ).to(tl.uint32)

    byte1 = tl.load(
        input_ptr + byte_indices + 1,
        mask=output_mask & ((byte_indices + 1) < compressed_numel),
        other=0,
    ).to(tl.uint32)

    byte2 = tl.load(
        input_ptr + byte_indices + 2,
        mask=output_mask & ((byte_indices + 2) < compressed_numel),
        other=0,
    ).to(tl.uint32)

    packed_window = byte0 | (byte1 << 8) | (byte2 << 16)
    restored = (packed_window >> bit_offsets) & 0x7FFF

    tl.store(
        output_ptr + output_offsets,
        restored,
        mask=output_mask,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_fn(data: Tensor, dtype=None, device=None) -> Tensor:
    """Remove and bit-pack the sign bit from positive fp16/bf16 data."""
    device = data.device if device is None else torch.device(device)

    data = data.contiguous()
    bits = data.view(torch.uint16).reshape(-1)

    numel = bits.numel()
    compressed_numel = (numel * 15 + 7) // 8

    if compressed_numel == 0:
        return torch.empty(0, dtype=torch.uint8, device=device)

    output = torch.empty(
        compressed_numel,
        dtype=torch.uint8,
        device=device,
    )

    block_size = 256
    grid = (triton.cdiv(compressed_numel, block_size),)

    _compress_15bit_kernel[grid](
        bits,
        output,
        numel=numel,
        compressed_numel=compressed_numel,
        BLOCK_SIZE=block_size,
    )
    return output


def uncompress_fn(
    compressed_tensor: Tensor,
    shape: torch.Size,
    dtype,
    device=None,
) -> Tensor:
    """Restore fp16/bf16 data previously produced by ``compress_fn``."""
    device = (
        compressed_tensor.device
        if device is None
        else torch.device(device)
    )

    numel = math.prod(shape)
    expected_bytes = (numel * 15 + 7) // 8

    compressed_tensor = compressed_tensor.contiguous().reshape(-1)

    if numel == 0:
        return torch.empty(shape, dtype=dtype, device=device)

    restored_bits = torch.empty(
        numel,
        dtype=torch.uint16,
        device=device,
    )

    block_size = 256
    grid = (triton.cdiv(numel, block_size),)

    _uncompress_15bit_kernel[grid](
        compressed_tensor,
        restored_bits,
        numel=numel,
        compressed_numel=expected_bytes,
        BLOCK_SIZE=block_size,
    )

    # Reinterpret the restored bits without performing a numeric conversion.
    return restored_bits.view(dtype).reshape(shape)
