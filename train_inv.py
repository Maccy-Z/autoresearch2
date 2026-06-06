import torch
import triton
import triton.language as tl

from prepare_inv import evaluate_kernel, pack_bool_mask


def compress_dense(dense, shape):
    """
    dense: 2D dense CUDA tensor
    shape: (rows, cols)
    Returns (vals, packed_mask) where vals are non-zero values in row-major
    order and packed_mask is uint8 bitmask.
    """
    rows, cols = shape
    flat = dense.flatten()
    mask = flat != 0
    vals = flat[mask]
    packed_mask = pack_bool_mask(mask)
    return vals, packed_mask


if __name__ == "__main__":
    evaluate_kernel(compress_dense)
