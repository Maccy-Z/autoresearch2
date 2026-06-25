import torch
from torch import Tensor

from shared.triton_operators import tile_pack, compact_vals
from shared.utils import tile_grid, BitsparseTensor, TensorBuffer

BLOCK_M, BLOCK_N = 128, 128

def _make_bitsparse(
    vals: Tensor, bitmask: Tensor, prefix: Tensor,
    vals_offset: Tensor,
    shape: tuple[int, int]
) -> BitsparseTensor:
    """Build a BitsparseTensor wrapper around packed values, bitmasks, and prefixes."""
    grid_m = (shape[0] + BLOCK_M - 1) // BLOCK_M
    grid_n = (shape[1] + BLOCK_N - 1) // BLOCK_N
    return BitsparseTensor(
        vals, bitmask, prefix,
        grid_m, grid_n, BLOCK_M, BLOCK_N, shape,
        vals_offset=vals_offset,
    )


def _dense_to_tilesparse_pack_impl(
    dense: Tensor, vals: Tensor, offset: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack a dense matrix into tile-sparse metadata and append values into the shared buffer."""
    M, N = dense.shape
    grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES = tile_grid(M, N, BLOCK_M, BLOCK_N)

    tile_counts = torch.empty(num_tiles, device=dense.device, dtype=torch.int32)
    tile_bitmasks = torch.empty(num_tiles * TILE_BYTES, device=dense.device, dtype=torch.uint8)

    tile_pack(dense, tile_counts, tile_bitmasks,
              M, N, grid_m, grid_n, BLOCK_M, BLOCK_N, TILE_NUMEL, TILE_BYTES)

    new_offset = offset.clone()

    tile_prefix = torch.empty(num_tiles + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(tile_counts, 0, out=tile_prefix[1:])
    tile_prefix[0] = 0

    compact_vals(dense, tile_prefix, vals, new_offset,
                 M, N, grid_n, num_tiles, BLOCK_M, BLOCK_N, TILE_NUMEL)

    offset.add_(tile_prefix[-1])
    return tile_bitmasks, tile_prefix, new_offset


def dense_to_tilesparse(
    dense: Tensor,
    sparse_data: TensorBuffer,
) -> BitsparseTensor:
    """Convert a dense activation matrix into a BitsparseTensor backed by sparse_data."""
    vals, offset = sparse_data.vals, sparse_data.offset
    bitmask, prefix, vals_offset = _dense_to_tilesparse_pack_impl(
        dense, vals, offset
    )
    return _make_bitsparse(vals, bitmask, prefix, vals_offset, dense.shape)

def true_fn(x):
    return x.sin_()

def false_fn(x):
    return -x

def main():
    buffer =TensorBuffer(256*256, "cuda", torch.float16)
    buffer.init_buffer()
    buffer.ready_buffer()

    x = torch.ones((250, 250), device="cuda", dtype=torch.float16)


    sparse_tensor = dense_to_tilesparse(x, buffer)
    print(sparse_tensor)

    with torch.no_grad():
        y = torch.cond(x[0, 0]>2, true_fn, false_fn, sparse_tensor.vals)

    print(y)
    print(x)



if __name__ == '__main__':
    main()
