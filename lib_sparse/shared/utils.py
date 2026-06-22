import torch

# Constant for RELU^2 scaling
RELU2_SCALE = 3 ** -0.5


def _tile_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> tuple[int, int, int, int, int]:
    """Return tile-grid dimensions and tile storage sizes for a dense matrix shape."""
    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_tiles = grid_m * grid_n
    return grid_m, grid_n, num_tiles, TILE_NUMEL, TILE_BYTES


def print_memory(msg):
    memory = torch.cuda.max_memory_allocated("cuda") / 1024 ** 2
    print(f'{msg}: {memory:.2f} MB')


@torch.no_grad()
def inplace_mm_(A, W, B=2048):
    """ A <- AW inplace operation. Done with batches. """
    m, n = A.shape
    assert W.shape == (n, n)

    x = torch.empty((B, n), device=A.device, dtype=A.dtype)
    y = torch.empty_like(x)

    for i in range(0, m, B):
        b = min(B, m - i)
        x[:b].copy_(A[i:i+b])
        torch.mm(x[:b], W, out=y[:b])
        A[i:i+b].copy_(y[:b])
    return A
