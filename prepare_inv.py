import torch
import triton
import time


def pack_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Packs a 1D bool CUDA tensor into uint8 bitmask.
    Bit i is stored in byte i // 8, bit position i % 8.
    """
    assert mask.is_cuda
    assert mask.ndim == 1

    N = mask.numel()
    n_bytes = triton.cdiv(N, 8)

    padded = torch.zeros(n_bytes * 8, device=mask.device, dtype=torch.uint8)
    padded[:N] = mask.to(torch.uint8)

    bits = padded.view(n_bytes, 8)
    shifts = torch.arange(8, device=mask.device, dtype=torch.uint8)

    return ((bits << shifts).sum(dim=1)).to(torch.uint8)


def generate_random_data(rows, cols, p=0.5):
    """Generate sparse dense tensor. Returns dense input and expected vals/packed_mask."""
    shape = (rows, cols)
    N = rows * cols

    dense_bool_mask = torch.rand(N, device="cuda") < p

    num_nonzero = dense_bool_mask.sum().item()
    nonzeros = torch.randn(num_nonzero, device="cuda")
    nonzeros = torch.where(nonzeros == 0, torch.tensor(1e-8, device="cuda"), nonzeros)

    dense = torch.zeros(N, device="cuda")
    dense[dense_bool_mask] = nonzeros
    dense = dense.reshape(shape)

    packed_mask = pack_bool_mask(dense_bool_mask)

    expected_vals = dense.flatten()[dense_bool_mask]

    return dense, shape, expected_vals, packed_mask


def dataloader():
    """Yield data for testing compression with different parameters."""
    for rows in [512, 2048, 4096]:
        for cols in [2048, 5000, 10000, 15000]:
            dense, shape, expected_vals, expected_mask = generate_random_data(rows, cols)
            yield dense, shape, expected_vals, expected_mask


def evaluate_kernel(compress_dense):
    torch.manual_seed(0)

    total_time = 0
    for dense, shape, expected_vals, expected_mask in dataloader():
        vals, packed_mask = None, None
        # Initial warmup:
        for _ in range(5):
            vals, packed_mask = compress_dense(dense, shape)

        # Time main run
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(2000):
            vals, packed_mask = compress_dense(dense, shape)
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Check accuracy on final run only, after timer ended.
        torch.testing.assert_close(vals, expected_vals)
        torch.testing.assert_close(packed_mask, expected_mask)

        time_taken = (end - start) / 100
        print(f"Shape {shape}: Time {time_taken:.2g} seconds")
        total_time += time_taken

    print("passed")
    print(f"Total time: {total_time:.4g}")
