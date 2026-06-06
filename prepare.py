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
    """ Generate data used for testing reconstruction, return packed values and the expected dense output for validation. """
    shape = (rows, cols)
    N = rows * cols

    dense_bool_mask = torch.rand(N, device="cuda") < 0.5
    packed_mask = pack_bool_mask(dense_bool_mask)

    vals = torch.randn(dense_bool_mask.sum().item(), device="cuda")

    expected = torch.zeros(N, device="cuda", dtype=vals.dtype)
    expected[dense_bool_mask] = vals
    expected = expected.reshape(shape)

    return vals, packed_mask, shape, expected

def dataloader():
    """ Yield data for testing reconstruction with different parameters. """
    for rows in [512, 2048, 4096]:
        for cols in [2048, 5000, 10000, 15000]:
            vals, packed_mask, shape, expected = generate_random_data(rows, cols)
            yield vals, packed_mask, shape, expected


def evaluate_kernel(reconstruct_bitmask):
    torch.manual_seed(0)

    total_time = 0
    for vals, packed_mask, shape, expected in dataloader():
        out = None
        # Initial warmup:
        for _ in range(5):
            out = reconstruct_bitmask(vals, packed_mask, shape)

        # Time main run
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            out = reconstruct_bitmask(vals, packed_mask, shape)
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Check accuracy on final run only, after timer ended.
        torch.testing.assert_close(out, expected)

        time_taken = (end - start) / 100
        print(f"Shape {shape}: Time {time_taken:.2g} seconds")
        total_time += time_taken

    print("passed")
    print(f"Total time: {total_time:.4g}")
