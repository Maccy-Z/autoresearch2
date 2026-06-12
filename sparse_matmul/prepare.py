import torch
import time

from sparse_pack import bitsparse_pack
from sparse_unpack import bitsparse_unpack


def generate_parameters(dim, expansion=4, shift=0.1, seed=1, device="cuda"):
    """Generate sparse A in bitsparse format and dense input x."""
    G = torch.Generator(device=device).manual_seed(seed)

    hdim = dim * expansion
    A_dense = torch.empty(hdim, dim, device=device)
    torch.nn.init.xavier_uniform_(A_dense, generator=G)
    A_dense = A_dense + shift * A_dense.std()
    A_dense = torch.relu(A_dense)

    vals, mask, row_offsets = bitsparse_pack(A_dense)
    A_shape = A_dense.shape

    x = torch.randn(dim, 10_000, device=device, generator=G)
    x = x + shift * x.std()
    return vals, mask, row_offsets, A_shape, x


def exact_solution(vals, mask, row_offsets, shape, x):
    """Compute relu(A @ x) via unpack -> matmul -> relu."""
    A_dense = bitsparse_unpack(vals, mask, row_offsets, shape)
    y = torch.relu(A_dense @ x)
    return y


def dataloader():
    for dim in [512, 2048, 4096]:
        for shift in [-0.1, 0.1]:
            vals, mask, row_offsets, shape, x = generate_parameters(dim, shift=shift)
            y_true = exact_solution(vals, mask, row_offsets, shape, x)
            yield vals, mask, row_offsets, shape, x, y_true


def evaluate_kernel(sparse_relu_fn, atol=1e-2, rtol=1e-5):
    torch.manual_seed(0)

    steps = 50
    total_time = 0
    for vals, mask, row_offsets, shape, x, y_true in dataloader():
        for _ in range(10):
            _ = sparse_relu_fn(vals, mask, row_offsets, shape, x)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            y = sparse_relu_fn(vals, mask, row_offsets, shape, x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        torch.testing.assert_close(y, y_true, atol=atol, rtol=rtol)

        fill_frac = vals.numel() / (shape[0] * shape[1])
        time_taken = 1000 * (end - start) / steps
        print(f"Shape {shape}, fill {fill_frac:.3f}: Time {time_taken:.3g}ms")
        total_time += time_taken

    print("passed")
    print(f"Total time: {total_time:.5g}ms")


def run_base():
    evaluate_kernel(exact_solution)


if __name__ == "__main__":
    run_base()
