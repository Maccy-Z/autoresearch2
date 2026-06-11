import torch
import torch.nn.functional as F
import time

from sparse_pack import bitsparse_pack
from sparse_unpack import  bitsparse_unpack

def generate_parameters(dim, expansion, shift=0., seed=1, device="cuda"):
    """ Fixed weights and inputs for consistency """
    G = torch.Generator(device=device).manual_seed(seed)

    hdim = dim * expansion
    W1 = torch.empty(hdim, dim, device=device)

    torch.nn.init.xavier_uniform_(W1, generator=G)

    x = torch.randn(10_000, dim, device=device, generator=G)

    # Shift
    W1 = W1 + 0.1*W1.std()
    x = x + shift*x.std()
    return W1, x


def exact_solution(W1, x):
    """ y = relu(Wx)"""

    x = F.linear(x, W1)
    y = F.relu(x)

    vals, mask = bitsparse_pack(y)

    return vals, mask


def dataloader():
    """ Yield data for testing reconstruction with different parameters. """
    for rows in [512, 2048, 4096]:
        for shift in [-0.1, 0.1]:
            W, x = generate_parameters(rows, 4, shift)
            vals_true, masks_true = exact_solution(W, x)
            yield W, x, vals_true, masks_true


def evaluate_kernel(relu_Ax_fn, atol=None, rtol=None):
    torch.manual_seed(0)

    steps = 50

    total_time = 0
    vals, mask = None, None
    for W, x, vals_true, masks_true in dataloader():
        # Initial warmup:
        for _ in range(10):
            _ = relu_Ax_fn(W, x)

        # Time main run
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            vals, mask = relu_Ax_fn(W, x)
        torch.cuda.synchronize()
        end = time.perf_counter()

        # Check accuracy on final run only, after timer ended.
        out_shape = [x.shape[0], W.shape[0]]
        y = bitsparse_unpack(vals, mask, out_shape)
        y_hat = bitsparse_unpack(vals_true, masks_true, out_shape)

        torch.testing.assert_close(y, y_hat, atol=atol, rtol=rtol)
        # Total nnz:
        numel = W.shape[0] * x.shape[0]          # Shape of y
        fill_frac = vals_true.numel() / numel

        time_taken = 1000*(end - start) / steps
        print(f"Shape {W.shape}, fill {fill_frac:.3f}: Time {time_taken:.3g}ms")
        total_time += time_taken

    print("passed")
    print(f"Total time: {total_time:.5g}ms")


def run_base():
    torch.set_float32_matmul_precision("high")

    evaluate_kernel(exact_solution)


if __name__ == "__main__":
    run_base()