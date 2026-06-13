import torch
import torch.nn.functional as F
import time

from sparse_unpack import bitsparse_unpack


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

    return y


def dataloader():
    """ Yield data for testing reconstruction with different parameters. """
    for rows in [512, 2048, 4096]:
        for shift in [-0.1, 0.1]:
            W, x = generate_parameters(rows, 4, shift)
            y = exact_solution(W, x)
            yield W, x, y


def check_out_dict(meta):
    size = 0
    for k, v in meta.items():
        if isinstance(v, int) or isinstance(v, float):
            continue
        size += v.nelement() * v.element_size() / 1024**2       # in MB

    return size


def evaluate_kernel(relu_Ax_fn, atol=1e-2, rtol=1e-5):
    torch.manual_seed(0)

    steps = 50

    total_time = 0
    vals, meta = None, None
    for W, x, y_true in dataloader():
        out_shape = [x.shape[0], W.shape[0]]

        # Initial warmup:
        for _ in range(10):
            _ = relu_Ax_fn(W, x)

        # Time main run
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            vals, meta = relu_Ax_fn(W, x)
            y = bitsparse_unpack(vals, meta, out_shape)

        torch.cuda.synchronize()
        end = time.perf_counter()

        # Check accuracy on final run only, after timer ended.
        torch.testing.assert_close(y, y_true, atol=atol, rtol=rtol)
        # Total nnz:
        numel = W.shape[0] * x.shape[0]          # Shape of y
        fill_frac = vals.numel() / numel

        # Check size is ok, with a bit extra.
        meta_size = check_out_dict(meta)
        tot_size = meta_size + vals.numel() * vals.element_size() / 1024**2
        full_size = y_true.numel() * y_true.element_size() / 1024**2
        print(f'{tot_size=:.2f}MB, {full_size=:.2f}MB')

        assert (tot_size < full_size * fill_frac * 1.1+0.02*tot_size)

        # Timing
        time_taken = 1000*(end - start) / steps
        print(f"Shape {W.shape}, fill {fill_frac:.3f}: Time {time_taken:.3g}ms")
        total_time += time_taken

    print("passed")
    print(f"Total time: {total_time:.5g}ms")


def run_base():
    from train import sparse_relu_Ax
    torch.set_float32_matmul_precision("high")

    evaluate_kernel(sparse_relu_Ax)


if __name__ == "__main__":
    run_base()