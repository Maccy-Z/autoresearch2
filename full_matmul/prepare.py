import torch
import torch.nn.functional as F
import time

from sparse_unpack import bitsparse_unpack


def generate_parameters(dim1, dim2, expansion=4, shift=0., seed=1, device="cuda"):
    G = torch.Generator(device=device).manual_seed(seed)

    hdim = dim1 * expansion
    W1 = torch.empty(hdim, dim1, device=device)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(dim2, hdim, device=device)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    x = torch.randn(10_000, dim1, device=device, generator=G)

    W1 = W1 + 0.1 * W1.std()
    W2 = W2 + 0.1 * W2.std()
    x = x + shift * x.std()
    return W1, W2, x


def exact_solution(W1, W2, x):
    y1 = F.relu(F.linear(x, W1))
    y2 = F.relu(F.linear(y1, W2))
    return y1, y2


def dataloader():
    for rows in [512, 2048, 4096]:
        for shift in [-0.1, 0.1]:
            W1, W2, x = generate_parameters(rows, 512, shift=shift)
            y1, y2 = exact_solution(W1, W2, x)
            yield W1, W2, x, y1, y2


def check_out_dict(meta):
    size = 0
    for k, v in meta.items():
        if isinstance(v, torch.Tensor):
            size += v.nelement() * v.element_size() / 1024**2
    return size


def evaluate_kernel():
    from train import sp_relu_Ax, sp_relu_spAx
    torch.manual_seed(0)
    atol = 2e-2
    rtol = 1e-3

    steps = 50
    total_time = 0
    for W1, W2, x, y1, y_true in dataloader():

        for _ in range(10):
            vals, meta = sp_relu_Ax(W1, x)
            _ = sp_relu_spAx(vals, meta, W2)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            vals, meta = sp_relu_Ax(W1, x)
            y = sp_relu_spAx(vals, meta, W2)

        torch.cuda.synchronize()
        end = time.perf_counter()

        torch.testing.assert_close(y, y_true, atol=atol, rtol=rtol)

        # Make sure sparsity is high enough.
        numel = W1.shape[0] * x.shape[0]
        fill_frac = (y1!=0).sum() / numel
        meta_size = check_out_dict(meta)
        tot_size = meta_size + vals.numel() * vals.element_size() / 1024**2
        full_size = numel * vals.element_size() / 1024**2
        assert tot_size < full_size * fill_frac * 1.1 + 0.02 * tot_size

        time_taken = 1000 * (end - start) / steps
        print(f"Shape {W1.shape}, fill {fill_frac:.3f}: Time {time_taken:.3g}ms")
        total_time += time_taken

    print("passed")
    print(f"Total time: {total_time:.5g}ms")


def run_base():
    torch.set_float32_matmul_precision("high")
    evaluate_kernel()


if __name__ == "__main__":
    run_base()
