import torch
import torch.nn.functional as F
import time
import gc


def generate_parameters(dim1, G, dtype, expansion=4, shift=0., device="cuda"):
    hdim = dim1 * expansion
    W1 = torch.empty(hdim, dim1, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(2048, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    x = torch.randn(10_000, dim1, device=device, generator=G, dtype=dtype)

    W1 = W1 + 0.1 * W1.std()
    W2 = W2 + 0.1 * W2.std()
    x = x + shift * x.std()
    return W1, W2, x


def exact_solution(W1, W2, x):
    y1 = F.relu(F.linear(x, W1))        # shape = [bs, 4*in_dim]
    y2 = F.relu(F.linear(y1, W2))       # shape = [bs, out_dim]
    y1_nz = (y1 != 0).sum()
    return y1_nz, y2


def check_out_dict(meta):
    size = 0
    for k, v in meta.items():
        if isinstance(v, torch.Tensor):
            size += v.nelement() * v.element_size() / 1024**2
    return size


def evaluate_step(rows, shift, G):
    from train import sp_relu_Ax, relu_spAx
    atol = 2e-2
    rtol = 1e-3
    n_tests = 15
    dtype = torch.bfloat16

    # Warmup
    W1, W2, x = generate_parameters(rows, G, shift=shift, dtype=dtype)
    for _ in range(5):
        vals, meta = sp_relu_Ax(W1, x)
        _ = relu_spAx(vals, meta, W2)
    del W1, W2, x
    gc.collect()
    torch.cuda.empty_cache()

    # Main loop. Do a few spmatmuls at a time.
    datasets = [generate_parameters(rows, G, shift=shift, dtype=dtype) for _ in range(n_tests)]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # First pass
    st1 = time.perf_counter()
    intermediates = []
    for W1, _, x in datasets:
        vals, meta = sp_relu_Ax(W1, x)
        intermediates.append((vals, meta))
    torch.cuda.synchronize()
    end1 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()

    # Second pass
    st2 = time.perf_counter()
    preds = []
    for i in range(len(intermediates)):
        vals, meta = intermediates[i]
        _, W2, _ = datasets[i]

        y = relu_spAx(vals, meta, W2)
        preds.append(y)
    torch.cuda.synchronize()
    end2 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()

    # Check accuracy — compute exact_solution only here
    for i in range(len(intermediates)):
        y_hat = preds[i]
        W1, W2, x = datasets[i]
        _, y_true = exact_solution(W1, W2, x)
        torch.testing.assert_close(y_hat, y_true, atol=atol, rtol=rtol)
        del W1, W2, x, y_true
    gc.collect()
    torch.cuda.empty_cache()

    # Make sure sparsity is high enough.
    for i in range(len(intermediates)):
        vals, meta = intermediates[i]
        W1, W2, x = datasets[i]
        y1_nz, _ = exact_solution(W1, W2, x)

        numel = W1.shape[0] * x.shape[0]
        fill_frac = y1_nz / numel
        meta_size = check_out_dict(meta)
        tot_size = meta_size + vals.numel() * vals.element_size() / 1024 ** 2
        full_size = numel * vals.element_size() / 1024 ** 2
        efficiency = tot_size / full_size
        assert efficiency < fill_frac +1.09

        del W2, x, _
        gc.collect()
        torch.cuda.empty_cache()

    time_taken = 1000 * (end1 - st1 + end2 - st2) / n_tests
    print(f"Shape {W1.shape}: Time {time_taken:.3g}ms")
    total_time = time_taken

    allocated = torch.cuda.memory_allocated("cuda")
    print(f"Allocated by tensors: {allocated / 1024 ** 2:.2f} MiB")

    return total_time


def evaluate_kernel():
    torch.manual_seed(0)
    G = torch.Generator(device="cuda").manual_seed(0)

    total_time = 0

    for rows in [512, 2048, 4096]:
        for shift in [-0.1, 0.1]:
            total_time += evaluate_step(rows, shift, G)

    print("passed")
    print(f"Total time: {total_time:.5g}ms")


def run_base():
    torch.set_float32_matmul_precision("high")
    evaluate_kernel()


if __name__ == "__main__":
    run_base()
