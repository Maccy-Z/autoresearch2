import math
import torch

# Constant for RELU^2 scaling
RELU2_SCALE = 3 ** -0.5

def print_memory(msg):
    memory = torch.cuda.max_memory_allocated("cuda")/1024**2
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
