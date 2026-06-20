import torch
import torch.nn as nn
import time
import math
import gc
import torch._logging
from torch.autograd import Function

from forward_methods import FFNSparse, ValueBuffer


class FFNv3(Function):
    """ Recompute relu gradient """
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [exp_fact*in_dim, in_dim]
        W2.shape = [dim, exp_fact*in_dim]

        returns:
            output: (BS, dim)
        """
        z = x @ W1.T           # shape = [BS, exp_fact*in_dim]
        z.relu_()
        output = z @ W2.T           # shape = [BS, dim]
        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, z = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, exp_fact*in_dim]
        grad_W2 = grad_output.T @ z        # [dim, exp_fact*in_dim]

        # z = relu(preact)
        grad_preact = torch.ops.aten.threshold_backward.grad_input(
        grad_z, z, 0, grad_input=grad_z
        )
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()
        del z

        # preact = x @ W1.T
        grad_x = None
        if needs_x:
            grad_x = grad_preact @ W1          # [BS, dim]

        grad_W1 = grad_preact.T @ x        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2, None, None


def generate_parameters(dim, G, dtype, expansion=5.25, device="cuda"):
    """Create one FFN layer's W1 and W2 parameters with deterministic initialisation."""
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01*shift*W1.std()
    W2 = W2 - 0.01*shift*W2.std()
    return W1, W2


class DeepFFN(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            W1, W2 = generate_parameters(hidm, G, dtype=dtype)

            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))

        # Simulate hook-based efficient optimiser, that applies gradient update as soon as possible and clears grads.
        self.setup_hooks()

    @staticmethod
    def hook(w):
        w.grad = None
        return

    def setup_hooks(self):
        for n, p in self.named_parameters():
            p.register_post_accumulate_grad_hook(self.hook)

    @torch.compile
    def forward_base(self, x):
        """ x.shape = [BS, dim] """
        for W1, W2 in zip(self.W1s, self.W2s):
            x_inner = x
            x = x + FFNv3.apply(x_inner, W1, W2)
        return x

    #@torch.compile
    def forward(self, x, buffer: ValueBuffer):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        buffer.ready_buffer()
        for W1, W2 in zip(self.W1s, self.W2s):
            x_inner = x
            x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
        return x


def run_step(x, model, buffer: ValueBuffer=None, sparse=False, steps=1):
    """Run forward/backward steps and return peak allocated VRAM plus average step time."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")

    if buffer is not None:
        buffer.init_buffer()
    start = time.perf_counter()

    for _ in range(steps):
        model.zero_grad()
        if sparse:
            y = model.forward(x, buffer)
        else:
            y = model.forward_base(x)

        loss = (y - x).pow(2).mean()
        loss.backward()

    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated("cuda") / 1024**2

    end = time.perf_counter()
    avg_time = (end - start) * 1000 / steps

    # Track gradient to ensure correctness
    tracking = [loss.detach().cpu()]
    for i, (n, p) in enumerate(model.named_parameters()):
        if p.grad is not None:
            tracking.append(p.grad.std().cpu())
    tracking = torch.stack(tracking) * 1e3
    return tracking, allocated, avg_time


def evaluate():
    """Build the benchmark model, run warmup and timed steps, and print memory results."""
    # Setup parameters
    hdim = 4096
    bs = 10_000
    layers = 12
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = DeepFFN(layers=layers, hidm=hdim, dtype=dtype)

    # Run baseline
    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=2)
    print(f'{vram_dn = :.2f} MB, {avg_time=:.2f} ms')
    print("-"*50)

    # Setup sparse buffer and run model
    hdim_expanded = math.floor(hdim * 5.25)
    buffer_size = int(bs * hdim_expanded * layers * 0.55)
    buffer = ValueBuffer(buffer_size, dtype=dtype, device="cuda")
    run_step(x, model, buffer, sparse=True, steps=1)
    # Main run
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=2)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f'{avg_time = :.2f} ms')

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):
        print(f'Predicted values are different.')
        print(f'{tracking_dn = }')
        print(f'{tracking = }')
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)

    # Make sure vram usage is low enough
    assert vram < vram_dn * 0.88


def run_base():
    """Configure deterministic/debug settings and launch the benchmark."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(
        graph_breaks=True,
    )

    evaluate()


if __name__ == "__main__":
    run_base()
