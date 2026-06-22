import torch
import torch.nn as nn
import time
import math
import gc
import torch._logging
from torch.autograd import Function

from forward_methods import FFNSparse, FFNSparse3, ValueBuffer


FFN_BLOCK_LAYERS = 3


class FFN(Function):
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
        del grad_z, z
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        # preact = x @ W1.T
        grad_x = None
        if needs_x:
            grad_x = grad_preact @ W1          # [BS, dim]

        grad_W1 = grad_preact.T @ x        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2, None, None


class FFN3(Function):
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        z1 = x @ W1.T
        z1.relu_()
        z2 = z1 @ W2.T
        z2.relu_()
        ctx.save_for_backward(x, z1, z2, W1, W2, W3)
        return z2 @ W3.T

    @staticmethod
    def backward(ctx, grad_output):
        x, a1, a2, W1, W2, W3 = ctx.saved_tensors
        needs_x = ctx.needs_input_grad[0]

        grad_a2 = grad_output @ W3
        grad_W3 = grad_output.T @ a2
        grad_z2 = torch.ops.aten.threshold_backward.grad_input(
            grad_a2, a2, 0, grad_input=grad_a2
        )

        grad_a1 = grad_z2 @ W2
        grad_W2 = grad_z2.T @ a1

        grad_z1 = torch.ops.aten.threshold_backward.grad_input(
            grad_a1, a1, 0, grad_input=grad_a1
        )
        del grad_a2, grad_z2, grad_a1, a1, a2
        if not torch.compiler.is_compiling():
            ctx.maybe_clear_saved_tensors()

        grad_x = grad_z1 @ W1 if needs_x else None
        grad_W1 = grad_z1.T @ x
        del grad_z1
        return grad_x, grad_W1, grad_W2, grad_W3


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


def generate_parameters_3(dim, G, dtype, expansion=5.25, device="cuda"):
    hdim = math.floor(dim * expansion)
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    W2 = torch.empty(hdim, hdim, device=device, dtype=dtype)
    W3 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)
    torch.nn.init.xavier_uniform_(W2, generator=G)
    torch.nn.init.xavier_uniform_(W3, generator=G)
    shift = torch.randn(1, generator=G, device=device, dtype=dtype)
    W1 = W1 + 0.01 * shift * W1.std()
    W2 = W2 + 0.01 * shift * W2.std()
    W3 = W3 - 0.01 * shift * W3.std()
    return W1, W2, W3


class DeepFFN(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096):
        """Construct a stack of residual FFN layers for the memory benchmark."""
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)
        self.block_layers = FFN_BLOCK_LAYERS
        self.W1s, self.W2s, self.W3s = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            if self.block_layers == 2:
                W1, W2 = generate_parameters(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
            else:
                W1, W2, W3 = generate_parameters_3(hidm, G, dtype=dtype)
                self.W1s.append(nn.Parameter(W1))
                self.W2s.append(nn.Parameter(W2))
                self.W3s.append(nn.Parameter(W3))

        # Simulate hook-based efficient optimiser, that applies gradient update as soon as possible and clears grads.
        self.setup_hooks()

    @staticmethod
    def hook(w):
        w.grad = None
        return

    def setup_hooks(self):
        for n, p in self.named_parameters():
            p.register_post_accumulate_grad_hook(self.hook)

    # @torch.compile
    def forward_base(self, x):
        """ x.shape = [BS, dim] """
        if self.block_layers == 2:
            self.block_forward = FFN.apply
        elif self.block_layers == 3:
            self.block_forward = FFN3.apply
        else:
            raise NotImplementedError

        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
                x = x + FFN.apply(x_inner, W1, W2)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + self.block_forward(x_inner, W1, W2, W3)
        return x

    #@torch.compile
    def forward(self, x, buffer: ValueBuffer):
        """Run the residual FFN stack while allocating sparse storage for this pass."""
        buffer.ready_buffer()
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = x
                x = x + FFNSparse.apply(x_inner, W1, W2, buffer)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = x
                x = x + FFNSparse3.apply(x_inner, W1, W2, W3, buffer)
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
    layers = 1
    dtype = torch.bfloat16
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(bs, hdim, dtype=dtype, device="cuda", generator=G, requires_grad=True)

    # Our model
    model = DeepFFN(layers=layers, hidm=hdim, dtype=dtype)

    # Run baseline
    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f'Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms')
    print("-"*50)

    # Setup sparse buffer and run model
    hdim_expanded = math.floor(hdim * 5.25)
    buffer_scale = 0.4 * (2 if FFN_BLOCK_LAYERS == 3 else 1)
    buffer_size = int(bs * hdim_expanded * layers * buffer_scale)
    buffer = ValueBuffer(buffer_size, dtype=dtype, device="cuda")
    run_step(x, model, buffer, sparse=True, steps=1)
    # Main run
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=3)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f'Total time: {avg_time:.2f} ms')

    # Check correctness
    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):
        print(f'Predicted values are different.')
        print(f'{tracking_dn = }')
        print(f'{tracking = }')
        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)

    # Make sure vram usage is low enough
    assert vram < vram_dn * 0.95


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
