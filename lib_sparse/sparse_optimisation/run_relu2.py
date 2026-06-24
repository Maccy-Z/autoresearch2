import torch
import torch.nn.functional as F
import math
import torch._logging

from forward_methods import TensorBuffer
from forward_relu2 import FFNSparseRelu2, FFNSparseRelu2_3
from shared.experiment import run_step, FFN_relu2_abc
from shared.utils import setup_hooks, remove_hooks

FFN_BLOCK_LAYERS = 3
LAYERS = 3
BATCH_SIZE = 10000
DIM = 4096


class DeepFFN(FFN_relu2_abc):
    def __init__(self, dtype):
        super().__init__(dtype, LAYERS, DIM, FFN_BLOCK_LAYERS)

    def forward(self, x, buffer):
        buffer.ready_buffer()
        if self.block_layers == 2:
            for W1, W2 in zip(self.W1s, self.W2s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNSparseRelu2.apply(x_inner, W1, W2, buffer)
        else:
            for W1, W2, W3 in zip(self.W1s, self.W2s, self.W3s):
                x_inner = F.rms_norm(x, x.shape[1:])
                x = x + FFNSparseRelu2_3.apply(x_inner, W1, W2, W3, buffer)
        return x


def evaluate():
    dtype = torch.bfloat16
    # Setup model
    G = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(BATCH_SIZE, DIM, dtype=dtype, device="cuda", generator=G, requires_grad=True)
    model = DeepFFN(dtype=dtype)
    setup_hooks(model)

    # Run baseline
    run_step(x, model, sparse=False, steps=1)
    tracking_dn, vram_dn, avg_time = run_step(x, model, sparse=False, steps=1)
    print(f'Baseline: {vram_dn = :.2f} MB, {avg_time=:.2f} ms')
    print("-" * 50)

    # Setup sparse buffer
    hdim_expanded = math.floor(DIM * 5.25)
    buffer_scale = 0.55 * (2 if FFN_BLOCK_LAYERS == 3 else 1)
    buffer_size = int(BATCH_SIZE * hdim_expanded * LAYERS * buffer_scale)
    buffer = TensorBuffer(buffer_size, dtype=dtype, device="cuda")
    # Run sparse model
    # run_step(x, model, model._sparse_data, sparse=True, steps=1)
    tracking, vram, avg_time = run_step(x, model, buffer, sparse=True, steps=1)
    print(f"VRAM allocated by tensors: {vram:.2f} MB")
    print(f'Total time: {avg_time:.2f} ms')

    tracking = tracking * 1e3
    tracking_dn = tracking_dn * 1e3
    print(f'{tracking_dn = }')
    print(f'{tracking = }')

    if not torch.allclose(tracking, tracking_dn, atol=3e-4, rtol=3e-4):

        torch.testing.assert_close(tracking, tracking_dn, atol=3e-4, rtol=3e-4)
        assert vram < vram_dn * 1.1



def run_base():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    torch._logging.set_logs(graph_breaks=True)
    evaluate()


if __name__ == "__main__":
    run_base()
