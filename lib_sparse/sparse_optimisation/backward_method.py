import torch
from torch import Tensor

from shared.triton_operators import relu_layer_sparse_
from shared.sparse_operators import AspB, spAB_block, ATspB_block
from shared.utils import print_memory


def FFN_backward_sparse(ctx, grad_output: Tensor):
    """Compute FFN gradients while keeping grad_z in the existing bit-sparse storage."""
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, z_sparse)
    # Combine grad_output @ W2, relu + masking. Updates z_sparse inplace.
    relu_layer_sparse_(grad_output, W2, z_sparse, BLOCK_K=32)
    grad_W1 = ATspB_block(x, z_sparse).T

    if needs_x:
        grad_x = spAB_block(z_sparse, W1)
    else:
        grad_x = None
    return grad_x, grad_W1, grad_W2, None


