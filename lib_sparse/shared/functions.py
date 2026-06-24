import torch
from torch import Tensor

from shared.triton_operators import unpack_batch, mask_with_bitmask_, relu2_grad_sparse_, relu2_layer_grad
from shared.sparse_operators import AspB, ATspB_block, ATspRelu2B_block, AspRelu2B, AspB_block, spAB_block
from shared.utils import BitsparseTensor, inplace_mm_, print_memory


BLOCK_M = 128
BLOCK_N = 128


def FFN_backward(ctx, grad_output: Tensor):
    """Compute FFN gradients."""
    x, W1, W2 = ctx.saved_tensors
    h: BitsparseTensor = ctx.h_sparse
    ctx.h_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, h)

    grad_h = grad_output @ W2
    grad_z = mask_with_bitmask_(grad_h, h)
    del h

    if needs_x:
        grad_x = grad_z @ W1
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN3_backward(ctx, grad_output: Tensor):
    """Compute 3-layer ReLU FFN gradients using sparse saved activations."""
    x, W1, W2, W3 = ctx.saved_tensors
    h1 = ctx.h1_sparse
    z2 = ctx.h2_sparse
    ctx.h1_sparse = None
    ctx.h2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = AspB(grad_output.T, z2)

    grad_h2 = grad_output @ W3
    grad_z2 = mask_with_bitmask_(grad_h2, z2)
    del z2, grad_h2
    grad_W2 = ATspB_block(grad_z2, h1)

    # grad_z1 = grad_z2 @ W2
    # Semi inplace variant to reduce peak memory.
    grad_h1 = inplace_mm_(grad_z2, W2)

    del grad_z2
    grad_z1 = mask_with_bitmask_(grad_h1, h1)
    del h1, grad_h1

    grad_x = grad_z1 @ W1 if needs_x else None
    grad_W1 = grad_z1.T @ x

    return grad_x, grad_W1, grad_W2, grad_W3, None


def FFN_relu2_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``."""
    x, W1, W2 = ctx.saved_tensors
    h = ctx.h_sparse
    ctx.h_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, h) # ATspRelu2B_block(grad_output, z) #

    grad_h2 = grad_output @ W2
    grad_z = relu2_grad_sparse_(grad_h2, h)
    del h

    grad_x = grad_z @ W1 if needs_x else None
    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
    """Backward keeping ``dpreact`` in sparse storage to reduce peak memory."""
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, z) # ATspRelu2B_block(grad_output, z) #

    relu2_layer_grad(grad_output, W2, z)
    grad_z = z

    grad_W1 = ATspB_block(x, grad_z).T
    grad_x = spAB_block(grad_z, W1) if needs_x else None

    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_3_backward(ctx, grad_output: Tensor):
    """Backward for ``z1 = k*relu(a1)^2``, ``z2 = k*relu(a2)^2`` using sparse caches."""
    x, W1, W2, W3 = ctx.saved_tensors
    h1 = ctx.h1_sparse
    h2 = ctx.h2_sparse
    ctx.h1_sparse = None
    ctx.h2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = ATspRelu2B_block(grad_output, h2)

    grad_h2_sq = grad_output @ W3
    grad_z2 = relu2_grad_sparse_(grad_h2_sq, h2)
    del h2, grad_h2_sq

    grad_W2 = ATspRelu2B_block(grad_z2, h1)

    grad_h1_sq = inplace_mm_(grad_z2, W2)        # grad_z1 = grad_z2 @ W2
    del grad_z2

    grad_z1 = relu2_grad_sparse_(grad_h1_sq, h1)
    del h1, grad_h1_sq

    grad_W1 = grad_z1.T @ x
    del x
    grad_x = grad_z1 @ W1 if needs_x else None
    del grad_z1

    return grad_x, grad_W1, grad_W2, grad_W3, None