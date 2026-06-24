import torch
from torch import Tensor

from shared.triton_operators import unpack_batch, mask_with_bitmask, relu2_grad_sparse, relu2_layer_grad
from shared.sparse_operators import AspB, ATspB_block, AspRelu2B_block, AspRelu2B, AspB_block, spAB_block
from shared.utils import BitsparseTensor, inplace_mm_


BLOCK_M = 128
BLOCK_N = 128


def FFN_backward(ctx, grad_output: Tensor):
    """Compute FFN gradients."""
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, z_sparse)

    grad_z = grad_output @ W2
    mask_with_bitmask(grad_z, z_sparse)
    del z_sparse

    if needs_x:
        grad_x = grad_z @ W1
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN3_backward(ctx, grad_output: Tensor):
    """Compute 3-layer ReLU FFN gradients using sparse saved activations."""
    x, W1, W2, W3 = ctx.saved_tensors
    z1 = ctx.z1_sparse
    z2 = ctx.z2_sparse
    ctx.z1_sparse = None
    ctx.z2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = AspB(grad_output.T, z2)

    grad_z2 = grad_output @ W3
    mask_with_bitmask(grad_z2, z2)
    del z2
    grad_W2 = ATspB_block(grad_z2, z1)

    # grad_z1 = grad_z2 @ W2
    # Semi inplace variant to reduce peak memory.
    grad_z1 = grad_z2
    inplace_mm_(grad_z1, W2)

    del grad_z2
    mask_with_bitmask(grad_z1, z1)
    del z1

    grad_x = grad_z1 @ W1 if needs_x else None
    grad_W1 = grad_z1.T @ x

    return grad_x, grad_W1, grad_W2, grad_W3, None


def FFN_relu2_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``."""
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, z) #AspRelu2B_block(grad_output, z)

    grad_z = grad_output @ W2
    relu2_grad_sparse(grad_z, z)
    del z

    grad_x = grad_z @ W1 if needs_x else None
    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
    """Backward keeping ``dpreact`` in sparse storage to reduce peak memory."""
    x, W1, W2 = ctx.saved_tensors
    r_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, r_sparse)
    relu2_layer_grad(grad_output, W2, r_sparse)
    grad_sparse = r_sparse
    grad_W1 = AspB_block(x.T, grad_sparse).T

    grad_x = spAB_block(grad_sparse, W1) if needs_x else None

    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_3_backward(ctx, grad_output: Tensor):
    """Backward for ``z1 = k*relu(a1)^2``, ``z2 = k*relu(a2)^2`` using sparse caches."""
    x, W1, W2, W3 = ctx.saved_tensors
    z1 = ctx.z1_sparse
    z2 = ctx.z2_sparse
    ctx.z1_sparse = None
    ctx.z2_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W3 = AspRelu2B_block(grad_output, z2)

    grad_z2 = grad_output @ W3
    relu2_grad_sparse(grad_z2, z2)
    del z2

    grad_W2 = AspRelu2B_block(grad_z2, z1)

    grad_z1 = grad_z2
    inplace_mm_(grad_z1, W2)        # grad_z1 = grad_z2 @ W2

    relu2_grad_sparse(grad_z1, z1)
    del z1

    grad_W1 = grad_z1.T @ x
    del x
    grad_x = grad_z1 @ W1 if needs_x else None
    del grad_z1, grad_z2

    return grad_x, grad_W1, grad_W2, grad_W3, None