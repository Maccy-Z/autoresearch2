from torch import Tensor

from backward_method import grad_z_sparse_inplace_op
from shared.triton_operators import relu2_grad_sparse
from shared.sparse_operators import spAB_block, AspRelu2B_block
from shared.functions import FFN_relu2_3_backward


def FFN_relu2_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``."""
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B_block(grad_output, z)

    grad_z = grad_output @ W2
    relu2_grad_sparse(grad_z, z)
    del z

    grad_x = grad_z @ W1 if needs_x else None
    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None


def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B_block(grad_output, z_sparse)
    grad_z_sparse = grad_z_sparse_inplace_op(grad_output, W2, z_sparse)
    del z_sparse
    grad_W1 = AspRelu2B_block(x, grad_z_sparse).T

    grad_x = spAB_block(grad_z_sparse, W1) if needs_x else None
    return grad_x, grad_W1, grad_W2, None
