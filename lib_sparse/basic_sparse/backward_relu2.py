from torch import Tensor

from shared.triton_operators import relu2_grad_sparse, relu2_layer_grad
from shared.sparse_operators import AspB_block, spAB_block, AspRelu2B, AspRelu2B_block
from shared.functions import FFN_relu2_3_backward



def FFN_relu2_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(x @ W1.T)^2 @ W2.T`` using sparse saved ``z``.
    """
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, z)

    grad_z = grad_output @ W2
    relu2_grad_sparse(grad_z, z)
    del z

    grad_x = grad_z @ W1 if needs_x else None
    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2


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

    if needs_x:
        grad_x = spAB_block(grad_sparse, W1)
    else:
        grad_x = None
    return grad_x, grad_W1, grad_W2


