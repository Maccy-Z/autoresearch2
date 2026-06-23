from torch import Tensor

from shared.sparse_operators import spAB_block, AspRelu2B, AspB_block
from shared.triton_operators import relu2_layer_grad


# def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
#     """Backward keeping ``dpreact`` in sparse storage to reduce peak memory."""
#
#     x, W1, W2 = ctx.saved_tensors
#     z_sparse = ctx.z_sparse
#     ctx.z_sparse = None
#     needs_x = ctx.needs_input_grad[0]
#
#     grad_W2 = AspRelu2B_block(grad_output, z_sparse)
#     # grad_z_sparse = grad_z_sparse_inplace_op(grad_output, W2, z_sparse)
#     relu2_layer_grad(grad_output, W2, z_sparse)
#     grad_z_sparse = z_sparse
#     del z_sparse
#     grad_W1 = AspRelu2B_block(x, grad_z_sparse).T
#
#     grad_x = spAB_block(grad_z_sparse, W1) if needs_x else None
#     return grad_x, grad_W1, grad_W2, None

def FFN_relu2_backward_sparse(ctx, grad_output: Tensor):
    """Backward keeping ``dpreact`` in sparse storage to reduce peak memory."""
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspRelu2B(grad_output.T, z_sparse)
    relu2_layer_grad(grad_output, W2, z_sparse)
    grad_sparse = z_sparse
    grad_W1 = AspB_block(x.T, grad_sparse).T

    grad_x = spAB_block(grad_sparse, W1) if needs_x else None

    return grad_x, grad_W1, grad_W2, None
