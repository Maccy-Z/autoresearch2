import torch
from torch import Tensor
from torch.autograd import Function

from shared.functions import (dense_to_tilesparse, FFN_backward, FFN3_backward, FFN_backward_sparse,
                              FFN_relu2_3_backward, FFN_relu2_backward)
from shared.sparse_operators import AspB, AspRelu2B
from shared.triton_operators import mask_with_bitmask_, relu2_grad_sparse_
from shared.utils import BitsparseTensor, RELU2_SCALE

# ------------------------------------------------------------
# ReLU layers
# ------------------------------------------------------------
class ReluLinear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data=None):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = dense_to_tilesparse(h, sparse_data)
        ctx.h_sparse = h_sparse
        y = h @ W.T
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        h: BitsparseTensor = ctx.h_sparse
        ctx.h_sparse = None
        grad_W2 = AspB(grad_output.T, h)
        grad_h = grad_output @ W
        grad_z = mask_with_bitmask_(grad_h, h)

        return grad_z, grad_W2, None


class FFNRelu:
    """ FFN block with relu activation"""
    @staticmethod
    def apply(x, W1, W2, sparse_data=None):
        z = x @ W1.T
        y = ReluLinear.apply(z, W2, sparse_data)

        return y


class FFNRelu_3:
    """ FFN block with relu activation, 3 linear layers. """
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data=None):
        z1 = x @ W1.T
        y1 = ReluLinear.apply(z1, W2, sparse_data)
        y2 = ReluLinear.apply(y1, W3, sparse_data)
        return y2

# ------------------------------------------------------------
# ReLU2 layers
# ------------------------------------------------------------
class Relu2Linear(Function):
    """y = relu(Wx)."""

    @staticmethod
    def forward(ctx, z, W, sparse_data):
        """ relu(Wx) layer. """
        ctx.save_for_backward(W)
        h = z.relu_()
        h_sparse = dense_to_tilesparse(h, sparse_data)
        ctx.h_sparse = h_sparse
        h.square_()
        h.mul_(RELU2_SCALE)
        y = h @ W.T
        return y

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output: Tensor):
        """Compute gradients."""
        W = ctx.saved_tensors[0]
        h: BitsparseTensor = ctx.h_sparse
        ctx.h_sparse = None
        grad_W2 = AspRelu2B(grad_output.T, h)
        grad_h = grad_output @ W
        grad_z = relu2_grad_sparse_(grad_h, h)

        return grad_z, grad_W2, None


class FFNRelu2:
    @staticmethod
    def apply(x, W1, W2, sparse_data=None):
        z = x @ W1.T
        y = Relu2Linear.apply(z, W2, sparse_data)
        return y


class FFNRelu2_3:
    @staticmethod
    def apply(x, W1, W2, W3, sparse_data=None):
        z1 = x @ W1.T
        y1 = Relu2Linear.apply(z1, W2, sparse_data)
        y2 = Relu2Linear.apply(y1, W3, sparse_data)
        return y2

# ------------------------------------------------------------
# Manual implemented layers
# ------------------------------------------------------------
BACKWARD_IMPL = FFN_backward
# BACKWARD_IMPL = FFN_backward_sparse
class FFNSparse(Function):
    """Forward of FFN."""

    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data=None):
        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h, sparse_data)
        return h @ W2.T

    backward = staticmethod(BACKWARD_IMPL)


class FFNSparse3(Function):
    """Autograd FFN block with two hidden ReLU layers."""
    @staticmethod
    def forward(ctx, x, W1, W2, W3, sparse_data=None):
        ctx.save_for_backward(x, W1, W2, W3)
        z1 = x @ W1.T
        h1 = z1.relu_()
        ctx.h1_sparse = dense_to_tilesparse(h1, sparse_data)
        z2 = h1 @ W2.T
        del z1, h1
        h2 = z2.relu_()
        ctx.h2_sparse = dense_to_tilesparse(h2, sparse_data)

        return h2 @ W3.T

    backward = staticmethod(FFN3_backward)


class FFNSparseRelu2(Function):
    """Autograd FFN using sparse storage for ReLU-squared hidden activation.
    Formula:
        z = x @ W1.T
        h = k * relu(z^2)
        out = z @ W2.T
        k = 1 / sqrt(3) matches the RMS of ReLU for standard-normal inputs.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, sparse_buffer=None):
        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h, sparse_buffer)
        h.square_()
        h.mul_(RELU2_SCALE)
        return h @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        return FFN_relu2_backward(ctx, grad_output)


class FFNSparseRelu2_3(Function):
    @staticmethod
    def forward(ctx, x, W1, W2, W3, sparse_buffer=None):
        ctx.save_for_backward(x, W1, W2, W3)
        z1 = x @ W1.T
        h1 = z1.relu_()
        ctx.h1_sparse = dense_to_tilesparse(h1, sparse_buffer)
        h1_sq = z1.square_()
        h1_sq.mul_(RELU2_SCALE)

        z2 = h1_sq @ W2.T
        del h1_sq, h1
        h2 = z2.relu_()
        ctx.h2_sparse = dense_to_tilesparse(h2, sparse_buffer)
        h2_sq = h2.square_()
        h2_sq.mul_(RELU2_SCALE)

        return h2_sq @ W3.T

    @staticmethod
    def backward(ctx, grad_output):
        return FFN_relu2_3_backward(ctx, grad_output)
