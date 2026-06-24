from torch.autograd import Function

from forward_methods import dense_to_tilesparse
from shared.utils import RELU2_SCALE
from shared.functions import FFN_relu2_3_backward, FFN_relu2_backward, FFN_relu2_backward_sparse


class FFNSparseRelu2(Function):
    """Autograd FFN using sparse storage for ReLU-squared hidden activation.
    Formula:
        z = x @ W1.T
        h = k * relu(z^2)
        out = z @ W2.T
        k = 1 / sqrt(3) matches the RMS of ReLU for standard-normal inputs.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, sparse_buffer):
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
    def forward(ctx, x, W1, W2, W3, sparse_buffer):
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
