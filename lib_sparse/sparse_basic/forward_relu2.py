from torch.autograd import Function

from forward_methods import dense_to_tilesparse
from shared.utils import RELU2_SCALE
from shared.functions import FFN_relu2_3_backward, FFN_relu2_backward


class FFNSparseRelu2(Function):
    """Autograd FFN using sparse storage for ReLU-squared hidden activation.
    Formula:
        z = x @ W1.T
        h = k * relu(z^2)
        out = z @ W2.T
        k = 1 / sqrt(3) matches the RMS of ReLU for standard-normal inputs.
    """
    @staticmethod
    def forward(ctx, x, W1, W2):
        """Compute FFN output and save pre-square ReLU values sparsely."""
        ctx.save_for_backward(x, W1, W2)
        z = x @ W1.T
        h = z.relu_()
        ctx.h_sparse = dense_to_tilesparse(h)
        h.square_()
        h.mul_(RELU2_SCALE)
        return h @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        return FFN_relu2_backward(ctx, grad_output)


class FFNSparseRelu2_3(Function):
    """Autograd FFN with two hidden ReLU-squared layers.

    Forward formula for ``x[B, D]``, ``W1[H, D]``, ``W2[H, H]``, ``W3[D, H]``:
    ``z1 = k*relu(x @ W1.T)^2``, ``z2 = k*relu(z1 @ W2.T)^2``, ``y = z2 @ W3.T``.
    """
    @staticmethod
    def forward(ctx, x, W1, W2, W3):
        ctx.save_for_backward(x, W1, W2, W3)
        z1 = x @ W1.T
        h1 = z1.relu_()
        ctx.h1_sparse = dense_to_tilesparse(h1)
        h1_sq = z1.square_()
        h1_sq.mul_(RELU2_SCALE)

        z2 = h1_sq @ W2.T
        h2 = z2.relu_()
        ctx.h2_sparse = dense_to_tilesparse(h2)
        h2_sq = h2.square_()
        h2_sq.mul_(RELU2_SCALE)

        return h2_sq @ W3.T

    @staticmethod
    def backward(ctx, grad_output):
        return FFN_relu2_3_backward(ctx, grad_output)
