from torch.autograd import Function

from forward_methods import dense_to_tilesparse


RELU2_SCALE = 3 ** -0.5


class FFNSparseRelu2(Function):
    """Autograd FFN using sparse storage for ReLU-squared hidden activation.

    Forward formula for ``x[B, D]``, ``W1[H, D]``, ``W2[D, H]``:
    ``z = k * relu(x @ W1.T)^2`` and ``y = z @ W2.T``, where
    ``k = 1 / sqrt(3)`` matches the RMS of ReLU for standard-normal inputs.
    """
    @staticmethod
    def forward(ctx, x, W1, W2):
        """Compute FFN output and save pre-square ReLU values sparsely."""
        ctx.save_for_backward(x, W1, W2)
        preact = x @ W1.T
        preact.relu_()
        ctx.z_sparse = dense_to_tilesparse(preact)
        preact.square_()
        preact.mul_(RELU2_SCALE)
        return preact @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        from backward_relu2 import FFN_relu2_backward

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
        z1.relu_()
        ctx.z1_sparse = dense_to_tilesparse(z1)
        z1.square_()
        z1.mul_(RELU2_SCALE)

        z2 = z1 @ W2.T
        z2.relu_()
        ctx.z2_sparse = dense_to_tilesparse(z2)
        z2.square_()
        z2.mul_(RELU2_SCALE)

        return z2 @ W3.T

    @staticmethod
    def backward(ctx, grad_output):
        from backward_relu2 import FFN_relu2_3_backward

        return FFN_relu2_3_backward(ctx, grad_output)
