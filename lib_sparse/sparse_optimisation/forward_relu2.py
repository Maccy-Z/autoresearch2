from torch.autograd import Function

from forward_methods import dense_to_tilesparse
from shared.utils import RELU2_SCALE
from shared.functions import FFN_relu2_3_backward, FFN_relu2_backward, FFN_relu2_backward_sparse


class FFNSparseRelu2(Function):
    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data):
        ctx.save_for_backward(x, W1, W2)
        preact = x @ W1.T
        preact.relu_()
        ctx.z_sparse = dense_to_tilesparse(preact, sparse_data)
        preact.square_()
        preact.mul_(RELU2_SCALE)
        return preact @ W2.T

    @staticmethod
    def backward(ctx, grad_output):
        return FFN_relu2_backward_sparse(ctx, grad_output)


class FFNSparseRelu2_3(Function):
    @staticmethod
    def forward(ctx, x, W1, W2, W3, sparse_data):
        ctx.save_for_backward(x, W1, W2, W3)
        z1 = x @ W1.T
        z1.relu_()
        ctx.z1_sparse = dense_to_tilesparse(z1, sparse_data)
        z1.square_()
        z1.mul_(RELU2_SCALE)

        z2 = z1 @ W2.T
        z2.relu_()
        ctx.z2_sparse = dense_to_tilesparse(z2, sparse_data)
        z2.square_()
        z2.mul_(RELU2_SCALE)

        return z2 @ W3.T

    @staticmethod
    def backward(ctx, grad_output):
        return FFN_relu2_3_backward(ctx, grad_output)
