from torch.autograd import Function
import torch.nn.functional as F

from sparse import dense_to_tilesparse, sp_relu_Ax


class FFNv1(Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [4*in_dim, in_dim]
        W2.shape = [dim, 4*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, 4*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        ctx.save_for_backward(x, W1, W2, preact, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, preact, z = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, 4*in_dim]
        grad_W2 = grad_output.T @ z        # [dim, 4*in_dim]

        # z = relu(preact)
        grad_preact = grad_z * (preact>0)

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [4*in_dim, dim]

        return grad_x, grad_W1, grad_W2


class FFNv2(Function):
    """ Cache activation mask instead of preactivation"""
    @staticmethod
    def forward(ctx, x, W1, W2):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [4*in_dim, in_dim]
        W2.shape = [dim, 4*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, 4*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        ctx.save_for_backward(x, W1, W2, (preact>0), z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, relu_grad, z = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, 4*in_dim]
        grad_W2 = grad_output.T @ z        # [dim, 4*in_dim]

        # z = relu(preact)
        grad_preact = grad_z * relu_grad

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [4*in_dim, dim]

        return grad_x, grad_W1, grad_W2


class FFNv3(Function):
    """ Recompute relu gradient """
    @staticmethod
    def forward(ctx, x, W1, W2):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [4*in_dim, in_dim]
        W2.shape = [dim, 4*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, 4*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, z = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, 4*in_dim]
        grad_W2 = grad_output.T @ z        # [dim, 4*in_dim]

        # z = relu(preact)
        grad_preact = grad_z * (z>0)

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [4*in_dim, dim]

        return grad_x, grad_W1, grad_W2


class FFNv4(Function):
    """ Recompute relu gradient """
    @staticmethod
    def forward(ctx, x, W1, W2):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [4*in_dim, in_dim]
        W2.shape = [dim, 4*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, 4*in_dim]
        z = F.relu(preact)

        output = z @ W2.T           # shape = [BS, dim]

        z_sparse = dense_to_tilesparse(z)

        # print(f'{z.numel() = }')
        # print(f'{z_sparse.vals.numel() =}')
        ctx.save_for_backward(x, W1, W2, (z>0), z_sparse)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, relu_grad, z_sparse = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, 4*in_dim]
        grad_W2 = grad_output.T @ z_sparse        # [dim, 4*in_dim]

        # z = relu(preact)
        grad_preact = grad_z * relu_grad

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [4*in_dim, dim]

        return grad_x, grad_W1, grad_W2


