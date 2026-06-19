from torch.autograd import Function
import torch.nn.functional as F
import torch


class FFNrelu2:
    @staticmethod
    def apply(x, W1, W2):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [exp_fact*in_dim, in_dim]
        W2.shape = [dim, exp_fact*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, exp_fact*in_dim]
        z = F.relu(preact)**2
        output = z @ W2.T           # shape = [BS, dim]

        return output


class FFNv1:
    @staticmethod
    def apply(x, W1, W2, a=None, b=None):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [exp_fact*in_dim, in_dim]
        W2.shape = [dim, exp_fact*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, exp_fact*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        return output


class FFNckpt:
    @staticmethod
    def apply(x, W1, W2):
        return torch.utils.checkpoint.checkpoint(FFNckpt.forward, x, W1, W2, use_reentrant=False)

    @staticmethod
    def forward(x, W1, W2):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [exp_fact*in_dim, in_dim]
        W2.shape = [dim, exp_fact*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, exp_fact*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        return output


class FFNv2(Function):
    """ Cache activation mask instead of preactivation"""
    @staticmethod
    def forward(ctx, x, W1, W2):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [exp_fact*in_dim, in_dim]
        W2.shape = [dim, exp_fact*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, exp_fact*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        ctx.save_for_backward(x, W1, W2, (preact>0), z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, relu_grad, z = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, exp_fact*in_dim]
        grad_W2 = grad_output.T @ z        # [dim, exp_fact*in_dim]

        # z = relu(preact)
        grad_preact = grad_z * relu_grad

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2


class FFNv3(Function):
    """ Recompute relu gradient """
    @staticmethod
    def forward(ctx, x, W1, W2, e1=None):
        """
        out = relu(x @ W1.T) @ W2.T

        x.shape = [BS, dim]
        W1.shape = [exp_fact*in_dim, in_dim]
        W2.shape = [dim, exp_fact*in_dim]

        returns:
            output: (BS, dim)
        """
        preact = x @ W1.T           # shape = [BS, exp_fact*in_dim]
        z = F.relu(preact)
        output = z @ W2.T           # shape = [BS, dim]

        ctx.save_for_backward(x, W1, W2, z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, z = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, exp_fact*in_dim]
        grad_W2 = grad_output.T @ z        # [dim, exp_fact*in_dim]

        # z = relu(preact)
        grad_preact = grad_z * (z>0)

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2, None, None
