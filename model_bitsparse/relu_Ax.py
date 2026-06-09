import torch
from torch.autograd import Function
import torch.nn as nn
from torch import Tensor

from sparse_pack import bitsparse_pack
from sparse_unpack import bitsparse_unpack


class BitsparseTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, metadata=None):
        # elem should usually be a real torch.Tensor
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.shape,
            dtype=elem.dtype,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )

    def __init__(self, dense_tensor: torch.Tensor, metadata=None):
        self.vals, self.packed_mask = bitsparse_pack(dense_tensor)

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, device={self.device}, "
                f"nnz={self.packed_mask.shape}, requires_grad={self.requires_grad})\n")

    def unpack(self) -> torch.Tensor:
        dense_tensor = bitsparse_unpack(self.vals, self.packed_mask, self.shape)
        return dense_tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Convert to dense before doing matmul.
        # x @ y for 2D tensors usually reaches aten.mm.default
        if func is torch.ops.aten.mm.default:
            lhs, rhs = args

            if isinstance(lhs, BitsparseTensor):
                lhs = lhs.unpack()

            if isinstance(rhs, BitsparseTensor):
                rhs = rhs.unpack()

            out = torch.ops.aten.mm.default(lhs, rhs)
            return out
        else:
            raise NotImplementedError("Operations other than matmul are not supported on BitsparseTensor yet.")


class LinearReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, sparse_in=False):
        """
        input:  (batch_size, in_features)
        weight: (out_features, in_features)
        sparse_in: Input is sparse, needs to be unpacked.
        Forward:  z = input @ W^T,  output = relu(z) = clamp(z, min=0)

        returns:
            output: (batch_size, out_features)
        """
        if sparse_in:
            input = input.unpack()

        z = input @ weight.t()
        output = torch.clamp(z, min=0)

        ctx.save_for_backward(input, weight, z)

        output = BitsparseTensor(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        returns gradients for:
            input, weight

        Backward:
            grad_z = grad_output * (z > 0)                       # relu mask
            grad_input  = grad_z @ W                              # (B, out) @ (out, in) -> (B, in)
            grad_weight = grad_z^T @ input                        # (out, B) @ (B, in) -> (out, in)
        """
        input, weight, z = ctx.saved_tensors

        grad_z = grad_output.clone()
        grad_z[z <= 0] = 0

        grad_input = grad_z @ weight
        grad_weight = grad_z.t() @ input

        return grad_input, grad_weight, None


class Model(nn.Module):
    def __init__(self, W1, W2):
        super().__init__()

        self.W1 = torch.nn.Parameter(W1.clone())
        self.W2 = torch.nn.Parameter(W2.clone())

    def forward(self, x: Tensor) -> Tensor:
        x = LinearReLUFunction.apply(x, self.W1, False)
        x = x @ self.W2.t()
        return x


def main():
    from standard_solution import generate_parameters, exact_solution
    device = "cuda"
    dim, expansion = 2048, 4

    W1, W2, x, y = generate_parameters(dim, expansion)
    exact_y_hat, exact_W1_g, exact_W2_g = exact_solution(W1, W2, x, y)


    model = Model(W1, W2)

    y_hat = model(x)
    loss = (y_hat - y).pow(2).mean()
    loss.backward()
    W1_g = model.W1.grad.detach().clone()
    W2_g = model.W2.grad.detach().clone()

    print("-" * 50)
    preds_same = torch.allclose(y_hat, exact_y_hat)
    W1_grads_same = torch.allclose(W1_g, exact_W1_g)
    W2_grads_same = torch.allclose(W2_g, exact_W2_g)

    print(f'{preds_same = }')
    print(f'{W1_grads_same = }')
    print(f'{W2_grads_same = }')


if __name__ == '__main__':
    main()
