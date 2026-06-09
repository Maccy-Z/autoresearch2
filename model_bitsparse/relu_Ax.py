import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sparse_pack import bitsparse_pack
from sparse_unpack import bitsparse_unpack


class BitsparseTensor(torch.Tensor):
    vals: Tensor            # Nonzero values
    packed_mask: Tensor     # Bitmask of nonzero values.

    @staticmethod
    def __new__(cls, dense: Tensor, metadata=None):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            dense.shape,
            dtype=dense.dtype,
            device=dense.device,
            requires_grad=dense.requires_grad,
        )

    def __init__(self, dense_tensor: torch.Tensor, metadata=None):
        super().__init__()
        self.vals, self.packed_mask = bitsparse_pack(dense_tensor)

    def __repr__(self):
        return (f"BitsparseTensor(shape={list(self.shape)}, device={self.device}, "
                f"nnz={self.packed_mask.shape}, requires_grad={self.requires_grad})\n")

    def unpack(self) -> torch.Tensor:
        dense_tensor = bitsparse_unpack(self.vals, self.packed_mask, self.shape)
        return dense_tensor

    @classmethod
    def _from_parts(cls, sparse: BitsparseTensor):
        # Create clone of tensor, used for .detach(), .clone() etc. Bypasses normal user-facing construction logic.
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            sparse.shape,
            strides=sparse.stride(),
            storage_offset=sparse.storage_offset(),
            dtype=sparse.dtype,
            layout=sparse.layout,
            device=sparse.device,
            requires_grad=sparse.requires_grad,
        )
        obj.vals = sparse.vals
        obj.packed_mask = sparse.packed_mask
        return obj


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
        if func is torch.ops.aten.detach.default:
            self = args[0]
            return BitsparseTensor._from_parts(self)
        else:
            raise NotImplementedError(f"Operations other than matmul are not supported on BitsparseTensor yet: {func}")


class LinearReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, sparse_in=False):
        """
        input:  (batch_size, in_features)
        weight: (out_features, in_features)
        sparse_in: Input is sparse, needs to be unpacked.
        Forward:    z = input @ W^T,
                    output = relu(z)
        returns:
            output: (batch_size, out_features)
        """
        if isinstance(input, BitsparseTensor):
            input_dense = input.unpack()
        else:
            input_dense = input

        z = input_dense @ weight.t()
        output = F.relu(z)
        output = BitsparseTensor(output)

        ctx.save_for_backward(input, weight, output)

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
        input, weight, output = ctx.saved_tensors
        output = output.unpack()

        grad_z = grad_output * (output > 0)

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
