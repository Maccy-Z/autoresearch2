import torch
from torch.autograd import Function
import torch.nn as nn
from torch.utils import _pytree as pytree

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
        if kwargs is None:
            kwargs = {}

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
        print(output)
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
    def __init__(self, in_features, out_features, hidden_size=64, num_layers=2, device=None, generator=None):
        super().__init__()

        # Build the list of (in_features, out_features) for each layer
        if num_layers == 1:
            shapes = [(in_features, out_features)]
        else:
            shapes = [(in_features, hidden_size)]
            shapes += [(hidden_size, hidden_size)] * (num_layers - 2)
            shapes += [(hidden_size, out_features)]

        weights = []
        for fan_in, fan_out in shapes:
            w = nn.Parameter(torch.empty(fan_out, fan_in, device=device))
            nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=generator)
            weights.append(w)

        self.weights = nn.ParameterList(weights)

    def forward(self, input):
        x = input
        for i, weight in enumerate(self.weights):
            if i < len(self.weights) - 1:
                sparse_in = (i > 0)
                x = LinearReLUFunction.apply(x, weight, sparse_in)
            else:
                print(f"Input to final layer: {x}")
                # x = x.unpack()
                x = x @ weight.t()
        return x

def trace_grad_fn(fn, depth=0):
    if fn is None:
        return
    print("  " * depth + str(fn))
    for next_fn, _ in fn.next_functions:
        trace_grad_fn(next_fn, depth + 1)


def main():
    device = "cuda"
    dim = 1024
    bs = 2048

    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(bs, dim, requires_grad=True, device=device, generator=gen)
    model = Model(dim, dim, hidden_size=dim*4, num_layers=2, device=device, generator=gen)

    y = model(x)
    loss = y.sum()
    loss.backward()


    print("-" * 50)
    # print(y)
    # trace_grad_fn(y.grad_fn)
    # exit(4)
    print(loss)
    for i, w in enumerate(model.weights):
        print(f'layer.weights[{i}].grad.shape = {w.grad = }')

    print(f'{x.grad.shape = }')


if __name__ == '__main__':
    main()
