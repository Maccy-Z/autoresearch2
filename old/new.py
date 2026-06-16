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

    def vram_size(self):
        val_size = self.vals.element_size() * self.vals.nelement()
        bitmask_size = self.packed_mask.element_size() * self.packed_mask.nelement()
        return (val_size + bitmask_size)/1024**2


class FFNV0(Function):
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

        sparse_z = BitsparseTensor(z)

        ctx.save_for_backward(x, W1, W2, sparse_z)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, z = ctx.saved_tensors

        # output = z @ W2.T
        # grad_output.shape = [BS, dim]

        grad_z = grad_output @ W2          # [BS, exp_fact*in_dim]
        grad_W2 = grad_output.T @ z        # [dim, exp_fact*in_dim]

        # z = relu(preact)
        packed = z.packed_mask
        relu_grad = ((packed[:, None] >> torch.arange(8, device=packed.device)) & 1).bool()
        grad_preact = grad_z * relu_grad.view(grad_z.shape)

        # preact = x @ W1.T
        grad_x = grad_preact @ W1          # [BS, dim]
        grad_W1 = grad_preact.T @ x        # [exp_fact*in_dim, dim]

        return grad_x, grad_W1, grad_W2


def generate_parameters(dim, G, dtype, expansion=4, device="cuda"):
    hdim = dim * expansion
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    W1 = W1 + 0.01*W1.std()
    W2 = W2 #+ W2.std()
    return W1, W2


class DeepFFN(nn.Module):
    def __init__(self, dtype, layers=12, hidm=4096):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)

        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            W1, W2 = generate_parameters(hidm, G, dtype=dtype)

            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))

    def forward(self, x):
        """ x.shape = [BS, dim] """
        for W1, W2 in zip(self.W1s, self.W2s):
            x_inner = F.layer_norm(x, x.shape[-1:])
            x = x + FFNV0.apply(x_inner, W1, W2)
        return x


def evaluate_step():
    layers = 12
    hdim = 4096
    dtype = torch.bfloat16

    model = DeepFFN(dtype=dtype)
    x = torch.randn(10_000, hdim, dtype=dtype, device="cuda")

    y = model(x)
    loss = (y - x).pow(2).mean()
    print(y.std())
    print(f'Loss = {loss.detach().item()}, y.std = {y.std().detach().item()}')

    allocated = torch.cuda.memory_allocated("cuda")
    print(f"VRAM allocated by tensors: {allocated / 1024**2:.2f} MB")

    loss.backward()

    for i, p in enumerate(model.parameters()):
        if i > 5:
            break
        print(p.grad.std())


def run_base():
    torch.set_float32_matmul_precision("high")
    # torch.set_printoptions(precision=10)
    torch.manual_seed(0)
    evaluate_step()


if __name__ == "__main__":
    run_base()

