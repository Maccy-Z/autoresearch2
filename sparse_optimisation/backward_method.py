import torch
from torch import Tensor

from sparse_kernels import _row_unpack_kernel, _row_mask_kernel
from sparse_utils import RowSparseTensor


@torch.jit.script
def _row_mask(grad: Tensor, bitmask: Tensor, N: int) -> Tensor:
    """Mask gradient using row_bitmask. Simple PyTorch loop."""
    return grad


def AspB(A: Tensor, B_sparse: RowSparseTensor, z_dense: Tensor) -> None:
    """Compute A @ B_sparse, writing dense z to z_dense and returning A @ z_dense."""
    vals = B_sparse.vals
    row_bitmask = B_sparse.row_bitmask
    row_offsets = B_sparse.row_offsets
    scales = B_sparse.scales
    M, N = B_sparse.shape

    ROW_BYTES = (N + 7) // 8
    BLOCK_COLS = 256

    _row_unpack_kernel[(M,)](
        vals, row_bitmask, row_offsets, scales,
        z_dense,
        M, N, N,
        ROW_BYTES=ROW_BYTES,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=8, num_stages=2,
    )

    return A @ z_dense


def FFN_backward(ctx, grad_output: Tensor):
    x, W1, W2 = ctx.saved_tensors
    z_sparse = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]
    M, N = z_sparse.shape

    z_dense = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    grad_W2 = AspB(grad_output.T, z_sparse, z_dense)

    grad_z = grad_output @ W2
    ROW_BYTES = (N + 7) // 8
    _row_mask_kernel[(M,)](
        grad_z, z_sparse.row_bitmask,
        M, N, N,
        ROW_BYTES=ROW_BYTES,
        BLOCK_COLS=256,
        num_warps=4, num_stages=2,
    )
    del z_dense, z_sparse

    if needs_x:
        grad_x = grad_z @ W1
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None
