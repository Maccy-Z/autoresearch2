import torch
from torch import Tensor

from sparse_kernels import _unpack_batch_kernel, _mask_with_bitmask_kernel
from sparse_utils import BitsparseTensor


def unpack_sparse(X: BitsparseTensor) -> Tensor:
    vals = X.vals
    bitmask = X.bitmask
    prefix = X.prefix
    BLOCK_M, BLOCK_N = X.BLOCK_M, X.BLOCK_N
    grid_m, grid_n = X.grid_m, X.grid_n
    M, N = X.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    num_tiles = grid_m * grid_n

    dense = torch.empty(M, N, device=vals.device, dtype=vals.dtype)
    _unpack_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8, num_stages=2,
    )

    return dense

def AspB(A: Tensor, B_sparse: BitsparseTensor) -> Tensor:
    """Compute ``A @ B`` where ``B`` is stored as ``BitsparseTensor``.

    Shapes: ``A[P, M]`` and sparse ``B[M, N]`` produce ``out[P, N]``.
    The current implementation unpacks ``B`` to dense before matmul.
    """
    vals = B_sparse.vals
    bitmask = B_sparse.bitmask
    prefix = B_sparse.prefix
    BLOCK_M, BLOCK_N = B_sparse.BLOCK_M, B_sparse.BLOCK_N
    grid_m, grid_n = B_sparse.grid_m, B_sparse.grid_n
    M, N = B_sparse.shape

    TILE_NUMEL = BLOCK_M * BLOCK_N
    TILE_BYTES = TILE_NUMEL // 8
    num_tiles = grid_m * grid_n

    dense = torch.empty(M, N, device=A.device, dtype=vals.dtype)
    _unpack_batch_kernel[(num_tiles,)](
        vals, bitmask, prefix,
        dense,
        0, grid_n, N, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        TILE_NUMEL=TILE_NUMEL, TILE_BYTES=TILE_BYTES,
        num_warps=8, num_stages=2,
    )
    return A @ dense


def FFN_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(x @ W1.T) @ W2.T`` using sparse saved ``z``.

    Shapes: ``grad_output[B, D]``, ``x[B, D]``, ``W1[H, D]``, ``W2[D, H]``.
    Gradients are ``dW2 = grad_output.T @ z``, ``dz = grad_output @ W2``,
    ``dpreact = dz * (z > 0)``, ``dx = dpreact @ W1``, and
    ``dW1 = dpreact.T @ x``.
    """
    x, W1, W2 = ctx.saved_tensors
    z = ctx.z_sparse
    ctx.z_sparse = None
    needs_x = ctx.needs_input_grad[0]

    grad_W2 = AspB(grad_output.T, z)

    grad_z = grad_output @ W2
    _mask_with_bitmask_kernel[(z.grid_m, z.grid_n)](
        grad_z, z.bitmask,
        z.shape[0], z.shape[1],
        BLOCK_M=z.BLOCK_M, BLOCK_N=z.BLOCK_N,
        TILE_BYTES=z.BLOCK_M * z.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )
    del z

    if needs_x:
        grad_x = grad_z @ W1
    else:
        grad_x = None

    grad_W1 = grad_z.T @ x
    return grad_x, grad_W1, grad_W2, None

def print_memory(msg):
    memory = torch.cuda.memory_allocated("cuda")/1024**2
    print(f'{msg}: {memory:.2f} MB')

def print_nbytes(t):
    print(f'{t.nbytes/1024**2:.1f}MB')

def FFN3_backward(ctx, grad_output: Tensor):
    """Backward for ``y = relu(relu(x @ W1.T) @ W2.T) @ W3.T`` using sparse caches."""
    x, W1, W2, W3 = ctx.saved_tensors
    z1 = ctx.z1_sparse
    z2 = ctx.z2_sparse
    ctx.z1_sparse = None
    ctx.z2_sparse = None
    needs_x = ctx.needs_input_grad[0]
    print_memory("Start allocated")

    grad_W3 = AspB(grad_output.T, z2)

    grad_z2 = grad_output @ W3
    _mask_with_bitmask_kernel[(z2.grid_m, z2.grid_n)](
        grad_z2, z2.bitmask,
        z2.shape[0], z2.shape[1],
        BLOCK_M=z2.BLOCK_M, BLOCK_N=z2.BLOCK_N,
        TILE_BYTES=z2.BLOCK_M * z2.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )
    del z2, grad_output

    grad_W2 = AspB(grad_z2.T, z1)

    z1.vals = None

    grad_z1 = grad_z2 @ W2

    _mask_with_bitmask_kernel[(z1.grid_m, z1.grid_n)](
        grad_z1, z1.bitmask,
        z1.shape[0], z1.shape[1],
        BLOCK_M=z1.BLOCK_M, BLOCK_N=z1.BLOCK_N,
        TILE_BYTES=z1.BLOCK_M * z1.BLOCK_N // 8,
        num_warps=4, num_stages=2,
    )
    del z1, grad_z2

    # ctx.maybe_clear_saved_tensors()
    if needs_x:
        grad_x = grad_z1 @ W1
    else:
        grad_x = None
    grad_W1 = grad_z1.T @ x
    del grad_z1

    print_memory("Alloc at end of block")


    return grad_x, grad_W1, grad_W2, grad_W3
