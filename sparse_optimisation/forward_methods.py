import torch
from torch import Tensor
from torch.autograd import Function

from backward_method import FFN_backward
from sparse_kernels import _row_pack_count_kernel, _row_pack_vals_kernel
from sparse_utils import RowSparseTensor, ValueBuffer


BACKWARD_IMPL = FFN_backward

ROW_BYTES_PER = 0  # computed dynamically


def _row_pack(dense: Tensor, sparse_data: ValueBuffer) -> RowSparseTensor:
    M, N = dense.shape
    ROW_BYTES = (N + 7) // 8
    BLOCK_COLS = 128
    stride_n = dense.stride(0)

    row_bitmask = torch.empty(M * ROW_BYTES, device=dense.device, dtype=torch.uint8)
    row_counts = torch.empty(M, device=dense.device, dtype=torch.int32)

    _row_pack_count_kernel[(M,)](
        dense, row_bitmask, row_counts,
        M, N, stride_n,
        ROW_BYTES=ROW_BYTES,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=4, num_stages=2,
    )

    row_offsets = torch.empty(M + 1, device=dense.device, dtype=torch.int32)
    torch.cumsum(row_counts, 0, out=row_offsets[1:])
    row_offsets[0] = 0
    total_nnz = row_offsets[-1].item()

    layer_offset = sparse_data._offset
    vals_slice = sparse_data.vals[layer_offset:layer_offset + total_nnz]
    scales_slice = sparse_data.row_scales[:M]

    _row_pack_vals_kernel[(M,)](
        dense, row_offsets, vals_slice, scales_slice,
        M, N, stride_n,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=4, num_stages=2,
    )

    sparse_data._offset += total_nnz
    return RowSparseTensor(vals_slice, row_bitmask, row_offsets, scales_slice, dense.shape)


class FFNSparse(Function):
    @staticmethod
    def forward(ctx, x, W1, W2, sparse_data):
        ctx.save_for_backward(x, W1, W2)
        preact = x @ W1.T
        preact.relu_()
        z_sparse = _row_pack(preact, sparse_data)
        ctx.z_sparse = z_sparse
        return preact @ W2.T

    backward = staticmethod(BACKWARD_IMPL)
