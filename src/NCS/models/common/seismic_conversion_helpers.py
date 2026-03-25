import torch


def ravel_multi_index(multi_index: torch.LongTensor, dims: tuple[int, ...]) -> torch.LongTensor:
    """PyTorch equivalent of numpy.ravel_multi_index.

    Converts a tensor of indices into a tensor of flat indices.

    Args:
        multi_index: Tensor of indices with shape (..., ndim)
        dims: Tuple of dimension sizes

    Returns:
        Tensor of raveled indices with shape (...)
    """
    ndim = len(dims)
    if multi_index.shape[-1] != ndim:
        raise ValueError(f"Last dimension of multi_index must be {ndim}, got {multi_index.shape[-1]}")

    # Calculate strides
    strides = torch.ones(ndim, dtype=torch.long, device=multi_index.device)
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]

    # Calculate raveled indices
    multi_index_long = multi_index.long()
    flat_indices = torch.zeros(multi_index.shape[:-1], dtype=torch.long, device=multi_index.device)
    for i in range(ndim):
        flat_indices += multi_index_long[..., i] * strides[i]

    return flat_indices
