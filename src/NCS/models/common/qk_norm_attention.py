# From QK-Norm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import logging


logger = logging.get_logger(__name__)


def scaling_factor(sequence_threshold) -> np.float32:
    """Initialize the scaling factor for QK-Norm."""
    return np.log2((sequence_threshold**2) - sequence_threshold).astype(np.float32)


class ScaleUp(nn.Module):
    ### Learned pararmeter used to scale up QKt before taking the softmax
    """ScaleUp."""

    def __init__(self, scale) -> None:
        """Init."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x) -> torch.Tensor:
        """Scale up tensor."""
        return x * self.scale


class L2Norm(nn.Module):
    """L2 Normalization."""

    def __init__(self, normalized_shape, p=2) -> None:
        """Init."""
        super().__init__()
        # ignoring normalized shape
        self.p = p
        # TODO: add eps?

    def forward(self, x) -> torch.Tensor:
        """L2 normalize along last dim."""
        return F.normalize(x, p=self.p, dim=-1)
