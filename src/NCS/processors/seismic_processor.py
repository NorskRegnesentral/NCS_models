"""Seismic data processor for ViT-based models."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from transformers import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor


NormalizationMode = Literal["std", "zscore"]


class SeismicProcessor(BaseImageProcessor):
    """Preprocesses seismic crops for ViT-based seismic foundation models.

    Handles seismic normalization, sigma clipping, axis reordering,
    and tensor conversion. Subclasses ``BaseImageProcessor`` so it can be
    serialized with ``save_pretrained`` / ``from_pretrained``.

    Args:
        do_normalize: Whether to apply normalization.
        normalize_mean: Cube-wide mean (set before inference via ``compute_cube_stats``).
        normalize_std: Cube-wide standard deviation.
        normalization_mode: ``"std"`` divides by the cube standard deviation only.
            ``"zscore"`` subtracts the mean and divides by the standard deviation.
        clip_sigma: Number of standard deviations for sigma clipping after normalization.
        model_type: One of ``"vit3d"``, ``"vit25d"``, ``"vit"`` — controls axis ordering.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_normalize: bool = True,
        normalize_mean: float | None = None,
        normalize_std: float | None = None,
        normalization_mode: NormalizationMode = "std",
        clip_sigma: float = 3.0,
        model_type: str = "vit3d",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if normalization_mode not in ("std", "zscore"):
            raise ValueError(f"Unknown normalization_mode: {normalization_mode}")
        self.do_normalize = do_normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.normalization_mode = normalization_mode
        self.clip_sigma = clip_sigma
        self.model_type = model_type

    def preprocess(
        self,
        images: np.ndarray | list[np.ndarray],
        do_normalize: bool | None = None,
        normalize_mean: float | None = None,
        normalize_std: float | None = None,
        normalization_mode: NormalizationMode | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:  # ty:ignore[invalid-method-override]
        """Normalize and prepare seismic crops for model input.

        Args:
            images: A single crop or list of crops as numpy arrays.

                - **3D** (``model_type="vit3d"``): each crop has shape ``(il, xl, t)`` in
                  seismic order. Converted to ``(t, il, xl)`` for PyTorch, then expanded
                  to ``(B, 1, T, H, W)``.
                - **2D** (``model_type="vit"``): each crop has shape ``(spatial, time)`` in seismic order.
                  Converted to ``(time, spatial)`` for PyTorch, then expanded to ``(B, 1, H, W)``.
                - **2.5D** (``model_type="vit25d"``): each crop has shape
                  ``(n_views, spatial, time)`` in seismic order. Converted to ``(n_views, time, spatial)`` for PyTorch, then expanded to ``(B, V, H, W)``.

            do_normalize: Override the instance ``do_normalize`` setting.
            normalize_mean: Override the instance ``normalize_mean``.
            normalize_std: Override the instance ``normalize_std``.
            normalization_mode: Override the instance ``normalization_mode``.
            return_tensors: Tensor type for output (default ``"pt"`` for PyTorch).

        Returns:
            ``BatchFeature`` with key ``"pixel_values"``.
        """
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        normalize_mean = normalize_mean if normalize_mean is not None else self.normalize_mean
        normalize_std = normalize_std if normalize_std is not None else self.normalize_std
        normalization_mode = normalization_mode if normalization_mode is not None else self.normalization_mode

        if isinstance(images, np.ndarray):
            image_list = [images] if images.ndim == self._expected_crop_ndim() else list(images)
        elif isinstance(images, Sequence) and not isinstance(images, str | bytes):
            image_list = [np.asarray(image) for image in images]
        else:
            raise TypeError("images must be a numpy array or a sequence of numpy arrays.")

        if return_tensors == "pt":
            pixel_values = self._preprocess_torch(
                image_list,
                do_normalize,
                normalize_mean,
                normalize_std,
                normalization_mode,
            )
            return BatchFeature(data={"pixel_values": pixel_values})

        processed = []
        for crop in image_list:
            crop = np.asarray(crop, dtype=np.float32)

            if do_normalize:
                if normalize_std is None:
                    raise ValueError(
                        "normalize_std must be set when do_normalize=True. "
                        "Use compute_cube_stats() to compute it from the seismic file."
                    )
                if normalization_mode == "zscore":
                    if normalize_mean is None:
                        raise ValueError(
                            "normalize_mean must be set when normalization_mode='zscore'. "
                            "Use compute_cube_stats() to compute it from the seismic file."
                        )
                    crop = (crop - normalize_mean) / (normalize_std + 1e-10)
                else:
                    crop = crop / (normalize_std + 1e-10)
                crop = np.clip(crop, -self.clip_sigma, self.clip_sigma)

            if self.model_type == "vit3d":
                # (il, xl, t) -> (t, il, xl), then add channel dim -> (1, T, H, W)
                crop = seismic_to_torch_3d(crop)
                crop = crop[np.newaxis]  # (1, T, H, W)
            elif self.model_type == "vit":
                # (spatial, time) -> (time, spatial), then add channel dim -> (3, H, W)
                crop = seismic_to_torch_2d(crop)
                crop = crop[np.newaxis]  # (1, H, W)
                crop = np.repeat(crop, 3, axis=0)  # (3, H, W)
            elif self.model_type == "vit25d":
                # (n_views, spatial, time) -> (n_views, time, spatial) for PyTorch. No channel dim.
                crop = seismic_to_torch_2d(crop)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            processed.append(crop)

        pixel_values = np.stack(processed, axis=0)  # (B, ...)
        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)

    def _preprocess_torch(
        self,
        images: list[np.ndarray],
        do_normalize: bool,
        normalize_mean: float | None,
        normalize_std: float | None,
        normalization_mode: NormalizationMode,
    ) -> torch.Tensor:
        crop_tensor = torch.as_tensor(np.asarray(images, dtype=np.float32))
        if crop_tensor.ndim == self._expected_crop_ndim():
            crop_tensor = crop_tensor.unsqueeze(0)

        if do_normalize:
            if normalize_std is None:
                raise ValueError(
                    "normalize_std must be set when do_normalize=True. "
                    "Use compute_cube_stats() to compute it from the seismic file."
                )
            if normalization_mode == "zscore":
                if normalize_mean is None:
                    raise ValueError(
                        "normalize_mean must be set when normalization_mode='zscore'. "
                        "Use compute_cube_stats() to compute it from the seismic file."
                    )
                crop_tensor = crop_tensor.sub(normalize_mean).div(normalize_std + 1e-10)
            else:
                crop_tensor = crop_tensor.div(normalize_std + 1e-10)
            crop_tensor = crop_tensor.clamp(-self.clip_sigma, self.clip_sigma)

        if self.model_type == "vit3d":
            crop_tensor = crop_tensor.movedim(-1, 1).unsqueeze(1)
        elif self.model_type == "vit":
            crop_tensor = crop_tensor.movedim(-1, 1).unsqueeze(1).repeat(1, 3, 1, 1)
        elif self.model_type == "vit25d":
            crop_tensor = crop_tensor.movedim(-1, -2)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return crop_tensor.contiguous()

    def _expected_crop_ndim(self) -> int:
        if self.model_type == "vit3d":
            return 3  # (il, xl, t)
        elif self.model_type == "vit":
            return 2  # (time, spatial)
        elif self.model_type == "vit25d":
            return 3  # (n_views, time, spatial)
        return 3


def seismic_to_torch_3d(data: np.ndarray) -> np.ndarray:
    """Permute seismic order ``(il, xl, t)`` to PyTorch order ``(t, il, xl)``.

    Works on bare numpy arrays (no batch/channel dims).
    """
    return np.moveaxis(data, -1, -3)  # (il, xl, t) -> (t, il, xl)


def seismic_to_torch_2d(data: np.ndarray) -> np.ndarray:
    """Permute seismic order ``(il/xl, t)`` to PyTorch order ``(t, il/xl)``.

    Works on bare numpy arrays (no batch/channel dims).
    """
    return np.moveaxis(data, -1, -2)  # (il/xl, t) -> (t, il/xl)
