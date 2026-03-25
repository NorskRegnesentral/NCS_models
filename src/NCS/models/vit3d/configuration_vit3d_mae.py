# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
# Modifications copyright (C) 2025 Norsk Regnesentral
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ViT-3D MAE model configuration."""

import collections.abc
import os
from typing import Literal

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

# DirectionTypes = Literal["dir0", "dir45", "dir90", "dir135", "dir180", "dir225", "dir270", "dir315"]
PosEmbTypes = Literal["3D"]  # , "3DRot", "35DRot"]


class ViT3DMAEConfig(PretrainedConfig):
    r"""This is the configuration class to store the configuration of a [`ViT3DModel`].

    It is used to instantiate an ViT-3D
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        cube_size (`int`, `tuple(int, int, int)`, *optional*, defaults to 224):
            The size (resolution) of each cube, in (inline, crossline, depth) dimensions.
        patch_size (`int`, `tuple(int, int, int)`, *optional*, defaults to 16):
            The size (resolution) of each patch, in (inline, crossline, depth) dimensions.
        num_channels (`int`, *optional*, defaults to 1):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, *optional*, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.
        pos_emb_type (`str`, *optional*, defaults to None):
            The type of positional embeddings to use. Choices are: ['3D', '3DRot', '35DRot'].
            If None, will default to '3D'.

    Example:
    ```python
    >>> from transformers import ViT3DConfig, ViT3DModel

    >>> # Initializing a ViT-3D vit-base-patch16-224 style configuration
    >>> configuration = ViT3DConfig()

    >>> # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
    >>> model = ViT3DModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "vit3d_mae"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        cube_size=224,
        patch_size=16,
        num_channels=1,
        qkv_bias=True,
        encoder_stride=16,
        ########## vit3d specifics ##########
        pos_emb_type: PosEmbTypes | None = None,
        use_ape: bool = False,
        block_diag: int = 2,
        use_liere: bool = True,
        ndims: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        cube_size = cube_size if isinstance(cube_size, collections.abc.Iterable) else (cube_size, cube_size, cube_size)
        self.cube_size = cube_size
        patch_size = (
            patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size, patch_size)
        )
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
        self.use_ape = use_ape
        self.block_diag = block_diag
        self.use_liere = use_liere
        self.ndims = ndims

        ######## parse our custom config ########
        if pos_emb_type is not None:
            if pos_emb_type not in PosEmbTypes.__args__:
                raise ValueError(f"pos_emb_type must be one of {PosEmbTypes}")
            self.pos_emb_type = pos_emb_type
        else:
            self.pos_emb_type = "3D"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            if config_dict["model_type"] != "vit3d_mae":
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )
        if (
            "pos_emb_type" in config_dict
            and hasattr(cls, "pos_emb_type")
            and config_dict["pos_emb_type"] != cls.pos_emb_type
        ):
            logger.warning(
                f"You are using a model with positional embeddings of type {config_dict['pos_emb_type']} to instantiate a model with positional embeddings of type "
                f"{cls.pos_emb_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
