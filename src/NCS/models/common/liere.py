# Portions derived from LieRE:
# https://github.com/StanfordMIMI/LieRE/blob/main/models/rope_vit.py
# Copyright (c) 2024 Stanford MIMI Lab (LieRE)
# Licensed under the MIT License (see NOTICE)
#
# Portions derived from Hugging Face Transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0
#
# Modifications and new code copyright (C) 2025 Norsk Regnesentral
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
"""LieRE (Lie Rotational Positional Encodings) attention layers for ViT-3D."""

import math
from collections.abc import Callable, Iterable

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.vit.modeling_vit import (
    BaseModelOutput,
    ViTIntermediate,
    ViTOutput,
    ViTSelfAttention,
    ViTSelfOutput,
    eager_attention_forward,
)
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging

from NCS.models.vit3d.configuration_vit3d import ViT3DConfig


logger = logging.get_logger(__name__)


class PositionEncoderBase(nn.Module):
    """Base Position Encoder returning normalized x, y, z coordinates in the range [0, 1] for each patch in the cube."""

    def __init__(self, config: ViT3DConfig) -> None:
        """Initialize the position encoder."""
        super().__init__()
        ndims = config.ndims
        cube_size, patch_size = config.cube_size, config.patch_size
        patch_size = patch_size if isinstance(patch_size, Iterable) else tuple(patch_size for _ in range(ndims))
        patch_size = (patch_size[-1], *patch_size[:-1])  # reorder to (depth, height, width)
        self.patch_size = patch_size
        cube_size = cube_size if isinstance(cube_size, Iterable) else tuple(cube_size for _ in range(ndims))
        cube_size = (cube_size[-1], *cube_size[:-1])  # reorder to (depth, height, width)
        self.cube_size = cube_size
        grid_size = tuple((cube_size[i] // patch_size[i]) for i in range(ndims))
        self.grid_size = grid_size

    def forward(self, cube_size: torch.Tensor) -> torch.Tensor:
        assert cube_size.shape[0] == 3, "cube_size should be a tensor of shape [3] representing (depth, height, width)"
        steps_per_axis = tuple(c for c in cube_size.tolist())

        normalized_positions = torch.cartesian_prod(
            *(torch.linspace(0, 1, steps, device=cube_size.device) for steps in steps_per_axis)
        )
        return normalized_positions.unsqueeze(0)


class LieREPositionEncoder(PositionEncoderBase):
    """Position Encoder for LieRE (Generalized Rotary Position Embeddings)."""

    def __init__(self, config: ViT3DConfig) -> None:
        """Initialize the LieRE position encoder."""
        super().__init__(config)
        self.ndims = config.ndims
        self.block_diag = config.block_diag
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        self.head_dim = (
            embed_dim // num_heads // self.block_diag
        )  # head_dim is the dimension of each attention head divided by the number of block-diagonal size

        # initialize the generator parameters
        # https://github.com/naver-ai/rope-vit/blob/c6aa201ee795daa4f841e2f9585164bb23a0b819/deit/models_v2_rope.py#L150C13-L150C76

        self.generator_raw_params = nn.Parameter(
            torch.rand(
                self.ndims,  # number of dimensions
                self.block_diag,  # Replace with proper value if you want to use a block-diagonal generator
                self.head_dim,
                self.head_dim,
            )
            * math.pi
            * 2  # RoPE-Mixed scaled by 2 pi, scaling by a constant https://github.com/naver-ai/rope-vit/blob/c6aa201ee795daa4f841e2f9585164bb23a0b819/deit/models_v2_rope.py#L25
        )

    def forward(self, volume_size: torch.Tensor, volume_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Generate the LieRE positional encodings.

        Args:
            volume_size (tuple[int, int, int]): Tuple representing the size of the cube.
            volume_indices (torch.Tensor | None): Tensor of shape [bs, ..., ndims] representing the indices in the volume.

        Returns:
            torch.Tensor: LieRE positional encodings of shape [bs, num_tokens, ndims, num_generators, head_dim, head_dim].
        """
        # Shape: [bs, num_tokens, dimensionality]
        org_dtype = self.generator_raw_params.dtype
        if volume_indices is None:
            positions = super().forward(volume_size)
        else:
            batch_size = volume_indices.shape[0]
            volume_indices = volume_indices.reshape(batch_size, -1, 3)
            VD, VH, VW = volume_size.tolist()

            # Normalize the volume indices to [0, 1] range
            volume_indices = volume_indices.float() / torch.tensor(
                [VD - 1, VH - 1, VW - 1], device=volume_indices.device, dtype=volume_indices.dtype
            ).view(1, 1, 3)

            # volume_indices is a tensor of shape [bs, num_tokens, dimensionality]
            positions = volume_indices

        # Shape: [generator_repeats, ndims, num_generators, generator_dim, generator_dim]
        upper_triangle = torch.triu(self.generator_raw_params, diagonal=1) - 0.5

        # Shape: [generator_repeats, ndims, num_generators, generator_dim, generator_dim]
        skew_bases = upper_triangle - torch.transpose(upper_triangle, -1, -2)

        # Shape: [bs, num_tokens, dimensionality, number_generators, head_dim, head_dim]
        in_basis_positions = positions.reshape(list(positions.shape) + [1] * 3) * skew_bases

        generator_pos = torch.sum(in_basis_positions, dim=-4)  # sum over dimensions

        # add an identity for the CLS token.
        cls_generator = torch.zeros_like(generator_pos[:, 0, ...]).unsqueeze(1)
        generator_pos = torch.cat((cls_generator, generator_pos), dim=1)

        return torch.matrix_exp(generator_pos.to(dtype=torch.float32)).to(dtype=org_dtype)


class ViTSelfLieREAttention(ViTSelfAttention):
    """Multi-head Attention block with generalized rotary position embeddings."""

    def apply_transforms(self, x: torch.Tensor, positional_transforms: torch.Tensor) -> torch.Tensor:
        """Apply the LieRE positional transforms to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_heads, num_tokens, head_size].
            positional_transforms (torch.Tensor): LieRE positional transforms of shape
                [num_tokens, num_rotators, generator_dim, generator_dim].

        Returns:
            torch.Tensor: Transformed tensor of shape [batch_size, num_heads, num_tokens, head_size].
        """
        generator_dim = positional_transforms.shape[-1]
        batch_size, num_heads, num_tokens, head_size = x.shape
        num_rotators = positional_transforms.shape[-3]
        # ipdb.set_trace()
        rotatable_dim = generator_dim * num_rotators
        assert head_size == self.attention_head_size  # self.dim_head, "Head dims and head size have to be the same"
        # Shape: [batch size, heads_num, tokens_num, head_dim]
        rotatable_states = x[..., :rotatable_dim]
        unrotatable_states = x[..., rotatable_dim:]
        states_split = rotatable_states.reshape((batch_size, num_heads, num_tokens, num_rotators, generator_dim, 1))
        # Shape: [batch, num_rotator , 65, heads_num, generator_dim, generator_dim]
        positional_transforms = positional_transforms.reshape(
            (
                batch_size,
                num_tokens,
                num_rotators,
                generator_dim,
                generator_dim,
            )
        )
        rotated_states = torch.matmul(positional_transforms[:, None], states_split)
        # rotated_states_einsum = torch.einsum("BNRIJ, BHMRKL -> BHMRIL",positional_transforms,states_split)
        return torch.cat([rotated_states.flatten(start_dim=-3), unrotatable_states], axis=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        positional_transforms,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        """Compute the attention scores and context layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, num_tokens, hidden_size].
            positional_transforms (torch.Tensor): LieRE positional transforms of shape
                [num_tokens, num_rotators, generator_dim, generator_dim].
            head_mask (torch.Tensor | None): Optional tensor to mask heads.
            output_attentions (bool): Whether to return attention probabilities.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]: A tuple containing the context layer and
                optionally the attention probabilities.
        """
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if positional_transforms is not None:
            # k is transformed in the next step.
            query_layer, key_layer = (
                self.apply_transforms(query_layer, positional_transforms),
                self.apply_transforms(key_layer, positional_transforms),
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTLieREAttention(nn.Module):
    def __init__(self, config: ViT3DConfig) -> None:
        super().__init__()
        self.attention = ViTSelfLieREAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_transforms: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_outputs = self.attention(hidden_states, positional_transforms, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTLieRELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViT3DConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTLieREAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_transforms: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            positional_transforms,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output, *outputs)

        return outputs


class ViTLieREEncoder(nn.Module):
    def __init__(self, config: ViT3DConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLieRELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        ndims = config.ndims

        cube_size, patch_size = config.cube_size, config.patch_size
        cube_size = cube_size if isinstance(cube_size, Iterable) else tuple(cube_size for _ in range(ndims))
        cube_size = (cube_size[-1], *cube_size[:-1])  # reorder to (depth, height, width)
        patch_size = patch_size if isinstance(patch_size, Iterable) else (patch_size for _ in range(ndims))
        patch_size = (patch_size[-1], *patch_size[:-1])  # reorder to (depth, height, width)
        self.cube_size = cube_size
        self.patch_size = patch_size

        self.position_encoder = LieREPositionEncoder(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        volume_indices: torch.Tensor | None = None,
        volume_size: torch.Tensor | None = None,
    ) -> tuple | BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if volume_size is None:
            volume_size = torch.tensor(self.cube_size, device=hidden_states.device)

            if volume_indices is not None:
                # Check if volume_indices are within the bounds of the cube size
                if torch.any(volume_indices < 0) or torch.any(volume_indices >= volume_size):
                    raise ValueError(
                        f"volume_indices contain values outside the bounds of volume_size {self.cube_size}. "
                        f"Got volume_indices with min {volume_indices.min().item()} and max {volume_indices.max().item()}."
                    )

        positional_transforms = self.position_encoder(
            volume_indices=volume_indices,
            volume_size=volume_size,
        )

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = (*all_hidden_states, hidden_states)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__, hidden_states, positional_transforms, layer_head_mask, output_attentions
                )  # type: ignore
            else:
                layer_outputs = layer_module(hidden_states, positional_transforms, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = (*all_self_attentions, layer_outputs[1])

        if output_hidden_states:
            all_hidden_states = (*all_hidden_states, hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
