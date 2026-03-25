# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
# Original source:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
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
"""PyTorch ViT-2.5D model."""

import collections.abc
from collections.abc import Callable

import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.vit.modeling_vit import (
    ViTConfig,
    ViTIntermediate,
    ViTOutput,
    ViTPooler,
    ViTPreTrainedModel,
    ViTSelfOutput,
    eager_attention_forward,
)
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging, torch_int

from NCS.models.common.qk_norm_attention import L2Norm, ScaleUp, scaling_factor
from NCS.models.vit25d.configuration_vit25d import ViT25DConfig


logger = logging.get_logger(__name__)


class ViT25DEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings. Optionally, also the mask token."""

    def __init__(self, config: ViT25DConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViT25DPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        pos_emb_channels = len(config.directions)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches * pos_emb_channels + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int, num_directions: int
    ) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        per_direction_num_positions = num_positions // len(self.config.directions)

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if (
            not torch.jit.is_tracing()
            and num_patches == num_positions
            and height == width
            and num_directions == len(self.config.directions)
        ):
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_per_direction_num_positions = torch_int(per_direction_num_positions**0.5)
        # Interpolating each channel/direction separately
        patch_pos_embed = patch_pos_embed.reshape(
            len(self.config.directions), sqrt_per_direction_num_positions, sqrt_per_direction_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_height, new_width), mode="bicubic", align_corners=False
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
        directions=None,
    ) -> torch.Tensor:
        batch_size, num_directions, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if directions is not None:
            # Directions are indices of [0,1,...,num_directions-1], which should be a subset of the directions in the config
            directions = directions.int()

            if directions.max() >= len(self.config.directions) or directions.min() < 0:
                raise ValueError(
                    f"The specified directions are not compatible with the configuration directions.\n\
                    Config directions: {self.config.directions}, got directions indices: {directions}."
                )

            if interpolate_pos_encoding:
                num_patches = (height // self.patch_size) ** 2
                pos_emb_indices = (
                    torch.arange(1, num_patches + 1, device=directions.device, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, num_directions, 1)
                )

                # add direction offset to pos emb indices, directions are indices of [0,1,...,num_directions-1]
                pos_emb_indices += directions[:, :, None] * num_patches
                pos_emb_indices = pos_emb_indices.reshape(batch_size, -1)
                interpolated_pos_embeddings = self.interpolate_pos_encoding(embeddings, height, width, num_directions)
                embeddings[:, 1:, :] += interpolated_pos_embeddings[0, pos_emb_indices, :]
                embeddings[:, 0, :] += interpolated_pos_embeddings[0, 0, :]  # add cls token pos emb

            else:
                # indices of pos emb [1,...,num_patches]
                pos_emb_indices = (
                    torch.arange(1, self.num_patches + 1, device=directions.device, dtype=torch.int32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(batch_size, num_directions, 1)
                )

                # add direction offset to pos emb indices, directions are indices of [0,1,...,num_directions-1]
                pos_emb_indices += directions[:, :, None] * self.num_patches
                pos_emb_indices = pos_emb_indices.reshape(batch_size, -1)

                # add position embeddings w/o cls token
                embeddings[:, 1:, :] += self.position_embeddings[0, pos_emb_indices, :]
                embeddings[:, 0, :] += self.position_embeddings[0, 0, :]  # add cls token pos emb
        else:
            # Assuming input is in the same order as the config directions and the same number of directions.
            if interpolate_pos_encoding:
                if num_directions != len(self.config.directions):
                    raise ValueError(
                        f"Either specify the directions as input or make sure that the number of directions in the input matches the configuration.\n\
                        Expected {len(self.config.directions)} but got {num_directions} directions."
                    )
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width, num_directions)
            else:
                embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViT25DPatchEmbeddings(nn.Module):
    """This class turns `pixel_values` of shape `(batch_size, num_directions, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config) -> None:
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_directions, hidden_size = len(config.directions), config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_directions = num_directions
        self.num_patches = num_patches
        self.hidden_size = hidden_size

        # NOTE: using the same projection for each view
        self.projection = nn.Conv2d(1, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_directions, height, width = pixel_values.shape
        # NOTE: we allow for different number of directions in the input than in the config
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = (
            self.projection(pixel_values.reshape(batch_size * num_directions, 1, height, width))
            .flatten(2)
            .transpose(1, 2)
        )
        embeddings = embeddings.reshape(batch_size, -1, self.hidden_size)
        return embeddings


class ViT25DPreTrainedModel(ViTPreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config = ViT25DConfig
    config_class = ViT25DConfig
    base_model_prefix = "vit25d"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViT25DEmbeddings", "ViTLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear | nn.Conv2d):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViT25DEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.cls_token.dtype)


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention
class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False
        self.qk_norm = config.qk_norm
        self.scale_norm = config.scale_norm
        self.learnable_scale = config.learnable_scale
        if self.learnable_scale:
            self.qk_scale = ScaleUp(scaling_factor(config.sequence_threshold))
            self.scaling = 1.0
        else:
            self.qk_scale = nn.Identity()
        if config.norm_layer.lower() == "layernorm":
            norm_layer = nn.LayerNorm
        elif config.norm_layer.lower() == "rmsnorm":
            # norm_layer = partial(nn.RMSNorm, eps=config.rms_norm_eps)
            norm_layer = nn.RMSNorm
        elif config.norm_layer.lower() == "l2norm":
            norm_layer = L2Norm
        else:
            error_msg = f"Could not find norm layer: {config.norm_layer}"
            raise KeyError(error_msg)

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.q_norm = norm_layer(self.attention_head_size) if self.qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.attention_head_size) if self.qk_norm else nn.Identity()

        self.norm = norm_layer(self.all_head_size) if self.scale_norm else nn.Identity()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: torch.Tensor | None = None, output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

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
            self.qk_scale(key_layer),
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        context_layer = self.norm(context_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTAttention
class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
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
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = None, output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTLayer
class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = None, output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
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

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder
class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple | BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__, hidden_states, layer_head_mask, output_attentions
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )


class ViT25DModel(ViT25DPreTrainedModel):
    def __init__(self, config: ViT25DConfig, add_pooling_layer: bool = False, use_mask_token: bool = False) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = ViT25DEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViT25DPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]) -> None:
        """Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        bool_masked_pos: torch.BoolTensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
        return_dict: bool | None = None,
        directions: torch.IntTensor | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        r"""bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
        Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
            directions=directions,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViT25DForImageClassification(ViT25DPreTrainedModel):
    def __init__(self, config: ViT25DConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViT25DModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
        return_dict: bool | None = None,
        directions: torch.IntTensor | None = None,
    ) -> tuple | ImageClassifierOutput:
        r"""Labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            directions=directions,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss, *output)) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


if __name__ == "__main__":
    # directions = ["dir0", "dir45", "dir90", "dir135"]
    dtype = torch.float32
    img_size = 512
    num_views = 2
    # vit25d_config = ViT25DConfig(mask_ratio=0.0, directions=directions)

    model = ViT25DForImageClassification.from_pretrained(
        "/nr/project/bild/VI-energy/seismic_cbir_data/trained_models/experiments_bamjo/seismic-cbir-pretrain-vit25d-mae-no-norm-crop-224",
        # config=vit25d_config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )

    outputs = model(
        pixel_values=torch.randn(1, num_views, img_size, img_size),
        directions=torch.tensor([[1, 2]]),
        # output_hidden_states=True,
        interpolate_pos_encoding=True,
    )
