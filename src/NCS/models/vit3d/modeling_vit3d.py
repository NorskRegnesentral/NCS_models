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
"""PyTorch ViT-3D model."""

import collections.abc
import math

import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.models.vit.modeling_vit import ViTEncoder, ViTPooler, ViTPreTrainedModel
from transformers.utils import torch_int

from NCS.models.common.liere import ViTLieREEncoder
from NCS.models.common.seismic_conversion_helpers import ravel_multi_index
from NCS.models.vit3d.configuration_vit3d import ViT3DConfig


class ViT3DEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings. Optionally, also the mask token."""

    def __init__(self, config: ViT3DConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViT3DPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        if config.use_ape:
            self.position_embeddings = nn.Parameter(
                torch.randn(1, self.num_patches * config.num_channels + 1, config.hidden_size)
            )
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        cube_size, patch_size = config.cube_size, config.patch_size
        cube_size = cube_size if isinstance(cube_size, collections.abc.Iterable) else (cube_size, cube_size, cube_size)
        cube_size = (cube_size[-1], cube_size[0], cube_size[1])  # reorder to (depth, height, width)
        patch_size = (
            patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size, patch_size)
        )
        patch_size = (patch_size[-1], patch_size[0], patch_size[1])  # reorder to (depth, height, width)
        grid_size = tuple(cube_size[i] // patch_size[i] for i in range(len(cube_size)))
        self.cube_size = cube_size
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, depth: int, height: int, width: int) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width and height == depth:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        new_depth = depth // self.patch_size[0]
        new_height = height // self.patch_size[1]
        new_width = width // self.patch_size[2]

        cube_root_num_positions = torch_int(math.cbrt(num_positions))

        patch_pos_embed = patch_pos_embed.reshape(
            1, cube_root_num_positions, cube_root_num_positions, cube_root_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 4, 1, 2, 3)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_depth, new_height, new_width), mode="trilinear", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
        volume_indices: torch.LongTensor | None = None,
        volume_size: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_volumes, depth, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            has_subvol_indices=volume_indices is not None,
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is None:
            embeddings = self.dropout(embeddings)
            return embeddings

        # add positional encoding to each token
        if interpolate_pos_encoding:
            pe = self.interpolate_pos_encoding(embeddings, depth, height, width)
        else:
            pe = self.position_embeddings

        if volume_indices is not None:
            if volume_size is not None:
                VD, VH, VW = volume_size.tolist()
            else:
                VD, VH, VW = self.grid_size

            volume_indices = volume_indices.reshape(batch_size, -1, 3)

            # ravel multi index to flat indices
            volume_indices = ravel_multi_index(volume_indices, dims=(VD, VH, VW)).unsqueeze(-1)
            # [batch_size, num_patches, 1]

            volume_indices += 1  # add 1 to account for cls token position

            # add cls token position
            cls_token_position = torch.zeros(
                (batch_size, 1, 1), dtype=volume_indices.dtype, device=volume_indices.device
            )
            volume_indices = torch.cat((cls_token_position, volume_indices), dim=1)  # [batch_size, num_patches + 1, 1]

            embeddings = embeddings + torch.gather(
                pe.expand(batch_size, -1, -1), 1, volume_indices.expand_as(embeddings)
            )
        else:
            embeddings = embeddings + pe.expand_as(embeddings)

        embeddings = self.dropout(embeddings)

        return embeddings


class ViT3DPatchEmbeddings(nn.Module):
    """This class turns `pixel_values` of shape `(batch_size, num_volumes, depth, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config) -> None:
        super().__init__()
        cube_size, patch_size = config.cube_size, config.patch_size
        hidden_size = config.hidden_size
        cube_size = cube_size if isinstance(cube_size, collections.abc.Iterable) else (cube_size, cube_size, cube_size)
        cube_size = (cube_size[-1], cube_size[0], cube_size[1])  # reorder to (depth, height, width)
        patch_size = (
            patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size, patch_size)
        )
        patch_size = (patch_size[-1], patch_size[0], patch_size[1])  # reorder to (depth, height, width)
        num_patches = (
            (cube_size[2] // patch_size[2]) * (cube_size[1] // patch_size[1]) * (cube_size[0] // patch_size[0])
        )
        self.cube_size = cube_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_size = hidden_size

        self.projection = nn.Conv3d(1, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False, has_subvol_indices: bool = False
    ) -> torch.Tensor:
        batch_size, num_volumes, depth, height, width = pixel_values.shape
        if not (interpolate_pos_encoding or has_subvol_indices):
            if depth != self.cube_size[0] or height != self.cube_size[1] or width != self.cube_size[2]:
                raise ValueError(
                    f"Input image size ({depth}*{height}*{width}) doesn't match model"
                    f" ({self.cube_size[0]}*{self.cube_size[1]}*{self.cube_size[2]})."
                )
        embeddings = (
            self.projection(pixel_values.reshape(batch_size * num_volumes, 1, depth, height, width))
            .flatten(2)
            .transpose(1, 2)
        )
        embeddings = embeddings.reshape(batch_size, -1, self.hidden_size)
        return embeddings


class ViT3DPreTrainedModel(ViTPreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViT3DConfig
    base_model_prefix = "vit3d"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViT3DEmbeddings", "ViTLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_flash_attn_3 = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module: nn.Linear | nn.Conv3d | nn.LayerNorm) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear | nn.Conv3d):
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
        elif isinstance(module, ViT3DEmbeddings):
            if hasattr(module, "position_embeddings") and module.position_embeddings is not None:
                module.position_embeddings.data = nn.init.trunc_normal_(
                    module.position_embeddings.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
                ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.cls_token.dtype)


class ViT3DModel(ViT3DPreTrainedModel):
    def __init__(self, config: ViT3DConfig, add_pooling_layer: bool = False, use_mask_token: bool = False) -> None:
        super().__init__(config)
        self.config = config
        self.use_liere = config.use_liere

        self.embeddings = ViT3DEmbeddings(config, use_mask_token=use_mask_token)
        if self.use_liere:
            self.encoder = ViTLieREEncoder(config)
        else:
            self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViT3DPatchEmbeddings:
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
        volume_indices: torch.LongTensor | None = None,
        volume_size: torch.LongTensor | None = None,
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
            volume_indices=volume_indices,
            volume_size=volume_size,
        )

        if isinstance(self.encoder, ViTLieREEncoder):
            encoder_outputs = self.encoder(
                embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                volume_indices=volume_indices,
                volume_size=volume_size,
            )
        else:
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


class ViT3DForImageClassification(ViT3DPreTrainedModel):
    def __init__(self, config: ViT3DConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViT3DModel(config, add_pooling_layer=False)

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
        volume_indices: torch.LongTensor | None = None,
        volume_size: torch.LongTensor | None = None,
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
            volume_indices=volume_indices,
            volume_size=volume_size,
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
    cube_size = 224
    num_volumes = 1
    patch_size = 16
    input_size = 224

    num_patches_per_dim = input_size // patch_size

    pixel_values = torch.randn(1, num_volumes, input_size, input_size, input_size)
    XX = YY = ZZ = torch.arange(start=0, end=input_size, step=patch_size)
    volume_indices = (
        torch.cartesian_prod(XX, YY, ZZ)
        .reshape(num_patches_per_dim, num_patches_per_dim, num_patches_per_dim, 3)
        .unsqueeze(0)
        .unsqueeze(0)
    ) + patch_size // 2  # center of each patch
    volume_size = torch.tensor([input_size, input_size, input_size])

    model = ViT3DForImageClassification.from_pretrained(
        "/nr/project/bild/VI-energy/seismic_cbir_data/trained_models/olivia/fall25/FMS-pretrain-vit3d-mae-patch-16-fullstack-time-split-summer-25-uniform-224-97046"
    )
    model.config.num_labels = 10
    labels = torch.tensor([1])
    outputs = model(
        pixel_values, volume_indices=volume_indices, volume_size=volume_size, labels=labels, output_hidden_states=True
    )

    print("model logits:", outputs.logits.shape)
    print("model loss:", outputs.loss)
    print("model hidden states:", outputs.hidden_states[0].shape)
