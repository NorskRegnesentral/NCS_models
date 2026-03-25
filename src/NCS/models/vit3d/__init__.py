from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

from .configuration_vit3d import ViT3DConfig
from .configuration_vit3d_mae import ViT3DMAEConfig
from .modeling_vit3d import ViT3DForImageClassification, ViT3DModel


AutoConfig.register("vit3d", ViT3DConfig)
AutoConfig.register("vit3d_mae", ViT3DMAEConfig)
AutoModel.register(ViT3DConfig, ViT3DModel)
AutoModelForImageClassification.register(ViT3DConfig, ViT3DForImageClassification)
