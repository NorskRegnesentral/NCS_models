from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

from .configuration_vit25d import ViT25DConfig
from .configuration_vit25d_mae import ViT25DMAEConfig
from .modeling_vit25d import ViT25DForImageClassification, ViT25DModel


AutoConfig.register("vit25d", ViT25DConfig)
AutoConfig.register("vit25d_mae", ViT25DMAEConfig)
AutoModel.register(ViT25DConfig, ViT25DModel)
AutoModelForImageClassification.register(ViT25DConfig, ViT25DForImageClassification)
