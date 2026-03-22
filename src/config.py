"""centralized configuration for paths, model registry, and environment."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIR_ASSETS = ROOT / "assets"
DIR_DATA = ROOT / "data"
DIR_PARAMS = ROOT / "params"
DIR_FONTS = DIR_ASSETS / "fonts"

PATH_CORPUS = DIR_ASSETS / "1000-corpus.txt"
PATH_TARGETS = DIR_ASSETS / "2014-targets.txt"
PATH_PRIME_TYPES = DIR_ASSETS / "2014-prime-types.txt"
PATH_PRIME_DATA = DIR_ASSETS / "2014-prime-data.json"
PATH_RANDOM_STRINGS = DIR_ASSETS / "random_strings.json"
PATH_NORM_STATS = DIR_DATA / "normalization_stats.json"

IMAGE_SIZE = 224

os.environ["TORCH_HOME"] = str(DIR_ASSETS / "model_cache")

_MODEL_REGISTRY = None


def _get_model_registry() -> dict[str, tuple]:
    """lazily build the model registry on first access."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        from torchvision import models

        _MODEL_REGISTRY = {
            "alexnet": (models.alexnet, models.AlexNet_Weights.DEFAULT),
            "densenet169": (models.densenet169, models.DenseNet169_Weights.DEFAULT),
            "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
            "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT),
            "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
            "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.DEFAULT),
            "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT),
            "vit_l_32": (models.vit_l_32, models.ViT_L_32_Weights.DEFAULT),
        }
    return _MODEL_REGISTRY


def __getattr__(name: str):
    """lazy module-level attribute access for MODEL_REGISTRY."""
    if name == "MODEL_REGISTRY":
        return _get_model_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_model(name: str):
    """instantiate a single pretrained model by name from the registry."""
    factory, weights = _get_model_registry()[name]
    return factory(weights=weights)
