"""__init__.py - MobileSAM backbone models."""

__version__ = "0.1.0"
from .model import MobileSAMBackbone

__all__ = [
    "MobileSAMBackbone",
    "mobilesam_vit_t"
]


def mobilesam_vit_t(pretrained=True, **kwargs):
    """MobileSAM with Tiny ViT backbone"""
    # Set default model_type if not provided
    if 'model_type' not in kwargs:
        kwargs['model_type'] = 'vit_t'
    return MobileSAMBackbone(pretrained=pretrained, **kwargs)
