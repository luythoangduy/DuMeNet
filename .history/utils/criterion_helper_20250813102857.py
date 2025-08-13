import torch.nn as nn


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_type = cfg["type"]
        
        # Try to get from current module first
        if loss_type in globals():
            loss_dict[loss_name] = globals()[loss_type](**cfg["kwargs"])
        else:
            # Try to import from contrastive_criterion module
            try:
                from utils.contrastive_criterion import (
                    ContrastiveCriterion,
                    ChannelContrastiveCriterion, 
                    SpatialContrastiveCriterion,
                    CrossModalContrastiveCriterion
                )
                contrastive_classes = {
                    'ContrastiveCriterion': ContrastiveCriterion,
                    'ChannelContrastiveCriterion': ChannelContrastiveCriterion,
                    'SpatialContrastiveCriterion': SpatialContrastiveCriterion,
                    'CrossModalContrastiveCriterion': CrossModalContrastiveCriterion
                }
                if loss_type in contrastive_classes:
                    loss_dict[loss_name] = contrastive_classes[loss_type](**cfg["kwargs"])
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")
            except ImportError:
                raise ValueError(f"Cannot import loss type: {loss_type}")
    return loss_dict
