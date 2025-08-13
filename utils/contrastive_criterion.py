import torch
import torch.nn as nn


class ContrastiveCriterion(nn.Module):
    """
    Criterion for contrastive loss in memory modules
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        """
        Extract contrastive loss from model output
        """
        if 'contrastive_loss' in input:
            return input['contrastive_loss']
        else:
            return torch.tensor(0.0, device=input['pred'].device)


class ChannelContrastiveCriterion(nn.Module):
    """
    Criterion for channel memory contrastive loss
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        if 'channel_contrastive' in input:
            return input['channel_contrastive']
        else:
            return torch.tensor(0.0, device=input['pred'].device)


class SpatialContrastiveCriterion(nn.Module):
    """
    Criterion for spatial memory contrastive loss
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        if 'spatial_contrastive' in input:
            return input['spatial_contrastive']
        else:
            return torch.tensor(0.0, device=input['pred'].device)


class CrossModalContrastiveCriterion(nn.Module):
    """
    Criterion for cross-modal contrastive loss between channel and spatial
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, input):
        if 'cross_modal_contrastive' in input:
            return input['cross_modal_contrastive']
        else:
            return torch.tensor(0.0, device=input['pred'].device)