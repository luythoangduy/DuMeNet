import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalRegionLoss(nn.Module):
    """
    Focal Region Loss from "Region-Attention-Transformer-for-Medical-Image-Restoration"
    Adapted for anomaly detection
    """
    def __init__(self, weight, beta=1.0, epsilon=1e-3, loss_type='l1'):
        super(FocalRegionLoss, self).__init__()
        self.weight = weight
        self.epsilon2 = epsilon * epsilon
        self.beta = beta
        self.loss_type = loss_type
        
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        mask = input.get("mask", None)
        
        if mask is None:
            mask = self.create_pseudo_mask(feature_rec, feature_align)
        
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        
        return self.compute_focal_region_loss(feature_rec, feature_align, mask)
    
    def create_pseudo_mask(self, feature_rec, feature_align, n_regions=5):
        b, c, h, w = feature_rec.shape
        
        if self.loss_type == 'l1':
            error = torch.abs(feature_rec - feature_align)
        else:
            error = (feature_rec - feature_align) ** 2
        
        error = torch.mean(error, dim=1)
        
        pseudo_masks = []
        for bi in range(b):
            error_flat = error[bi].flatten().cpu().numpy()
            quantiles = np.linspace(0, 1, n_regions + 1)
            thresholds = np.quantile(error_flat, quantiles)
            
            mask_bi = torch.zeros_like(error[bi])
            for i in range(n_regions):
                if i == n_regions - 1:
                    region_mask = error[bi] >= thresholds[i]
                else:
                    region_mask = (error[bi] >= thresholds[i]) & (error[bi] < thresholds[i + 1])
                mask_bi[region_mask] = i
            
            pseudo_masks.append(mask_bi)
        
        return torch.stack(pseudo_masks, dim=0)
    
    def compute_focal_region_loss(self, pred, target, mask):
        if self.loss_type == 'l1':
            loss_metric = F.l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            loss_metric = F.mse_loss(pred, target, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        loss_metric = torch.mean(loss_metric, dim=1)
        weight = loss_metric.clone().detach()
        
        b = weight.shape[0]
        for bi in range(b):
            mask_bi = mask[bi]
            total_area = 0
            
            for cls_i in range(int(mask_bi.max().item()) + 1):
                region = (mask_bi == cls_i)
                area_i = torch.sum(region)
                
                if area_i > 0:
                    avg_i = torch.mean(weight[bi][region])
                    weight[bi][region] = avg_i
                    total_area += area_i
        
        weight = weight / (weight.max() + self.epsilon2)
        weight = torch.clamp(weight, min=0.0, max=1.0)
        
        weighted_loss = loss_metric * (weight * self.beta + 1)
        return torch.mean(weighted_loss)


class FeatureFocalRegionLoss(nn.Module):
    """Direct replacement for FeatureMSELoss using Focal Region Loss"""
    def __init__(self, weight, beta=1.0, epsilon=1e-3, loss_type='l1'):
        super().__init__()
        self.weight = weight
        self.focal_region_loss = FocalRegionLoss(
            weight=1.0, beta=beta, epsilon=epsilon, loss_type=loss_type
        )
    
    def forward(self, input):
        return self.focal_region_loss(input)


class CombinedMSEFocalRegionLoss(nn.Module):
    """Combines MSE loss and Focal Region Loss"""
    def __init__(self, weight, mse_weight=0.4, focal_weight=0.6, 
                 beta=1.0, epsilon=1e-3, loss_type='l1'):
        super().__init__()
        self.weight = weight
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        
        self.mse_loss = nn.MSELoss()
        self.focal_region_loss = FocalRegionLoss(
            weight=1.0, beta=beta, epsilon=epsilon, loss_type=loss_type
        )
    
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        
        mse_component = self.mse_loss(feature_rec, feature_align)
        focal_component = self.focal_region_loss(input)
        
        total_loss = self.mse_weight * mse_component + self.focal_weight * focal_component
        return total_loss


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


class RegionFocalLoss(nn.Module):
    """
    Region Focal Loss for anomaly detection
    """
    
    def __init__(self, weight, alpha=0.25, gamma=2.0, reduction='mean', 
                 region_weight_factor=2.0, smooth_factor=1e-6):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.region_weight_factor = region_weight_factor
        self.smooth_factor = smooth_factor
        
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        mask = input.get("mask", None)
        
        # Compute reconstruction error
        reconstruction_error = torch.mean((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        
        if mask is not None:
            return self.compute_region_focal_loss(reconstruction_error, mask)
        else:
            return self.compute_adaptive_focal_loss(reconstruction_error)
    
    def compute_region_focal_loss(self, pred_error, gt_mask):
        # Normalize prediction error to [0, 1]
        pred_error_norm = torch.sigmoid(pred_error)
        targets = gt_mask.float()
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy(pred_error_norm, targets, reduction='none')
        p_t = pred_error_norm * targets + (1 - pred_error_norm) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t + self.smooth_factor) ** self.gamma
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Apply additional weight for anomaly regions
        region_weights = 1.0 + (self.region_weight_factor - 1.0) * targets
        focal_loss = focal_loss * region_weights
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def compute_adaptive_focal_loss(self, pred_error):
        pred_error_norm = torch.sigmoid(pred_error)
        threshold = torch.quantile(pred_error_norm.flatten(), 0.9)
        pseudo_targets = (pred_error_norm > threshold).float()
        
        ce_loss = F.binary_cross_entropy(pred_error_norm, pseudo_targets, reduction='none')
        p_t = pred_error_norm * pseudo_targets + (1 - pred_error_norm) * (1 - pseudo_targets)
        alpha_t = self.alpha * pseudo_targets + (1 - self.alpha) * (1 - pseudo_targets)
        focal_weight = (1 - p_t + self.smooth_factor) ** self.gamma
        
        focal_loss = alpha_t * focal_weight * ce_loss
        region_weights = 1.0 + (self.region_weight_factor - 1.0) * pseudo_targets
        focal_loss = focal_loss * region_weights
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FeatureRegionFocalLoss(nn.Module):
    """
    Direct replacement for FeatureMSELoss using Region Focal Loss
    """
    
    def __init__(self, weight, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.focal_loss = RegionFocalLoss(
            weight=1.0, 
            alpha=alpha, 
            gamma=gamma, 
            reduction=reduction
        )
        
    def forward(self, input):
        return self.focal_loss(input)


class CombinedLoss(nn.Module):
    """
    Combines MSE loss and Region Focal Loss
    """
    
    def __init__(self, weight, mse_weight=0.5, focal_weight=0.5, **focal_kwargs):
        super().__init__()
        self.weight = weight
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        
        self.mse_loss = nn.MSELoss()
        self.focal_loss = RegionFocalLoss(weight=1.0, **focal_kwargs)
        
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        mse_component = self.mse_loss(feature_rec, feature_align)
        focal_component = self.focal_loss(input)
        
        total_loss = self.mse_weight * mse_component + self.focal_weight * focal_component
        return total_loss


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
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
