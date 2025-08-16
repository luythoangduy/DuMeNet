import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


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
        """Create pseudo mask using reconstruction error clustering"""
        b, c, h, w = feature_rec.shape
        
        if self.loss_type == 'l1':
            error = torch.abs(feature_rec - feature_align)
        else:
            error = (feature_rec - feature_align) ** 2
        
        error = torch.mean(error, dim=1)  # [B, H, W]
        
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
        
        return torch.stack(pseudo_masks, dim=0)  # [B, H, W]
    
    def compute_focal_region_loss(self, pred, target, mask):
        """Compute the focal region loss following the original paper"""
        # Compute base loss
        if self.loss_type == 'l1':
            loss_metric = F.l1_loss(pred, target, reduction='none')  # [B, C, H, W]
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            loss_metric = F.mse_loss(pred, target, reduction='none')  # [B, C, H, W]
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # Average across channels to get [B, H, W]
        loss_metric = torch.mean(loss_metric, dim=1)
        
        # Initialize weight same as loss_metric
        weight = loss_metric.clone().detach()
        
        b = weight.shape[0]
        for bi in range(b):
            mask_bi = mask[bi]  # [H, W]
            total_area = 0
            
            # Process each region (following original implementation)
            for cls_i in range(int(mask_bi.max().item()) + 1):
                region = (mask_bi == cls_i)
                area_i = torch.sum(region)
                
                if area_i > 0:
                    # Compute average loss for this region
                    avg_i = torch.mean(weight[bi][region])
                    # Assign this average to all pixels in the region
                    weight[bi][region] = avg_i
                    total_area += area_i
            
            # Verify total area (optional check)
            expected_area = mask_bi.shape[0] * mask_bi.shape[1]
            if total_area != expected_area:
                print(f"Warning: Total area mismatch for batch {bi}: {total_area} vs {expected_area}")
        
        # Normalize weights to [0, 1] (following original paper)
        weight = weight / (weight.max() + self.epsilon2)
        weight = torch.clamp(weight, min=0.0, max=1.0)
        
        # Apply focal weighting: loss * (weight * beta + 1)
        weighted_loss = loss_metric * (weight * self.beta + 1)
        
        return torch.mean(weighted_loss)


class FeatureFocalRegionLoss(nn.Module):
    """Direct replacement for FeatureMSELoss using Focal Region Loss"""
    def __init__(self, weight, beta=1.0, epsilon=1e-3, loss_type='l1'):
        super().__init__()
        self.weight = weight
        self.focal_region_loss = FocalRegionLoss(
            weight=1.0, 
            beta=beta, 
            epsilon=epsilon, 
            loss_type=loss_type
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
            weight=1.0, 
            beta=beta, 
            epsilon=epsilon, 
            loss_type=loss_type
        )
    
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        
        # MSE component
        mse_component = self.mse_loss(feature_rec, feature_align)
        
        # Focal region component  
        focal_component = self.focal_region_loss(input)
        
        # Combined loss
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
