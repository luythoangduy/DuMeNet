import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileSAMBypass"]


class MobileSAMBypass(nn.Module):
    """
    Direct bypass from MobileSAM backbone to UniAD autoencoder.
    Takes single feature output from MobileSAM and passes it directly to autoencoder.
    Includes feature normalization options to match EfficientNet range.
    """
    def __init__(self, inplanes, outplanes, instrides, outstrides, 
                 normalize_features=True, feature_norm_type='layer'):
        super(MobileSAMBypass, self).__init__()
        
        assert isinstance(inplanes, list) and len(inplanes) == 1, "MobileSAM outputs single feature map"
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1  
        assert isinstance(outstrides, list) and len(outstrides) == 1
        
        self.inplanes = inplanes   # [256] from MobileSAM
        self.outplanes = outplanes # Should match inplanes for direct bypass
        self.instrides = instrides # [16] from MobileSAM
        self.outstrides = outstrides # [16] target stride
        self.normalize_features = normalize_features
        self.feature_norm_type = feature_norm_type
        
        # MobileSAM already outputs at 14x14 with stride 16
        # No additional processing needed for direct bypass
        assert inplanes[0] == outplanes[0], f"Input planes {inplanes[0]} must match output planes {outplanes[0]} for direct bypass"
        assert instrides[0] == outstrides[0], f"Input stride {instrides[0]} must match output stride {outstrides[0]} for direct bypass"
        
        # Add feature normalization layer if needed
        if self.normalize_features:
            if feature_norm_type == 'layer':
                self.feature_norm = nn.LayerNorm(inplanes[0])
            elif feature_norm_type == 'batch':
                self.feature_norm = nn.BatchNorm2d(inplanes[0])
            elif feature_norm_type == 'instance':
                self.feature_norm = nn.InstanceNorm2d(inplanes[0])
            else:
                self.feature_norm = None
                self.normalize_features = False
        
        print(f"MobileSAM Bypass initialized:")
        print(f"  Input planes: {inplanes}")
        print(f"  Output planes: {outplanes}")
        print(f"  Input strides: {instrides}")  
        print(f"  Output strides: {outstrides}")
        print(f"  Feature normalization: {normalize_features}")
        print(f"  Normalization type: {feature_norm_type if normalize_features else 'None'}")

    def forward(self, input):
        """
        Forward with optional feature normalization.
        
        Args:
            input: dict with "features" key containing [B, 256, 14, 14] tensor
        Returns:
            dict with "feature_align" key for UniAD autoencoder input
        """
        features = input["features"]
        assert len(features) == 1, "MobileSAM should output single feature map"
        
        feature_align = features[0]  # B x 256 x 14 x 14
        
        # Apply feature normalization if enabled
        if self.normalize_features and self.feature_norm is not None:
            if self.feature_norm_type == 'layer':
                # LayerNorm expects (B, H, W, C) format
                B, C, H, W = feature_align.shape
                feature_align = feature_align.permute(0, 2, 3, 1)  # B x H x W x C
                feature_align = self.feature_norm(feature_align)
                feature_align = feature_align.permute(0, 3, 1, 2)  # B x C x H x W
            elif self.feature_norm_type in ['batch', 'instance']:
                # BatchNorm2d and InstanceNorm2d expect (B, C, H, W)
                feature_align = self.feature_norm(feature_align)
            
            # Additional statistical normalization to match EfficientNet range
            # Normalize to zero mean, unit variance per channel
            B, C, H, W = feature_align.shape
            feature_flat = feature_align.view(B, C, -1)
            mean = feature_flat.mean(dim=2, keepdim=True).unsqueeze(-1)  # B x C x 1 x 1
            std = feature_flat.std(dim=2, keepdim=True).unsqueeze(-1)    # B x C x 1 x 1
            feature_align = (feature_align - mean) / (std + 1e-6)
        
        return {
            "feature_align": feature_align, 
            "outplane": self.get_outplanes()
        }
    
    def get_outplanes(self):
        return self.outplanes
        
    def get_outstrides(self):
        return self.outstrides