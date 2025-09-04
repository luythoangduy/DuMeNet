import torch
import torch.nn as nn
import numpy as np
from torch import no_grad
import os

try:
    from mobile_sam import sam_model_registry, SamPredictor
    from mobile_sam.utils.transforms import ResizeLongestSide
except ImportError:
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from mobile_sam import sam_model_registry, SamPredictor
        from mobile_sam.utils.transforms import ResizeLongestSide
    except ImportError:
        print("Warning: MobileSAM not found, falling back to basic implementation")
        sam_model_registry = None


class MobileSAMBackbone(nn.Module):
    def __init__(self, model_type='vit_t', checkpoint_path='', pretrained=True, outlayers=[4], **kwargs):
        super().__init__()
        
        if checkpoint_path == '':
            checkpoint_path = 'C:\\IAD\\UniAD\\mobile_sam_checkpoint\\mobile_sam.pt'
            
        self.outlayers = outlayers
        self.model_type = model_type
        print("Initializing MobileSAMBackbone with:")
        print(f" - Model Type: {self.model_type}")
        print(f" - Checkpoint Path: {checkpoint_path}")
        print(f" - Outlayers: {self.outlayers}")
        print(f'model registry: {sam_model_registry}' )
        if sam_model_registry is not None and os.path.exists(checkpoint_path):
            self.mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.image_encoder = self.mobile_sam.image_encoder
        else:
            raise ImportError("MobileSAM model registry not available or checkpoint not found")
            
        self.img_size = self.image_encoder.img_size
        self.transformation = ResizeLongestSide(self.img_size) if 'ResizeLongestSide' in globals() else None
        
        # Get actual feature dimensions from MobileSAM encoder
        try:
            # Get the actual output dimension from MobileSAM encoder
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
                dummy_features = self.image_encoder(dummy_input)
                self.actual_channels = dummy_features.shape[1]
        except:
            # Fallback to known dimension
            self.actual_channels = 256  # MobileSAM typically outputs 256 channels
        
        # Simplified - only use single layer output (no multi-scale)
        self.output_channels = self.actual_channels
        self.output_stride = 16  # MobileSAM standard stride
        
        # Freeze the MobileSAM image encoder by default if pretrained
        if pretrained:
            self.freeze_image_encoder()
    
    def get_outplanes(self):
        return [self.output_channels]
    
    def get_outstrides(self):
        return [self.output_stride]
    
    def freeze_image_encoder(self):
        """Freeze the MobileSAM image encoder parameters"""
        print("Freezing MobileSAM image encoder...")
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Set to eval mode to freeze batch norm layers
        self.image_encoder.eval()
        print("MobileSAM image encoder frozen successfully")
    
    def train(self, mode=True):
        """Override train method to keep image encoder in eval mode when frozen"""
        super().train(mode)
        
        # Keep image encoder in eval mode if it's frozen
        if hasattr(self, 'image_encoder'):
            # Check if parameters are frozen
            encoder_frozen = all(not param.requires_grad for param in self.image_encoder.parameters())
            if encoder_frozen:
                self.image_encoder.eval()
                print("Keeping MobileSAM image encoder in eval mode (frozen)")
        
        return self
    
    def preprocess_image(self, image):
        """Preprocess image for MobileSAM"""
        if self.transformation is not None:
            # Apply MobileSAM's transformation
            if isinstance(image, torch.Tensor):
                # Convert to float32 and normalize to [0, 1] if it's in [0, 255]
                if image.dtype == torch.uint8:
                    image = image.float() / 255.0
                elif image.dtype != torch.float32:
                    image = image.float()
                    
                image_np = image.cpu().numpy().transpose(1, 2, 0)
                # Ensure image_np is in [0, 255] range for MobileSAM transformation
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255.0).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                    
                transformed = self.transformation.apply_image(image_np)
                # Convert back to float32 tensor and normalize to [0, 1]
                transformed_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0
                return transformed_tensor.unsqueeze(0)
            else:
                # Handle numpy array input
                if image.max() <= 1.0:
                    image = (image * 255.0).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                    
                transformed = self.transformation.apply_image(image)
                transformed_tensor = torch.from_numpy(transformed).permute(2, 0, 1).float() / 255.0
                return transformed_tensor.unsqueeze(0)
        else:
            # Simple resize fallback
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                if image.max() > 1.0:
                    image = image / 255.0
                image = image.unsqueeze(0)
            else:
                # Convert to float32 and normalize
                if image.dtype == torch.uint8:
                    image = image.float() / 255.0
                elif image.dtype != torch.float32:
                    image = image.float()
                    
            return torch.nn.functional.interpolate(image, size=(self.img_size, self.img_size), mode='bilinear')

    def forward(self, input_dict):
        """Forward pass compatible with UniAD architecture"""
        image = input_dict['image']
        batch_size = image.shape[0]
        
        # Preprocess images
        processed_images = []
        for i in range(batch_size):
            img = image[i]
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img)
        
        batch_images = torch.cat(processed_images, dim=0).to(image.device)
        
        # Get features from image encoder (no multi-scale)
        encoder_frozen = all(not param.requires_grad for param in self.image_encoder.parameters())
        if encoder_frozen:
            with torch.no_grad():
                features = self.image_encoder(batch_images)
        else:
            features = self.image_encoder(batch_images)
        
        # Downsample features to 14x14 to match EfficientNet output size
        features = torch.nn.functional.interpolate(
            features, 
            size=(14, 14), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Return single feature map (no multi-scale processing)
        return {'features': [features]}