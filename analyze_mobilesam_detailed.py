import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append('C:\\IAD\\UniAD')

from models.backbones.mobileSAM.model import MobileSAMBackbone

def analyze_with_fork_features():
    """Analyze MobileSAM with different fork_features settings"""
    
    print("=== MobileSAM Fork Features Analysis ===\n")
    
    # Test both fork_features=True and False
    configs = [
        {"fork_features": False, "outlayers": [4]},
        {"fork_features": True, "outlayers": [4]},
        {"fork_features": True, "outlayers": [1, 2, 3]}  # Multiple layers
    ]
    
    batch_size = 2
    input_images = torch.randn(batch_size, 3, 224, 224)
    
    for i, config in enumerate(configs):
        print(f"=== Configuration {i+1}: fork_features={config['fork_features']}, outlayers={config['outlayers']} ===")
        
        try:
            backbone = MobileSAMBackbone(
                model_type='vit_t',
                checkpoint_path='C:\\IAD\\UniAD\\mobile_sam_checkpoint\\mobile_sam.pt',
                pretrained=True,
                outlayers=config['outlayers'],
                fork_features=config['fork_features']
            )
            backbone.eval()
            
            # Forward pass
            with torch.no_grad():
                input_dict = {'image': input_images}
                output = backbone(input_dict)
                features_list = output['features']
            
            print(f"Number of output branches: {len(features_list)}")
            print(f"Output channels per branch: {backbone.get_outplanes()}")
            print(f"Output strides: {backbone.get_outstrides()}")
            
            # Analyze each branch
            total_features = 0
            for j, features in enumerate(features_list):
                print(f"  Branch {j+1}: Shape {features.shape}")
                
                # Feature statistics
                feat_np = features.cpu().numpy()
                min_val = feat_np.min()
                max_val = feat_np.max()
                mean_val = feat_np.mean()
                std_val = feat_np.std()
                
                print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
                print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                
                # Activation analysis
                zero_pct = np.sum(feat_np == 0) / feat_np.size * 100
                neg_pct = np.sum(feat_np < 0) / feat_np.size * 100
                pos_pct = np.sum(feat_np > 0) / feat_np.size * 100
                
                print(f"    Zeros: {zero_pct:.1f}%, Negative: {neg_pct:.1f}%, Positive: {pos_pct:.1f}%")
                
                # Sparsity analysis
                sparse_thresholds = [0.001, 0.01, 0.1]
                for thresh in sparse_thresholds:
                    sparse_pct = np.sum(np.abs(feat_np) < thresh) / feat_np.size * 100
                    print(f"    Sparsity (|x| < {thresh}): {sparse_pct:.1f}%")
                
                total_features += feat_np.size
            
            print(f"Total feature elements: {total_features}")
            print(f"Memory footprint: ~{total_features * 4 / 1024 / 1024:.2f} MB (float32)")
            
        except Exception as e:
            print(f"Error with configuration {i+1}: {e}")
        
        print("-" * 80)
    
def analyze_encoder_layers():
    """Analyze features from different encoder layers"""
    
    print("\n=== Encoder Layer Analysis ===\n")
    
    # Initialize backbone
    backbone = MobileSAMBackbone(
        model_type='vit_t',
        checkpoint_path='C:\\IAD\\UniAD\\mobile_sam_checkpoint\\mobile_sam.pt',
        pretrained=True,
        outlayers=[4],
        fork_features=False
    )
    backbone.eval()
    
    # Test with different input patterns
    test_inputs = [
        ("Random noise", torch.randn(1, 3, 224, 224)),
        ("Zeros", torch.zeros(1, 3, 224, 224)),
        ("Ones", torch.ones(1, 3, 224, 224)),
        ("Checkerboard", create_checkerboard_pattern()),
        ("Gaussian blob", create_gaussian_blob())
    ]
    
    for name, input_tensor in test_inputs:
        print(f"=== Input: {name} ===")
        print(f"Input stats - Min: {input_tensor.min():.3f}, Max: {input_tensor.max():.3f}, Mean: {input_tensor.mean():.3f}")
        
        with torch.no_grad():
            input_dict = {'image': input_tensor}
            output = backbone(input_dict)
            features = output['features'][0]  # First (and only) branch
            
            feat_np = features.cpu().numpy()
            
            print(f"Output shape: {features.shape}")
            print(f"Value range: [{feat_np.min():.6f}, {feat_np.max():.6f}]")
            print(f"Mean: {feat_np.mean():.6f}, Std: {feat_np.std():.6f}")
            
            # Check activation patterns
            channel_activity = np.abs(feat_np).mean(axis=(0, 2, 3))  # Activity per channel
            active_channels = np.sum(channel_activity > 0.001)
            print(f"Active channels (>0.001): {active_channels}/{features.shape[1]} ({active_channels/features.shape[1]*100:.1f}%)")
            
            # Spatial activity
            spatial_activity = np.abs(feat_np).mean(axis=(0, 1))  # Activity per spatial location
            max_activity_loc = np.unravel_index(np.argmax(spatial_activity), spatial_activity.shape)
            print(f"Peak spatial activity at: {max_activity_loc}, value: {spatial_activity.max():.6f}")
            
        print("-" * 50)

def create_checkerboard_pattern():
    """Create a checkerboard pattern input"""
    pattern = torch.zeros(1, 3, 224, 224)
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            if (i // 16 + j // 16) % 2 == 0:
                pattern[:, :, i:i+16, j:j+16] = 1.0
    return pattern

def create_gaussian_blob():
    """Create a Gaussian blob in the center"""
    x = torch.linspace(-1, 1, 224)
    y = torch.linspace(-1, 1, 224)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    gaussian = torch.exp(-(X**2 + Y**2) / 0.3)
    return gaussian.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

if __name__ == "__main__":
    try:
        analyze_with_fork_features()
        analyze_encoder_layers()
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()