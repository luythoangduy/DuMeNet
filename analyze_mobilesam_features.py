import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import sys

# Add the project root to Python path
sys.path.append('C:\\IAD\\UniAD')

from models.backbones.mobileSAM.model import MobileSAMBackbone

def analyze_mobilesam_features():
    """Analyze MobileSAM encoder output features"""
    
    print("=== MobileSAM Feature Analysis ===\n")
    
    # Initialize the MobileSAM backbone
    try:
        backbone = MobileSAMBackbone(
            model_type='vit_t',
            checkpoint_path='C:\\IAD\\UniAD\\mobile_sam_checkpoint\\mobile_sam.pt',
            pretrained=True,
            outlayers=[4],
            fork_features=True
        )
        backbone.eval()
        print("MobileSAM backbone loaded successfully\n")
    except Exception as e:
        print(f"Error loading MobileSAM backbone: {e}")
        return
    
    # Create dummy input data (batch of images)
    batch_size = 4
    input_images = torch.randn(batch_size, 3, 224, 224)  # Random RGB images
    
    print(f"Input shape: {input_images.shape}")
    print(f"Input value range: [{input_images.min():.3f}, {input_images.max():.3f}]")
    print(f"Input mean: {input_images.mean():.3f}, std: {input_images.std():.3f}\n")
    
    # Forward pass through the backbone
    with torch.no_grad():
        input_dict = {'image': input_images}
        output = backbone(input_dict)
        features_list = output['features']
    
    print(f"Number of feature branches: {len(features_list)}")
    print(f"Expected output channels: {backbone.get_outplanes()}")
    print(f"Expected output strides: {backbone.get_outstrides()}\n")
    
    # Analyze each feature branch
    feature_stats = []
    
    for i, features in enumerate(features_list):
        print(f"=== Feature Branch {i+1} ===")
        print(f"Shape: {features.shape}")
        print(f"Data type: {features.dtype}")
        
        # Convert to numpy for analysis
        feat_np = features.cpu().numpy()
        
        # Basic statistics
        min_val = feat_np.min()
        max_val = feat_np.max()
        mean_val = feat_np.mean()
        std_val = feat_np.std()
        median_val = np.median(feat_np)
        
        print(f"Value range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"Mean: {mean_val:.6f}")
        print(f"Std: {std_val:.6f}")
        print(f"Median: {median_val:.6f}")
        
        # Percentiles
        p25 = np.percentile(feat_np, 25)
        p75 = np.percentile(feat_np, 75)
        p95 = np.percentile(feat_np, 95)
        p99 = np.percentile(feat_np, 99)
        
        print(f"Percentiles - 25th: {p25:.6f}, 75th: {p75:.6f}, 95th: {p95:.6f}, 99th: {p99:.6f}")
        
        # Check for special values
        num_zeros = np.sum(feat_np == 0)
        num_negatives = np.sum(feat_np < 0)
        num_positives = np.sum(feat_np > 0)
        
        print(f"Zero values: {num_zeros} ({num_zeros/feat_np.size*100:.2f}%)")
        print(f"Negative values: {num_negatives} ({num_negatives/feat_np.size*100:.2f}%)")
        print(f"Positive values: {num_positives} ({num_positives/feat_np.size*100:.2f}%)")
        
        # Channel-wise statistics
        channel_means = feat_np.mean(axis=(0, 2, 3))  # Average over batch and spatial dims
        channel_stds = feat_np.std(axis=(0, 2, 3))
        
        print(f"Channel means - min: {channel_means.min():.6f}, max: {channel_means.max():.6f}")
        print(f"Channel stds - min: {channel_stds.min():.6f}, max: {channel_stds.max():.6f}")
        
        # Spatial statistics (average over batch and channel)
        spatial_mean = feat_np.mean(axis=(0, 1))  # Shape: (H, W)
        spatial_std = feat_np.std(axis=(0, 1))
        
        print(f"Spatial activation pattern - mean range: [{spatial_mean.min():.6f}, {spatial_mean.max():.6f}]")
        print(f"Spatial variation - std range: [{spatial_std.min():.6f}, {spatial_std.max():.6f}]")
        
        feature_stats.append({
            'branch': i,
            'shape': features.shape,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'p25': p25,
            'p75': p75,
            'p95': p95,
            'p99': p99,
            'zeros_pct': num_zeros/feat_np.size*100,
            'neg_pct': num_negatives/feat_np.size*100,
            'pos_pct': num_positives/feat_np.size*100,
            'channel_mean_range': (channel_means.min(), channel_means.max()),
            'channel_std_range': (channel_stds.min(), channel_stds.max()),
            'features': feat_np
        })
        
        print("-" * 50)
    
    # Summary comparison
    print("\n=== Summary Comparison ===")
    print("Branch | Shape | Value Range | Mean±Std | Zeros% | Neg% | Pos%")
    print("-" * 80)
    
    for stat in feature_stats:
        print(f"  {stat['branch']+1}   | {str(stat['shape']):15} | "
              f"[{stat['min']:+.3f}, {stat['max']:+.3f}] | "
              f"{stat['mean']:+.3f}±{stat['std']:.3f} | "
              f"{stat['zeros_pct']:5.1f}% | "
              f"{stat['neg_pct']:5.1f}% | "
              f"{stat['pos_pct']:5.1f}%")
    
    # Distribution analysis
    print("\n=== Distribution Analysis ===")
    
    # Analyze distribution characteristics
    for i, stat in enumerate(feature_stats):
        feat_flat = stat['features'].flatten()
        
        print(f"\nBranch {i+1} Distribution:")
        
        # Check if distribution is approximately normal
        from scipy import stats
        _, p_value = stats.normaltest(feat_flat[:10000])  # Use sample for large arrays
        is_normal = p_value > 0.05
        print(f"  Normality test p-value: {p_value:.6f} ({'Normal' if is_normal else 'Non-normal'})")
        
        # Skewness and kurtosis
        skewness = stats.skew(feat_flat)
        kurt = stats.kurtosis(feat_flat)
        print(f"  Skewness: {skewness:.6f} ({'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'})")
        print(f"  Kurtosis: {kurt:.6f} ({'Heavy-tailed' if kurt > 0 else 'Light-tailed' if kurt < 0 else 'Normal-tailed'})")
        
        # Activation sparsity
        activation_threshold = 0.01  # Consider values below this as inactive
        sparse_pct = np.sum(np.abs(feat_flat) < activation_threshold) / len(feat_flat) * 100
        print(f"  Sparsity (|x| < {activation_threshold}): {sparse_pct:.1f}%")
        
        # Dynamic range
        dynamic_range = stat['max'] - stat['min']
        print(f"  Dynamic range: {dynamic_range:.6f}")
        
        # Effective range (5th to 95th percentile)
        p5 = np.percentile(feat_flat, 5)
        p95 = np.percentile(feat_flat, 95)
        effective_range = p95 - p5
        print(f"  Effective range (5-95%): [{p5:.6f}, {p95:.6f}] = {effective_range:.6f}")
    
    print("\n=== Feature Analysis Complete ===")
    
    return feature_stats

if __name__ == "__main__":
    try:
        feature_stats = analyze_mobilesam_features()
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()