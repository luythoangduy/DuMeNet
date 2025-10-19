# DuMeNet: Dual-Memory based Reconstruction Method for Image Anomaly Detection

## Overview

Image anomaly detection (IAD) has emerged as a critical task in numerous real-world applications. Among existing techniques, reconstruction based approaches have shown promise; however, they often suffer from the identical shortcut problem, wherein both normal and anomalous regions are reconstructed with similar fidelity. This undermines the discriminative power of anomaly scores and limits detection performance. To address this limitation, we introduce DuMeNet, a novel dual-memory reconstruction network that incorporates two complementary memory mechanisms. Specifically, we proposed a spatial memory bank that utilizes the Structural Similarity Index Measure (SSIM) to emphasize spatial pattern reconstruction. On the MVTec-AD benchmark, DuMeNet achieves 98.5% image-level, 97.4% pixel-level AUROC, outperforming many baselines and has the best Out-of-Distribution detection performance on CIFAR-10. These results demonstrate the strong generalizability of DuMeNet and its practical utility in vision-field anomaly detection scenarios.

## Key Features

- **Dual Memory Architecture**: Dual memory architecture at channel-wise feature and spatial structural levels addressing the identical shortcut problem.
- **Transformer-based Reconstruction**: Leverages transformer encoder-decoder architecture for robust feature reconstruction
- **Spatial Attention Mechanism**: Utilizes SSIM similarity for preserving spatial patterns.
- **Continuous Query Mechanism**: Prevents the decoder from drifting away from memory-constrained representations by continuously referencing learned normal patterns

## Architecture

![DuMeNet Architecture Overview](images/screenshot.png)

## MVTec-AD
- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move some to `./data/MVTec-AD/`. The MVTec-AD dataset directory should be as follows. 

```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
        |-- json_vis_decoder
        |-- train.json
        |-- test.json
```

## Results

### MVTec-AD Dataset Results

DuMeNet achieves state-of-the-art performance on the MVTec-AD dataset:

| Category | Image-level AUC | Pixel-level AUC |
|----------|----------------|-----------------|
| Objects  | 98.5%          | 97.4%           |
| Textures | 99.8%          | 98.4%           |
| **Mean** | **98.5%**      | **97.4%**       |

Compared to previous methods like DRAEM (96.6%, 96.6%), UniAD (97.6%, 97.4%), and others, DuMeNet consistently shows superior performance across most categories.

### CIFAR-10 Anomaly Detection Results

DuMeNet also performs exceptionally well on the CIFAR-10 dataset under the unified case:

| Normal Indices | AUROC |
|----------------|-------|
| {01234}        | 85.3  |
| {56789}        | 81.0  |
| {02468}        | 93.3  |
| {13579}        | 90.4  |
| **Mean**       | **87.5** |

These results surpass previous methods including UniAD (87.2), MKD (72.1), PANDA (72.4), FCDD+OE (78.9), and US (55.9).

## Key Advantages

- **Robust to Noise**: The dual memory architecture enhances the model's robustness to noise in training data
- **Comprehensive Feature Extraction**: Multi-scale approach captures both local and long-term dependency information
- **Improved Memory Representation**: Dual memory-augmented encoder extracts both global typical patterns and local common features
- **Adaptive Memory**: Memory items are learned during training, allowing for better adaptation to different domains

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- CUDA 11.1+ (for GPU acceleration)
- scikit-learn
- OpenCV
- numpy
- einops
- easydict

## Usage

### Configuration

The framework uses a YAML configuration file to set parameters. Key configuration options include:

```yaml
net:
  - name: reconstruction
    prev: neck
    type: models.reconstructions.UniADMemory
    kwargs:
      memory_mode: 'both'  # Options: 'channel', 'spatial', 'both', 'none'
      fusion_mode: 'concat'  # Options: 'concat', 'add', 'multiply', 'attention', 'gate'
      channel_memory_size: 256
      spatial_memory_size: 256
```

### Training

To train the model on a specific class:

```bash
python tools/train_val.py --config tools/config.yaml --class_name bottle --single_gpu
```

For multi-class training:

```bash
python tools/train_single.py
```

### Evaluation

To evaluate a trained model:

```bash
python tools/train_val.py --config tools/config.yaml --class_name bottle --evaluate
```

