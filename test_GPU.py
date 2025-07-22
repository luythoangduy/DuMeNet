import torch
print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ CUDA version in PyTorch:", torch.version.cuda)
print("✓ GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("✓ GPU name:", torch.cuda.get_device_name(0))
    # Test tạo tensor trên GPU
    x = torch.randn(3, 3).cuda()
    print("✓ Successfully created tensor on GPU:", x.device)
