import torch
import torchvision

# Check PyTorch version
print("PyTorch Version:", torch.__version__)

# Check TorchVision version
print("TorchVision Version:", torchvision.__version__)

# Create a sample 2x3 tensor
sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Sample Tensor:\n", sample_tensor)