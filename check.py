import ultralytics
import torch

# Check if ultralytics is ready
ultralytics.checks()

# Ensure it targets your GPU instead of CPU
print("CUDA GPU Available:", torch.cuda.is_available())