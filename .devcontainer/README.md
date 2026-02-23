# Dev Container Configurations

This project offers two devcontainer configurations to support both GPU and CPU-only machines:

## GPU Configuration

**Location**: `.devcontainer/gpu/`

**Use when:**
- You have an NVIDIA GPU
- Running on Linux or WSL2
- Docker has GPU support enabled

**Features:**
- Pre-installed PyTorch with CUDA 12.6 support
- GPU-accelerated training
- Based on `gperdrizet/deeplearning-gpu` image
- Includes TensorFlow, PyTorch, and Jupyter pre-installed

**Requirements:**
- NVIDIA drivers ≥545
- Docker with GPU support (nvidia-container-toolkit)
- 15GB free disk space

## CPU Configuration

**Location**: `.devcontainer/cpu/`

**Use when:**
- Running on Mac (Intel or Apple Silicon)
- No GPU available
- Don't have Docker GPU support
- Just want to develop/test code without GPU-accelerated training

**Features:**
- PyTorch 2.10 CPU-only (pre-installed)
- TensorFlow 2.16 (pre-installed)
- Works on any machine
- Based on `gperdrizet/deeplearning-cpu` image
- Good for code development, testing, and documentation

**Requirements:**
- Docker Desktop
- 10GB free disk space

## How to Choose

When you open this project in VS Code, it will prompt you to select a configuration:

1. **DeepLearning GPU** - for GPU machines (Linux/WSL2 with NVIDIA GPU)
2. **DeepLearning CPU** - for Macs and CPU-only machines

The code automatically detects your hardware, so both configurations work seamlessly. If you're using the CPU configuration, the notebooks will run slower but will still work correctly.

## Switching Configurations

To switch between configurations:

1. Close VS Code
2. Reopen the project folder
3. Select the other configuration when prompted

Or use `F1` > "Dev Containers: Reopen in Container" > choose configuration

## Performance Notes

**GPU configuration:**
- Recommended for training models
- Typical training time: minutes to hours
- Can handle larger batch sizes

**CPU configuration:**
- Suitable for code development and testing
- Training will be slower (10-50x slower than GPU)
- May need to reduce batch sizes in notebook configuration cells
- Perfect for following along with the material without a GPU

## Verification

After opening in either configuration, verify your setup:

```python
import torch

# Check PyTorch version
print(f'PyTorch: {torch.__version__}')

# Check CUDA availability (GPU config should show True, CPU should show False)
print(f'CUDA available: {torch.cuda.is_available()}')

# If CUDA is available, show GPU info
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('Running in CPU mode')
```
