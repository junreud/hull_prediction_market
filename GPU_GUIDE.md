# GPU Configuration Guide

## 🎯 Overview

This project supports GPU acceleration for LightGBM training:
- **Kaggle**: CUDA GPU (NVIDIA)
- **Local Mac**: MPS (Apple Silicon M1/M2/M3)
- **Fallback**: CPU

## 🚀 Quick Start

### 1. Test GPU Availability

```bash
python scripts/test_gpu.py
```

This will:
- Detect available GPU (CUDA, MPS, or CPU)
- Test LightGBM GPU training
- Show recommended configuration

### 2. Kaggle Notebook (Auto-configured)

The notebook automatically detects and configures GPU:

```python
# Cell 1: Setup & GPU Detection
DATASET_NAME = "prediction-market-modules"
# GPU is automatically configured based on environment
# Kaggle → CUDA GPU
# Mac → MPS
# Other → CPU
```

**No manual configuration needed!** ✅

### 3. Local Training

#### Option A: Auto-detect (Recommended)
```bash
# Run test to see recommended config
python scripts/test_gpu.py

# Scripts will use config from conf/params.yaml
python scripts/optimize_return_model.py
python scripts/optimize_risk_model.py
python scripts/optimize_position_strategy.py
```

#### Option B: Manual Override

Edit `conf/params.yaml`:

**For Kaggle (CUDA GPU):**
```yaml
model_return:
  lightgbm:
    device: 'gpu'
    gpu_platform_id: 0
    gpu_device_id: 0
```

**For Mac (Apple Silicon):**
```yaml
model_return:
  lightgbm:
    device: 'mps'
```

**For CPU:**
```yaml
model_return:
  lightgbm:
    device: 'cpu'
```

## 📊 Performance Comparison

| Device | Training Time | Speedup |
|--------|---------------|---------|
| **CUDA GPU** (Kaggle T4) | ~3-5 min | **6-8x** |
| **MPS** (M1 Pro) | ~5-7 min | **3-4x** |
| **CPU** (8 cores) | ~20-30 min | 1x |

*Times for full optimization pipeline (return + risk models)*

## 🔧 Installation

### Kaggle (Already Installed)
```bash
# PyTorch with CUDA support is pre-installed
# LightGBM with GPU support is pre-installed
✅ No action needed
```

### Mac (Apple Silicon)

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# LightGBM should already support MPS
pip install lightgbm
```

### Linux/Windows with NVIDIA GPU

```bash
# Install CUDA toolkit (version 11.x or 12.x)
# Follow: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install LightGBM with GPU support
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
```

## 🐛 Troubleshooting

### Issue: "GPU training failed"

**Solution 1: Check GPU availability**
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")
```

**Solution 2: Verify LightGBM GPU build**
```python
import lightgbm as lgb
print(lgb.__version__)
# Should be 3.3.0 or higher for MPS support
```

**Solution 3: Fallback to CPU**
```yaml
# In conf/params.yaml
model_return:
  lightgbm:
    device: 'cpu'
```

### Issue: "CUDA out of memory"

**Solution: Reduce batch size or use CPU**
```yaml
model_return:
  lightgbm:
    device: 'cpu'  # Or reduce num_leaves, max_depth
```

### Issue: MPS not working on Mac

**Requirements:**
- macOS 12.3+ (Monterey)
- Apple Silicon (M1/M2/M3)
- PyTorch 1.12+

**Update PyTorch:**
```bash
pip install --upgrade torch torchvision torchaudio
```

## 📝 Configuration Reference

### Full GPU Config (conf/params.yaml)

```yaml
model_return:
  lightgbm:
    # Core parameters
    objective: "regression"
    metric: "rmse"
    num_leaves: 31
    learning_rate: 0.05
    
    # GPU settings (auto-configured in Kaggle notebook)
    device: "cpu"  # Options: 'cpu', 'gpu', 'mps'
    gpu_platform_id: 0  # For CUDA only
    gpu_device_id: 0    # For CUDA only
    gpu_use_dp: false   # Double precision (slower, more accurate)

risk:
  lightgbm:
    fixed_params:
      objective: "regression"
      metric: "rmse"
      
      # GPU settings (auto-configured)
      device: "cpu"
      gpu_platform_id: 0
      gpu_device_id: 0
```

## ✅ Validation

After configuration, validate GPU is working:

```bash
# Should show GPU usage
python scripts/optimize_return_model.py 2>&1 | grep -i "device\|gpu\|mps"

# Expected output:
# "Using device: gpu" or "Using device: mps"
```

## 🎯 Best Practices

1. **Kaggle Submission**: Always use GPU for faster training
2. **Local Development**: Use MPS (Mac) or CPU for testing
3. **Production**: Test both GPU and CPU to ensure compatibility
4. **Debugging**: Use CPU first, then enable GPU for speed

## 📚 References

- [LightGBM GPU Tutorial](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
