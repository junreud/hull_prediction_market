"""
Test GPU availability and LightGBM GPU support.

Usage:
    python scripts/test_gpu.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import lightgbm as lgb

print("="*80)
print("GPU AVAILABILITY TEST")
print("="*80)

# Test 1: Check PyTorch GPU support
print("\n1. PyTorch GPU Support:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        recommended_device = "gpu"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"   ✓ MPS (Apple Silicon) available")
        recommended_device = "mps"
    else:
        print(f"   ℹ️  No GPU available, using CPU")
        recommended_device = "cpu"
except ImportError:
    print(f"   ⚠️  PyTorch not installed")
    recommended_device = "cpu"

# Test 2: Check LightGBM GPU support
print("\n2. LightGBM GPU Support:")
print(f"   LightGBM version: {lgb.__version__}")

try:
    # Create small test dataset
    X_train = np.random.rand(1000, 10)
    y_train = np.random.rand(1000)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Try GPU training
    print("\n   Testing GPU training...")
    params_gpu = {
        'objective': 'regression',
        'metric': 'rmse',
        'device': recommended_device,
        'verbose': -1,
        'num_iterations': 10
    }
    
    if recommended_device == "gpu":
        params_gpu['gpu_platform_id'] = 0
        params_gpu['gpu_device_id'] = 0
    
    try:
        model = lgb.train(params_gpu, train_data)
        print(f"   ✅ {recommended_device.upper()} training successful!")
    except Exception as e:
        print(f"   ❌ {recommended_device.upper()} training failed: {str(e)}")
        print(f"   Falling back to CPU...")
        recommended_device = "cpu"
        
        params_cpu = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'cpu',
            'verbose': -1,
            'num_iterations': 10
        }
        model = lgb.train(params_cpu, train_data)
        print(f"   ✅ CPU training successful!")

except Exception as e:
    print(f"   ❌ Error: {str(e)}")

# Test 3: Configuration recommendation
print("\n3. Configuration Recommendation:")
print("="*80)
print(f"\nRecommended LightGBM config:")
print(f"  device: '{recommended_device}'")

if recommended_device == "gpu":
    print(f"  gpu_platform_id: 0")
    print(f"  gpu_device_id: 0")
    print(f"  gpu_use_dp: false")

print("\nUpdate your conf/params.yaml:")
print("```yaml")
print("model_return:")
print("  lightgbm:")
print(f"    device: '{recommended_device}'")
if recommended_device == "gpu":
    print(f"    gpu_platform_id: 0")
    print(f"    gpu_device_id: 0")
print("```")

print("\n✅ GPU test complete!")
