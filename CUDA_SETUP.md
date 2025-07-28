# CUDA Support for AI Stock Screener

This document provides comprehensive instructions for setting up and using CUDA GPU acceleration with the AI Stock Screener.

## Overview

The AI Stock Screener now supports GPU acceleration for machine learning model training and calculations, which can significantly improve performance for large datasets and complex models.

### Supported GPU Frameworks

- **XGBoost GPU**: Native GPU support for gradient boosting
- **cuML (RAPIDS)**: GPU-accelerated RandomForest and other ML algorithms
- **CuPy**: GPU array processing (NumPy-compatible)

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- Minimum 4GB GPU memory (8GB+ recommended)
- CUDA-compatible driver

### Software Requirements

- CUDA Toolkit 11.2 or higher
- Python 3.11+
- Compatible NVIDIA drivers

## Installation

### 1. Install CUDA Toolkit

Download and install the CUDA Toolkit from [NVIDIA's official website](https://developer.nvidia.com/cuda-toolkit).

### 2. Install GPU Dependencies

Install the AI Stock Screener with GPU support:

```bash
# Install with GPU dependencies
poetry install -E gpu

# Or if using pip
pip install ai-stock-screener[gpu]
```

### 3. Verify Installation

Test your GPU setup:

```bash
# Check GPU information
poetry run screener --gpu_info

# Run GPU model tests
python test_gpu_models.py
```

## Usage

### Command Line Options

The following new CLI options are available for GPU control:

#### `--gpu_info`
Display detailed GPU information and exit.

```bash
poetry run screener --gpu_info
```

Example output:
```
🚀 GPU Information:
🚀 GPU Acceleration Status:
   • CUDA Available: ✅
   • cuML Available: ✅
   • PyTorch CUDA: ❌
   • GPU Count: 1
   • GPU Memory: 8 GB

Detailed GPU Info:
  cuda_available: True
  cuml_available: True
  torch_available: False
  gpu_count: 1
  gpu_memory_gb: 8
```

#### `--no_gpu`
Disable GPU acceleration and force CPU-only mode.

```bash
# Force CPU-only mode
poetry run screener --mode eval --tickers AAPL,NVDA --no_gpu
```

### GPU-Accelerated Models

#### XGBoost with GPU

XGBoost automatically uses GPU acceleration when available:

```bash
# XGBoost with GPU acceleration (default when GPU available)
poetry run screener --mode eval --tickers AAPL,NVDA --model xgboost

# XGBoost with CPU only
poetry run screener --mode eval --tickers AAPL,NVDA --model xgboost --no_gpu
```

#### RandomForest with cuML

RandomForest automatically attempts to use cuML GPU acceleration:

```bash
# RandomForest with GPU acceleration (cuML when available)
poetry run screener --mode eval --tickers AAPL,NVDA --model random_forest

# RandomForest with CPU only
poetry run screener --mode eval --tickers AAPL,NVDA --model random_forest --no_gpu
```

### Performance Comparison

Example performance improvements with GPU acceleration:

| Model | Dataset Size | CPU Time | GPU Time | Speedup |
|-------|-------------|----------|----------|---------|
| XGBoost | 1000 samples | 2.3s | 0.8s | 2.9x |
| RandomForest (cuML) | 1000 samples | 1.5s | 0.6s | 2.5x |

*Results may vary based on hardware configuration and dataset characteristics.*

## Configuration

### Automatic GPU Detection

The system automatically detects available GPU capabilities:

- **CUDA**: Detected via CuPy
- **cuML**: Detected via RAPIDS cuML library
- **Memory**: GPU memory information retrieved

### Fallback Behavior

The system gracefully falls back to CPU when:

- GPU libraries are not installed
- CUDA is not available
- GPU memory is insufficient
- GPU initialization fails

### Model-Specific GPU Settings

#### XGBoost GPU Parameters

When GPU is available, XGBoost uses:
```python
{
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "predictor": "gpu_predictor"
}
```

When GPU is not available:
```python
{
    "tree_method": "hist",
    "n_jobs": -1
}
```

#### cuML RandomForest Parameters

GPU-optimized parameters:
```python
{
    "n_estimators": 100,
    "max_depth": 16,  # Reasonable default instead of None
    "max_features": "sqrt",
    "random_state": 42,
    "n_streams": 1,
    "n_bins": 128  # Conservative default to avoid warnings
}
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Detected

**Problem**: `CUDA Available: ❌`

**Solutions**:
- Verify CUDA Toolkit installation
- Check NVIDIA driver compatibility
- Ensure CuPy is installed: `pip install cupy-cuda11x`

#### 2. cuML Not Available

**Problem**: `cuML Available: ❌`

**Solutions**:
- Install RAPIDS cuML: `conda install -c rapidsai cuml`
- Or use pip: `pip install cuml-cu11`
- Check CUDA version compatibility

#### 3. Out of Memory Errors

**Problem**: GPU memory errors during training

**Solutions**:
- Reduce batch size or dataset size
- Use `--no_gpu` flag to fall back to CPU
- Close other GPU-intensive applications

#### 4. Performance Not Improved

**Problem**: GPU mode is slower than CPU

**Possible Causes**:
- Small dataset size (GPU overhead)
- Memory transfer bottlenecks
- Suboptimal GPU utilization

**Solutions**:
- Use larger datasets for GPU benefits
- Consider CPU mode for small datasets
- Monitor GPU utilization

### Debugging Commands

```bash
# Check GPU status
poetry run screener --gpu_info

# Test GPU models
python test_gpu_models.py

# Run with CPU only for comparison
poetry run screener --mode eval --tickers AAPL --no_gpu

# Run with GPU (default)
poetry run screener --mode eval --tickers AAPL
```

### Environment Variables

You can set these environment variables for additional control:

```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Limit GPU memory growth (for TensorFlow/PyTorch)
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Best Practices

### When to Use GPU

**Recommended for GPU**:
- Large datasets (>1000 samples)
- Complex models with many estimators
- Discovery mode (S&P 500 screening)
- Grid search with multiple parameter combinations

**Recommended for CPU**:
- Small datasets (<500 samples)
- Quick evaluations
- Limited GPU memory
- Development/testing

### Performance Optimization

1. **Use appropriate model types**:
   - XGBoost generally has better GPU acceleration
   - cuML RandomForest for specific use cases

2. **Optimize data size**:
   - Larger datasets benefit more from GPU
   - Consider data preprocessing on GPU

3. **Monitor resources**:
   - Check GPU memory usage
   - Monitor GPU utilization
   - Balance CPU and GPU workloads

## Support

For GPU-related issues:

1. Check the troubleshooting section above
2. Run the test scripts to isolate issues
3. Verify your CUDA installation
4. Check compatibility with your GPU model

For additional help, please refer to:
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [XGBoost GPU Documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html)