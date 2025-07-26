# ai_stock_screener/gpu_utils.py

import sys
import warnings
from typing import Optional, Dict, Any
from .output_formatter import console

class GPUManager:
    """Manages GPU detection and configuration for machine learning models."""
    
    def __init__(self):
        self.cuda_available = False
        self.cuml_available = False
        self.torch_available = False
        self.gpu_count = 0
        self.gpu_memory = 0
        self._detect_gpu_capabilities()
    
    def _detect_gpu_capabilities(self):
        """Detect available GPU capabilities and libraries."""
        
        # Check CUDA availability via CuPy
        try:
            import cupy as cp
            self.cuda_available = True
            self.gpu_count = cp.cuda.runtime.getDeviceCount()
            if self.gpu_count > 0:
                # Get memory info for the first GPU
                meminfo = cp.cuda.runtime.memGetInfo()
                self.gpu_memory = meminfo[1] // (1024**3)  # Convert to GB
        except ImportError:
            pass
        except Exception as e:
            console.print(f"⚠️ CUDA detection warning: {e}")
        
        # Check cuML availability
        try:
            import cuml
            self.cuml_available = True
        except ImportError:
            pass
        except Exception as e:
            console.print(f"⚠️ cuML detection warning: {e}")
        
        # Check PyTorch CUDA availability
        try:
            import torch
            self.torch_available = torch.cuda.is_available()
        except ImportError:
            pass
        except Exception as e:
            console.print(f"⚠️ PyTorch CUDA detection warning: {e}")
    
    def is_gpu_available(self) -> bool:
        """Check if any GPU acceleration is available."""
        return self.cuda_available or self.torch_available
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        return {
            "cuda_available": self.cuda_available,
            "cuml_available": self.cuml_available,
            "torch_available": self.torch_available,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory
        }
    
    def print_gpu_status(self):
        """Print GPU status information."""
        if self.is_gpu_available():
            console.print("🚀 GPU Acceleration Status:")
            console.print(f"   • CUDA Available: {'✅' if self.cuda_available else '❌'}")
            console.print(f"   • cuML Available: {'✅' if self.cuml_available else '❌'}")
            console.print(f"   • PyTorch CUDA: {'✅' if self.torch_available else '❌'}")
            if self.gpu_count > 0:
                console.print(f"   • GPU Count: {self.gpu_count}")
                console.print(f"   • GPU Memory: {self.gpu_memory} GB")
        else:
            console.print("💻 Using CPU-only mode (no GPU acceleration detected)")
    
    def get_xgboost_gpu_params(self, use_gpu: bool = True, dataset_size: int = None, 
                              feature_count: int = None, available_memory_gb: float = None) -> Dict[str, Any]:
        """Get optimized XGBoost parameters for GPU acceleration with dynamic tuning.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            dataset_size: Number of training samples for optimization
            feature_count: Number of features for memory optimization
            available_memory_gb: Available GPU memory in GB for optimization
            
        Returns:
            Dictionary of optimized XGBoost parameters
        """
        if use_gpu and self.cuda_available:
            # Base GPU parameters
            gpu_params = {
                "tree_method": "gpu_hist",
                "gpu_id": 0,
                "predictor": "gpu_predictor"
            }
            
            # Dynamic optimization based on dataset characteristics
            if dataset_size is not None:
                # Optimize based on dataset size thresholds from benchmark results
                if dataset_size <= 200:  # Small datasets (~8-20 tickers)
                    # For small datasets, use CPU-optimized settings on GPU
                    gpu_params.update({
                        "max_bin": 64,  # Reduced bins for small datasets
                        "grow_policy": "lossguide",  # More efficient for small data
                        "max_leaves": 31,  # Conservative leaf count
                        "subsample": 0.8,  # Reduce overfitting
                        "colsample_bytree": 0.8
                    })
                elif dataset_size <= 1000:  # Medium datasets (~20-50 tickers)
                    # Balanced settings for medium datasets
                    gpu_params.update({
                        "max_bin": 128,  # Standard bins
                        "grow_policy": "depthwise",  # Standard growth
                        "subsample": 0.9,
                        "colsample_bytree": 0.9
                    })
                else:  # Large datasets (50+ tickers)
                    # GPU-optimized settings for large datasets
                    gpu_params.update({
                        "max_bin": 256,  # More bins for better accuracy
                        "grow_policy": "depthwise",
                        "subsample": 1.0,  # Use all data
                        "colsample_bytree": 1.0,
                        "single_precision_histogram": True  # GPU memory optimization
                    })
            
            # Memory optimization based on available GPU memory
            if available_memory_gb is not None and available_memory_gb < 8:
                # Conservative settings for limited GPU memory
                gpu_params.update({
                    "max_bin": min(gpu_params.get("max_bin", 128), 64),
                    "single_precision_histogram": True,
                    "max_cached_hist_node": 32768  # Reduce cache size
                })
            
            # Feature-based optimization
            if feature_count is not None and feature_count > 100:
                # Optimize for high-dimensional data
                gpu_params.update({
                    "colsample_bytree": min(gpu_params.get("colsample_bytree", 1.0), 0.8),
                    "colsample_bylevel": 0.8,
                    "colsample_bynode": 0.8
                })
            
            return gpu_params
        else:
            # CPU parameters with optimization
            cpu_params = {
                "tree_method": "hist",
                "n_jobs": -1
            }
            
            # CPU-specific optimizations
            if dataset_size is not None and dataset_size > 1000:
                cpu_params.update({
                    "max_bin": 255,  # CPU can handle more bins efficiently
                    "grow_policy": "depthwise"
                })
            
            return cpu_params
    
    def get_cuml_random_forest(self, n_samples=None, **kwargs):
        """Get cuML RandomForest if available, otherwise return None.
        
        Args:
            n_samples: Number of training samples to adjust n_bins accordingly
            **kwargs: Additional parameters for the RandomForest
        """
        if not self.cuml_available:
            return None
        
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            
            # Dynamically adjust n_bins based on training data size to avoid warnings
            # Use a conservative approach since cuML processes features individually
            # and some features may have fewer unique values than the total sample count
            default_n_bins = kwargs.get("n_bins", 128)
            if n_samples is not None:
                # Use a conservative value that's much smaller than typical sample sizes
                # This ensures n_bins is always less than unique values per feature
                conservative_n_bins = min(32, max(n_samples // 4, 8))
                adjusted_n_bins = min(default_n_bins, conservative_n_bins)
            else:
                # Use a conservative default when n_samples is not provided
                adjusted_n_bins = min(default_n_bins, 32)
            
            # Set default parameters optimized for GPU, being careful with cuML-specific constraints
            gpu_params = {
                "n_estimators": kwargs.get("n_estimators", 100),
                "random_state": kwargs.get("random_state", 42),
                "n_streams": 1,  # GPU streams
                "n_bins": adjusted_n_bins,
            }
            
            # Handle max_depth carefully - cuML doesn't like None
            max_depth = kwargs.get("max_depth", 16)  # Use reasonable default instead of None
            if max_depth is not None:
                gpu_params["max_depth"] = max_depth
            
            # Handle max_features - cuML has different options
            max_features = kwargs.get("max_features", "sqrt")
            if max_features in ["sqrt", "log2"]:
                gpu_params["max_features"] = max_features
            elif max_features == "auto":
                gpu_params["max_features"] = "sqrt"  # cuML equivalent
            # Skip max_features if it's None or other unsupported values
            
            return cuRF(**gpu_params)
        except ImportError:
            console.print("⚠️ cuML RandomForest not available, falling back to CPU")
            return None
        except Exception as e:
            console.print(f"⚠️ Error creating cuML RandomForest: {e}")
            return None
    
    def cleanup_cuml_model(self, model):
        """Safely cleanup cuML model resources to prevent GPU memory leaks.
        
        Args:
            model: cuML model to cleanup
        """
        if model is None:
            return
            
        try:
            # Check if this is a cuML model
            if hasattr(model, '__module__') and 'cuml' in str(model.__module__):
                # Force garbage collection of GPU resources
                if hasattr(model, '_clear_model'):
                    model._clear_model()
                elif hasattr(model, 'clear'):
                    model.clear()
                
                # Clear any internal references
                if hasattr(model, '_model'):
                    model._model = None
                if hasattr(model, 'handle'):
                    model.handle = None
                    
                # Force Python garbage collection
                import gc
                gc.collect()
                
                # Try to clear CUDA cache if available
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
                    
        except Exception as e:
            # Silently handle cleanup errors to avoid masking the main application
            pass

# Global GPU manager instance
gpu_manager = GPUManager()

def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    return gpu_manager

def is_gpu_available() -> bool:
    """Quick check if GPU is available."""
    return gpu_manager.is_gpu_available()

def print_gpu_status():
    """Print GPU status."""
    gpu_manager.print_gpu_status()