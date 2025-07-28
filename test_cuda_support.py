#!/usr/bin/env python3
"""
Test script to verify CUDA support in AI Stock Screener
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_stock_screener'))

from ai_stock_screener.gpu_utils import get_gpu_manager, print_gpu_status
from ai_stock_screener.cli import main

def test_gpu_detection():
    """Test GPU detection capabilities"""
    print("=" * 60)
    print("🧪 Testing GPU Detection")
    print("=" * 60)
    
    gpu_manager = get_gpu_manager()
    print_gpu_status()
    
    gpu_info = gpu_manager.get_gpu_info()
    print(f"\nDetailed GPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nGPU Available: {gpu_manager.is_gpu_available()}")
    
    return gpu_manager.is_gpu_available()

def test_xgboost_gpu_params():
    """Test XGBoost GPU parameter generation"""
    print("\n" + "=" * 60)
    print("🧪 Testing XGBoost GPU Parameters")
    print("=" * 60)
    
    gpu_manager = get_gpu_manager()
    
    # Test GPU parameters
    gpu_params = gpu_manager.get_xgboost_gpu_params(use_gpu=True)
    print(f"GPU Parameters: {gpu_params}")
    
    # Test CPU parameters
    cpu_params = gpu_manager.get_xgboost_gpu_params(use_gpu=False)
    print(f"CPU Parameters: {cpu_params}")

def test_cuml_random_forest():
    """Test cuML RandomForest availability"""
    print("\n" + "=" * 60)
    print("🧪 Testing cuML RandomForest")
    print("=" * 60)
    
    gpu_manager = get_gpu_manager()
    
    if gpu_manager.cuml_available:
        try:
            cuml_rf = gpu_manager.get_cuml_random_forest(
                n_estimators=10,
                random_state=42
            )
            if cuml_rf is not None:
                print("✅ cuML RandomForest created successfully")
                print(f"Model type: {type(cuml_rf)}")
            else:
                print("❌ cuML RandomForest creation returned None")
        except Exception as e:
            print(f"❌ cuML RandomForest creation failed: {e}")
    else:
        print("⚠️ cuML not available")

def test_cli_gpu_info():
    """Test CLI GPU info functionality"""
    print("\n" + "=" * 60)
    print("🧪 Testing CLI GPU Info")
    print("=" * 60)
    
    # Override sys.argv to test --gpu_info
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["screener", "--gpu_info"]
        print("Testing --gpu_info flag...")
        main()
    except SystemExit as e:
        if e.code == 0:
            print("✅ --gpu_info flag works correctly")
        else:
            print(f"❌ --gpu_info flag failed with exit code: {e.code}")
    except Exception as e:
        print(f"❌ --gpu_info flag failed with exception: {e}")
    finally:
        sys.argv = original_argv

def test_model_training_cpu_mode():
    """Test model training in CPU-only mode"""
    print("\n" + "=" * 60)
    print("🧪 Testing Model Training (CPU Mode)")
    print("=" * 60)
    
    original_argv = sys.argv.copy()
    try:
        # Test with CPU-only mode
        sys.argv = [
            "screener", 
            "--mode", "eval", 
            "--tickers", "AAPL", 
            "--period", "1mo",
            "--no_gpu",
            "--n_estimators", "10"  # Small number for quick test
        ]
        print("Testing CPU-only mode with --no_gpu flag...")
        main()
        print("✅ CPU-only mode completed successfully")
    except Exception as e:
        print(f"❌ CPU-only mode failed: {e}")
    finally:
        sys.argv = original_argv

def test_model_training_gpu_mode():
    """Test model training with GPU enabled (if available)"""
    print("\n" + "=" * 60)
    print("🧪 Testing Model Training (GPU Mode)")
    print("=" * 60)
    
    gpu_manager = get_gpu_manager()
    if not gpu_manager.is_gpu_available():
        print("⚠️ Skipping GPU mode test - no GPU available")
        return
    
    original_argv = sys.argv.copy()
    try:
        # Test with GPU mode
        sys.argv = [
            "screener", 
            "--mode", "eval", 
            "--tickers", "AAPL", 
            "--period", "1mo",
            "--model", "xgboost",  # XGBoost has better GPU support
            "--n_estimators", "10"  # Small number for quick test
        ]
        print("Testing GPU mode with XGBoost...")
        main()
        print("✅ GPU mode completed successfully")
    except Exception as e:
        print(f"❌ GPU mode failed: {e}")
    finally:
        sys.argv = original_argv

def main_test():
    """Run all tests"""
    print("🚀 AI Stock Screener CUDA Support Test Suite")
    print("=" * 60)
    
    # Test 1: GPU Detection
    gpu_available = test_gpu_detection()
    
    # Test 2: XGBoost GPU Parameters
    test_xgboost_gpu_params()
    
    # Test 3: cuML RandomForest
    test_cuml_random_forest()
    
    # Test 4: CLI GPU Info
    test_cli_gpu_info()
    
    # Test 5: CPU Mode (always test this)
    test_model_training_cpu_mode()
    
    # Test 6: GPU Mode (only if GPU available)
    if gpu_available:
        test_model_training_gpu_mode()
    
    print("\n" + "=" * 60)
    print("🎉 Test Suite Completed")
    print("=" * 60)
    
    if gpu_available:
        print("✅ GPU acceleration is available and tested")
    else:
        print("⚠️ GPU acceleration not available - CPU-only mode tested")

if __name__ == "__main__":
    main_test()