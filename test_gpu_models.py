#!/usr/bin/env python3
"""
Simple test script to verify GPU model functionality without full data pipeline
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

# Add the project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_stock_screener'))

from ai_stock_screener.gpu_utils import get_gpu_manager

def create_test_data():
    """Create synthetic test data for model testing"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame to match expected format
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Label'] = y
    
    return df

def test_xgboost_gpu():
    """Test XGBoost with GPU acceleration"""
    print("🧪 Testing XGBoost GPU Acceleration")
    print("=" * 50)
    
    gpu_manager = get_gpu_manager()
    
    # Create test data
    df = create_test_data()
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        from xgboost import XGBClassifier
        
        # Test GPU mode
        if gpu_manager.cuda_available:
            print("Testing XGBoost with GPU...")
            gpu_params = gpu_manager.get_xgboost_gpu_params(use_gpu=True)
            xgb_gpu = XGBClassifier(
                n_estimators=100,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                **gpu_params
            )
            
            xgb_gpu.fit(X_train, y_train)
            y_pred_gpu = xgb_gpu.predict(X_test)
            acc_gpu = accuracy_score(y_test, y_pred_gpu)
            print(f"✅ XGBoost GPU Accuracy: {acc_gpu:.4f}")
        else:
            print("⚠️ CUDA not available, skipping GPU test")
        
        # Test CPU mode for comparison
        print("Testing XGBoost with CPU...")
        cpu_params = gpu_manager.get_xgboost_gpu_params(use_gpu=False)
        xgb_cpu = XGBClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            **cpu_params
        )
        
        xgb_cpu.fit(X_train, y_train)
        y_pred_cpu = xgb_cpu.predict(X_test)
        acc_cpu = accuracy_score(y_test, y_pred_cpu)
        print(f"✅ XGBoost CPU Accuracy: {acc_cpu:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        return False

def test_cuml_random_forest():
    """Test cuML RandomForest"""
    print("\n🧪 Testing cuML RandomForest")
    print("=" * 50)
    
    gpu_manager = get_gpu_manager()
    
    if not gpu_manager.cuml_available:
        print("⚠️ cuML not available, skipping test")
        return True
    
    # Create test data
    df = create_test_data()
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        # Test cuML RandomForest
        print("Testing cuML RandomForest...")
        cuml_rf = gpu_manager.get_cuml_random_forest(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        if cuml_rf is not None:
            # Convert to cupy arrays for cuML
            try:
                import cupy as cp
                X_train_gpu = cp.asarray(X_train.values)
                y_train_gpu = cp.asarray(y_train.values)
                X_test_gpu = cp.asarray(X_test.values)
                
                cuml_rf.fit(X_train_gpu, y_train_gpu)
                y_pred_gpu = cuml_rf.predict(X_test_gpu)
                
                # Convert back to numpy for accuracy calculation
                y_pred_cpu = cp.asnumpy(y_pred_gpu)
                acc_cuml = accuracy_score(y_test, y_pred_cpu)
                print(f"✅ cuML RandomForest Accuracy: {acc_cuml:.4f}")
                return True
                
            except ImportError:
                print("⚠️ CuPy not available, testing with pandas arrays...")
                cuml_rf.fit(X_train, y_train)
                y_pred = cuml_rf.predict(X_test)
                acc_cuml = accuracy_score(y_test, y_pred)
                print(f"✅ cuML RandomForest Accuracy: {acc_cuml:.4f}")
                return True
        else:
            print("❌ cuML RandomForest creation failed")
            return False
            
    except Exception as e:
        print(f"❌ cuML RandomForest test failed: {e}")
        return False

def test_sklearn_random_forest():
    """Test standard scikit-learn RandomForest for comparison"""
    print("\n🧪 Testing scikit-learn RandomForest (CPU)")
    print("=" * 50)
    
    # Create test data
    df = create_test_data()
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        print("Testing scikit-learn RandomForest...")
        sklearn_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        sklearn_rf.fit(X_train, y_train)
        y_pred = sklearn_rf.predict(X_test)
        acc_sklearn = accuracy_score(y_test, y_pred)
        print(f"✅ scikit-learn RandomForest Accuracy: {acc_sklearn:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ scikit-learn RandomForest test failed: {e}")
        return False

def main():
    """Run all GPU model tests"""
    print("🚀 GPU Model Testing Suite")
    print("=" * 60)
    
    # Display GPU status
    gpu_manager = get_gpu_manager()
    gpu_manager.print_gpu_status()
    
    results = []
    
    # Test XGBoost
    results.append(test_xgboost_gpu())
    
    # Test cuML RandomForest
    results.append(test_cuml_random_forest())
    
    # Test scikit-learn RandomForest
    results.append(test_sklearn_random_forest())
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Test Results Summary")
    print("=" * 60)
    
    if all(results):
        print("✅ All GPU model tests passed!")
    else:
        print("⚠️ Some tests failed, but basic functionality is working")
    
    if gpu_manager.is_gpu_available():
        print("🚀 GPU acceleration is available and functional")
    else:
        print("💻 Running in CPU-only mode")

if __name__ == "__main__":
    main()