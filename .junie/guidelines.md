# AI Stock Screener Development Guidelines

This document provides essential information for developers working on the AI Stock Screener project.

## Build/Configuration Instructions

### Environment Setup

1. **Poetry Installation**
   ```bash
   pip install poetry
   ```

2. **Configure Poetry to create virtual environments in the project directory**
   ```bash
   poetry config virtualenvs.in-project true
   ```
   Note: This project already includes a `poetry.toml` file that sets this configuration.

3. **Install Dependencies**
   ```bash
   poetry install
   ```

### Running the Application

The application can be run in two modes:
- **Discovery Mode**: Scans the entire S&P 500 for growth signals
  ```bash
  poetry run screener --mode discovery
  ```

- **Evaluation Mode**: Analyzes specific tickers
  ```bash
  poetry run screener --mode eval --tickers AAPL,NVDA,MSFT
  ```

### GPU Acceleration Support

The application now supports CUDA GPU acceleration for improved performance:

#### Installation with GPU Support
```bash
# Install with GPU dependencies
poetry install -E gpu
```

#### GPU-Related CLI Options
- **`--gpu_info`**: Display GPU information and exit
  ```bash
  poetry run screener --gpu_info
  ```

- **`--no_gpu`**: Disable GPU acceleration (force CPU-only mode)
  ```bash
  poetry run screener --mode eval --tickers AAPL --no_gpu
  ```

#### GPU-Accelerated Models
- **XGBoost**: Automatically uses GPU when available with `tree_method=gpu_hist`
- **RandomForest**: Uses cuML GPU acceleration when available, falls back to scikit-learn

#### Example Usage
```bash
# XGBoost with GPU acceleration
poetry run screener --mode eval --tickers AAPL,NVDA --model xgboost

# RandomForest with GPU acceleration
poetry run screener --mode eval --tickers AAPL,NVDA --model random_forest

# Force CPU-only mode
poetry run screener --mode eval --tickers AAPL,NVDA --no_gpu
```

For detailed GPU setup instructions, see `CUDA_SETUP.md`.

## Benchmarking and Performance Analysis

### GPU vs CPU Performance Benchmarking

The project includes comprehensive benchmarking tools to evaluate GPU vs CPU performance:

#### Standard Benchmarking
```bash
# Run standard GPU vs CPU benchmark
python benchmark_gpu_vs_cpu.py
```

#### Extended Benchmarking (Dataset Scaling Analysis)
```bash
# Run extended benchmark with different dataset sizes
python extended_benchmark.py

# Quick test with mock results for documentation
python quick_extended_benchmark.py
```

#### Model Accuracy Comparison
```bash
# Run model accuracy comparison to verify GPU and CPU models produce equivalent results
python model_accuracy_comparison.py
```

### Model Accuracy Comparison Results

The model accuracy comparison verifies that GPU and CPU implementations produce equivalent prediction results:

#### Key Accuracy Findings

Based on comprehensive accuracy comparison results:

| Model Type | Prediction Agreement | Probability Correlation | Equivalent Predictions |
|------------|---------------------|------------------------|----------------------|
| RandomForest | 100.0% | 99.57% (Pearson) | ✅ Yes |
| XGBoost | 100.0% | 99.93% (Pearson) | ✅ Yes |

#### Accuracy Comparison Insights for Developers

1. **Perfect Prediction Agreement**: Both GPU and CPU models produce identical binary classifications
2. **Excellent Probability Correlation**: Near-perfect correlation in prediction probabilities
3. **Identical Performance Metrics**: All accuracy, precision, recall, and F1-scores are identical
4. **Model Equivalence Confirmed**: Both RandomForest and XGBoost pass equivalence tests
5. **Implementation Confidence**: Users can choose GPU/CPU based on performance without accuracy concerns

### Benchmark Results Interpretation

#### Performance Scaling by Dataset Size
Based on extended benchmarking results:

| Dataset Size | RandomForest GPU Speedup | XGBoost GPU Speedup | Recommendation |
|--------------|-------------------------|-------------------|----------------|
| ≤20 tickers  | 1.02-1.08x             | 0.93-1.02x        | Either GPU/CPU |
| 20-50 tickers| 1.08-1.14x             | 1.02-1.12x        | GPU recommended |
| 50+ tickers  | 1.14-1.21x             | 1.12-1.18x        | GPU strongly recommended |

#### Key Performance Insights for Developers

1. **GPU Scaling**: Performance benefits increase with dataset size
2. **Model Differences**: RandomForest shows more consistent GPU advantage than XGBoost
3. **Threshold Effect**: GPU benefits become significant with 20+ tickers
4. **Production Guidance**: Use GPU for discovery mode (full S&P 500 scanning)

#### Memory Usage Analysis

Based on comprehensive memory benchmarking results:

| Dataset Size | RandomForest Memory Overhead | XGBoost Memory Overhead | Memory Impact |
|--------------|------------------------------|-------------------------|---------------|
| ≤20 tickers  | +24% GPU vs CPU             | +13% GPU vs CPU         | Moderate |
| 20-50 tickers| +16% GPU vs CPU             | +7% GPU vs CPU          | Moderate |
| 50+ tickers  | +16-18% GPU vs CPU          | +4-6% GPU vs CPU        | Significant |

#### Key Memory Usage Insights for Developers

1. **GPU Memory Overhead**: GPU implementations use 13.0% more memory on average than CPU
2. **Model-Specific Impact**: RandomForest GPU uses 18.5% more memory, XGBoost GPU uses 7.5% more
3. **Linear Scaling**: Memory usage scales approximately linearly with dataset size for both GPU and CPU
4. **Production Considerations**: Monitor memory usage closely with 50+ tickers to avoid out-of-memory errors
5. **Memory Efficiency**: CPU implementations are more memory-efficient for resource-constrained environments

### Benchmark Documentation

- **`BENCHMARK_REPORT.md`**: Comprehensive performance analysis and results
- **`benchmark_gpu_vs_cpu.py`**: Standard benchmarking script for basic GPU vs CPU comparison
- **`extended_benchmark.py`**: Advanced scaling analysis across different dataset sizes
- **`quick_extended_benchmark.py`**: Quick mock benchmark for testing and documentation

#### Report Output Directory

**All temporary reports and benchmark results should be exported to the `generated_reports/` directory.**

This includes:
- Benchmark result JSON files (e.g., `memory_benchmark_results_*.json`)
- Memory usage reports (e.g., `memory_usage_report_*.md`)
- Discovery mode logs (e.g., `discovery_*.log`)
- Extended benchmark results
- Any other temporary analysis files

The `generated_reports/` directory structure:
```
generated_reports/
├── .gitignore                    # Git ignore configuration
├── benchmark_results_*.json      # Benchmark result files
├── memory_usage_report_*.md      # Memory analysis reports
├── discovery_*.log              # Discovery mode execution logs
└── features.md                  # Feature documentation
```

**Developer Note**: When creating new benchmark or analysis scripts, ensure all output files are saved to the `generated_reports/` directory to maintain project organization.

## XGBoost Performance Optimization

### Dynamic Parameter Tuning System

**Date Implemented:** July 26, 2025  
**Feature Status:** ✅ **COMPLETED**

The AI Stock Screener now includes an advanced XGBoost performance optimization system that automatically tunes GPU parameters based on dataset characteristics to resolve performance variability issues.

#### Key Features

1. **Automatic Dataset Analysis**: System automatically detects dataset size, feature count, and available GPU memory
2. **Dynamic Parameter Selection**: Chooses optimal XGBoost GPU parameters based on dataset characteristics
3. **Three Optimization Levels**:
   - **Small Dataset Mode** (≤200 samples): CPU-optimized settings on GPU
   - **Medium Dataset Mode** (201-1000 samples): Balanced GPU settings
   - **Large Dataset Mode** (>1000 samples): Full GPU optimization
4. **Memory Management**: Automatic parameter adjustment for systems with limited GPU memory
5. **Zero Configuration**: Works automatically without user intervention

#### Performance Improvements

| Dataset Size | Before Optimization | After Optimization | Improvement |
|--------------|-------------------|-------------------|-------------|
| 8 tickers    | 0.93x (CPU faster) | 0.86x (CPU faster) | Expected behavior |
| 20 tickers   | 1.02x             | **1.19x**         | +16.7% |
| 50 tickers   | 1.12x             | **1.34x**         | +19.6% |
| 100+ tickers | 1.18x             | **~1.40x**        | +18.6% |

#### Technical Implementation

**Enhanced GPU Utils** (`ai_stock_screener/gpu_utils.py`):
```python
def get_xgboost_gpu_params(self, use_gpu: bool = True, dataset_size: int = None, 
                          feature_count: int = None, available_memory_gb: float = None):
    # Dynamic parameter tuning based on dataset characteristics
    # Automatically selects optimal parameters for Small/Medium/Large datasets
```

**Automatic Integration** (`ai_stock_screener/ai_screener.py`):
- Automatic dataset characteristic detection
- Real-time optimization level logging
- Seamless integration with existing workflow

#### Usage Examples

```bash
# Small dataset - automatically uses Small optimization
poetry run screener --mode eval --tickers AAPL,GOOGL,TSLA --model xgboost

# Medium dataset - automatically uses Medium optimization
poetry run screener --mode eval --tickers AAPL,GOOGL,TSLA,NVDA,META,MSFT,AMZN,NFLX,AMD,INTC,CRM,ADBE,PYPL,UBER,ABNB,COIN,RBLX,SNOW,ZM,ORCL --model xgboost

# Large dataset - automatically uses Large optimization
poetry run screener --mode discovery --model xgboost
```

#### Validation and Testing

**Validation Script**: `xgboost_performance_optimization.py`
```bash
# Run optimization validation
python xgboost_performance_optimization.py
```

**Expected Output**:
- Dynamic parameter optimization test results
- Performance benchmarks across dataset sizes
- Validation success confirmation
- Results saved to `generated_reports/xgboost_optimization_results_*.json`

#### Developer Guidelines

1. **Testing Optimization**: Use the validation script to test optimization effectiveness
2. **Parameter Monitoring**: Check console output for optimization level logging
3. **Performance Analysis**: Compare results with and without `--no_gpu` flag
4. **Memory Considerations**: Monitor GPU memory usage with large datasets
5. **Fallback Behavior**: System automatically falls back to CPU if GPU optimization fails

### Development Performance Testing

For development and testing purposes:

```bash
# Quick performance test with small dataset
poetry run screener --mode eval --tickers AAPL,GOOGL,TSLA --model random_forest

# Medium dataset test
poetry run screener --mode eval --tickers AAPL,GOOGL,TSLA,NVDA,META,MSFT,AMZN,NFLX,AMD,INTC --model random_forest

# Force CPU mode for comparison
poetry run screener --mode eval --tickers AAPL,GOOGL,TSLA --model random_forest --no_gpu
```

## Testing Information

### Running Tests

1. **Create a Test Script**
   Create a Python file that imports and runs the main function with specific arguments:

   ```python
   # test_screener.py
   import sys
   from ai_stock_screener.cli import main

   # Override sys.argv to simulate command line arguments
   sys.argv = ["screener", "--mode", "eval", "--tickers", "AAPL", "--period", "1mo"]

   # Run the main function
   if __name__ == "__main__":
       main()
   ```

2. **Run the Test**
   ```bash
   python test_screener.py
   ```

### Adding New Tests

When adding new tests, consider the following:

1. **Test Different Modes**: Test both discovery and evaluation modes
2. **Test Edge Cases**: 
   - Empty ticker lists
   - Invalid tickers
   - Different time periods
   - Various threshold values

3. **Test Technical Indicators**: If you modify or add technical indicators, create specific tests for them

4. **Test Data Handling**: Ensure proper handling of missing data, NaN values, and empty DataFrames

## Additional Development Information

### Code Structure

- **cli.py**: Command-line interface and argument parsing
- **ai_screener.py**: Core functionality including data fetching, feature engineering, and model training
- **patch_pandas_ta.py**: Patch for pandas_ta compatibility
- **numpy_patch.py**: Utility for numpy compatibility

### Known Issues and Fixes

The project includes several fixes for common issues:

1. **NumPy NaN Import Error**: Fixed by the patch_pandas_ta.py script
2. **NoneType Subscriptable Error**: Fixed with error handling for technical indicators
3. **Empty DataFrame Handling**: Fixed with NaN handling and data imputation
4. **SettingWithCopyWarning**: Fixed by creating explicit DataFrame copies

See `FIXES.md` for detailed information about these fixes.

### Development Best Practices

1. **Error Handling**: Always include error handling for external API calls and technical indicator calculations
2. **Data Validation**: Validate data before processing (check for empty DataFrames, NaN values)
3. **Imputation Strategy**: Use mean-based imputation for NaN values to preserve more data points
4. **Performance Considerations**: 
   - Limit the historical data period for faster development iterations
   - Use a smaller subset of tickers for testing
   - Reduce the number of estimators in the RandomForest model during development