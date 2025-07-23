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