# 📈 AI Stock Screener CLI

This command-line tool uses technical and fundamental indicators to screen for high-growth stocks using machine learning. It supports **discovery mode** (scan entire S&P 500) and **evaluation mode** (analyze your own ticker list).

---

## 🚀 Features

- Discovery mode: Scans the S&P 500 for growth signals
- Evaluation mode: Lets you supply your own tickers
- Uses RSI, SMA(50/200), Volume trends, and P/E ratio
- Built-in machine learning classifier (Random Forest)

---

## ⚙️ Setup

### 1. Install Poetry
```bash
pip install poetry
```

### 2. Set Poetry to create venv 
```bash
poetry config virtualenvs.in-project true
```

### 3. Install Dependencies
```bash
poetry install
```

### 4. Usage
##### Discovery mode (scan S&P 500)
```bash
poetry run screener --mode discovery
```

##### Evaluation Mode (your own tickers)
```bash
poetry run screener --mode eval --tickers AAPL,NVDA,AMZN
```

## 📊 Sample Output Examples

### Discovery Mode Output
```bash
poetry run screener --mode discovery --period 1y --threshold 0.05 --future_days 10
```

```
🔍 AI Stock Screener - Discovery Mode
📊 Scanning S&P 500 for growth opportunities...
📅 Using 1y historical data, predicting 10 days ahead
🎯 Growth threshold: 5.0%

📈 Fetching S&P 500 constituent data...
✅ Found 503 tickers in S&P 500

🤖 Training Random Forest model with 100 estimators...
📊 Processing technical indicators for 503 stocks...

⚡ Grid Search Results:
Best parameters: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}
Cross-validation score: 0.847

🎯 MODEL PERFORMANCE:
Accuracy: 82.3%
Precision: 0.79
Recall: 0.74
F1-Score: 0.76

📈 TOP GROWTH PREDICTIONS (Probability > 0.70):

┌─────────┬──────────────┬─────────────┬────────┬─────────┬──────────┬─────────────┐
│ Ticker  │ Company      │ Growth Prob │ RSI    │ Price   │ Vol Chg  │ P/E Ratio   │
├─────────┼──────────────┼─────────────┼────────┼─────────┼──────────┼─────────────┤
│ NVDA    │ NVIDIA Corp  │ 0.94        │ 45.2   │ $875.28 │ +127%    │ 73.5        │
│ META    │ Meta Plat.   │ 0.89        │ 52.1   │ $485.75 │ +89%     │ 24.8        │
│ GOOGL   │ Alphabet     │ 0.87        │ 48.7   │ $142.56 │ +76%     │ 23.1        │
│ AAPL    │ Apple Inc    │ 0.82        │ 41.9   │ $189.87 │ +45%     │ 29.2        │
│ MSFT    │ Microsoft    │ 0.79        │ 55.3   │ $411.22 │ +67%     │ 32.4        │
│ TSLA    │ Tesla Inc    │ 0.76        │ 38.4   │ $248.42 │ +156%    │ 62.1        │
│ AMD     │ AMD Inc      │ 0.74        │ 44.8   │ $137.89 │ +134%    │ 45.7        │
│ CRM     │ Salesforce   │ 0.72        │ 46.2   │ $267.31 │ +92%     │ 58.9        │
└─────────┴──────────────┴─────────────┴────────┴─────────┴──────────┴─────────────┘

🔥 HIGH CONFIDENCE PICKS (> 85% probability):
• NVDA - Strong momentum, oversold RSI, massive volume spike
• META - Technical breakout pattern, strong fundamentals
• GOOGL - Consolidation phase ending, good value

📊 Market Context:
S&P 500 Trend: BULLISH (+2.1% this week)
VIX Level: 18.2 (Moderate volatility)
Integration: ENABLED (market trends considered)

✅ Scan complete! 8 high-probability growth candidates identified.
```

### Evaluation Mode Output
```bash
poetry run screener --mode eval --tickers NVDA,AAPL,TSLA,AMD,GOOGL --period 6mo --threshold 0.07
```

```
🔍 AI Stock Screener - Evaluation Mode
📊 Analyzing your portfolio: NVDA, AAPL, TSLA, AMD, GOOGL
📅 Using 6mo historical data, predicting 5 days ahead
🎯 Growth threshold: 7.0%

🤖 Training Random Forest model with 100 estimators...
📊 Processing technical indicators for 5 stocks...

⚡ Grid Search enabled - optimizing hyperparameters...
Best parameters: {'max_depth': 8, 'min_samples_split': 3, 'n_estimators': 150}

🎯 MODEL PERFORMANCE:
Accuracy: 78.9%
Precision: 0.82
Recall: 0.71
F1-Score: 0.76
Cross-validation score: 0.791

📈 DETAILED STOCK ANALYSIS:

┌─────────┬──────────────────────┬─────────┬────────┬─────────┬──────────┬─────────────┐
│ Ticker  │ Current Analysis     │ Prob    │ RSI    │ Price   │ Vol Chg  │ Prediction  │
├─────────┼──────────────────────┼─────────┼────────┼─────────┼──────────┼─────────────┤
│ NVDA    │ 🟢 STRONG BUY       │ 0.91    │ 42.1   │ $875.28 │ +198%    │ GROWTH      │
│         │ • Oversold + Volume  │         │        │         │          │             │
│         │ • Breaking resistance│         │        │         │          │             │
├─────────┼──────────────────────┼─────────┼────────┼─────────┼──────────┼─────────────┤
│ GOOGL   │ 🟡 MODERATE BUY      │ 0.73    │ 51.8   │ $142.56 │ +67%     │ GROWTH      │
│         │ • Neutral RSI        │         │        │         │          │             │
│         │ • Good volume        │         │        │         │          │             │
├─────────┼──────────────────────┼─────────┼────────┼─────────┼──────────┼─────────────┤
│ AAPL    │ 🟡 MODERATE BUY      │ 0.68    │ 58.9   │ $189.87 │ +34%     │ GROWTH      │
│         │ • Near overbought    │         │        │         │          │             │
│         │ • Steady momentum    │         │        │         │          │             │
├─────────┼──────────────────────┼─────────┼────────┼─────────┼──────────┼─────────────┤
│ AMD     │ 🔶 HOLD              │ 0.58    │ 61.2   │ $137.89 │ +89%     │ HOLD        │
│         │ • Overbought RSI     │         │        │         │          │             │
│         │ • Mixed signals      │         │        │         │          │             │
├─────────┼──────────────────────┼─────────┼────────┼─────────┼──────────┼─────────────┤
│ TSLA    │ 🔴 WEAK               │ 0.34    │ 71.3   │ $248.42 │ -23%     │ NO GROWTH   │
│         │ • Very overbought    │         │        │         │          │             │
│         │ • Volume declining   │         │        │         │          │             │
└─────────┴──────────────────────┴─────────┴────────┴─────────┴──────────┴─────────────┘

📊 TECHNICAL INDICATORS BREAKDOWN:

NVDA Analysis:
├── RSI: 42.1 (Oversold territory)
├── SMA50: $789.43 (Price above)
├── SMA200: $634.21 (Strong uptrend)
├── Volume: +198% vs avg (Massive interest)
├── P/E Ratio: 73.5 (High but justified by growth)
└── 🎯 Prediction: 91.2% probability of 7%+ gain in 5 days

GOOGL Analysis:
├── RSI: 51.8 (Neutral zone)
├── SMA50: $136.78 (Price above)
├── SMA200: $128.45 (Uptrend confirmed)
├── Volume: +67% vs avg (Good participation)
├── P/E Ratio: 23.1 (Reasonable valuation)
└── 🎯 Prediction: 73.4% probability of 7%+ gain in 5 days

📈 PORTFOLIO SUMMARY:
✅ Strong Growth Candidates: 2 (NVDA, GOOGL)
🟡 Moderate Growth: 1 (AAPL)
🔶 Hold Positions: 1 (AMD)
🔴 Weak/Avoid: 1 (TSLA)

🎯 RECOMMENDED ACTIONS:
1. 🚀 NVDA - High conviction buy (91% probability)
2. 📈 GOOGL - Good entry point (73% probability)
3. ⚖️ AAPL - Small position or wait for pullback
4. ⏸️ AMD - Monitor for better entry
5. ⚠️ TSLA - Consider profit-taking if holding

📊 Market Context:
S&P 500 Trend: BULLISH (+1.8% this week)
Tech Sector: OUTPERFORMING (+3.2% vs S&P)
Market Integration: ENABLED

⏱️ Analysis completed in 23.4 seconds
🎯 Next update recommended: Check back in 2-3 trading days
```

## 🛠️ CLI Parameters Explained

| Parameter         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--mode`         | Operation mode: `discovery` (scan S&P 500) or `eval` (custom tickers). Required. |
| `--tickers`      | Comma-separated list of stock symbols (required for `eval` mode).           |
| `--period`       | Historical data period for yfinance (e.g., `6mo`, `1y`). Default: `6mo`.    |
| `--future_days`  | Days ahead to calculate return and create label. Default: `5`.               |
| `--threshold`    | Return threshold to label a day as `growth`. Default: `0.0`.                 |
| `--n_estimators` | Number of trees in the Random Forest model. Default: `100`.                 |
| `--model`        | Model type to use: `random_forest` or `xgboost`. Default: `random_forest`.  |
| `--grid_search`  | Enable grid search for hyperparameters (1=enabled, 0=disabled). Default: `1`. |
| `--ensemble_runs`| Number of models to train and average. Default: `1`.                        |
| `--use_sharpe_labeling` | Enable return-volatility labeling with given threshold. Default: `1.0`. |
| `--no_integrate_market` | Disable market trend integration in predictions. Default: False.           |

- **Mode Options**:
  - `discovery`: Scans entire S&P 500 for growth opportunities
  - `eval`: Analyzes specific tickers provided via `--tickers` parameter

- **Market Integration**:
  - By default, the model considers overall market trends in predictions
  - Use `--no_integrate_market` to analyze stocks independently of market conditions

- **Threshold Usage**:
  - `0.0`: Any positive return counts as growth.
  - `0.02`: Only returns greater than 2% are labeled as growth.

- **Number of Trees (n_estimators)**:
  - A higher value generally improves accuracy.
  - More trees increase training time and memory usage.
  - Example: `--n_estimators 200` uses 200 decision trees.

  
  ### Discovery Mode Example
  Scan the S&P 500 for stocks with strong growth potential using a 2-year lookback period and 30-day future prediction:
  ```bash
  poetry run screener --mode discovery \
    --period 2y \
    --threshold 0.07 \
    --future_days 30 \
    --n_estimators 1000
  ```
  This configuration:
  - Uses 2 years of historical data
  - Labels stocks as growth if they return >7% in 30 days
  - Uses 1000 trees for higher model accuracy

  ### Evaluation Mode Example
  Analyze a specific list of tech and semiconductor stocks using a 1-year lookback:
  ```bash
  poetry run screener --mode eval \
    --tickers NVDA,QCOM,AVGO,APP,TSLA,SPY,GOOGL,PLTR,HIMS,QQQ,MSTR,AAPL,XBI \
    --period 1y \
    --threshold 0.07 \
    --future_days 30 \
    --n_estimators 300
  ```
  This configuration:
  - Evaluates 13 specific stocks
  - Uses 1 year of historical data
  - Labels stocks as growth if they return >7% in 30 days
  - Uses 300 trees for balanced performance

## 🎯 Key Features Explained

### 🤖 **Machine Learning Models**
- **Random Forest**: Default model with ensemble of decision trees
- **XGBoost**: Alternative gradient boosting model for comparison
- **Grid Search**: Automatic hyperparameter optimization
- **Cross-validation**: Robust performance estimation

### 📊 **Technical Indicators**
- **RSI**: Relative Strength Index for momentum analysis
- **SMA 50/200**: Simple moving averages for trend identification  
- **Volume Analysis**: Trading volume changes vs historical average
- **Price Action**: Support/resistance and breakout patterns

### 🎯 **Prediction System**
- **Probability Scores**: Confidence level for each prediction (0-1)
- **Growth Labeling**: Customizable return thresholds
- **Future Timeframe**: Adjustable prediction horizon (1-30 days)
- **Market Integration**: Considers overall market conditions

### 📈 **Portfolio Tools**
- **Discovery Mode**: Find new opportunities across S&P 500
- **Evaluation Mode**: Analyze your existing watchlist
- **Risk Assessment**: Visual indicators for position sizing
- **Action Recommendations**: Clear buy/hold/sell guidance

## 🚀 GPU Acceleration & Performance

### **GPU-Accelerated Models**
The AI Stock Screener supports GPU acceleration for significantly improved performance with larger datasets:

- **RandomForest**: Uses cuML GPU acceleration when available
- **XGBoost**: Features **dynamic parameter optimization** with automatic GPU tuning based on dataset characteristics
- **Automatic Fallback**: Seamlessly falls back to CPU if GPU unavailable
- **Scaling Benefits**: Performance improvements increase with dataset size
- **Smart Optimization**: Automatically selects optimal parameters for Small/Medium/Large datasets

### **GPU Setup**
```bash
# Install with GPU dependencies
poetry install -E gpu

# Check GPU status
poetry run screener --gpu_info

# Force CPU-only mode if needed
poetry run screener --mode eval --tickers AAPL,NVDA --no_gpu
```

### **Performance Benchmarks**
Based on extended benchmarking across different dataset sizes with **XGBoost optimization enabled**:

| Dataset Size | RandomForest GPU Speedup | XGBoost GPU Speedup | XGBoost Optimization |
|--------------|-------------------------|-------------------|---------------------|
| 8 tickers    | 1.02x                  | 0.86x (CPU faster) | Small Dataset Mode |
| 20 tickers   | 1.08x                  | **1.19x** ⬆️        | Medium Dataset Mode |
| 50 tickers   | 1.14x                  | **1.34x** ⬆️        | Large Dataset Mode |
| 100 tickers  | **1.21x**              | **~1.40x** ⬆️       | Large Dataset Mode |

**Key Performance Insights:**
- GPU benefits increase with larger datasets
- RandomForest shows consistent GPU advantage
- **XGBoost optimization significantly improved**: Medium datasets now show 1.19x speedup (vs 1.02x), Large datasets show 1.34x speedup (vs 1.18x)
- **Dynamic parameter tuning**: Automatically optimizes XGBoost GPU parameters based on dataset size
- Best performance: XGBoost with 50+ tickers using automatic optimization (34%+ speedup)

### **Recommended Configurations**
```bash
# Small datasets (≤20 tickers): Either GPU or CPU
poetry run screener --mode eval --tickers AAPL,GOOGL,TSLA --model random_forest

# Medium datasets (20-50 tickers): GPU recommended
poetry run screener --mode eval --tickers [20-50 tickers] --model random_forest

# Large datasets (50+ tickers): GPU strongly recommended
poetry run screener --mode discovery --model random_forest  # Uses full S&P 500
```

### **Extended Benchmarking**
For detailed performance analysis across different configurations, see:
- `BENCHMARK_REPORT.md` - Comprehensive GPU vs CPU performance analysis
- `benchmark_gpu_vs_cpu.py` - Standard benchmarking script
- `extended_benchmark.py` - Scaling analysis across dataset sizes

## 🧠 Understanding the Output

### **Probability Scores**
- **> 0.85**: High confidence predictions
- **0.70-0.85**: Moderate confidence  
- **0.50-0.70**: Low confidence
- **< 0.50**: Negative/bearish prediction

### **Visual Indicators**
- **🟢 STRONG BUY**: High probability + favorable technicals
- **🟡 MODERATE BUY**: Good probability with some caution
- **🔶 HOLD**: Mixed signals or overbought conditions
- **🔴 WEAK**: Low probability or bearish indicators

### **Technical Analysis**
- **RSI < 30**: Oversold (potential buy opportunity)
- **RSI > 70**: Overbought (consider taking profits)
- **Volume +**: Increased participation confirms moves
- **SMA Positioning**: Price vs moving averages shows trend

## 🚨 Important Disclaimers

- **Past performance** does not guarantee future results
- **Machine learning predictions** are probabilistic, not certainties
- **Market conditions** can change rapidly and invalidate technical analysis
- **Risk management** is essential - never invest more than you can afford to lose
- **Do your own research** - this tool is for informational purposes only

## 🤝 Contributing

This tool is part of the larger ai-stock-screener project. Feel free to suggest improvements or report issues!

## 📜 License

AI Stock Screener by Ilia Buleh.

---

**Happy screening! 🚀📊**
  