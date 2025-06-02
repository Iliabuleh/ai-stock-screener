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
poetry run screener --mode eval --tickers AAPL NVDA AMZN
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
  