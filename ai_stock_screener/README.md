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

| Parameter         | Default | Description                                                                 |
|------------------|---------|-----------------------------------------------------------------------------|
| `--mode`         | None | Operation mode: `discovery` (scan indexes) or `eval` (custom tickers). Required unless using `--hot-stocks`. |
| `--tickers`      | None | Comma-separated list of stock symbols (required for `eval` mode).           |
| `--index`        | `sp500` | Stock index(es): `sp500`, `russell1000`, `nasdaq`, `all`, or comma-separated combination. |
| `--period`       | `1y` | Historical data period for yfinance (e.g., `6mo`, `1y`, `2y`). |
| `--future_days`  | `5` | Days ahead to calculate return and create label. |
| `--threshold`    | `0.0` | Return threshold to label a day as `growth` (ML training labels). |
| `--n_estimators` | `300` | Number of trees in the Random Forest model. |
| `--use_sharpe_labeling` | `1.0` | Enable return-volatility labeling with given threshold. |
| `--model`        | `random_forest` | Model type: `random_forest` or `xgboost`. |
| `--grid_search`  | `0` | Enable grid search for hyperparameters (1=enabled, 0=disabled). |
| `--ensemble_runs`| `1` | Number of models to train and average. |
| `--no_integrate_market` | `False` | Disable market trend integration in predictions. |
| `--news`         | `False` | Enable news sentiment analysis (adds processing time). |
| `--sector`       | None | Filter by sector: Technology, Healthcare, Financials, etc. |
| `--hot-stocks`   | `0` | Pure momentum scanner: find top N trending stocks (independent mode). |

### 🔧 Advanced Configuration

### 🤖 **Discovery/Eval Modes (ML-Based Analysis)**

These modes use machine learning models with **fixed technical indicators** to ensure consistent feature engineering:

| Parameter | Default | Purpose | Impact | Notes |
|-----------|---------|---------|--------|-------|
| `--ml-probability-threshold` | `0.70` | Results display filter | Shows only stocks above this ML confidence | Display only - doesn't affect model |
| `--period` | `1y` | Historical data window | More data = better training | Core ML parameter |
| `--threshold` | `0.0` | Growth definition for training | What % return counts as "growth" | Core ML parameter |
| `--n_estimators` | `300` | Number of ML model trees | More trees = better accuracy, slower | Core ML parameter |
| `--future_days` | `5` | Prediction timeframe | How many days ahead to predict | Core ML parameter |
| `--model` | `random_forest` | ML algorithm selection | Choose between RandomForest/XGBoost | Core ML parameter |
| `--grid_search` | `0` | Hyperparameter optimization | Enable automatic parameter tuning | Core ML parameter |
| `--ensemble_runs` | `1` | Model ensemble size | Multiple models averaged together | Core ML parameter |

**Important**: Technical indicator periods (RSI, EMA, etc.) are **fixed** to match model training data and cannot be changed.

### 🔥 **Hot-Stocks Mode (Technical Momentum Analysis)**

This mode uses pure technical analysis with **fully configurable** momentum scoring:

| Parameter | Default | Purpose | Impact | Notes |
|-----------|---------|---------|--------|-------|
| `--trend-weight` | `0.35` | Long-term trend importance | Higher = favor strong secular trends | Momentum scoring |
| `--setup-weight` | `0.20` | Entry timing precision | Higher = precise SMA crossover timing | Momentum scoring |
| `--volume-weight` | `0.20` | Volume confirmation requirement | Higher = must have volume breakouts | Momentum scoring |
| `--momentum-data-period` | `18mo` | Historical analysis window | Longer = more trend context, slower | Data calculation |
| `--rsi-period` | `14` | RSI calculation sensitivity | Lower = faster signals, more noise | Technical indicator |
| `--ema-short` | `13` | Short-term momentum sensitivity | Lower = faster trend detection | Technical indicator |
| `--ema-long` | `48` | Long-term trend baseline | Higher = stronger trend confirmation | Technical indicator |

**Note**: Momentum weights automatically normalize to 1.0. Remaining weights (Price vs 20SMA: 15%, EMA: 5%, RSI: 5%) are kept at defaults.

### ⚠️ **Mode-Specific Usage**

| Mode | Configuration | Why |
|------|---------------|-----|
| **Discovery** | ML parameters + display threshold only | Uses pre-trained model with fixed indicators |
| **Evaluation** | ML parameters + display threshold only | Same as discovery mode |
| **Hot-Stocks** | All momentum weights + technical periods | Pure technical analysis, fully customizable |

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

## 🎯 Advanced Usage Examples

### 📊 **Original Scanner (Discovery/Eval) Examples**

These examples use the **ML-based scanner** with limited advanced configuration:

#### Conservative Discovery Scan
```bash
# Higher confidence threshold for safer picks
python -m ai_stock_screener.cli --mode discovery \
  --ml-probability-threshold 0.80 \
  --sector Technology \
  --rsi-period 21
```
**Effect**: Only shows stocks with >80% ML probability, uses slower RSI signals.

#### Aggressive Discovery Scan
```bash
# Lower threshold for more opportunities  
python -m ai_stock_screener.cli --mode discovery \
  --ml-probability-threshold 0.60 \
  --rsi-period 9
```
**Effect**: Shows stocks with >60% ML probability, uses faster RSI signals.

### 🔥 **Hot Stocks Mode Examples**

These examples use the **momentum scoring algorithm** with full configuration:

### Conservative Long-Term Investor
```bash
# Focus on strong secular trends with high confidence threshold
python -m ai_stock_screener.cli --hot-stocks 10 \
  --trend-weight 0.50 \
  --setup-weight 0.15 \
  --volume-weight 0.15 \
  --momentum-data-period "2y" \
  --rsi-period 21 \
  --ml-probability-threshold 0.80
```
**When to use**: Looking for long-term growth stories with strong fundamentals.

### Aggressive Momentum Trader  
```bash
# Focus on volume breakouts with faster signals
python -m ai_stock_screener.cli --hot-stocks 20 \
  --volume-weight 0.35 \
  --trend-weight 0.25 \
  --setup-weight 0.25 \
  --rsi-period 9 \
  --ema-short 8 \
  --ema-long 21 \
  --ml-probability-threshold 0.65
```
**When to use**: Day trading or short-term momentum plays.

### Balanced Growth Screening
```bash
# Technology sector with balanced weighting
python -m ai_stock_screener.cli --mode discovery \
  --sector Technology \
  --ml-probability-threshold 0.75 \
  --rsi-period 14
```
**When to use**: Medium-term position building in specific sectors.

### Swing Trading Setup
```bash
# Focus on precise entry timing
python -m ai_stock_screener.cli --hot-stocks 15 \
  --setup-weight 0.35 \
  --trend-weight 0.30 \
  --volume-weight 0.20 \
  --rsi-period 14 \
  --ema-short 13 \
  --ema-long 34
```
**When to use**: 1-4 week holding periods with technical precision.

## ⚖️ Weight Configuration Guidelines

### Trend Weight (--trend-weight)
- **High (0.45-0.60)**: Focus on secular growth stories, longer holding periods
- **Medium (0.30-0.45)**: Balanced approach, most common use case  
- **Low (0.20-0.30)**: Short-term trading, less concerned with long-term direction

### Setup Weight (--setup-weight)  
- **High (0.25-0.35)**: Precise entry timing, swing trading
- **Medium (0.15-0.25)**: Balanced momentum approach
- **Low (0.10-0.15)**: Less concerned with perfect timing

### Volume Weight (--volume-weight)
- **High (0.25-0.35)**: Breakout trading, momentum confirmation critical
- **Medium (0.15-0.25)**: Standard momentum analysis
- **Low (0.10-0.15)**: Volume less important for strategy

## 📊 Technical Indicator Guidelines

### RSI Period
- **Short (9-12)**: Day trading, faster signals, more noise
- **Standard (14)**: Most common, good balance  
- **Long (18-21)**: Swing trading, smoother signals, less noise

### EMA Periods
- **Fast (8/21)**: Day trading setups, quick trend changes
- **Standard (13/48)**: Balanced momentum analysis
- **Slow (21/100)**: Position trading, stronger trend confirmation

### Data Period
- **1y**: Faster analysis, less trend context
- **18mo**: Good balance (default)
- **2y**: Best trend analysis, slower execution

## 🎪 Mode-Specific Features

### Hot Stocks Mode (`--hot-stocks N`)
- **Purpose**: Find trending momentum plays
- **Focus**: Technical momentum scoring  
- **Uses**: All momentum configuration flags
- **Output**: Ranked list of top N momentum stocks

### Discovery Mode (`--mode discovery`)
- **Purpose**: Find ML-predicted growth candidates
- **Focus**: Machine learning probability  
- **Uses**: `--ml-probability-threshold` flag
- **Output**: Stocks above probability threshold

### Evaluation Mode (`--mode eval`)
- **Purpose**: Analyze specific tickers
- **Focus**: Detailed analysis of your picks
- **Uses**: Same as discovery mode
- **Output**: Full analysis of provided tickers

## 🔄 Automatic Weight Normalization

The system automatically ensures momentum weights sum to 1.0:

```bash
# Input weights sum to 0.90
--trend-weight 0.40 --setup-weight 0.25 --volume-weight 0.25

# System normalizes to:
# trend: 0.444 (0.40/0.90), setup: 0.278 (0.25/0.90), volume: 0.278 (0.25/0.90) 
# Remaining: price_sma20: 0.167, ema: 0.056, rsi: 0.056
```

**Warning displayed**: `⚠️ Warning: Momentum weights sum to 0.900, normalizing to 1.0`
  