# Pro Screener - Advanced MCP-Style Technical Analysis Tool

## Overview
Pro Screener is a standalone advanced technical analysis tool that incorporates all the sophisticated methods from MCP-trader, enhanced with original scoring algorithms and discovery capabilities.

## Key Features

### 🎯 **MCP-Trader Coverage (100%)**
- **analyze-stock**: Comprehensive individual stock analysis
- **relative-strength**: Multi-timeframe RS analysis vs benchmarks  
- **volume-profile**: Point of Control (POC) + Value Area analysis
- **detect-patterns**: Advanced pattern recognition (16+ patterns)
- **position-size**: Risk-based position sizing with R:R targets
- **suggest-stops**: Multiple stop-loss methodologies

### 🧮 **Enhanced Technical Analysis**
- **ADRP** (Average Daily Range Percentage) - volatility assessment
- **Multi-SMA trend alignment** analysis
- **MACD crossover** detection
- **Volume vs average** comparison
- **Advanced pattern recognition**: H&S, triangles, wedges, double tops/bottoms

### 🔍 **Discovery Mode**
- Scan **full S&P 500** (503 stocks), NASDAQ-100, Russell 1000
- **Original technical scoring** algorithm (0-100%)
- **Ranking and filtering** by score thresholds
- **Real-time fetching** of index constituents

### 📊 **Technical Score Algorithm**
**3-Pillar Scoring System (0-100%):**
1. **Volume Profile (30%)**: POC positioning + Value Area analysis
2. **Relative Strength (40%)**: 63-day performance vs SPY
3. **Pattern Recognition (30%)**: Bullish/bearish pattern weighting

## Installation & Usage

### Individual Stock Analysis
```bash
# Detailed analysis of single stock
python pro_screener.py GOOGL --detailed

# Quick comparison of multiple stocks  
python pro_screener.py AAPL NVDA TSLA
```

### Discovery Mode
```bash
# Find top 10 stocks with 40%+ technical scores from S&P 500
python pro_screener.py --discovery --min-score 0.4 --top-n 10 --indices sp500

# Scan all major indices for high-scoring opportunities
python pro_screener.py --discovery --min-score 0.5 --top-n 20 --indices all

# Focus on specific index with lower threshold
python pro_screener.py --discovery --min-score 0.3 --top-n 15 --indices nasdaq
```

### Advanced Options
```bash
python pro_screener.py --help
```

## Sample Output

### Discovery Mode Results
```
🔥 High-Scoring Stock Opportunities (Ranked by Technical Score)
┌──────┬────────┬─────────┬────────────┬────────┬────────┬─────────┬───────┐
│ Rank │ Symbol │ Price   │ Tech Score │ RS 63d │ ADRP   │ Volume  │ Trend │
├──────┼────────┼─────────┼────────────┼────────┼────────┼─────────┼───────┤
│ 1    │ ADSK   │ $298.76 │ 110%       │ 61.1   │ 1.6%   │ 0.2x    │ 55/100│
│ 2    │ GWW    │ $1077.81│ 100%       │ 53.5   │ 1.4%   │ 0.2x    │ 40/100│
│ 3    │ HUBB   │ $393.08 │ 100%       │ 59.1   │ 2.0%   │ 0.2x    │ 55/100│
└──────┴────────┴─────────┴────────────┴────────┴────────┴─────────┴───────┘
```

### Individual Analysis
```
🎯 VOLUME PROFILE ANALYSIS
   Point of Control (POC): $163.53
   Value Area: $153.60 - $173.46
   ✅ Price above POC (bullish)

💪 RELATIVE STRENGTH vs SPY
   63d: 50.1 - Slight Outperformance ⭐
       Stock: +10.0% | SPY: +9.9% | Excess: +0.1%

🔍 PATTERN RECOGNITION
   Double Bottom: $161.7 (Bullish Reversal)
   Inverse Head and Shoulders: $148.2 (Bullish Reversal)
```

## Technical Score Interpretation
- **70%+ = Strong** opportunity
- **50%+ = Moderate** opportunity  
- **30%+ = Weak** opportunity
- **<30% = Very Weak** opportunity

## Dependencies
- `yfinance` - Stock data
- `pandas` - Data manipulation
- `pandas_ta` - Technical indicators
- `numpy` - Numerical calculations
- `rich` - Beautiful console output

## Architecture
- **`pro_screener.py`** - Main analysis engine
- **`helper.py`** - Index ticker fetching functions
- **Standalone** - No external dependencies on main AI screener

## Performance
- **503 S&P 500 stocks** analyzed in ~5 minutes
- **Real-time data** fetching and analysis
- **Parallel processing** of technical indicators
- **Memory efficient** with streaming analysis 