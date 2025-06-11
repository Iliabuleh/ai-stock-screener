# 📈 AI Stock Screener Suite

A comprehensive collection of AI-powered stock analysis tools for technical analysis, fundamental screening, and options strategies.

## 🛠️ Tools Overview

### 🤖 AI Stock Screener
**Machine Learning-Powered Stock Discovery & Analysis**

Advanced ML-driven stock screening using Random Forest and XGBoost models with technical and fundamental indicators.

**Key Features:**
- **Discovery Mode**: Scan entire S&P 500 for growth signals
- **Evaluation Mode**: Analyze custom ticker lists
- **ML Models**: Random Forest & XGBoost with grid search optimization
- **Market Intelligence**: VIX, Fear & Greed, yield curve integration
- **News Analysis**: Sentiment analysis and news impact scoring
- **Regime Adjustment**: Dynamic scoring based on market conditions

**Usage:**
```bash
# Discover growth opportunities in S&P 500
poetry run screener --mode discovery

# Analyze specific stocks
poetry run screener --mode eval --tickers AAPL NVDA AMZN
```

📚 **[Full Documentation](ai_stock_screener/README.md)**

---

### 🎯 Pro Screener
**Advanced MCP-Style Technical Analysis Tool**

Professional-grade technical analysis incorporating all MCP-trader methodologies with enhanced scoring and discovery capabilities.

**Key Features:**
- **MCP-Trader Coverage**: All 6 tools (analyze-stock, relative-strength, volume-profile, detect-patterns, position-size, suggest-stops)
- **Volume Profile Analysis**: Point of Control (POC) + Value Area institutional zones
- **Advanced Patterns**: 16+ patterns including H&S, triangles, wedges, double tops/bottoms
- **Discovery Mode**: Scan 500+ stocks across S&P 500, NASDAQ, Russell 1000
- **Technical Scoring**: Original 3-pillar algorithm (Volume Profile + Relative Strength + Patterns)
- **Risk Management**: Multiple stop-loss strategies and position sizing

**Usage:**
```bash
# Discover high-scoring technical opportunities
poetry run pro-screener --discovery --min-score 0.4 --top-n 10

# Detailed technical analysis
poetry run pro-screener GOOGL --detailed

# Multi-index scanning
poetry run pro-screener --discovery --indices sp500 nasdaq
```

📚 **[Full Documentation](pro_screener/README.md)**

---

### 📊 Covered Call Analyzer
**Options Strategy Analysis & Income Generation**

Specialized tool for analyzing covered call strategies and dividend-focused income generation.

**Key Features:**
- **Covered Call Analysis**: Premium collection strategies
- **Dividend Analysis**: Yield optimization and payout schedules
- **Risk Assessment**: Downside protection and upside capture
- **Income Optimization**: Best risk-adjusted returns

**Usage:**
```bash
# Analyze covered call opportunities
poetry run covered-call-div --help
```

📚 **[Full Documentation](covered_call_analyzer/README.md)**

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd ai-stock-screener

# Install dependencies
poetry install
```

### Usage Examples

```bash
# 1. Find ML-predicted growth stocks
poetry run screener --mode discovery --threshold 0.05

# 2. Technical analysis discovery
poetry run pro-screener --discovery --min-score 0.6 --detailed

# 3. Analyze specific stocks with both tools
poetry run screener --mode eval --tickers AAPL NVDA
poetry run pro-screener AAPL NVDA --detailed

# 4. Options income strategies
poetry run covered-call-div --analysis-mode
```

## 🎯 Tool Comparison

| Feature | AI Screener | Pro Screener | Covered Call |
|---------|------------|--------------|--------------|
| **ML Models** | ✅ RF/XGBoost | ❌ | ❌ |
| **Technical Analysis** | ⚠️ Basic | ✅ Advanced | ⚠️ Basic |
| **Volume Profile** | ❌ | ✅ POC/Value Area | ❌ |
| **Pattern Recognition** | ❌ | ✅ 16+ Patterns | ❌ |
| **Discovery Mode** | ✅ S&P 500 | ✅ Multi-Index | ⚠️ Limited |
| **News Analysis** | ✅ Sentiment | ❌ | ❌ |
| **Options Analysis** | ❌ | ❌ | ✅ Covered Calls |
| **Risk Management** | ⚠️ Basic | ✅ Advanced | ✅ Income Focus |

## 📋 Dependencies

- **Python**: >=3.11
- **Core**: pandas, numpy, yfinance, scikit-learn
- **ML**: xgboost, joblib
- **Technical**: pandas-ta
- **UI**: rich, typer, matplotlib
- **Data**: lxml, textblob, fear-and-greed

## 🔧 Development

```bash
# Install in development mode
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .
```

## 📊 Performance

- **AI Screener**: Analyzes 500+ stocks in ~2 minutes
- **Pro Screener**: Technical analysis of 500+ stocks in ~5 minutes  
- **Covered Call**: Options analysis varies by scope

## 🎪 Use Cases

**Day Traders**: Pro Screener for technical setups and patterns
**Swing Traders**: AI Screener for ML-predicted momentum + Pro Screener for entry/exit
**Income Investors**: Covered Call Analyzer for premium collection strategies
**Institutional**: Combined analysis for comprehensive stock evaluation

---

*Each tool is designed to complement the others - use AI Screener for discovery, Pro Screener for technical validation, and Covered Call Analyzer for income strategies.* 