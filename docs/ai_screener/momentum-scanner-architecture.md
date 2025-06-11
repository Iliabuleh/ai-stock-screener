# Momentum Scanner & Configuration Architecture

## Overview

The AI Stock Screener implements a sophisticated momentum detection system with centralized configuration management. The system supports two distinct scanning approaches:

1. **Hot-Stocks Mode**: Configurable momentum scanner for technical analysis
2. **Discovery/Eval Mode**: ML-powered growth prediction with limited configuration

## 🔧 Centralized Configuration System

### Configuration Architecture

The system uses a centralized `DEFAULT_CONFIG` dictionary with organized sections:

```python
DEFAULT_CONFIG = {
    "momentum_weights": {
        "trend": 0.35,      # Long-term trend (SMA150 slope)
        "setup": 0.20,      # SMA crossover timing
        "price_sma20": 0.15, # Price vs 20SMA positioning
        "volume": 0.20,     # Volume confirmation
        "ema": 0.05,        # EMA crossover signal
        "rsi": 0.05         # RSI momentum
    },
    "hot_stocks_weights": {
        "ema": 0.35,        # EMA crossover strength
        "candle": 0.30,     # Price action patterns
        "rsi": 0.25,        # RSI positioning
        "price_momentum": 0.10  # Recent price velocity
    },
    "indicators": {
        "rsi_period": 14,
        "ema_short": 13,
        "ema_long": 48,
        "sma_short": 20,
        "sma_trend": 150,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2
    },
    "momentum_thresholds": {
        "trend_strong": 0.02,
        "trend_weak": -0.005,
        "setup_bullish": 1.0,
        "volume_surge": 1.5,
        "volume_strong": 1.2,
        "volume_weak": 0.8,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "rsi_momentum_min": 45,
        "rsi_momentum_max": 65
    },
    "system": {
        "discovery_threshold": 0.70,
        "momentum_data_period": "18mo",
        "min_data_points": 150,
        "trend_filter_pct": 0.98
    }
}
```

### Configuration Management

- **get_effective_config()**: Deep merges user overrides with defaults
- **CLI Integration**: Advanced flags override specific configuration values
- **Weight Normalization**: Automatically ensures momentum weights sum to 1.0
- **Validation**: Warns users about weight adjustments and invalid configurations

## 🔥 Hot-Stocks Momentum Scanner

### Philosophy
The hot-stocks scanner focuses on **configurable technical momentum signals** to identify stocks with strong movement potential. All scoring weights and thresholds can be customized via CLI flags or configuration overrides.

### Architecture

#### Time Horizon & Data Requirements
- **Historical Period**: 18 months (configurable via `--momentum-data-period`)
- **Minimum Data Points**: 150 days
- **Trend Filter**: Must be above 98% of 150SMA (configurable)

### Current Momentum Scoring Algorithm

The momentum score uses **6 weighted components** with **configurable weights**:

```
Final Score = (
    Trend * 0.35 +           # Long-term trend strength
    Setup * 0.20 +           # SMA crossover timing
    Price_SMA20 * 0.15 +     # Price positioning vs 20SMA
    Volume * 0.20 +          # Volume confirmation
    EMA * 0.05 +             # EMA crossover signal
    RSI * 0.05               # RSI momentum
)
```

#### 1. Trend Component (35% Default Weight)
**Purpose**: Measure long-term trend strength using SMA150 slope

**Calculation**:
```python
slope = (current_sma150 - sma150_30days_ago) / sma150_30days_ago
if slope > trend_strong_threshold (0.02):
    score = 1.0
elif slope > trend_weak_threshold (-0.005):
    score = 0.6
else:
    score = 0.2
```

**CLI Configuration**: `--trend-weight 0.50` (higher = favor strong trends)

#### 2. Setup Component (20% Default Weight)
**Purpose**: Detect SMA crossover timing for entry signals

**Calculation**:
```python
if sma20 > sma50:  # Bullish crossover
    days_since_cross = calculate_crossover_days()
    if days_since_cross <= 10:
        score = 1.0  # Recent crossover
    else:
        score = 0.8  # Established uptrend
else:
    score = 0.2  # Bearish setup
```

**CLI Configuration**: `--setup-weight 0.30` (higher = precise entry timing)

#### 3. Price vs SMA20 Component (15% Default Weight)
**Purpose**: Assess short-term positioning relative to key support

**Calculation**:
```python
price_vs_sma20 = (current_price - sma20) / sma20
if 0 <= price_vs_sma20 <= 0.08:
    score = 1.0  # Healthy extension
elif -0.05 <= price_vs_sma20 < 0:
    score = 0.8  # Near support
elif 0.08 < price_vs_sma20 <= 0.15:
    score = 0.6  # Getting extended
else:
    score = 0.3  # Too extended or too weak
```

#### 4. Volume Component (20% Default Weight)
**Purpose**: Confirm momentum with participation

**Calculation**:
```python
volume_ratio = current_volume / average_volume_20d
if volume_ratio >= volume_surge_threshold (1.5):
    score = 1.0
elif volume_ratio >= volume_strong_threshold (1.2):
    score = 0.8
elif volume_ratio >= volume_weak_threshold (0.8):
    score = 0.6
else:
    score = 0.4
```

**CLI Configuration**: `--volume-weight 0.35` (higher = require volume breakouts)

#### 5. EMA Component (5% Default Weight)
**Purpose**: Short-term momentum signal

**Calculation**:
```python
ema_separation = (ema_short - ema_long) / ema_long
if ema_short > ema_long:
    score = min(1.0, ema_separation / 0.05)
else:
    score = 0.1
```

#### 6. RSI Component (5% Default Weight)
**Purpose**: Momentum confirmation in optimal range

**Calculation**:
```python
if rsi_momentum_min <= rsi <= rsi_momentum_max:  # 45-65 range
    score = 1.0
elif rsi_oversold <= rsi < rsi_momentum_min:     # 30-45 range
    score = 0.8
elif rsi_momentum_max < rsi <= rsi_overbought:   # 65-70 range
    score = 0.6
else:
    score = 0.3
```

**CLI Configuration**: `--rsi-period 9` (faster signals) or `--rsi-period 21` (smoother)

### Hot Stocks Scoring Algorithm (Alternative)

For `run_hot_stocks_scanner()`, a different algorithm is used:

```
Hot Score = (
    EMA * 0.35 +           # EMA crossover strength
    Candle * 0.30 +        # Price action patterns
    RSI * 0.25 +           # RSI positioning
    Price_Momentum * 0.10  # Recent velocity
)
```

This algorithm focuses more on short-term price action and momentum.

### Critical Trend Filter

**Hard Requirement**: All candidates must pass the trend filter:
```python
trend_filter_passed = current_price >= (sma150 * trend_filter_pct)
# Default: price >= (SMA150 * 0.98)
```

**Rationale**: Prevents momentum trading against major downtrends

## 🎯 Configuration Modes

### Hot-Stocks Mode Configuration
**Full Configuration Available**:
- All momentum weights configurable
- Technical indicator periods adjustable
- Threshold values customizable
- Data period selectable

### Discovery/Eval Mode Configuration
**Limited Configuration**:
- Only `--ml-probability-threshold` affects results display
- Technical indicator periods affect ML feature calculation
- Core ML algorithm and weights are fixed

## 🔧 CLI Configuration Interface

### Advanced Configuration Flags

| Flag | Default | Impact |
|------|---------|--------|
| `--trend-weight` | `0.35` | Long-term trend importance |
| `--setup-weight` | `0.20` | Entry timing precision |
| `--volume-weight` | `0.20` | Volume confirmation requirement |
| `--rsi-period` | `14` | RSI calculation sensitivity |
| `--ema-short` | `13` | Short EMA period |
| `--ema-long` | `48` | Long EMA period |
| `--momentum-data-period` | `18mo` | Historical data window |
| `--ml-probability-threshold` | `0.70` | Results display filter |

### Weight Normalization

The system automatically normalizes momentum weights:
```python
total_weight = sum(momentum_weights.values())
if abs(total_weight - 1.0) > 0.01:
    for key in momentum_weights:
        momentum_weights[key] /= total_weight
```

Users receive a warning: `⚠️ Warning: Momentum weights sum to 0.900, normalizing to 1.0`

## 📊 Technical Implementation

### Data Architecture
- **18-month historical window** (configurable)
- **150-day minimum requirement** for complete indicators
- **Real-time processing** with progress indicators
- **Graceful error handling** for missing data

### Performance Features
- **Parallel processing** where possible
- **Intelligent caching** of expensive calculations
- **Progress tracking** for user feedback
- **Memory optimization** with pandas operations

### Quality Assurance
- **Configuration validation** with user warnings
- **Debug functions** for individual stock analysis
- **Transparent scoring** that reveals component contributions
- **Comprehensive error handling** with graceful degradation

## 🚀 Usage Examples

### Basic Hot-Stocks Scanning
```bash
python -m ai_stock_screener.cli --hot-stocks 10
```

### Conservative Long-Term Configuration
```bash
python -m ai_stock_screener.cli --hot-stocks 10 \
  --trend-weight 0.50 \
  --setup-weight 0.15 \
  --volume-weight 0.15 \
  --momentum-data-period "2y" \
  --rsi-period 21
```

### Aggressive Momentum Trading
```bash
python -m ai_stock_screener.cli --hot-stocks 20 \
  --volume-weight 0.35 \
  --trend-weight 0.25 \
  --setup-weight 0.25 \
  --rsi-period 9 \
  --ema-short 8 \
  --ema-long 21
```

### Discovery Mode with Custom Threshold
```bash
python -m ai_stock_screener.cli --mode discovery \
  --ml-probability-threshold 0.80 \
  --rsi-period 21
```

## 🎪 Configuration Philosophy

### Design Principles
1. **Sensible Defaults**: Works well out-of-the-box
2. **Full Customization**: Advanced users can tune everything
3. **Automatic Normalization**: Prevents configuration errors
4. **Clear Impact**: Users understand what changes when they adjust parameters
5. **Mode Separation**: Different algorithms for different use cases

### Trading Style Adaptation
- **Day Trading**: Lower periods, higher volume weight
- **Swing Trading**: Balanced weights, setup timing focus
- **Position Trading**: Higher trend weight, longer periods
- **Breakout Trading**: Maximum volume weight, momentum focus

---

*This architecture provides professional-grade momentum detection with complete configurability while maintaining ease of use for beginners.* 