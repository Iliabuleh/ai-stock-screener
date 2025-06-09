# REGIME Calculation Documentation

## 🏛️ REGIME Calculation Components

The REGIME multiplier analyzes **overall market conditions** and adjusts your stock predictions accordingly. Here's what it includes:

### **1. VIX-Based Volatility Adjustment** 📊
```python
# Fear/Volatility levels
VIX > 30:   ×0.70  # High fear environment (-30%)
VIX > 25:   ×0.85  # Elevated volatility (-15%)
VIX < 15:   ×1.10  # Complacency/low volatility (+10%)
```

### **2. Market Overextension Analysis** 📈
```python
# How far market is stretched from trend
Overextension > 15%: ×0.80  # Very overextended (-20%)
Overextension > 10%: ×0.90  # Moderately overextended (-10%)
Overextension < 3%:  ×1.05  # Not overextended (+5%)
```

### **3. Fear & Greed Index** 🎭
```python
# CNN Fear & Greed (contrarian approach)
F&G > 80 (Extreme Greed): ×0.85  # Contrarian reduction (-15%)
F&G > 60 (Greed):        ×0.95  # Slight reduction (-5%)
F&G < 20 (Extreme Fear): ×1.15  # Contrarian opportunity (+15%)
F&G < 40 (Fear):         ×1.05  # Slight boost (+5%)
```

### **4. Yield Curve Analysis** 📉
```python
# Recession indicator
Yield Curve Inverted: ×0.75  # Recession risk (-25%)
```

### **5. Risk Appetite & Market Stress** 🌡️
```python
# Additional factors
Risk-Off Environment:  ×0.90  # Reduce confidence (-10%)
Risk-On Environment:   ×1.05  # Slight boost (+5%)
High Market Stress:    ×0.85  # Significant reduction (-15%)
Low Market Stress:     ×1.05  # Slight boost (+5%)
```

## 📊 Final REGIME Multiplier Range
- **Minimum**: 0.60 (60%) - Very bearish market conditions
- **Maximum**: 1.20 (120%) - Very bullish market conditions
- **Typical Range**: 0.85 - 1.15 (±15%)

## 🎯 What This Means
The REGIME component asks: **"Given current market conditions, should we be more or less confident in our stock predictions?"**

**Examples:**
- **Bear Market** (High VIX, Fear, Overextended): Reduce all predictions by 20-40%
- **Bull Market** (Low VIX, Greed, Not Extended): Boost all predictions by 5-20%
- **Uncertain Market** (Mixed signals): Minimal adjustment (±5%)

This ensures your AI adapts to changing market environments rather than giving the same confidence regardless of conditions! 🚀

## 📍 Code Location
The REGIME calculation is implemented in:
- `ai_stock_screener/ai_screener.py` - `calculate_dynamic_regime_multiplier()`
- `ai_stock_screener/ai_screener.py` - `apply_regime_adjustment()` 