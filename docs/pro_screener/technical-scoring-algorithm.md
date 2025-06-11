# Pro Screener Technical Scoring Algorithm

## Overview
The Pro Screener uses a sophisticated 3-pillar scoring system to evaluate stocks on a scale of 0-100%, combining institutional volume analysis, relative performance metrics, and advanced pattern recognition.

## Algorithm Architecture

### 🎯 Core Formula
```
Technical Score = (Volume Profile Score × 30%) + (Relative Strength Score × 40%) + (Pattern Score × 30%)
```

**Weight Distribution:**
- **Volume Profile**: 30% - Institutional positioning analysis
- **Relative Strength**: 40% - Performance vs benchmark (highest weight)
- **Pattern Recognition**: 30% - Technical setup identification

---

## 📊 Pillar 1: Volume Profile Analysis (30% Weight)

### Point of Control (POC) & Value Area
**Data Source:** 6-month price/volume history divided into 10 price bins

**Value Area Calculation:**
- Sort price bins by volume (highest to lowest)
- Accumulate bins until 70% of total volume is captured
- Value Area = Price range spanning these high-volume bins

**Scoring Logic:**
```python
score = 0.0

# Component 1: Value Area Position (0.2 points max)
if value_area_low <= current_price <= value_area_high:
    score += 0.2  # In institutional comfort zone

# Component 2: POC Position (0.1 points max)  
if current_price > point_of_control:
    score += 0.1  # Above most-traded price (bullish)

# Final: score / 0.3 = percentage of volume profile component
```

**Interpretation:**
- **In Value Area**: Institutional interest/support zone
- **Above POC**: Bullish positioning with momentum
- **Outside Value Area**: Extended price, higher risk

---

## 🚀 Pillar 2: Relative Strength Analysis (40% Weight)

### RS Score Calculation
**Formula:** `RS = 50 + (Stock Return % - SPY Return %)`

**Process:**
1. Calculate 63-day returns for stock and SPY
2. Compute excess return (outperformance/underperformance)
3. Add excess to 50 baseline for 1-100 scale
4. Cap between 1-99 to avoid extremes

**Scoring Tiers:**
```python
if rs_score > 70:      # 20%+ outperformance
    score += 0.4       # Maximum points
elif rs_score > 60:    # 10%+ outperformance  
    score += 0.3       # Good points
elif rs_score > 50:    # Any outperformance
    score += 0.2       # Minimal points
# rs_score <= 50 = 0 points (underperformance)
```

**Why 63-Day Period:**
- Captures quarterly performance cycle
- Balances recent momentum with trend stability
- Commonly used in institutional analysis

**Why 40% Weight:**
- Momentum is primary driver of stock performance
- Relative strength predicts continuation
- Higher weight than volume or patterns

---

## 🔍 Pillar 3: Pattern Recognition (30% Weight)

### Bullish Pattern Detection
**Patterns Adding Points:**
- **Resistance Breakout** (+0.2 high confidence, +0.1 medium)
- **Near Resistance** (+0.2 high confidence, +0.1 medium)
- **Inverse Head & Shoulders** (+0.2 high confidence, +0.1 medium)
- **Ascending Triangle** (+0.1 medium confidence)
- **Falling Wedge** (+0.1 medium confidence)
- **Double Bottom** (+0.1 medium confidence)

### Bearish Pattern Penalties
**Patterns Reducing Score:**
- **Head & Shoulders** (-0.1 high confidence, -0.05 medium)
- **Descending Triangle** (-0.1 high confidence, -0.05 medium)
- **Rising Wedge** (-0.1 high confidence, -0.05 medium)
- **Double Top** (-0.05 medium confidence)

### Pattern Detection Methodology

**Double Tops/Bottoms:**
```python
# Requirements:
- Similar price levels (within 3%)
- 10-60 day separation
- Significant peak/trough between (5%+ difference)
- Detected in recent 60-day window
```

**Head & Shoulders:**
```python
# Requirements:
- 3 consecutive peaks/troughs
- Center peak higher than shoulders (or lower for inverse)
- Shoulders roughly equal (within 5%)
- Head significantly different (3%+ from shoulders)
```

**Triangles:**
```python
# Ascending: flat resistance + rising support
# Descending: flat support + falling resistance  
# Symmetrical: converging resistance + support
# Detected via trend line slope analysis
```

---

## 🎯 Score Interpretation

### Score Ranges
- **70-100%**: Strong technical setup
- **50-69%**: Moderate opportunity
- **30-49%**: Weak setup
- **0-29%**: Very weak/avoid

### Discovery Mode Filtering
**Default Threshold: 50%**
- Filters for "moderate or better" opportunities
- Adjustable via `--min-score` parameter
- Higher thresholds = more selective screening

---

## 🔧 Algorithm Advantages

### 1. Multi-Dimensional Analysis
- **Not just price action** - includes volume and institutional behavior
- **Balanced weighting** prevents any single factor from dominating
- **Complementary signals** strengthen overall conviction

### 2. Institutional Insight
- **Volume Profile** reveals where institutions accumulate/distribute
- **Value Area** identifies support/resistance zones with volume backing
- **POC positioning** shows bullish/bearish institutional bias

### 3. Momentum Integration
- **Relative Strength** captures outperformance trends
- **63-day timeframe** balances recency with stability
- **Benchmark comparison** provides market-relative context

### 4. Pattern Confluence
- **Multiple pattern types** increase setup identification
- **Confidence weighting** emphasizes higher-probability patterns
- **Bearish penalties** reduce scores for negative technical setups

---

## 📈 Practical Examples

### High-Scoring Stock (ADSK - 110%)
```
Volume Profile: 0.3/0.3 (100%)
- In Value Area: +0.2
- Above POC: +0.1

Relative Strength: 0.3/0.4 (75%)  
- RS 63d: 61.0 (11% outperformance) → 0.3 points

Patterns: 0.4/0.3 (133%, capped at 100%)
- Ascending Triangle: +0.1
- Inverse H&S: +0.2  
- Double Bottom: +0.1

Total: 1.0/1.0 = 100% (displayed as 110% due to pattern bonus)
```

### Low-Scoring Stock (AMZN - 30%)
```
Volume Profile: 0.1/0.3 (33%)
- Outside Value Area: +0.0
- Above POC: +0.1

Relative Strength: 0.2/0.4 (50%)
- RS 63d: 51.9 (minimal outperformance) → 0.2 points

Patterns: 0.0/0.3 (0%)
- Multiple Double Tops: -0.3 (penalties)
- Limited bullish patterns: +0.0

Total: 0.3/1.0 = 30%
```

---

## 🛠️ Implementation Notes

### Data Requirements
- **6 months OHLCV data** for comprehensive analysis
- **SPY benchmark data** for relative strength calculation
- **Minimum 60 days** for pattern recognition accuracy

### Performance Considerations
- **Single-pass analysis** per stock for efficiency
- **Cached calculations** where possible
- **Error handling** for data quality issues

### Extensibility
- **Modular design** allows component weight adjustment
- **Pattern library** easily expandable
- **Additional timeframes** can be integrated

---

## 📚 References

### Theoretical Foundation
- **Volume Profile Analysis**: Market Profile theory (J. Peter Steidlmayer)
- **Relative Strength**: IBD methodology (William O'Neil)
- **Pattern Recognition**: Technical Analysis of Financial Markets (John Murphy)

### MCP-Trader Integration
- **Base data methods** from MCP-trader project
- **Enhanced with scoring** for actionable insights
- **Discovery capabilities** beyond raw data provision

---

*This algorithm represents an evolution of MCP-trader capabilities, adding intelligent synthesis and actionable ranking to raw technical analysis data.* 