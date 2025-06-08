import yfinance as yf
import fear_and_greed as fg
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd

class MarketRegime(Enum):
    """Market regime classifications for model selection"""
    BULL_LOW_VOL = "bull_low_vol"      # Bull market + VIX < 15
    BULL_HIGH_VOL = "bull_high_vol"    # Bull market + VIX > 25  
    BEAR_LOW_VOL = "bear_low_vol"      # Bear market + VIX < 15
    BEAR_HIGH_VOL = "bear_high_vol"    # Bear market + VIX > 25
    SIDEWAYS_LOW_VOL = "sideways_low_vol"  # Sideways + VIX < 15
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"  # Sideways + VIX > 25
    BULL_NORMAL_VOL = "bull_normal_vol"    # Bull market + 15 <= VIX <= 25
    BEAR_NORMAL_VOL = "bear_normal_vol"    # Bear market + 15 <= VIX <= 25
    SIDEWAYS_NORMAL_VOL = "sideways_normal_vol"  # Sideways + 15 <= VIX <= 25

@dataclass
class MarketIntelligence:
    """Comprehensive market intelligence data structure"""
    # Basic indicators
    spy_trend: str
    vix_level: float
    vix_classification: str
    fear_greed_value: Optional[int]
    fear_greed_classification: str
    yield_curve_spread: Optional[float]
    yield_curve_classification: str
    
    # Advanced regime analysis
    current_regime: MarketRegime
    regime_confidence: float
    regime_description: str
    
    # Market context
    sma_values: Dict[str, float]
    overextension_pct: float
    crossovers: List[str]
    
    # Risk indicators
    risk_appetite: str  # Risk-on, Risk-off, Neutral
    market_stress_level: str  # Low, Medium, High
    
    def get_regime_context(self) -> str:
        """Get human-readable regime context for predictions"""
        return f"{self.regime_description} (Confidence: {self.regime_confidence:.1%})"

# 🏭 SECTOR ETF TRACKING
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV", 
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB"
}

@dataclass 
class SectorPerformance:
    """Individual sector performance metrics"""
    sector_name: str
    sector_etf: str
    current_price: float
    performance_1d: float
    performance_5d: float
    performance_1m: float
    performance_3m: float
    relative_strength_spy: float  # vs SPY over 1 month
    momentum_score: float  # Combined momentum indicator
    rotation_signal: str  # "Inflow", "Outflow", "Neutral"

@dataclass
class SectorIntelligence:
    """Comprehensive sector analysis and rotation intelligence"""
    sector_performances: Dict[str, SectorPerformance]
    leading_sectors: List[str]  # Top 3 performing sectors
    lagging_sectors: List[str]  # Bottom 3 performing sectors
    rotation_trend: str  # "Growth to Value", "Value to Growth", "Defensive Rotation", "Risk-On", "Neutral"
    rotation_strength: float  # 0-1 confidence in rotation signal
    sector_breadth: float  # % of sectors outperforming SPY
    cross_sector_correlation: float  # Average correlation between sectors (diversification indicator)

def get_spy_trend():
    # Load 18 months of SPY data for long-term SMA calculation
    df = yf.Ticker("SPY").history(period="18mo")

    if df.empty or "Close" not in df.columns:
        return {
            "trend": "❌ SPY data unavailable.",
            "trend_classification": "unknown",
            "sma_values": {},
            "crossovers": [],
            "overextension": "N/A",
            "overextension_pct": 0.0
        }

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df = df.dropna()

    if df.empty:
        return {
            "trend": "❌ Not enough data after SMA calculation.",
            "trend_classification": "unknown",
            "sma_values": {},
            "crossovers": [],
            "overextension": "N/A",
            "overextension_pct": 0.0
        }

    latest = df.iloc[-1]
    close_price = latest["Close"]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    sma200 = latest["SMA200"]

    # 🔹 Determine Trend Classification
    if sma20 > sma50 and sma50 > sma200:
        trend = "🟢 Strong Bullish (All SMAs aligned)"
        trend_classification = "bull"
    elif sma20 > sma50:
        trend = "🟢 Bullish (SMA20 > SMA50)"
        trend_classification = "bull"
    elif sma20 < sma50 and sma50 < sma200:
        trend = "🔴 Strong Bearish (All SMAs aligned)"
        trend_classification = "bear"
    elif sma20 < sma50:
        trend = "🔴 Bearish (SMA20 < SMA50)"
        trend_classification = "bear"
    else:
        trend = "🟡 Sideways (SMA20 ≈ SMA50)"
        trend_classification = "sideways"

    # 🔹 Crossovers (check last N days)
    crossovers = []
    window = 20  # how far back we check
    recent = df.iloc[-(window+1):]  # +1 for prev day access

    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]

        # SMA20/SMA50 crossover
        if prev["SMA20"] < prev["SMA50"] and curr["SMA20"] > curr["SMA50"]:
            crossovers.append(f"📈 SMA20 crossed above SMA50 ({window - i} days ago)")
        elif prev["SMA20"] > prev["SMA50"] and curr["SMA20"] < curr["SMA50"]:
            crossovers.append(f"📉 SMA20 crossed below SMA50 ({window - i} days ago)")

        # SMA50/SMA200 crossover
        if prev["SMA50"] < prev["SMA200"] and curr["SMA50"] > curr["SMA200"]:
            crossovers.append(f"🌟 Golden Cross: SMA50 above SMA200 ({window - i} days ago)")
        elif prev["SMA50"] > prev["SMA200"] and curr["SMA50"] < curr["SMA200"]:
            crossovers.append(f"💀 Death Cross: SMA50 below SMA200 ({window - i} days ago)")

        # SMA20/SMA200 crossover
        if prev["SMA20"] < prev["SMA200"] and curr["SMA20"] > curr["SMA200"]:
            crossovers.append(f"📈 SMA20 crossed above SMA200 ({window - i} days ago)")
        elif prev["SMA20"] > prev["SMA200"] and curr["SMA20"] < curr["SMA200"]:
            crossovers.append(f"📉 SMA20 crossed below SMA200 ({window - i} days ago)")

    # 🔹 Overextension from SMA200
    over_pct = ((close_price - sma200) / sma200) * 100
    if over_pct > 10:
        overext = f"🔴 Overbought (+{over_pct:.2f}% above SMA200)"
    elif 5 < over_pct <= 10:
        overext = f"🟡 Stretched (+{over_pct:.2f}%)"
    elif over_pct <= 5:
        overext = f"🟢 Normal (+{over_pct:.2f}%)"
    else:
        overext = f"⚠️ Below SMA200 ({over_pct:.2f}%)"

    # 🧾 Return full summary
    return {
        "trend": trend,
        "trend_classification": trend_classification,
        "sma_values": {
            "SMA20": round(sma20, 2),
            "SMA50": round(sma50, 2),
            "SMA200": round(sma200, 2),
            "Close": round(close_price, 2)
        },
        "crossovers": crossovers or ["No recent crossovers."],
        "overextension": overext,
        "overextension_pct": over_pct
    }


def get_vix_level():
    vix = yf.download("^VIX", period="1mo", interval="1d", progress=False, auto_adjust=True)
    if vix.empty or "Close" not in vix.columns:
        return {"level": None, "classification": "Unknown", "description": "VIX data unavailable"}

    try:
        latest_close = vix["Close"].dropna().iloc[-1].item()
    except Exception:
        return {"level": None, "classification": "Unknown", "description": "VIX data error"}

    if latest_close < 15:
        classification = "low"
        description = "🟢 Low (<15)"
    elif latest_close < 25:
        classification = "normal"
        description = "🟡 Medium (15–25)"
    else:
        classification = "high"
        description = "🔴 High (>25)"
    
    return {
        "level": latest_close,
        "classification": classification,
        "description": description
    }


def get_fear_and_greed_level():
    try:
        index = fg.get()
        value = index.value

        if value >= 75:
            classification = "extreme_greed"
            description = "🟢 Extreme Greed (75+)"
        elif value >= 55:
            classification = "greed"
            description = "🟢 Greed (55–74)"
        elif value >= 45:
            classification = "neutral"
            description = "🟡 Neutral (45–54)"
        elif value >= 25:
            classification = "fear"
            description = "🟠 Fear (25–44)"
        else:
            classification = "extreme_fear"
            description = "🔴 Extreme Fear (<25)"
        
        return {
            "value": value,
            "classification": classification,
            "description": description
        }
    except Exception as e:
        return {
            "value": None,
            "classification": "unknown",
            "description": f"⚠️ Fear & Greed unavailable ({str(e).splitlines()[0]})"
        }


def get_yield_curve_level():
    try:
        # 10Y yield
        y10 = yf.download("^TNX", period="1mo", interval="1d", progress=False, auto_adjust=True)
        # 2Y yield
        y2 = yf.download("^IRX", period="1mo", interval="1d", progress=False, auto_adjust=True)

        if y10.empty or y2.empty:
            raise ValueError("Missing yield data")

        y10_latest = y10["Close"].dropna().iloc[-1].item()
        y2_latest = y2["Close"].dropna().iloc[-1].item()

        spread = y10_latest - y2_latest

        if spread < 0:
            classification = "inverted"
            description = "🔴 Inverted (Recession signal)"
        elif spread < 1:
            classification = "flat"
            description = "🟡 Flat (Caution)"
        else:
            classification = "normal"
            description = "🟢 Normal (>1pt spread)"
        
        return {
            "spread": spread,
            "classification": classification,
            "description": description
        }

    except Exception as e:
        return {
            "spread": None,
            "classification": "unknown",
            "description": f"⚠️ Yield curve data unavailable ({str(e).splitlines()[0]})"
        }


def classify_market_regime(spy_trend_data: dict, vix_data: dict, fear_greed_data: dict) -> tuple[MarketRegime, float, str]:
    """
    Classify current market regime based on trend and volatility
    Returns: (regime, confidence_score, description)
    """
    trend_classification = spy_trend_data.get("trend_classification", "unknown")
    vix_classification = vix_data.get("classification", "unknown")
    
    # Base confidence calculation
    confidence = 0.8  # Start with 80% base confidence
    
    # Adjust confidence based on data quality
    if trend_classification == "unknown" or vix_classification == "unknown":
        confidence *= 0.5
    
    # Strong trend alignment boosts confidence
    if "Strong" in spy_trend_data.get("trend", ""):
        confidence = min(0.95, confidence + 0.1)
    
    # Recent crossovers may indicate regime change (lower confidence)
    recent_crossovers = [c for c in spy_trend_data.get("crossovers", []) if "days ago" in c and "1 days ago" in c]
    if recent_crossovers:
        confidence *= 0.7
    
    # Map trend + volatility to regime
    if trend_classification == "bull":
        if vix_classification == "low":
            regime = MarketRegime.BULL_LOW_VOL
            description = "Bullish with low volatility - favorable for growth stocks"
        elif vix_classification == "high":
            regime = MarketRegime.BULL_HIGH_VOL
            description = "Bullish but volatile - proceed with caution"
        else:
            regime = MarketRegime.BULL_NORMAL_VOL
            description = "Bullish with normal volatility - good risk/reward"
    
    elif trend_classification == "bear":
        if vix_classification == "low":
            regime = MarketRegime.BEAR_LOW_VOL
            description = "Bearish with low volatility - potential bottoming"
        elif vix_classification == "high":
            regime = MarketRegime.BEAR_HIGH_VOL
            description = "Bearish with high volatility - high risk environment"
        else:
            regime = MarketRegime.BEAR_NORMAL_VOL
            description = "Bearish with normal volatility - defensive positioning"
    
    else:  # sideways or unknown
        if vix_classification == "low":
            regime = MarketRegime.SIDEWAYS_LOW_VOL
            description = "Sideways with low volatility - range-bound trading"
        elif vix_classification == "high":
            regime = MarketRegime.SIDEWAYS_HIGH_VOL
            description = "Sideways with high volatility - uncertain direction"
        else:
            regime = MarketRegime.SIDEWAYS_NORMAL_VOL
            description = "Sideways with normal volatility - neutral conditions"
    
    return regime, confidence, description


def assess_risk_appetite(fear_greed_data: dict, yield_curve_data: dict, vix_data: dict) -> str:
    """Assess overall market risk appetite"""
    fear_greed_class = fear_greed_data.get("classification", "unknown")
    yield_curve_class = yield_curve_data.get("classification", "unknown")
    vix_class = vix_data.get("classification", "unknown")
    
    risk_on_indicators = 0
    risk_off_indicators = 0
    
    # Fear & Greed assessment
    if fear_greed_class in ["greed", "extreme_greed"]:
        risk_on_indicators += 1
    elif fear_greed_class in ["fear", "extreme_fear"]:
        risk_off_indicators += 1
    
    # Yield curve assessment
    if yield_curve_class == "normal":
        risk_on_indicators += 1
    elif yield_curve_class == "inverted":
        risk_off_indicators += 1
    
    # VIX assessment
    if vix_class == "low":
        risk_on_indicators += 1
    elif vix_class == "high":
        risk_off_indicators += 1
    
    if risk_on_indicators > risk_off_indicators:
        return "Risk-On"
    elif risk_off_indicators > risk_on_indicators:
        return "Risk-Off"
    else:
        return "Neutral"


def assess_market_stress(vix_data: dict, yield_curve_data: dict, spy_trend_data: dict) -> str:
    """Assess overall market stress level"""
    stress_score = 0
    
    # VIX stress
    vix_level = vix_data.get("level", 20)
    if vix_level and vix_level > 30:
        stress_score += 2
    elif vix_level and vix_level > 25:
        stress_score += 1
    
    # Yield curve stress
    if yield_curve_data.get("classification") == "inverted":
        stress_score += 2
    elif yield_curve_data.get("classification") == "flat":
        stress_score += 1
    
    # Trend stress (recent crossovers indicate uncertainty)
    crossovers = spy_trend_data.get("crossovers", [])
    recent_crossovers = [c for c in crossovers if "days ago" in c and any(str(i) + " days ago" in c for i in range(1, 6))]
    if recent_crossovers:
        stress_score += 1
    
    if stress_score >= 4:
        return "High"
    elif stress_score >= 2:
        return "Medium"
    else:
        return "Low"


def get_market_intelligence() -> MarketIntelligence:
    """
    Get comprehensive market intelligence for regime-aware predictions
    This is the main function to call for market context
    """
    # Gather all market data
    spy_data = get_spy_trend()
    vix_data = get_vix_level()
    fear_greed_data = get_fear_and_greed_level()
    yield_curve_data = get_yield_curve_level()
    
    # Classify market regime
    regime, regime_confidence, regime_description = classify_market_regime(spy_data, vix_data, fear_greed_data)
    
    # Assess market conditions
    risk_appetite = assess_risk_appetite(fear_greed_data, yield_curve_data, vix_data)
    market_stress = assess_market_stress(vix_data, yield_curve_data, spy_data)
    
    # Create comprehensive intelligence object
    intelligence = MarketIntelligence(
        # Basic indicators
        spy_trend=spy_data.get("trend", "Unknown"),
        vix_level=vix_data.get("level", 0.0),
        vix_classification=vix_data.get("description", "Unknown"),
        fear_greed_value=fear_greed_data.get("value"),
        fear_greed_classification=fear_greed_data.get("description", "Unknown"),
        yield_curve_spread=yield_curve_data.get("spread"),
        yield_curve_classification=yield_curve_data.get("description", "Unknown"),
        
        # Advanced regime analysis
        current_regime=regime,
        regime_confidence=regime_confidence,
        regime_description=regime_description,
        
        # Market context
        sma_values=spy_data.get("sma_values", {}),
        overextension_pct=spy_data.get("overextension_pct", 0.0),
        crossovers=spy_data.get("crossovers", []),
        
        # Risk indicators
        risk_appetite=risk_appetite,
        market_stress_level=market_stress
    )
    
    return intelligence


def market_clock():
    """Enhanced market clock with regime detection"""
    intelligence = get_market_intelligence()
    
    print("\n🕒 Enhanced Market Intelligence")
    print("=" * 50)
    
    # Basic Market Data
    print("\n📊 Market Overview:")
    print(f"📈 SPY Trend:           {intelligence.spy_trend}")
    print(f"⚡ VIX Volatility:      {intelligence.vix_classification}")
    print(f"😬 Fear & Greed:        {intelligence.fear_greed_classification}")
    print(f"📉 Yield Curve:         {intelligence.yield_curve_classification}")
    
    # Market Regime Analysis
    print("\n🎯 Market Regime Analysis:")
    print(f"🏛️  Current Regime:      {intelligence.current_regime.value.replace('_', ' ').title()}")
    print(f"🎯 Confidence:          {intelligence.regime_confidence:.1%}")
    print(f"📝 Description:         {intelligence.regime_description}")
    
    # Risk Assessment
    print("\n⚠️ Risk Assessment:")
    print(f"🎲 Risk Appetite:       {intelligence.risk_appetite}")
    print(f"📊 Market Stress:       {intelligence.market_stress_level}")
    
    # Technical Details
    print("\n📈 Technical Analysis:")
    print("SMA Values:", intelligence.sma_values)
    print(f"📐 Overextension:       SPY {intelligence.overextension_pct:+.2f}% vs SMA200")
    print("🔀 Recent Events:")
    for crossover in intelligence.crossovers:
        print(f"   - {crossover}")
    
    print("\n" + "=" * 50)
    print(f"🧠 Regime Context: {intelligence.get_regime_context()}")
    print()
    
    return intelligence

def get_sector_for_stock(ticker: str) -> str:
    """Get sector classification for a stock ticker using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        
        # Normalize sector names to match our ETF mapping
        if sector == 'Technology':
            return 'Technology'
        elif sector == 'Healthcare':
            return 'Healthcare'
        elif sector == 'Financial Services':
            return 'Financials'
        elif sector == 'Consumer Cyclical':
            return 'Consumer Discretionary'
        elif sector == 'Communication Services':
            return 'Communication Services'
        elif sector == 'Industrials':
            return 'Industrials'
        elif sector == 'Consumer Defensive':
            return 'Consumer Staples'
        elif sector == 'Energy':
            return 'Energy'
        elif sector == 'Utilities':
            return 'Utilities'
        elif sector == 'Real Estate':
            return 'Real Estate'
        elif sector == 'Basic Materials':
            return 'Materials'
        else:
            return 'Unknown'
            
    except Exception as e:
        print(f"⚠️ Could not fetch sector for {ticker}: {e}")
        return "Unknown"

def calculate_dynamic_sector_multiplier(sector_name: str, sector_intel: 'SectorIntelligence') -> float:
    """Calculate dynamic sector adjustment based on actual performance"""
    
    # If we don't have sector performance data, no adjustment
    sector_perf = sector_intel.sector_performances.get(sector_name)
    if not sector_perf:
        return 1.0
    
    # Base multiplier on actual relative strength vs SPY
    relative_strength = sector_perf.relative_strength_spy
    
    # Dynamic adjustment based on performance
    if relative_strength > 3:
        return 1.08  # Strong outperformance: +8%
    elif relative_strength > 1:
        return 1.04  # Moderate outperformance: +4%
    elif relative_strength > -1:
        return 1.0   # Neutral performance: no adjustment
    elif relative_strength > -3:
        return 0.96  # Moderate underperformance: -4%
    else:
        return 0.92  # Strong underperformance: -8%

def fetch_sector_etf_data(sector_etf: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Fetch sector ETF price data"""
    try:
        etf = yf.Ticker(sector_etf)
        data = etf.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"⚠️ Error fetching {sector_etf}: {e}")
        return None

def calculate_sector_performance(sector_name: str, sector_etf: str, spy_data: pd.DataFrame) -> Optional[SectorPerformance]:
    """Calculate comprehensive sector performance metrics"""
    
    # Fetch sector ETF data
    sector_data = fetch_sector_etf_data(sector_etf)
    if sector_data is None or len(sector_data) < 2:  # Need at least 2 data points
        return None
    
    try:
        current_price = sector_data['Close'].iloc[-1]
        
        # Performance calculations with safe indexing
        perf_1d = ((sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-2]) - 1) * 100 if len(sector_data) >= 2 else 0
        perf_5d = ((sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-6]) - 1) * 100 if len(sector_data) >= 6 else 0
        perf_1m = ((sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-21]) - 1) * 100 if len(sector_data) >= 21 else 0
        perf_3m = ((sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-63]) - 1) * 100 if len(sector_data) >= 63 else 0
        
        # Relative strength vs SPY (1 month) with proper null handling
        if spy_data is not None and len(spy_data) >= 21 and len(sector_data) >= 21:
            spy_return_1m = ((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-21]) - 1) * 100
            relative_strength = perf_1m - spy_return_1m
        else:
            relative_strength = 0
            
        # Momentum Score (weighted average of timeframes)
        momentum_score = (perf_1d * 0.1 + perf_5d * 0.2 + perf_1m * 0.4 + perf_3m * 0.3)
        
        # Rotation Signal based on recent performance
        if perf_5d > 2 and relative_strength > 1:
            rotation_signal = "Inflow"
        elif perf_5d < -2 and relative_strength < -1:
            rotation_signal = "Outflow"  
        else:
            rotation_signal = "Neutral"
            
        return SectorPerformance(
            sector_name=sector_name,
            sector_etf=sector_etf,
            current_price=current_price,
            performance_1d=perf_1d,
            performance_5d=perf_5d,
            performance_1m=perf_1m,
            performance_3m=perf_3m,
            relative_strength_spy=relative_strength,
            momentum_score=momentum_score,
            rotation_signal=rotation_signal
        )
        
    except Exception as e:
        print(f"⚠️ Error calculating performance for {sector_name}: {e}")
        return None

def analyze_sector_rotation(sector_performances: Dict[str, SectorPerformance]) -> tuple[str, float]:
    """Analyze overall sector rotation patterns"""
    
    if not sector_performances:
        return "Neutral", 0.0
    
    # Get valid performances
    valid_sectors = [s for s in sector_performances.values() if s is not None]
    if len(valid_sectors) < 5:
        return "Neutral", 0.0
    
    # Classify sector types for rotation analysis
    growth_sectors = ["Technology", "Communication Services", "Consumer Discretionary"]
    value_sectors = ["Financials", "Energy", "Materials", "Industrials"] 
    defensive_sectors = ["Utilities", "Consumer Staples", "Healthcare", "Real Estate"]
    
    # Calculate average performance by category with division by zero protection
    growth_sector_list = [s for s in valid_sectors if s.sector_name in growth_sectors]
    value_sector_list = [s for s in valid_sectors if s.sector_name in value_sectors]
    defensive_sector_list = [s for s in valid_sectors if s.sector_name in defensive_sectors]
    
    growth_perf = sum([s.performance_1m for s in growth_sector_list]) / len(growth_sector_list) if growth_sector_list else 0
    value_perf = sum([s.performance_1m for s in value_sector_list]) / len(value_sector_list) if value_sector_list else 0
    defensive_perf = sum([s.performance_1m for s in defensive_sector_list]) / len(defensive_sector_list) if defensive_sector_list else 0
    
    # Determine rotation trend
    perf_diff_growth_value = growth_perf - value_perf
    perf_diff_growth_defensive = growth_perf - defensive_perf
    perf_diff_value_defensive = value_perf - defensive_perf
    
    # Rotation logic
    if perf_diff_growth_value > 3:
        rotation_trend = "Growth Rotation"
        strength = min(1.0, abs(perf_diff_growth_value) / 10)
    elif perf_diff_growth_value < -3:
        rotation_trend = "Value Rotation"  
        strength = min(1.0, abs(perf_diff_growth_value) / 10)
    elif defensive_perf > max(growth_perf, value_perf) + 2:
        rotation_trend = "Defensive Rotation"
        strength = min(1.0, abs(perf_diff_growth_defensive) / 8)
    elif growth_perf > defensive_perf + 2 and value_perf > defensive_perf + 2:
        rotation_trend = "Risk-On"
        strength = min(1.0, (perf_diff_growth_defensive + perf_diff_value_defensive) / 16)
    else:
        rotation_trend = "Neutral"
        strength = 0.0
        
    return rotation_trend, strength

def get_sector_intelligence() -> SectorIntelligence:
    """
    Get comprehensive sector intelligence for sector-aware predictions
    This is the main function to call for sector context
    """
    
    # Get SPY data for relative strength calculations with better error handling
    spy_data = None
    try:
        spy = yf.Ticker("SPY")
        spy_data = spy.history(period="3mo")
        if spy_data.empty:
            print("⚠️ SPY data is empty, relative strength calculations may be limited")
            spy_data = None
    except Exception as e:
        print(f"⚠️ Failed to fetch SPY data: {e}, relative strength calculations will be limited")
        spy_data = None
    
    # Calculate performance for each sector
    sector_performances = {}
    
    for sector_name, sector_etf in SECTOR_ETFS.items():
        perf = calculate_sector_performance(sector_name, sector_etf, spy_data)
        if perf is not None:
            sector_performances[sector_name] = perf
    
    # Find leading and lagging sectors
    valid_sectors = [s for s in sector_performances.values() if s is not None]
    if valid_sectors:
        sorted_by_momentum = sorted(valid_sectors, key=lambda x: x.momentum_score, reverse=True)
        leading_sectors = [s.sector_name for s in sorted_by_momentum[:3]]
        lagging_sectors = [s.sector_name for s in sorted_by_momentum[-3:]]
    else:
        leading_sectors = []
        lagging_sectors = []
    
    # Analyze rotation trends
    rotation_trend, rotation_strength = analyze_sector_rotation(sector_performances)
    
    # Calculate sector breadth (% outperforming SPY)
    outperforming_count = sum(1 for s in valid_sectors if s.relative_strength_spy > 0)
    sector_breadth = outperforming_count / len(valid_sectors) if valid_sectors else 0.5
    
    # Simple correlation proxy (lower is better for diversification)
    # In a real implementation, would calculate actual correlations
    cross_sector_correlation = 0.6 if sector_breadth > 0.7 else 0.4 if sector_breadth < 0.3 else 0.5
    
    return SectorIntelligence(
        sector_performances=sector_performances,
        leading_sectors=leading_sectors,
        lagging_sectors=lagging_sectors, 
        rotation_trend=rotation_trend,
        rotation_strength=rotation_strength,
        sector_breadth=sector_breadth,
        cross_sector_correlation=cross_sector_correlation
    )
