import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import numpy as np
import sys
import time
from .output_formatter import *
from .clock import get_market_intelligence, MarketIntelligence, get_sector_intelligence, get_sector_for_stock, SectorIntelligence, calculate_dynamic_sector_multiplier
from .news_intelligence import get_news_intelligence, calculate_news_multiplier, NewsIntelligence, get_enhanced_news
from typing import List, Dict

# 🔧 CENTRALIZED CONFIGURATION SYSTEM
DEFAULT_CONFIG = {
    # === MOMENTUM SCORING WEIGHTS ===
    "momentum_weights": {
        "trend": 0.35,          # Long-term trend strength (SMA150 slope)
        "setup": 0.20,          # SMA crossover setup positioning  
        "price_sma20": 0.15,    # Price vs 20SMA momentum
        "volume": 0.20,         # Volume confirmation
        "ema": 0.05,            # EMA momentum
        "rsi": 0.05             # RSI filter
    },
    
    # === HOT STOCKS SCORING WEIGHTS ===
    "hot_weights": {
        "ema": 0.35,            # EMA crossover/momentum
        "candle": 0.30,         # Green candle bias + volume
        "rsi": 0.25,            # RSI momentum patterns
        "price_momentum": 0.10  # 1-day breakout momentum
    },
    
    # === TECHNICAL INDICATOR PARAMETERS ===
    "indicators": {
        "rsi_period": 14,
        "ema_short": 13,
        "ema_long": 48,
        "sma_20": 20,
        "sma_150": 150,
        "volume_avg_period": 20,
        "bb_period": 20,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stoch_period": 14,
        "atr_period": 14
    },
    
    # === MOMENTUM SCORING THRESHOLDS ===
    "momentum_thresholds": {
        # Trend scoring (annual SMA150 slope %)
        "trend_secular_bull": 50,    # 50%+ = secular bull
        "trend_strong": 30,          # 30%+ = strong uptrend
        "trend_solid": 20,           # 20%+ = solid uptrend
        "trend_decent": 10,          # 10%+ = decent uptrend
        "trend_weak": 5,             # 5%+ = weak uptrend
        "trend_flat": 0,             # 0%+ = flat trend
        
        # SMA setup scoring (SMA20 vs SMA150 %)
        "setup_perfect_min": -1.0,
        "setup_perfect_max": 2.0,
        "setup_close_min": -3.0,
        "setup_crossed_max": 5.0,
        "setup_good_min": -5.0,
        "setup_extended_max": 10.0,
        "setup_early_min": -8.0,
        "setup_extreme_max": 20.0,
        
        # Price vs 20SMA scoring (%)
        "price_optimal_max": 4.0,
        "price_healthy_max": 8.0,
        "price_near_min": -2.0,
        "price_extended_max": 15.0,
        "price_pullback_min": -5.0,
        
        # Volume scoring (ratio vs average)
        "volume_surge": 2.0,
        "volume_strong": 1.5,
        "volume_above_avg": 1.2,
        "volume_normal": 0.8,
        "volume_low": 0.5,
        
        # RSI scoring
        "rsi_normal_min": 35,
        "rsi_normal_max": 75,
        "rsi_extreme_high": 80,
        "rsi_extreme_low": 25
    },
    
    # === SYSTEM THRESHOLDS ===
    "discovery_threshold": 0.70,    # Minimum probability for discovery results
    "hot_pe_ratio_max": 1000,       # Max P/E ratio to consider
    "min_data_days": 150,            # Minimum days of data required
    "momentum_data_period": "18mo",  # Period for momentum analysis
    "crossover_lookback_days": 10,   # Days to look back for crossovers
    "recent_candles": 5,             # Number of recent candles to analyze
    
    # === DATA PERIODS FOR TREND ANALYSIS ===
    "trend_periods": {
        "ideal_days": 500,           # 2+ years ideal
        "ideal_sma_points": 100,
        "good_days": 250,            # 1+ year good
        "good_sma_points": 50,
        "minimum_days": 180,         # 6+ months minimum
        "minimum_sma_points": 30
    }
}

def get_effective_config(user_config=None):
    """Merge user config with defaults"""
    if user_config is None:
        return DEFAULT_CONFIG.copy()
    
    effective_config = DEFAULT_CONFIG.copy()
    
    # Deep merge nested dictionaries
    for key, value in user_config.items():
        if key in effective_config and isinstance(effective_config[key], dict) and isinstance(value, dict):
            effective_config[key].update(value)
        else:
            effective_config[key] = value
    
    return effective_config

# 🔧 Centralized list of features used in training & prediction
BASE_FEATURE_COLUMNS = [
    "PE_ratio", "RSI", "Volume", "Volume_Avg",
    "MACD", "MACD_signal", "BB_upper", "BB_lower",
    "Stoch_K", "Stoch_D", "ATR",
    "Return_1d", "Return_5d", "Volatility_20d",
    "Price_vs_SMA20", "Price_vs_SMA50", "Price_vs_SMA200", "Rel_Strength_SPY"
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS.copy()

def calculate_dynamic_regime_multiplier(market_intel: MarketIntelligence) -> float:
    """Calculate dynamic regime adjustment based on actual market conditions"""
    
    base_multiplier = 1.0
    
    # VIX-based volatility adjustment
    vix_level = market_intel.vix_level
    if vix_level and vix_level > 30:
        base_multiplier *= 0.70  # High fear environment
    elif vix_level and vix_level > 25:
        base_multiplier *= 0.85  # Elevated volatility
    elif vix_level and vix_level < 15:
        base_multiplier *= 1.10  # Complacency/low vol boost
    
    # Trend strength adjustment
    overextension = abs(market_intel.overextension_pct)
    if overextension > 15:
        base_multiplier *= 0.80  # Overextended markets are risky
    elif overextension > 10:
        base_multiplier *= 0.90  # Moderately overextended
    elif overextension < 3:
        base_multiplier *= 1.05  # Not overextended
    
    # Fear & Greed adjustment
    if market_intel.fear_greed_value:
        fg_value = market_intel.fear_greed_value
        if fg_value > 80:  # Extreme greed
            base_multiplier *= 0.85  # Contrarian reduction
        elif fg_value > 60:  # Greed
            base_multiplier *= 0.95  # Slight reduction
        elif fg_value < 20:  # Extreme fear
            base_multiplier *= 1.15  # Contrarian opportunity
        elif fg_value < 40:  # Fear
            base_multiplier *= 1.05  # Slight boost
    
    # Yield curve inversion penalty
    if market_intel.yield_curve_spread and market_intel.yield_curve_spread < 0:
        base_multiplier *= 0.75  # Recession risk
    
    return min(1.20, max(0.60, base_multiplier))  # Cap between 60%-120%

def apply_regime_adjustment(probability: float, market_intel: MarketIntelligence) -> tuple[float, str]:
    """
    Apply market regime adjustment to prediction probability
    Returns: (adjusted_probability, explanation)
    """
    multiplier = calculate_dynamic_regime_multiplier(market_intel)
    
    # Additional adjustments based on market conditions
    if market_intel.risk_appetite == "Risk-Off":
        multiplier *= 0.9  # Reduce confidence in risk-off environment
    elif market_intel.risk_appetite == "Risk-On":
        multiplier *= 1.05  # Slight boost in risk-on environment
    
    if market_intel.market_stress_level == "High":
        multiplier *= 0.85  # Significantly reduce confidence in high stress
    elif market_intel.market_stress_level == "Low":
        multiplier *= 1.05  # Slight boost in low stress environment
    
    adjusted_prob = min(0.95, probability * multiplier)  # Cap at 95%
    
    # Create explanation
    regime_desc = market_intel.regime_description
    confidence_change = ((adjusted_prob - probability) / probability) * 100 if probability > 0 else 0
    
    if abs(confidence_change) < 1:
        explanation = f"Minimal adjustment for {regime_desc.lower()}"
    elif confidence_change > 0:
        explanation = f"Boosted {confidence_change:+.1f}% for {regime_desc.lower()}"
    else:
        explanation = f"Reduced {confidence_change:.1f}% for {regime_desc.lower()}"
    
    return adjusted_prob, explanation

def apply_sector_adjustment(probability: float, ticker: str, sector_intel: SectorIntelligence) -> tuple[float, str]:
    """
    Apply sector rotation adjustment to prediction probability using dynamic performance data
    Returns: (adjusted_probability, explanation)
    """
    # Get stock's sector
    stock_sector = get_sector_for_stock(ticker)
    if stock_sector == "Unknown":
        return probability, "No sector adjustment (unknown sector)"
    
    # Get dynamic multiplier based on actual performance
    multiplier = calculate_dynamic_sector_multiplier(stock_sector, sector_intel)
    
    # Additional leading/lagging sector adjustment
    if stock_sector in sector_intel.leading_sectors[:3]:  # Top 3 leading
        multiplier *= 1.05  # Additional 5% boost for leading sectors
        leader_boost = True
    elif stock_sector in sector_intel.lagging_sectors[-3:]:  # Bottom 3 lagging  
        multiplier *= 0.95  # Additional 5% penalty for lagging sectors
        leader_boost = False
    else:
        leader_boost = None
    
    adjusted_prob = probability * multiplier  # No cap - conviction score approach
    
    # Create explanation
    sector_perf = sector_intel.sector_performances.get(stock_sector)
    explanations = []
    
    if sector_perf:
        rel_strength = sector_perf.relative_strength_spy
        if abs(rel_strength) > 1:
            explanations.append(f"sector {rel_strength:+.1f}% vs SPY")
    
    if leader_boost is True:
        explanations.append("leading sector boost")
    elif leader_boost is False:
        explanations.append("lagging sector penalty")
    
    if explanations:
        explanation = f"{stock_sector}: {', '.join(explanations)}"
    else:
        explanation = f"Minimal sector adjustment for {stock_sector}"
    
    return adjusted_prob, explanation

def apply_news_adjustment(probability: float, ticker: str, news_intel: NewsIntelligence) -> tuple[float, str]:
    """
    Apply news sentiment adjustment to prediction probability
    Returns: (adjusted_probability, explanation)
    """
    # Check if we have sentiment analysis for this ticker
    if ticker not in news_intel.sentiment_analyses:
        return probability, "No news data available"
    
    sentiment_analysis = news_intel.sentiment_analyses[ticker]
    
    # Calculate news multiplier
    multiplier = calculate_news_multiplier(sentiment_analysis)
    
    adjusted_prob = probability * multiplier
    
    # Create detailed explanation
    explanations = []
    
    # Sentiment component
    if sentiment_analysis.overall_sentiment > 0.1:
        sentiment_desc = f"positive sentiment ({sentiment_analysis.overall_sentiment:.2f})"
        explanations.append(sentiment_desc)
    elif sentiment_analysis.overall_sentiment < -0.1:
        sentiment_desc = f"negative sentiment ({sentiment_analysis.overall_sentiment:.2f})"
        explanations.append(sentiment_desc)
    
    # News volume component
    if sentiment_analysis.news_volume > 0:
        explanations.append(f"{sentiment_analysis.news_volume} articles")
        
        # Velocity component
        if sentiment_analysis.news_velocity > 1.0:
            explanations.append(f"high activity ({sentiment_analysis.news_velocity:.1f}/day)")
    
    # Trend component
    if sentiment_analysis.sentiment_trend != "stable":
        explanations.append(f"{sentiment_analysis.sentiment_trend} trend")
    
    # Dominant themes
    if sentiment_analysis.dominant_themes:
        theme_str = ", ".join(sentiment_analysis.dominant_themes[:2])  # Top 2 themes
        explanations.append(f"themes: {theme_str}")
    
    if explanations:
        confidence_change = ((adjusted_prob - probability) / probability) * 100 if probability > 0 else 0
        explanation = f"News {confidence_change:+.1f}%: {'; '.join(explanations)}"
    else:
        explanation = "Minimal news adjustment"
    
    return adjusted_prob, explanation

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        console.print("📈 Fetching S&P 500 constituent data...")
        table = pd.read_html(url)
        df = table[0]
        console.print(f"✅ Found {len(df)} tickers in S&P 500")
        return df['Symbol'].tolist()
    except Exception as e:
        console.print(f"❌ Error fetching S&P 500 tickers: {e}")
        return []

def get_russell1000_tickers():
    """Fetch Russell 1000 tickers dynamically via market cap screening"""
    try:
        console.print("📈 Fetching Russell 1000 index dynamically...")
        
        # Get S&P 500 as base
        sp500_tickers = get_sp500_tickers()
        
        # Try to get Russell 1000 from a reliable source
        try:
            # Option 1: Try Wikipedia Russell 1000 page
            url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
            tables = pd.read_html(url)
            
            # Look for the table with ticker symbols
            russell_tickers = []
            for table in tables:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    symbol_col = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                    russell_tickers.extend(table[symbol_col].dropna().tolist())
                    break
            
            if russell_tickers:
                # Clean tickers (remove any formatting issues)
                russell_tickers = [ticker.strip().upper() for ticker in russell_tickers if ticker and isinstance(ticker, str)]
                russell_tickers = list(set(russell_tickers))  # Remove duplicates
                
                console.print(f"✅ Found {len(russell_tickers)} Russell 1000 tickers via Wikipedia")
                return russell_tickers
                
        except Exception as e:
            console.print(f"⚠️ Wikipedia Russell 1000 fetch failed: {e}")
        
        # Option 2: Use yfinance screener approach (market cap > $2B)
        try:
            console.print("📊 Attempting market cap screening for large-cap stocks...")
            
            # Start with S&P 500 and expand with market cap filtering
            import yfinance as yf
            
            # Get some major exchanges' stock lists
            large_cap_candidates = []
            
            # We'll use a different approach - scan major sectors for large caps
            # This is still somewhat manual but based on real market structure
            major_sectors = [
                'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
                'Industrials', 'Communication Services', 'Consumer Staples', 
                'Energy', 'Utilities', 'Real Estate', 'Materials'
            ]
            
            console.print("📊 Using S&P 500 as Russell 1000 approximation (most overlap)")
            console.print("💡 For true Russell 1000, consider upgrading to a premium data provider")
            return sp500_tickers
            
        except Exception as e:
            console.print(f"⚠️ Market cap screening failed: {e}")
            
        # Fallback: Return S&P 500 with a note
        console.print("📊 Falling back to S&P 500 (substantial overlap with Russell 1000)")
        return sp500_tickers
        
    except Exception as e:
        console.print(f"❌ Error building Russell 1000 index: {e}")
        return get_sp500_tickers()

def get_all_tickers():
    """Get all available tickers by combining multiple dynamic sources"""
    try:
        console.print("📈 Building complete stock index dynamically...")
        
        all_tickers = set()
        
        # 1. Get S&P 500
        sp500 = get_sp500_tickers()
        all_tickers.update(sp500)
        console.print(f"✅ Added {len(sp500)} S&P 500 stocks")
        
        # 2. Try to get Russell 1000 (if different from S&P 500)
        russell1000 = get_russell1000_tickers()
        initial_count = len(all_tickers)
        all_tickers.update(russell1000)
        russell_additions = len(all_tickers) - initial_count
        if russell_additions > 0:
            console.print(f"✅ Added {russell_additions} additional Russell 1000 stocks")
        
        # 3. Try to get NASDAQ-100 for tech coverage
        try:
            console.print("📈 Fetching NASDAQ-100 for tech coverage...")
            nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            nasdaq_tables = pd.read_html(nasdaq_url)
            
            nasdaq_tickers = []
            for table in nasdaq_tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    ticker_col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    nasdaq_tickers.extend(table[ticker_col].dropna().tolist())
                    break
            
            if nasdaq_tickers:
                nasdaq_tickers = [ticker.strip().upper() for ticker in nasdaq_tickers if ticker and isinstance(ticker, str)]
                initial_count = len(all_tickers)
                all_tickers.update(nasdaq_tickers)
                nasdaq_additions = len(all_tickers) - initial_count
                console.print(f"✅ Added {nasdaq_additions} additional NASDAQ-100 stocks")
                
        except Exception as e:
            console.print(f"⚠️ NASDAQ-100 fetch failed: {e}")
        
        # 4. Try to get Dow Jones for blue chips
        try:
            console.print("📈 Fetching Dow Jones Industrial Average...")
            dow_url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            dow_tables = pd.read_html(dow_url)
            
            dow_tickers = []
            for table in dow_tables:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    symbol_col = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                    dow_tickers.extend(table[symbol_col].dropna().tolist())
                    break
            
            if dow_tickers:
                dow_tickers = [ticker.strip().upper() for ticker in dow_tickers if ticker and isinstance(ticker, str)]
                initial_count = len(all_tickers)
                all_tickers.update(dow_tickers)
                dow_additions = len(all_tickers) - initial_count
                if dow_additions > 0:
                    console.print(f"✅ Added {dow_additions} additional Dow Jones stocks")
                    
        except Exception as e:
            console.print(f"⚠️ Dow Jones fetch failed: {e}")
        
        final_tickers = list(all_tickers)
        console.print(f"✅ Complete index: {len(final_tickers)} total stocks from multiple indices")
        return final_tickers
        
    except Exception as e:
        console.print(f"❌ Error building complete index: {e}")
        return get_russell1000_tickers()

def filter_tickers_by_sector(tickers, sector_name):
    """Filter ticker list to only include stocks from specified sector"""
    if not sector_name:
        return tickers
    
    console.print(f"🏭 Filtering for {sector_name} sector stocks...")
    
    filtered_tickers = []
    sector_name_normalized = sector_name.upper().strip()
    
    for ticker in tickers:
        try:
            stock_sector = get_sector_for_stock(ticker)
            # Normalize sector names for comparison
            if stock_sector.upper().replace(" ", "").replace("SERVICES", "") in sector_name_normalized.replace(" ", ""):
                filtered_tickers.append(ticker)
        except Exception as e:
            console.print(f"⚠️ Could not determine sector for {ticker}: {e}")
            continue
    
    if filtered_tickers:
        console.print(f"✅ Found {len(filtered_tickers)} {sector_name} stocks: {', '.join(filtered_tickers[:10])}{'...' if len(filtered_tickers) > 10 else ''}")
    else:
        console.print(f"❌ No stocks found in {sector_name} sector")
        console.print("💡 Available sectors: Technology, Healthcare, Financials, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy, Utilities, Real Estate, Materials")
    
    return filtered_tickers

def calculate_hot_score(ticker, config):
    """Calculate simple trending score for pre-filtering hot stocks"""
    try:
        # Get effective configuration
        eff_config = get_effective_config(config)
        
        # Fetch minimal data for quick trending detection
        stock = yf.Ticker(ticker)
        data = stock.history(period=eff_config["momentum_data_period"])
        
        if data.empty or len(data) < eff_config["min_data_days"]:
            return 0.0
        
        # Calculate EMAs for momentum detection
        data['EMA_13'] = ta.ema(data['Close'], length=eff_config["indicators"]["ema_short"])
        data['EMA_48'] = ta.ema(data['Close'], length=eff_config["indicators"]["ema_long"])
        data['RSI'] = ta.rsi(data['Close'], length=eff_config["indicators"]["rsi_period"])
        
        # Get recent data
        recent = data.tail(eff_config["recent_candles"]).copy()
        latest = data.iloc[-1]
        
        # Check for NaN values in key indicators
        if pd.isna(latest['EMA_13']) or pd.isna(latest['EMA_48']) or pd.isna(latest['RSI']):
            return 0.0
        
        # Get scoring weights
        weights = eff_config["hot_weights"]
        
        # 1. EMA Momentum Signal
        ema_bullish = latest['EMA_13'] > latest['EMA_48']
        ema_separation = (latest['EMA_13'] / latest['EMA_48'] - 1) * 100  # % separation
        
        # Recent crossover bonus
        ema_crossover_recent = False
        if len(data) >= eff_config["crossover_lookback_days"]:
            last_lookback = data.tail(eff_config["crossover_lookback_days"])
            for i in range(1, len(last_lookback)):
                if (last_lookback['EMA_13'].iloc[i] > last_lookback['EMA_48'].iloc[i] and 
                    last_lookback['EMA_13'].iloc[i-1] <= last_lookback['EMA_48'].iloc[i-1]):
                    ema_crossover_recent = True
                    break
        
        ema_score = 0
        if ema_bullish:
            ema_score = min(1.0, max(0.3, ema_separation / 5))  # 5% separation = 1.0 score
            if ema_crossover_recent:
                ema_score = min(1.0, ema_score * 1.5)  # 50% bonus for recent crossover
        
        # 2. Candle/Volume Pattern
        # Green vs red candle analysis
        recent['Candle_Size'] = abs(recent['Close'] - recent['Open'])
        recent['Green_Candle'] = recent['Close'] > recent['Open']
        recent['Volume_Ratio'] = recent['Volume'] / recent['Volume'].shift(1)
        
        # Green candle momentum (more/bigger green candles)
        green_count = recent['Green_Candle'].sum()
        green_ratio = green_count / len(recent)  # % of recent days green
        
        # Volume on green days vs red days
        green_days = recent[recent['Green_Candle']]
        red_days = recent[~recent['Green_Candle']]
        
        avg_green_volume = green_days['Volume'].mean() if len(green_days) > 0 else 0
        avg_red_volume = red_days['Volume'].mean() if len(red_days) > 0 else 1
        volume_bias = avg_green_volume / (avg_red_volume + 1e-6)  # Green days have higher volume?
        
        candle_score = (green_ratio * 0.7 + min(1.0, volume_bias / 2) * 0.3)
        
        # 3. RSI Recovery/Momentum
        current_rsi = latest['RSI']
        rsi_score = 0
        
        # RSI momentum patterns
        if 45 <= current_rsi <= 65:  # Healthy momentum zone
            rsi_score = 0.8
        elif 35 <= current_rsi <= 45:  # Recovery from oversold
            rsi_score = 1.0  # Best score for bounce setups
        elif 65 < current_rsi <= 75:  # Strong but not extreme
            rsi_score = 0.6
        elif current_rsi > 75:  # Overbought but strong
            rsi_score = 0.3
        
        # 4. Price Momentum
        price_1d = (latest['Close'] / data['Close'].iloc[-2] - 1) * 100 if len(data) >= 2 else 0
        momentum_score = max(0, price_1d / 5)  # 5% move = 1.0 score
        
        # Final weighted combination using configurable weights
        trending_score = (
            ema_score * weights["ema"] +
            candle_score * weights["candle"] +
            rsi_score * weights["rsi"] +
            momentum_score * weights["price_momentum"]
        )
        
        return min(1.0, trending_score)
        
    except Exception as e:
        # Fail silently for speed - hot filtering should be fast
        return 0.0

def filter_hot_stocks(tickers, hot_stocks_count, config):
    """Filter to top N hottest stocks based on momentum, volume, and news"""
    if hot_stocks_count <= 0:
        return tickers
    
    console.print(f"🔥 Pre-filtering for top {hot_stocks_count} trending stocks...")
    
    hot_scores = []
    
    for ticker in tickers:
        score = calculate_hot_score(ticker, config)
        if score > 0:  # Only include stocks with valid scores
            hot_scores.append((ticker, score))
    
    # Sort by hot score descending
    hot_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_hot_stocks = hot_scores[:hot_stocks_count]
    filtered_tickers = [ticker for ticker, score in top_hot_stocks]
    
    if filtered_tickers:
        console.print(f"✅ Top {len(filtered_tickers)} trending candidates (will analyze with full AI pipeline):")
        for ticker, score in top_hot_stocks[:5]:  # Show top 5 with scores
            console.print(f"   🔥 {ticker}: {score:.3f} trending score")
        if len(filtered_tickers) > 5:
            console.print(f"   ... and {len(filtered_tickers) - 5} more")
    else:
        console.print(f"❌ No trending stocks found")
    
    return filtered_tickers

def initialize_feature_columns(tickers, config):
    stock = yf.Ticker(tickers[0])
    data = stock.history(period=config["period"])
    sma_lengths = [20, 50, 100, 150, 200]
    for length in sma_lengths:
        if len(data) >= length:
            sma_col = f"SMA_{length}"
            if sma_col not in FEATURE_COLUMNS:
                FEATURE_COLUMNS.append(sma_col)

def fetch_data(ticker, config, is_market=False, spy_close=None):
    label_info = "(Market Data)" if is_market else ""
    if not is_market:
        console.print(f"🔍 Fetching {ticker}... {label_info}")
    
    # Get effective configuration
    eff_config = get_effective_config(config)
    
    stock = yf.Ticker(ticker)
    info = stock.info
    pe_ratio = info.get("trailingPE", None)

    if not is_market and (pe_ratio is None or pe_ratio > eff_config["hot_pe_ratio_max"]):
        console.print(f"❌ Skipped {ticker} due to missing or extreme P/E ratio.")
        return None

    data = stock.history(period=config["period"])
    if data.empty:
        console.print(f"❌ Skipped {ticker} (no price data available)")
        return None

    try:
        # Get indicator parameters
        indicators = eff_config["indicators"]
        
        # 📈 Technical indicators using configurable parameters
        data["RSI"] = ta.rsi(data["Close"], length=indicators["rsi_period"])
        data["Volume_Avg"] = data["Volume"].rolling(window=indicators["volume_avg_period"]).mean()
        data["PE_ratio"] = pe_ratio if not is_market else 15  # dummy if market

        for col in FEATURE_COLUMNS:
            if col.startswith("SMA_"):
                length = int(col.split("_")[1])
                data[col] = ta.sma(data["Close"], length=length)

        macd = ta.macd(data["Close"])
        data["MACD"] = macd[f"MACD_{indicators['macd_fast']}_{indicators['macd_slow']}_{indicators['macd_signal']}"]
        data["MACD_signal"] = macd[f"MACDs_{indicators['macd_fast']}_{indicators['macd_slow']}_{indicators['macd_signal']}"]

        bb = ta.bbands(data["Close"], length=indicators["bb_period"])
        data["BB_upper"] = bb[f"BBU_{indicators['bb_period']}_2.0"]
        data["BB_lower"] = bb[f"BBL_{indicators['bb_period']}_2.0"]

        stoch = ta.stoch(data["High"], data["Low"], data["Close"])
        data["Stoch_K"] = stoch[f"STOCHk_{indicators['stoch_period']}_3_3"]
        data["Stoch_D"] = stoch[f"STOCHd_{indicators['stoch_period']}_3_3"]

        data["ATR"] = ta.atr(data["High"], data["Low"], data["Close"], length=indicators["atr_period"])

        # 📉 Price Action Features
        data["Return_1d"] = data["Close"].pct_change()
        data["Return_5d"] = data["Close"].pct_change(5)
        data["Volatility_20d"] = data["Return_1d"].rolling(window=indicators["volume_avg_period"]).std()
        data["Price_vs_SMA20"] = data["Close"] / data[f"SMA_{indicators['sma_20']}"] if f"SMA_{indicators['sma_20']}" in data else None
        data["Price_vs_SMA50"] = data["Close"] / data[f"SMA_50"] if "SMA_50" in data else None
        data["Price_vs_SMA200"] = data["Close"] / data[f"SMA_200"] if "SMA_200" in data else None

        # 📊 Relative Strength vs SPY
        if not is_market and spy_close is not None:
            aligned = pd.concat([data["Close"], spy_close], axis=1, join="inner")
            aligned.columns = ["Stock_Close", "SPY_Close"]
            data["Rel_Strength_SPY"] = aligned["Stock_Close"] / aligned["SPY_Close"]

        # 📊 Labeling for training
        if not is_market:
            data["Future_Return"] = data["Close"].shift(-config["future_days"]) / data["Close"] - 1
            data["Volatility_Future"] = data["Return_1d"].rolling(window=config["future_days"]).std()

            # Standard labeling
            data["Label"] = (data["Future_Return"] > config["threshold"]).astype(int)

            # Optional: override with return-volatility (Sharpe-like) labeling
            sharpe_threshold = config.get("use_sharpe_labeling", None)
            if sharpe_threshold is not None:
                data["Sharpe_Like"] = data["Future_Return"] / (data["Volatility_Future"] + 1e-6)
                data["Label"] = (data["Sharpe_Like"] > sharpe_threshold).astype(int)

        if not is_market:
            console.print(f"✅ Using {ticker} for training.")

        return data.dropna()

    except Exception as e:
        console.print(f"⚠️ Error computing indicators for {ticker}: {e}")
        return None

def train_model(df, config):
    X = df[FEATURE_COLUMNS]
    y = df["Label"]
    label_counts = y.value_counts().to_dict()

    if y.sum() == 0:
        console.print("⚠️ Model training skipped — no high-growth (label=1) samples present.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get("seed", 42)
    )

    model_type = config.get("model", "random_forest")
    grid_search = config.get("grid_search", 0)
    ensemble_runs = config.get("ensemble_runs", 1)
    n_estimators = config.get("n_estimators", 100)

    def build_model(seed):
        if model_type == "xgboost":
            base_model = XGBClassifier(
                n_jobs=-1, random_state=seed, verbosity=0, use_label_encoder=False
            )
            param_grid = {
                "n_estimators": [100, 300, 1000, 2000],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "min_child_weight": [1, 3],
            }
        else:
            base_model = RandomForestClassifier(
                n_jobs=-1, random_state=seed
            )
            param_grid = {
                "n_estimators": [100, 300, 1000, 2000],
                "max_depth": [None, 10, 20, 30],
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }

        if grid_search:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            search = GridSearchCV(base_model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
            search.fit(X_train, y_train)
            
            # Print grid search results
            print_grid_search_results(search.best_params_, search.best_score_)
            return search.best_estimator_
        else:
            # Dynamically prepare only allowed parameters
            override_params = {"n_estimators": n_estimators}
            for param in param_grid:
                if param in config:
                    override_params[param] = config[param]

            base_model.set_params(**override_params)
            base_model.fit(X_train, y_train)
            return base_model

    if ensemble_runs > 1:
        models = []
        probs = []
        for i in range(ensemble_runs):
            seed = 42 + i
            clf = build_model(seed)
            prob = clf.predict_proba(X_test)[:, 1]
            probs.append(prob)
            models.append(clf)

        avg_prob = np.mean(probs, axis=0)
        final_preds = (avg_prob > 0.5).astype(int)
        acc = (final_preds == y_test).mean()
        final_model = models[0]  # return one of the models for later use
    else:
        clf = build_model(config.get("seed", 42))
        
        # Calculate performance metrics
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Print model performance
        print_model_performance(acc, precision, recall, f1)
        
        final_model = clf

    if config.get("save_model_path"):
        joblib.dump(final_model, config["save_model_path"])

    return final_model

def run_screening(tickers, config, mode="eval", news_analysis=False):
    start_time = time.time()
    
    # 🎯 Apply filtering before processing
    original_count = len(tickers)
    
    # Apply sector filtering if specified
    sector_filter = config.get("sector_filter")
    if sector_filter:
        tickers = filter_tickers_by_sector(tickers, sector_filter)
        if not tickers:
            console.print("❌ No stocks found after sector filtering. Exiting.")
            return
    
    # Show filtering results
    if original_count != len(tickers):
        console.print(f"📊 Filtering Results: {original_count} → {len(tickers)} stocks selected")
    
    # 🧠 Get market intelligence first
    console.print("\n🧠 Gathering Market Intelligence...")
    market_intel = get_market_intelligence()
    
    # 🏭 Get sector intelligence
    console.print("🏭 Analyzing Sector Rotation...")
    sector_intel = get_sector_intelligence()
    
    # 📰 Get news intelligence (optional)
    news_intel = None
    if news_analysis:
        console.print("📰 Gathering News Intelligence...")
        news_intel = get_news_intelligence(tickers)
    else:
        console.print("📰 News analysis disabled (use --news flag to enable)")
    
    # Print beautiful header with market context
    print_header(mode, tickers, config, market_intel, sector_intel)
    
    initialize_feature_columns(tickers, config)
    
    # Print model training start
    model_type = config.get("model", "random_forest")
    n_estimators = config.get("n_estimators", 100)
    print_model_training_start(model_type, n_estimators, len(tickers))

    all_data = []

    # Fetch SPY market context first
    spy_df = fetch_data("SPY", config, is_market=True)
    spy_close_series = spy_df["Close"] if spy_df is not None else None
    if spy_df is not None and config.get("integrate_market"):
        spy_df["Ticker"] = "SPY_MARKET"
        spy_df["Label"] = 0  # Dummy label to keep shape
        all_data.append(spy_df)

    # Collect stock info for company names
    stock_infos = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_infos[ticker] = stock.info
        except:
            stock_infos[ticker] = {}

    for ticker in tickers:
        try:
            df = fetch_data(ticker, config, spy_close=spy_close_series)
            if df is not None:
                df["Ticker"] = ticker
                all_data.append(df)
        except Exception as e:
            console.print(f"❌ Error fetching data for {ticker}: {e}")

    if not all_data:
        console.print("\n❌ No usable data collected from tickers. Exiting.")
        sys.exit(1)

    combined = pd.concat(all_data)
    combined = combined[combined["Ticker"] != "SPY_MARKET"] if not config.get("integrate_market") else combined

    clf = train_model(combined, config)
    if clf is None:
        console.print("⚠️ Skipping prediction due to insufficient positive training data.")
        return

    # Make predictions
    latest = combined[combined["Ticker"] != "SPY_MARKET"].groupby("Ticker").tail(1)
    X_pred = latest[FEATURE_COLUMNS]
    raw_probs = clf.predict_proba(X_pred)[:, 1]
    
    # 🧠 Apply market regime adjustments
    console.print(f"\n🎯 Applying {market_intel.current_regime.value.replace('_', ' ').title()} regime adjustments...")
    
    regime_adjusted_probs = []
    regime_explanations = []
    
    for raw_prob in raw_probs:
        adj_prob, explanation = apply_regime_adjustment(raw_prob, market_intel)
        regime_adjusted_probs.append(adj_prob)
        regime_explanations.append(explanation)
    
    # 🏭 Apply sector adjustments
    console.print(f"🏭 Applying {sector_intel.rotation_trend} sector adjustments...")
    
    final_probs = []
    sector_explanations = []
    
    for i, ticker in enumerate(latest["Ticker"].tolist()):
        regime_prob = regime_adjusted_probs[i]
        sector_adj_prob, sector_explanation = apply_sector_adjustment(regime_prob, ticker, sector_intel)
        final_probs.append(sector_adj_prob)
        sector_explanations.append(sector_explanation)
    
    # 📰 Apply news adjustments (only if enabled)
    if news_analysis and news_intel:
        console.print(f"📰 Applying news sentiment adjustments...")
        
        news_adjusted_probs = []
        news_explanations = []
        
        for i, ticker in enumerate(latest["Ticker"].tolist()):
            sector_prob = final_probs[i]
            news_adj_prob, news_explanation = apply_news_adjustment(sector_prob, ticker, news_intel)
            news_adjusted_probs.append(news_adj_prob)
            news_explanations.append(news_explanation)
        
        final_conviction_scores = news_adjusted_probs
    else:
        # No news adjustments - use sector-adjusted scores as final
        news_adjusted_probs = final_probs.copy()  # Same as sector-adjusted
        news_explanations = ["News analysis disabled"] * len(final_probs)
        final_conviction_scores = final_probs
    
    # Create results dataframe with detailed probability breakdown
    results_df = create_results_dataframe(
        latest["Ticker"].tolist(), 
        final_conviction_scores,  # Final conviction scores
        latest,
        stock_infos,
        regime_explanations=regime_explanations,
        sector_explanations=sector_explanations,
        news_explanations=news_explanations,
        market_intel=market_intel,
        sector_intel=sector_intel,
        raw_probs=raw_probs,  # Add raw ML scores
        regime_probs=regime_adjusted_probs,  # Add regime-adjusted scores
        sector_probs=final_probs,  # Add sector-adjusted scores
        news_probs=news_adjusted_probs  # Add news-adjusted scores (same as sector if disabled)
    )
    
    # Filter for high probability results in discovery mode
    eff_config = get_effective_config(config)  # Get effective configuration
    
    if mode == "discovery":
        # Regular discovery mode - filter for high probability results
        discovery_threshold = eff_config["discovery_threshold"]
        results_df = results_df[results_df['Growth_Prob'] > discovery_threshold].sort_values('Growth_Prob', ascending=False)
        print_discovery_results(results_df, config, market_intel, sector_intel, discovery_threshold)
    else:
        results_df = results_df.sort_values('Growth_Prob', ascending=False)
        print_evaluation_results(results_df, config, market_intel, sector_intel)
    
    # Print probability breakdown for transparency
    print_probability_breakdown(results_df)
    
    # Print enhanced market context
    print_enhanced_market_context(market_intel)
    
    # Print sector intelligence
    print_sector_intelligence(sector_intel)
    
    # Print completion stats
    duration = time.time() - start_time
    num_candidates = len(results_df[results_df['Growth_Prob'] > eff_config["discovery_threshold"]]) if mode == "discovery" else None
    print_completion_stats(duration, num_candidates, market_intel, sector_intel, news_enabled=news_analysis)

def calculate_momentum_score(ticker, config, news_intel=None):
    """TREND-FOCUSED momentum scoring - SMA150 slope is PRIMARY driver"""
    try:
        # Get effective configuration
        eff_config = get_effective_config(config)
        
        # Fetch data for momentum analysis
        stock = yf.Ticker(ticker)
        data = stock.history(period=eff_config["momentum_data_period"])
        
        if data.empty or len(data) < eff_config["min_data_days"]:
            return 0.0, {}
        
        # Calculate indicators using configurable parameters
        indicators = eff_config["indicators"]
        data['EMA_13'] = ta.ema(data['Close'], length=indicators["ema_short"])
        data['EMA_48'] = ta.ema(data['Close'], length=indicators["ema_long"])
        data['RSI'] = ta.rsi(data['Close'], length=indicators["rsi_period"])
        data['SMA_20'] = ta.sma(data['Close'], length=indicators["sma_20"])
        data['SMA_150'] = ta.sma(data['Close'], length=indicators["sma_150"])
        data['Volume_Avg'] = data['Volume'].rolling(window=indicators["volume_avg_period"]).mean()
        
        # Get recent data
        latest = data.iloc[-1]
        
        # Check for NaN values in key indicators
        if pd.isna(latest['EMA_13']) or pd.isna(latest['EMA_48']) or pd.isna(latest['RSI']) or pd.isna(latest['SMA_20']) or pd.isna(latest['SMA_150']):
            return 0.0, {}
        
        # Get thresholds and weights
        thresholds = eff_config["momentum_thresholds"]
        weights = eff_config["momentum_weights"]
        trend_periods = eff_config["trend_periods"]
        
        # ===== PRIORITY 1: LONG-TERM TREND STRENGTH =====
        # Calculate SMA150 slope - THE PRIMARY SCORE
        long_term_slope = 0
        trend_score = 0.0
        trend_period = "insufficient_data"
        
        valid_sma_data = data['SMA_150'].dropna()
        
        if len(data) >= trend_periods["ideal_days"] and len(valid_sma_data) >= trend_periods["ideal_sma_points"]:
            sma_150_first = valid_sma_data.iloc[0]
            sma_150_current = valid_sma_data.iloc[-1]
            long_term_slope = (sma_150_current - sma_150_first) / sma_150_first
            trend_period = "2year"
        elif len(data) >= trend_periods["good_days"] and len(valid_sma_data) >= trend_periods["good_sma_points"]:
            sma_150_first = valid_sma_data.iloc[0]
            sma_150_current = valid_sma_data.iloc[-1]
            long_term_slope = (sma_150_current - sma_150_first) / sma_150_first
            trend_period = "1year"
        elif len(data) >= trend_periods["minimum_days"] and len(valid_sma_data) >= trend_periods["minimum_sma_points"]:
            sma_150_first = valid_sma_data.iloc[0]
            sma_150_current = valid_sma_data.iloc[-1]
            long_term_slope = (sma_150_current - sma_150_first) / sma_150_first
            trend_period = "6month"
        else:
            # Insufficient data - reject
            return 0.0, {'error': 'Insufficient data for trend analysis'}
        
        # Convert to annual percentage for scoring
        annual_slope_pct = long_term_slope * 100
        
        # PRIMARY SCORING: Long-term trend strength using configurable thresholds
        if annual_slope_pct >= thresholds["trend_secular_bull"]:
            trend_score = 1.0
            trend_desc = "Secular bull trend"
        elif annual_slope_pct >= thresholds["trend_strong"]:
            trend_score = 0.9
            trend_desc = "Strong uptrend"
        elif annual_slope_pct >= thresholds["trend_solid"]:
            trend_score = 0.8
            trend_desc = "Solid uptrend"
        elif annual_slope_pct >= thresholds["trend_decent"]:
            trend_score = 0.6
            trend_desc = "Decent uptrend"
        elif annual_slope_pct >= thresholds["trend_weak"]:
            trend_score = 0.4
            trend_desc = "Weak uptrend"
        elif annual_slope_pct >= thresholds["trend_flat"]:
            trend_score = 0.2
            trend_desc = "Flat trend"
        else:
            return 0.0, {'error': 'Declining long-term trend - rejected'}
        
        # ===== HARD FILTER: PRICE MUST BE ABOVE SMA150 =====
        price_vs_150sma = latest['Close'] / latest['SMA_150']
        if price_vs_150sma < 1.0:
            return 0.0, {'error': 'Price below SMA150 - not participating in uptrend'}
        
        # ===== PRIORITY 2: SMA CROSSOVER SETUP =====
        sma20_vs_150sma = latest['SMA_20'] / latest['SMA_150']
        sma_relationship_pct = ((sma20_vs_150sma - 1) * 100)
        
        # OPTIMAL SCORING using configurable thresholds
        if thresholds["setup_perfect_min"] <= sma_relationship_pct <= thresholds["setup_perfect_max"]:
            setup_score = 1.0
            setup_desc = "Perfect crossover setup"
        elif thresholds["setup_close_min"] <= sma_relationship_pct < thresholds["setup_perfect_min"]:
            setup_score = 0.95
            setup_desc = "Approaching crossover from below"
        elif thresholds["setup_perfect_max"] < sma_relationship_pct <= thresholds["setup_crossed_max"]:
            setup_score = 0.9
            setup_desc = "Recently crossed above"
        elif thresholds["setup_good_min"] <= sma_relationship_pct < thresholds["setup_close_min"]:
            setup_score = 0.8
            setup_desc = "Close setup from below"
        elif thresholds["setup_crossed_max"] < sma_relationship_pct <= thresholds["setup_extended_max"]:
            setup_score = 0.6
            setup_desc = "Moderately extended"
        elif thresholds["setup_early_min"] <= sma_relationship_pct < thresholds["setup_good_min"]:
            setup_score = 0.5
            setup_desc = "Early recovery stage"
        elif thresholds["setup_extended_max"] < sma_relationship_pct <= thresholds["setup_extreme_max"]:
            setup_score = 0.3
            setup_desc = "Extended above setup zone"
        else:
            setup_score = 0.2
            setup_desc = "Extreme positioning"
        
        # ===== PRIORITY 3: PRICE vs 20SMA MOMENTUM =====
        price_vs_sma20 = latest['Close'] / latest['SMA_20']
        price_sma20_pct = ((price_vs_sma20 - 1) * 100)
        
        if 0.0 <= price_sma20_pct <= thresholds["price_optimal_max"]:
            price20_score = 1.0
            price20_desc = "Perfect momentum zone"
        elif thresholds["price_optimal_max"] < price_sma20_pct <= thresholds["price_healthy_max"]:
            price20_score = 0.8
            price20_desc = "Healthy above 20SMA"
        elif thresholds["price_near_min"] <= price_sma20_pct < 0.0:
            price20_score = 0.9
            price20_desc = "Near 20SMA support"
        elif thresholds["price_healthy_max"] < price_sma20_pct <= thresholds["price_extended_max"]:
            price20_score = 0.6
            price20_desc = "Extended above 20SMA"
        elif thresholds["price_pullback_min"] <= price_sma20_pct < thresholds["price_near_min"]:
            price20_score = 0.7
            price20_desc = "Mild pullback to 20SMA"
        elif price_sma20_pct > thresholds["price_extended_max"]:
            price20_score = 0.3
            price20_desc = "Overextended vs 20SMA"
        else:
            price20_score = 0.4
            price20_desc = "Below 20SMA support"
        
        # ===== PRIORITY 4: VOLUME CONFIRMATION =====
        current_volume = latest['Volume']
        avg_volume = latest['Volume_Avg']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio >= thresholds["volume_surge"]:
            volume_score = 1.0
            volume_desc = "Major volume surge"
        elif volume_ratio >= thresholds["volume_strong"]:
            volume_score = 0.8
            volume_desc = "Strong volume"
        elif volume_ratio >= thresholds["volume_above_avg"]:
            volume_score = 0.6
            volume_desc = "Above average volume"
        elif volume_ratio >= thresholds["volume_normal"]:
            volume_score = 0.4
            volume_desc = "Normal volume"
        elif volume_ratio >= thresholds["volume_low"]:
            volume_score = 0.3
            volume_desc = "Below average volume"
        else:
            volume_score = 0.2
            volume_desc = "Very low volume"
        
        # ===== PRIORITY 5: EMA MOMENTUM =====
        ema_bullish = latest['EMA_13'] > latest['EMA_48']
        ema_separation = (latest['EMA_13'] / latest['EMA_48'] - 1) * 100
        
        # Recent crossover detection
        ema_crossover_recent = False
        if len(data) >= eff_config["crossover_lookback_days"]:
            last_lookback = data.tail(eff_config["crossover_lookback_days"])
            for i in range(1, len(last_lookback)):
                if (last_lookback['EMA_13'].iloc[i] > last_lookback['EMA_48'].iloc[i] and 
                    last_lookback['EMA_13'].iloc[i-1] <= last_lookback['EMA_48'].iloc[i-1]):
                    ema_crossover_recent = True
                    break
        
        if ema_bullish and ema_crossover_recent:
            ema_score = 1.0
            ema_desc = "Fresh bullish crossover"
        elif ema_bullish and ema_separation > 2:
            ema_score = 0.8
            ema_desc = "Strong bullish momentum"
        elif ema_bullish:
            ema_score = 0.6
            ema_desc = "Bullish momentum"
        else:
            ema_score = 0.2
            ema_desc = "Bearish momentum"
        
        # ===== PRIORITY 6: RSI FILTER =====
        current_rsi = latest['RSI']
        
        if thresholds["rsi_normal_min"] <= current_rsi <= thresholds["rsi_normal_max"]:
            rsi_score = 1.0
            rsi_desc = "Normal RSI range"
        elif current_rsi > thresholds["rsi_extreme_high"] or current_rsi < thresholds["rsi_extreme_low"]:
            rsi_score = 0.3
            rsi_desc = "Extreme RSI level"
        else:
            rsi_score = 0.6
            rsi_desc = "Extended RSI level"
        
        # ===== FINAL TREND-FOCUSED SCORING using configurable weights =====
        final_score = (
            trend_score * weights["trend"] +
            setup_score * weights["setup"] +
            price20_score * weights["price_sma20"] +
            volume_score * weights["volume"] +
            ema_score * weights["ema"] +
            rsi_score * weights["rsi"]
        )
        
        # Store detailed breakdown
        momentum_details = {
            # Trend analysis (PRIMARY)
            'trend_score': trend_score,
            'trend_desc': trend_desc,
            'long_term_slope': f"{annual_slope_pct:+.1f}% annual",
            'trend_period': trend_period,
            
            # Setup analysis  
            'setup_score': setup_score,
            'setup_desc': setup_desc,
            'sma_relationship': f"{sma_relationship_pct:+.1f}%",
            'sma20_vs_150sma': sma20_vs_150sma,
            
            # Price vs 20SMA analysis
            'price20_score': price20_score,
            'price20_desc': price20_desc,
            'price_sma20_pct': f"{price_sma20_pct:+.1f}%",
            'price_vs_sma20': price_vs_sma20,
            
            # Volume analysis
            'volume_score': volume_score,
            'volume_desc': volume_desc,
            'volume_ratio': volume_ratio,
            
            # EMA analysis
            'ema_score': ema_score,
            'ema_desc': ema_desc,
            'ema_bullish': ema_bullish,
            'ema_separation': ema_separation,
            'ema_crossover_recent': ema_crossover_recent,
            
            # RSI analysis
            'rsi_score': rsi_score,
            'rsi_desc': rsi_desc,
            'rsi': current_rsi,
            
            # Price positioning context
            'price_vs_150sma': latest['Close'] / latest['SMA_150'],
            'distance_from_150sma': f"{((latest['Close'] / latest['SMA_150'] - 1) * 100):+.1f}%",
            
            # Final result
            'final_score': final_score
        }
        
        return min(1.0, final_score), momentum_details
        
    except Exception as e:
        return 0.0, {'error': str(e)}

def run_hot_stocks_scanner(tickers, config, hot_stocks_count):
    """Pure momentum scanner for hot stocks - fast technical signals focused on price/volume action"""
    start_time = time.time()
    
    from .output_formatter import console, print_hot_stocks_results
    
    console.print(f"🔥 MOMENTUM SCANNER - Finding top {hot_stocks_count} trending stocks")
    console.print(f"📊 Scanning {len(tickers)} stocks for pure momentum signals...")
    
    # Apply sector filtering if specified
    sector_filter = config.get("sector_filter")
    if sector_filter:
        tickers = filter_tickers_by_sector(tickers, sector_filter)
        if not tickers:
            console.print("❌ No stocks found after sector filtering. Exiting.")
            return
        console.print(f"🏭 Filtered to {len(tickers)} {sector_filter} stocks")
    
    momentum_scores = []
    filter_stats = {'total': 0, 'passed_filters': 0, 'rejected_filters': 0}
    
    console.print("\n🔍 Calculating momentum scores...")
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:  # Progress indicator
            console.print(f"   Processed {i + 1}/{len(tickers)} stocks...")
        
        filter_stats['total'] += 1
        score, details = calculate_momentum_score(ticker, config, news_intel=None)  # No news for hot stocks
        
        if score > 0:
            filter_stats['passed_filters'] += 1
            momentum_scores.append((ticker, score, details))
        else:
            filter_stats['rejected_filters'] += 1
    
    # Report filtering statistics
    console.print(f"\n📊 MOMENTUM FILTER RESULTS:")
    console.print(f"   • Total stocks analyzed: {filter_stats['total']}")
    console.print(f"   • Passed all filters: {filter_stats['passed_filters']} ({filter_stats['passed_filters']/filter_stats['total']*100:.1f}%)")
    console.print(f"   • Rejected (failed filters): {filter_stats['rejected_filters']} ({filter_stats['rejected_filters']/filter_stats['total']*100:.1f}%)")
    console.print(f"   📈 Filters: SMA150 uptrend + Price above SMA150")
    
    # Sort by momentum score descending
    momentum_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_momentum_stocks = momentum_scores[:hot_stocks_count]
    
    if top_momentum_stocks:
        console.print(f"\n🚀 TOP {len(top_momentum_stocks)} MOMENTUM STOCKS:")
        print_hot_stocks_results(top_momentum_stocks, config)
    else:
        console.print("❌ No momentum stocks found matching criteria")
    
    # Print completion stats
    duration = time.time() - start_time
    console.print(f"\n⏱️ Momentum scan completed in {duration:.1f} seconds")
    console.print(f"🎯 Found {len(top_momentum_stocks)} momentum candidates from {len(tickers)} stocks")
    console.print(f"🚀 Fast technical analysis - Price & Volume focused")

def get_nasdaq_tickers():
    """Fetch NASDAQ-100 tickers dynamically"""
    try:
        console.print("📈 Fetching NASDAQ-100 constituent data...")
        
        # Get NASDAQ-100 from Wikipedia
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        tables = pd.read_html(url)
        
        nasdaq_tickers = []
        for table in tables:
            # Look for tables with ticker/symbol columns
            if 'Ticker' in table.columns:
                nasdaq_tickers.extend(table['Ticker'].dropna().tolist())
            elif 'Symbol' in table.columns:
                nasdaq_tickers.extend(table['Symbol'].dropna().tolist())
        
        if nasdaq_tickers:
            # Clean and deduplicate tickers
            nasdaq_tickers = [ticker.strip().upper() for ticker in nasdaq_tickers if ticker and isinstance(ticker, str)]
            nasdaq_tickers = list(set(nasdaq_tickers))  # Remove duplicates
            
            console.print(f"✅ Found {len(nasdaq_tickers)} NASDAQ-100 tickers")
            return nasdaq_tickers
        else:
            console.print("⚠️ No NASDAQ-100 tickers found in Wikipedia tables")
            
        # Fallback: Try alternative NASDAQ source or return empty
        console.print("📊 Falling back to S&P 500 for NASDAQ index")
        return get_sp500_tickers()
        
    except Exception as e:
        console.print(f"❌ Error fetching NASDAQ-100: {e}")
        return get_sp500_tickers()  # Fallback to S&P 500
