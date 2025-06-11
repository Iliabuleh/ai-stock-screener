#!/usr/bin/env python3
"""
Pro Screener - Advanced MCP-Style Technical Analysis Tool
Incorporates all advanced technical analysis methods from MCP-trader that were missing from our implementation.

Key Features Added:
- ADRP (Average Daily Range Percentage) - volatility assessment
- Multi-SMA trend alignment analysis
- MACD crossover detection
- Advanced pattern recognition (double tops/bottoms, head & shoulders, triangles, wedges)
- Volume vs average comparison
- Multiple stop-loss methodologies
- Detailed relative strength classifications
- Volume Profile Analysis (POC + Value Area)
- Discovery mode with dynamic symbol fetching
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import dynamic ticker functions from local helper
from .helper import get_sp500_tickers, get_russell1000_tickers, get_nasdaq_tickers, get_all_tickers

console = Console()

class VolumeProfileAnalysis:
    """Advanced volume analysis for identifying key price levels (from enhanced_technical.py)"""
    
    @staticmethod
    def analyze_volume_profile(df: pd.DataFrame, num_bins: int = 10) -> Dict[str, Any]:
        """
        Create volume profile analysis by price level
        Identifies Point of Control (POC) and Value Area
        """
        try:
            if len(df) < 20:
                return {"error": "Not enough data for volume profile analysis"}
            
            # Use OHLCV columns consistently
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            volume_col = 'Volume' if 'Volume' in df.columns else 'volume'
            
            price_min = df[low_col].min()
            price_max = df[high_col].max()
            bin_width = (price_max - price_min) / num_bins
            
            profile = {
                "price_min": price_min,
                "price_max": price_max,
                "bin_width": bin_width,
                "bins": []
            }
            
            # Calculate volume by price bin
            for i in range(num_bins):
                bin_low = price_min + i * bin_width
                bin_high = bin_low + bin_width
                bin_mid = (bin_low + bin_high) / 2
                
                # Filter data in this price range
                mask = (df[low_col] <= bin_high) & (df[high_col] >= bin_low)
                volume_in_bin = df.loc[mask, volume_col].sum()
                
                # Calculate percentage of total volume
                total_volume = df[volume_col].sum()
                volume_percent = (volume_in_bin / total_volume * 100) if total_volume > 0 else 0
                
                profile["bins"].append({
                    "price_low": round(bin_low, 2),
                    "price_high": round(bin_high, 2),
                    "price_mid": round(bin_mid, 2),
                    "volume": int(volume_in_bin),
                    "volume_percent": round(volume_percent, 2)
                })
            
            # Find Point of Control (POC) - price with highest volume
            poc_bin = max(profile["bins"], key=lambda x: x["volume"])
            profile["point_of_control"] = round(poc_bin["price_mid"], 2)
            
            # Find Value Area (70% of volume)
            sorted_bins = sorted(profile["bins"], key=lambda x: x["volume"], reverse=True)
            cumulative_volume = 0
            value_area_bins = []
            
            for bin_data in sorted_bins:
                value_area_bins.append(bin_data)
                cumulative_volume += bin_data["volume_percent"]
                if cumulative_volume >= 70:
                    break
            
            if value_area_bins:
                profile["value_area_low"] = round(min(b["price_low"] for b in value_area_bins), 2)
                profile["value_area_high"] = round(max(b["price_high"] for b in value_area_bins), 2)
            
            return profile
            
        except Exception as e:
            return {"error": f"Volume profile analysis failed: {str(e)}"}

class ComprehensiveTechnicalAnalysis:
    """
    Complete technical analysis incorporating all MCP-trader methods
    """
    
    def __init__(self):
        self.console = Console()
    
    def add_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators including missing MCP-trader ones"""
        try:
            # === BASIC TREND INDICATORS ===
            df["sma_20"] = ta.sma(df["Close"], length=20)
            df["sma_50"] = ta.sma(df["Close"], length=50)
            df["sma_200"] = ta.sma(df["Close"], length=200)
            
            # === MCP-TRADER MISSING: ADRP (Average Daily Range Percentage) ===
            # This is a KEY volatility measure they use
            daily_range = df["High"].sub(df["Low"])
            adr = daily_range.rolling(window=20).mean()
            df["adrp"] = adr.div(df["Close"]).mul(100)  # Volatility as % of price
            
            # === MCP-TRADER MISSING: Volume vs Average Analysis ===
            df["avg_20d_vol"] = df["Volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["Volume"] / df["avg_20d_vol"]  # Current vs average
            
            # === MOMENTUM INDICATORS ===
            df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
            df["rsi"] = ta.rsi(df["Close"], length=14)
            
            # === MCP-TRADER MISSING: MACD Analysis ===
            macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
                # Add MACD crossover detection
                df["macd_bullish"] = df["MACD_12_26_9"] > df["MACDs_12_26_9"]
            
            return df
            
        except Exception as e:
            console.print(f"❌ Error calculating indicators: {e}")
            return df
    
    def analyze_trend_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        MCP-TRADER MISSING: Comprehensive trend status analysis
        Checks multiple SMA alignments and crossovers
        """
        if df.empty:
            return {"error": "Empty dataframe"}
        
        latest = df.iloc[-1]
        
        # Multi-SMA trend alignment (missing from our implementation)
        # Add null checks to prevent NoneType comparison errors
        current_price = latest["Close"]
        sma_20 = latest["sma_20"] if pd.notnull(latest["sma_20"]) else current_price
        sma_50 = latest["sma_50"] if pd.notnull(latest["sma_50"]) else current_price
        sma_200 = latest["sma_200"] if pd.notnull(latest["sma_200"]) else current_price
        rsi = latest["rsi"] if pd.notnull(latest["rsi"]) else 50
        adrp = latest["adrp"] if pd.notnull(latest["adrp"]) else 2.0
        volume_ratio = latest["volume_ratio"] if pd.notnull(latest["volume_ratio"]) else 1.0
        
        trend_status = {
            "above_20sma": current_price > sma_20,
            "above_50sma": current_price > sma_50, 
            "above_200sma": current_price > sma_200,
            "20_50_bullish": sma_20 > sma_50,  # Golden cross component
            "50_200_bullish": sma_50 > sma_200,  # Golden cross 
            "rsi": rsi,
            "adrp": adrp,  # NEW: Volatility measure
            "volume_ratio": volume_ratio,  # NEW: Volume analysis
        }
        
        # Add MACD analysis if available
        if "macd_bullish" in df.columns and pd.notnull(latest.get("macd_bullish")):
            trend_status["macd_bullish"] = latest["macd_bullish"]
        
        # Calculate trend strength score (0-100)
        trend_score = 0
        if trend_status["above_200sma"]: trend_score += 25
        if trend_status["above_50sma"]: trend_score += 20
        if trend_status["above_20sma"]: trend_score += 15
        if trend_status["20_50_bullish"]: trend_score += 20
        if trend_status["50_200_bullish"]: trend_score += 20
        
        trend_status["trend_strength"] = trend_score
        
        # Trend classification
        if trend_score >= 90:
            trend_status["trend_classification"] = "Very Strong Uptrend"
        elif trend_score >= 70:
            trend_status["trend_classification"] = "Strong Uptrend"
        elif trend_score >= 50:
            trend_status["trend_classification"] = "Moderate Uptrend"
        elif trend_score >= 30:
            trend_status["trend_classification"] = "Weak Uptrend"
        else:
            trend_status["trend_classification"] = "No Clear Uptrend"
        
        return trend_status
    
    def calculate_relative_strength_detailed(
        self, 
        symbol: str, 
        benchmark: str = "SPY",
        lookback_periods: List[int] = [21, 63, 126, 252]
    ) -> Dict[str, Any]:
        """
        MCP-TRADER STYLE: Detailed relative strength with classifications
        """
        try:
            # Get data
            max_period = max(lookback_periods) + 10
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max_period + 50)
            
            stock = yf.Ticker(symbol)
            benchmark_ticker = yf.Ticker(benchmark)
            
            stock_df = stock.history(start=start_date, end=end_date)
            benchmark_df = benchmark_ticker.history(start=start_date, end=end_date)
            
            if stock_df.empty or benchmark_df.empty:
                return {"error": "Failed to fetch data"}
            
            rs_analysis = {}
            
            for period in lookback_periods:
                if len(stock_df) <= period or len(benchmark_df) <= period:
                    continue
                
                # Calculate returns
                stock_return = (stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-period] - 1) * 100
                benchmark_return = (benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-period] - 1) * 100
                
                # Relative performance
                relative_performance = stock_return - benchmark_return
                
                # Convert to 1-100 RS score
                rs_score = min(max(50 + relative_performance, 1), 99)
                
                # MCP-TRADER STYLE: Detailed classifications
                if rs_score >= 80:
                    classification = "Strong Outperformance ⭐⭐⭐"
                elif rs_score >= 65:
                    classification = "Moderate Outperformance ⭐⭐"
                elif rs_score >= 50:
                    classification = "Slight Outperformance ⭐"
                elif rs_score >= 35:
                    classification = "Slight Underperformance ⚠️"
                elif rs_score >= 20:
                    classification = "Moderate Underperformance ⚠️⚠️"
                else:
                    classification = "Strong Underperformance ⚠️⚠️⚠️"
                
                rs_analysis[f"RS_{period}d"] = {
                    "score": round(rs_score, 2),
                    "classification": classification,
                    "stock_return": round(stock_return, 2),
                    "benchmark_return": round(benchmark_return, 2),
                    "excess_return": round(relative_performance, 2)
                }
            
            return rs_analysis
            
        except Exception as e:
            return {"error": f"Relative strength calculation failed: {str(e)}"}
    
    def detect_advanced_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        MCP-TRADER MISSING: Advanced pattern recognition
        Detects double tops, double bottoms, and sophisticated patterns
        """
        try:
            if len(df) < 60:
                return {"patterns": [], "message": "Not enough data for pattern detection"}
            
            patterns = []
            recent_df = df.tail(60).copy()
            
            # Find local minima and maxima (5-period rolling)
            recent_df["is_min"] = (
                recent_df["Low"].rolling(window=5, center=True).min() == recent_df["Low"]
            )
            recent_df["is_max"] = (
                recent_df["High"].rolling(window=5, center=True).max() == recent_df["High"]
            )
            
            minima = recent_df[recent_df["is_min"]].copy()
            maxima = recent_df[recent_df["is_max"]].copy()
            
            # === DOUBLE BOTTOM DETECTION ===
            if len(minima) >= 2:
                for i in range(len(minima) - 1):
                    for j in range(i + 1, len(minima)):
                        price1 = minima.iloc[i]["Low"]
                        price2 = minima.iloc[j]["Low"]
                        date1 = minima.iloc[i].name
                        date2 = minima.iloc[j].name
                        
                        # Check if similar price levels (within 3%)
                        if abs(price1 - price2) / price1 < 0.03:
                            # Check time separation (10-60 days)
                            days_apart = (date2 - date1).days
                            if 10 <= days_apart <= 60:
                                # Check for peak in between (5%+ higher)
                                mask = (recent_df.index > date1) & (recent_df.index < date2)
                                if mask.any():
                                    max_between = recent_df.loc[mask, "High"].max()
                                    if max_between > price1 * 1.05:
                                        patterns.append({
                                            "type": "Double Bottom",
                                            "start_date": date1.strftime("%Y-%m-%d"),
                                            "end_date": date2.strftime("%Y-%m-%d"),
                                            "price_level": round((price1 + price2) / 2, 2),
                                            "confidence": "Medium",
                                            "signal": "Bullish Reversal"
                                        })
            
            # === DOUBLE TOP DETECTION ===
            if len(maxima) >= 2:
                for i in range(len(maxima) - 1):
                    for j in range(i + 1, len(maxima)):
                        price1 = maxima.iloc[i]["High"]
                        price2 = maxima.iloc[j]["High"]
                        date1 = maxima.iloc[i].name
                        date2 = maxima.iloc[j].name
                        
                        if abs(price1 - price2) / price1 < 0.03:
                            days_apart = (date2 - date1).days
                            if 10 <= days_apart <= 60:
                                mask = (recent_df.index > date1) & (recent_df.index < date2)
                                if mask.any():
                                    min_between = recent_df.loc[mask, "Low"].min()
                                    if min_between < price1 * 0.95:
                                        patterns.append({
                                            "type": "Double Top",
                                            "start_date": date1.strftime("%Y-%m-%d"),
                                            "end_date": date2.strftime("%Y-%m-%d"),
                                            "price_level": round((price1 + price2) / 2, 2),
                                            "confidence": "Medium",
                                            "signal": "Bearish Reversal"
                                        })
            
            # === BREAKOUT PATTERNS ===
            current_close = df["Close"].iloc[-1]
            recent_high_20 = df["High"].iloc[-20:].max()
            recent_low_20 = df["Low"].iloc[-20:].min()
            
            # Resistance breakout
            if current_close > recent_high_20 * 0.999:
                patterns.append({
                    "type": "Resistance Breakout",
                    "price_level": round(recent_high_20, 2),
                    "confidence": "Medium",
                    "signal": "Bullish Continuation"
                })
            
            # Support breakdown
            if current_close < recent_low_20 * 1.001:
                patterns.append({
                    "type": "Support Breakdown",
                    "price_level": round(recent_low_20, 2),
                    "confidence": "Medium",
                    "signal": "Bearish Continuation"
                })
            
            # === ENHANCED: Near Support/Resistance Detection (from enhanced_technical.py) ===
            resistance_distance = (recent_high_20 - current_close) / current_close
            support_distance = (current_close - recent_low_20) / current_close
            
            # Near resistance (1-3% below resistance)
            if 0.01 < resistance_distance < 0.03:
                patterns.append({
                    "type": "Near Resistance",
                    "price_level": round(recent_high_20, 2),
                    "confidence": "High",
                    "signal": "Watch for breakout"
                })
            
            # Near support (1-3% above support)
            if 0.01 < support_distance < 0.03:
                patterns.append({
                    "type": "Near Support",
                    "price_level": round(recent_low_20, 2), 
                    "confidence": "High",
                    "signal": "Watch for bounce"
                })
            
            # === NEW: HEAD & SHOULDERS PATTERN DETECTION ===
            if len(maxima) >= 3:
                # Look for Head & Shoulders pattern in last 3 peaks
                for i in range(len(maxima) - 2):
                    left_shoulder = maxima.iloc[i]["High"]
                    head = maxima.iloc[i + 1]["High"] 
                    right_shoulder = maxima.iloc[i + 2]["High"]
                    
                    # Head should be higher than both shoulders
                    if head > left_shoulder and head > right_shoulder:
                        # Shoulders should be roughly equal (within 5%)
                        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                        if shoulder_diff < 0.05:
                            # Head should be significantly higher (at least 3%)
                            if head > left_shoulder * 1.03:
                                patterns.append({
                                    "type": "Head and Shoulders",
                                    "price_level": round((left_shoulder + right_shoulder) / 2, 2),
                                    "confidence": "High", 
                                    "signal": "Bearish Reversal"
                                })
            
            # === NEW: INVERSE HEAD & SHOULDERS PATTERN ===
            if len(minima) >= 3:
                for i in range(len(minima) - 2):
                    left_shoulder = minima.iloc[i]["Low"]
                    head = minima.iloc[i + 1]["Low"]
                    right_shoulder = minima.iloc[i + 2]["Low"]
                    
                    # Head should be lower than both shoulders
                    if head < left_shoulder and head < right_shoulder:
                        # Shoulders should be roughly equal (within 5%)
                        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                        if shoulder_diff < 0.05:
                            # Head should be significantly lower (at least 3%)
                            if head < left_shoulder * 0.97:
                                patterns.append({
                                    "type": "Inverse Head and Shoulders",
                                    "price_level": round((left_shoulder + right_shoulder) / 2, 2),
                                    "confidence": "High",
                                    "signal": "Bullish Reversal"
                                })
            
            # === NEW: TRIANGLE PATTERN DETECTION ===
            if len(recent_df) >= 30:
                # Get highs and lows from recent data
                highs = recent_df["High"].values
                lows = recent_df["Low"].values
                
                # Check for ascending triangle (horizontal resistance, rising support)
                recent_highs = highs[-10:]
                recent_lows = lows[-10:]
                
                # Ascending Triangle: resistance flat, support rising
                resistance_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
                support_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
                
                if abs(resistance_trend) < 0.1 and support_trend > 0.2:
                    patterns.append({
                        "type": "Ascending Triangle",
                        "price_level": round(np.mean(recent_highs), 2),
                        "confidence": "Medium",
                        "signal": "Bullish Breakout Expected"
                    })
                
                # Descending Triangle: support flat, resistance falling
                elif abs(support_trend) < 0.1 and resistance_trend < -0.2:
                    patterns.append({
                        "type": "Descending Triangle", 
                        "price_level": round(np.mean(recent_lows), 2),
                        "confidence": "Medium",
                        "signal": "Bearish Breakdown Expected"
                    })
                
                # Symmetrical Triangle: both converging
                elif resistance_trend < -0.1 and support_trend > 0.1:
                    patterns.append({
                        "type": "Symmetrical Triangle",
                        "price_level": round((np.mean(recent_highs) + np.mean(recent_lows)) / 2, 2),
                        "confidence": "Medium", 
                        "signal": "Breakout Direction Unclear"
                    })
            
            # === NEW: WEDGE PATTERN DETECTION ===
            if len(recent_df) >= 20:
                # Rising Wedge: both rising but resistance rises faster
                if resistance_trend > 0 and support_trend > 0 and resistance_trend > support_trend * 1.5:
                    patterns.append({
                        "type": "Rising Wedge",
                        "price_level": round(current_close, 2),
                        "confidence": "Medium",
                        "signal": "Bearish Reversal Expected"
                    })
                
                # Falling Wedge: both falling but support falls faster  
                elif resistance_trend < 0 and support_trend < 0 and support_trend < resistance_trend * 1.5:
                    patterns.append({
                        "type": "Falling Wedge",
                        "price_level": round(current_close, 2),
                        "confidence": "Medium",
                        "signal": "Bullish Reversal Expected"
                    })
            
            return {"patterns": patterns}
            
        except Exception as e:
            return {"patterns": [], "error": f"Pattern detection failed: {str(e)}"}
    
    def suggest_multiple_stops(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        MCP-TRADER MISSING: Multiple stop-loss methodologies
        Provides various stop-loss approaches
        """
        try:
            if len(df) < 20:
                return {"error": "Not enough data for stop analysis"}
            
            latest_close = df["Close"].iloc[-1]
            
            # Calculate ATR for ATR-based stops
            atr = df["atr"].iloc[-1] if "atr" in df.columns else (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
            
            stops = {
                # === ATR-BASED STOPS (MCP-trader style) ===
                "atr_1x_conservative": round(latest_close - 1 * atr, 2),
                "atr_2x_moderate": round(latest_close - 2 * atr, 2),
                "atr_3x_aggressive": round(latest_close - 3 * atr, 2),
                
                # === PERCENTAGE-BASED STOPS ===
                "percent_2_tight": round(latest_close * 0.98, 2),
                "percent_5_moderate": round(latest_close * 0.95, 2),
                "percent_8_wide": round(latest_close * 0.92, 2),
                
                # === SMA-BASED STOPS ===
                "sma_20_support": round(df["sma_20"].iloc[-1], 2) if "sma_20" in df.columns else None,
                "sma_50_support": round(df["sma_50"].iloc[-1], 2) if "sma_50" in df.columns else None,
                "sma_200_support": round(df["sma_200"].iloc[-1], 2) if "sma_200" in df.columns else None,
                
                # === TECHNICAL SUPPORT STOPS ===
                "recent_swing_low": round(df["Low"].iloc[-20:].min(), 2),
                "weekly_low": round(df["Low"].iloc[-5:].min(), 2),
            }
            
            # Calculate risk percentages for each stop
            for stop_name, stop_price in stops.items():
                if stop_price and stop_price > 0:
                    risk_pct = ((latest_close - stop_price) / latest_close) * 100
                    stops[f"{stop_name}_risk_pct"] = round(risk_pct, 2)
            
            return stops
            
        except Exception as e:
            return {"error": f"Stop level analysis failed: {str(e)}"}
    
    def calculate_position_sizing(
        self,
        current_price: float,
        stop_price: float,
        account_size: float = 100000,
        risk_per_trade: float = 1000,
        max_risk_percent: float = 2.0
    ) -> Dict[str, Any]:
        """
        MCP-TRADER STYLE: Position sizing with multiple approaches
        """
        try:
            if current_price <= 0 or account_size <= 0:
                return {"error": "Invalid price or account size"}
            
            if current_price <= stop_price:
                return {"error": "Stop price must be below current price"}
            
            # Risk per share
            risk_per_share = current_price - stop_price
            
            # Position size based on dollar risk
            shares_dollar_risk = int(risk_per_trade / risk_per_share)
            
            # Position size based on % risk
            max_risk_dollars = account_size * (max_risk_percent / 100)
            shares_pct_risk = int(max_risk_dollars / risk_per_share)
            
            # Take the smaller (more conservative)
            recommended_shares = min(shares_dollar_risk, shares_pct_risk)
            
            # Calculate metrics
            position_cost = recommended_shares * current_price
            actual_risk = recommended_shares * risk_per_share
            risk_reward_1to1 = current_price + risk_per_share
            risk_reward_2to1 = current_price + 2 * risk_per_share
            risk_reward_3to1 = current_price + 3 * risk_per_share
            
            return {
                "recommended_shares": recommended_shares,
                "position_cost": round(position_cost, 2),
                "actual_risk": round(actual_risk, 2),
                "risk_per_share": round(risk_per_share, 2),
                "risk_reward_targets": {
                    "r1_target": round(risk_reward_1to1, 2),
                    "r2_target": round(risk_reward_2to1, 2),
                    "r3_target": round(risk_reward_3to1, 2)
                },
                "account_risk_pct": round((actual_risk / account_size) * 100, 2)
            }
            
        except Exception as e:
            return {"error": f"Position sizing failed: {str(e)}"}
    
    def comprehensive_stock_analysis(
        self,
        symbol: str,
        benchmark: str = "SPY",
        account_size: float = 100000,
        risk_per_trade: float = 1000
    ) -> Dict[str, Any]:
        """
        Complete MCP-trader style analysis of a single stock
        """
        try:
            console.print(f"\n🔍 [bold cyan]Comprehensive Analysis: {symbol}[/bold cyan]")
            
            # Fetch 6 months of data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")
            
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Add all technical indicators
            df = self.add_comprehensive_indicators(df)
            
            current_price = df["Close"].iloc[-1]
            
            # 1. Volume Profile Analysis (NEW from enhanced_technical.py)
            volume_profile_analyzer = VolumeProfileAnalysis()
            volume_profile = volume_profile_analyzer.analyze_volume_profile(df)
            
            # 2. Trend Status Analysis
            trend_analysis = self.analyze_trend_status(df)
            
            # 3. Relative Strength Analysis
            rs_analysis = self.calculate_relative_strength_detailed(symbol, benchmark)
            
            # 4. Advanced Pattern Recognition
            pattern_analysis = self.detect_advanced_patterns(df)
            
            # 5. Multiple Stop Loss Suggestions
            stop_analysis = self.suggest_multiple_stops(df)
            
            # 6. Position Sizing (using best ATR stop)
            if "atr_2x_moderate" in stop_analysis:
                position_analysis = self.calculate_position_sizing(
                    current_price, stop_analysis["atr_2x_moderate"], 
                    account_size, risk_per_trade
                )
            else:
                position_analysis = {"error": "Cannot calculate position sizing"}
            
            # 7. Technical Score Calculation (NEW from enhanced_technical.py)
            technical_score = self._calculate_technical_score(
                volume_profile, rs_analysis, pattern_analysis, current_price
            )
            
            # 8. Volatility Analysis (ADRP)
            adrp = df["adrp"].iloc[-1] if "adrp" in df.columns else None
            volume_ratio = df["volume_ratio"].iloc[-1] if "volume_ratio" in df.columns else None
            
            # Compile comprehensive results
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "analysis_date": datetime.now().isoformat(),
                
                # Core analyses
                "volume_profile": volume_profile,  # NEW: Volume profile analysis
                "trend_analysis": trend_analysis,
                "relative_strength": rs_analysis,
                "pattern_analysis": pattern_analysis,
                "stop_analysis": stop_analysis,
                "position_analysis": position_analysis,
                "technical_score": technical_score,  # NEW: Overall technical score
                
                # Key metrics (with null checks)
                "volatility_adrp": round(adrp, 2) if adrp is not None and pd.notnull(adrp) else None,
                "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None and pd.notnull(volume_ratio) else None,
                "rsi": round(df["rsi"].iloc[-1], 1) if "rsi" in df.columns and pd.notnull(df["rsi"].iloc[-1]) else None,
                
                # Price levels (with null checks)
                "sma_20": round(df["sma_20"].iloc[-1], 2) if "sma_20" in df.columns and pd.notnull(df["sma_20"].iloc[-1]) else None,
                "sma_50": round(df["sma_50"].iloc[-1], 2) if "sma_50" in df.columns and pd.notnull(df["sma_50"].iloc[-1]) else None,
                "sma_200": round(df["sma_200"].iloc[-1], 2) if "sma_200" in df.columns and pd.notnull(df["sma_200"].iloc[-1]) else None,
            }
            
        except Exception as e:
            return {"error": f"Analysis failed for {symbol}: {str(e)}"}
    
    def _calculate_technical_score(
        self, 
        volume_profile: Dict, 
        rs_analysis: Dict, 
        patterns: Dict,
        current_price: float
    ) -> float:
        """
        Calculate overall technical score (0-1) from enhanced_technical.py
        Combines Volume Profile (30%) + Relative Strength (40%) + Patterns (30%)
        """
        try:
            score = 0.0
            max_score = 0.0
            
            # Volume profile score (0.3 weight)
            if "point_of_control" in volume_profile and "value_area_low" in volume_profile:
                poc = volume_profile["point_of_control"]
                va_low = volume_profile["value_area_low"]
                va_high = volume_profile["value_area_high"]
                
                # Score based on price position relative to volume areas
                if va_low <= current_price <= va_high:
                    score += 0.2  # In value area
                if current_price > poc:
                    score += 0.1  # Above POC
                    
            max_score += 0.3
            
            # Relative strength score (0.4 weight)
            if "RS_63d" in rs_analysis:
                rs_63d = rs_analysis["RS_63d"]["score"]
                if rs_63d > 70:
                    score += 0.4
                elif rs_63d > 60:
                    score += 0.3
                elif rs_63d > 50:
                    score += 0.2
            max_score += 0.4
            
            # Pattern score (0.3 weight)
            bullish_patterns = ["Resistance Breakout", "Near Resistance", "Inverse Head and Shoulders", 
                               "Ascending Triangle", "Falling Wedge"]
            bearish_patterns = ["Head and Shoulders", "Descending Triangle", "Rising Wedge"]
            
            if patterns and "patterns" in patterns:
                for pattern in patterns["patterns"]:
                    pattern_type = pattern.get("type")
                    confidence = pattern.get("confidence")
                    
                    if pattern_type in bullish_patterns:
                        if confidence == "High":
                            score += 0.2
                        else:
                            score += 0.1
                    elif pattern_type in bearish_patterns:
                        # Bearish patterns reduce score
                        if confidence == "High":
                            score -= 0.1
                        else:
                            score -= 0.05
            
            max_score += 0.3
            
            return round(score / max_score if max_score > 0 else 0, 3)
            
        except Exception as e:
            return 0.0

def print_comprehensive_analysis(analysis: Dict[str, Any]):
    """Print comprehensive analysis results in a beautiful format"""
    if "error" in analysis:
        console.print(f"❌ [red]Error: {analysis['error']}[/red]")
        return
    
    symbol = analysis["symbol"]
    current_price = analysis["current_price"]
    
    # Header
    console.print(Panel(
        f"[bold blue]Comprehensive Technical Analysis: {symbol}[/bold blue]\n"
        f"Current Price: ${current_price}\n"
        f"Analysis Time: {analysis['analysis_date'][:19]}",
        title="MCP-Style Analysis", border_style="blue"
    ))
    
    # === TREND ANALYSIS ===
    trend = analysis["trend_analysis"]
    console.print(f"\n📈 [bold green]TREND STATUS ANALYSIS[/bold green]")
    console.print(f"   Trend Strength: {trend.get('trend_strength', 0)}/100 - {trend.get('trend_classification', 'Unknown')}")
    console.print(f"   Above 20 SMA: {'✅' if trend.get('above_20sma') else '❌'}")
    console.print(f"   Above 50 SMA: {'✅' if trend.get('above_50sma') else '❌'}")
    console.print(f"   Above 200 SMA: {'✅' if trend.get('above_200sma') else '❌'}")
    console.print(f"   20/50 Bullish Cross: {'✅' if trend.get('20_50_bullish') else '❌'}")
    console.print(f"   50/200 Bullish Cross: {'✅' if trend.get('50_200_bullish') else '❌'}")
    if "macd_bullish" in trend:
        console.print(f"   MACD Bullish: {'✅' if trend.get('macd_bullish') else '❌'}")
    
    # === VOLATILITY & VOLUME ===
    console.print(f"\n📊 [bold yellow]VOLATILITY & VOLUME ANALYSIS[/bold yellow]")
    adrp = analysis.get("volatility_adrp")
    volume_ratio = analysis.get("volume_ratio")
    if adrp:
        volatility_desc = "High" if adrp > 5 else "Normal" if adrp > 2 else "Low"
        console.print(f"   ADRP (Volatility): {adrp:.2f}% ({volatility_desc})")
    if volume_ratio:
        volume_desc = "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.8 else "Low"
        console.print(f"   Volume Ratio: {volume_ratio:.2f}x ({volume_desc})")
    
    # === VOLUME PROFILE (NEW) ===
    volume_profile = analysis.get("volume_profile", {})
    if "point_of_control" in volume_profile:
        console.print(f"\n🎯 [bold blue]VOLUME PROFILE ANALYSIS[/bold blue]")
        poc = volume_profile["point_of_control"]
        va_low = volume_profile.get("value_area_low")
        va_high = volume_profile.get("value_area_high")
        
        console.print(f"   Point of Control (POC): ${poc:.2f}")
        if va_low and va_high:
            console.print(f"   Value Area: ${va_low:.2f} - ${va_high:.2f}")
            if va_low <= current_price <= va_high:
                console.print(f"   ✅ Price is in Value Area (institutional interest zone)")
            else:
                console.print(f"   ⚠️ Price outside Value Area")
        
        if current_price > poc:
            console.print(f"   ✅ Price above POC (bullish)")
        else:
            console.print(f"   ❌ Price below POC")
    
    # === TECHNICAL SCORE (NEW) ===
    technical_score = analysis.get("technical_score")
    if technical_score is not None:
        score_pct = technical_score * 100
        if score_pct >= 70:
            score_desc = "Strong"
            score_color = "green"
        elif score_pct >= 50:
            score_desc = "Moderate" 
            score_color = "yellow"
        elif score_pct >= 30:
            score_desc = "Weak"
            score_color = "red"
        else:
            score_desc = "Very Weak"
            score_color = "red"
        
        console.print(f"\n🎯 [bold {score_color}]OVERALL TECHNICAL SCORE: {score_pct:.1f}% ({score_desc})[/bold {score_color}]")
        console.print(f"   Combines: Volume Profile (30%) + Relative Strength (40%) + Patterns (30%)")
    
    # === RELATIVE STRENGTH ===
    rs = analysis["relative_strength"]
    if "error" not in rs:
        console.print(f"\n💪 [bold magenta]RELATIVE STRENGTH vs SPY[/bold magenta]")
        for period, data in rs.items():
            if period.startswith("RS_"):
                days = period.split("_")[1]
                console.print(f"   {days}: {data['score']:.1f} - {data['classification']}")
                console.print(f"       Stock: {data['stock_return']:+.1f}% | SPY: {data['benchmark_return']:+.1f}% | Excess: {data['excess_return']:+.1f}%")
    
    # === PATTERN ANALYSIS ===
    patterns = analysis["pattern_analysis"]["patterns"]
    console.print(f"\n🔍 [bold cyan]PATTERN RECOGNITION[/bold cyan]")
    if patterns:
        for pattern in patterns:
            signal_color = "green" if "Bullish" in pattern.get("signal", "") else "red"
            console.print(f"   [{signal_color}]{pattern['type']}[/{signal_color}]: ${pattern['price_level']} ({pattern.get('signal', 'Unknown')})")
    else:
        console.print("   No significant patterns detected")
    
    # === STOP LOSS ANALYSIS ===
    stops = analysis["stop_analysis"]
    if "error" not in stops:
        console.print(f"\n🛡️  [bold red]STOP LOSS SUGGESTIONS[/bold red]")
        console.print("   ATR-Based Stops:")
        for stop_type in ["atr_1x_conservative", "atr_2x_moderate", "atr_3x_aggressive"]:
            if stop_type in stops:
                risk_key = f"{stop_type}_risk_pct"
                risk = stops.get(risk_key, 0)
                console.print(f"     {stop_type.replace('_', ' ').title()}: ${stops[stop_type]} ({risk:.1f}% risk)")
        
        console.print("   Percentage-Based Stops:")
        for stop_type in ["percent_2_tight", "percent_5_moderate", "percent_8_wide"]:
            if stop_type in stops:
                console.print(f"     {stop_type.replace('_', ' ').title()}: ${stops[stop_type]}")
    
    # === POSITION SIZING ===
    position = analysis["position_analysis"]
    if "error" not in position:
        console.print(f"\n💰 [bold green]POSITION SIZING RECOMMENDATION[/bold green]")
        console.print(f"   Recommended Shares: {position['recommended_shares']}")
        console.print(f"   Position Cost: ${position['position_cost']:,.2f}")
        console.print(f"   Risk Amount: ${position['actual_risk']:.2f} ({position['account_risk_pct']:.2f}% of account)")
        console.print(f"   Risk/Reward Targets:")
        targets = position["risk_reward_targets"]
        console.print(f"     R1 (1:1): ${targets['r1_target']:.2f}")
        console.print(f"     R2 (2:1): ${targets['r2_target']:.2f}")
        console.print(f"     R3 (3:1): ${targets['r3_target']:.2f}")

def analyze_multiple_stocks(symbols: List[str], benchmark: str = "SPY"):
    """Analyze multiple stocks and create a comparison table"""
    console.print(f"\n🔍 [bold blue]Multi-Stock MCP-Style Analysis[/bold blue]")
    console.print(f"Analyzing {len(symbols)} stocks vs {benchmark}")
    
    analyzer = ComprehensiveTechnicalAnalysis()
    results = []
    
    for symbol in symbols:
        console.print(f"\n📊 Processing {symbol}...")
        analysis = analyzer.comprehensive_stock_analysis(symbol, benchmark)
        results.append(analysis)
    
    # Create comparison table
    table = Table(title="Technical Analysis Comparison", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Price", style="green")
    table.add_column("Trend Score", style="yellow")
    table.add_column("RSI", style="magenta")
    table.add_column("ADRP", style="blue")
    table.add_column("Volume", style="white")
    table.add_column("Pattern", style="red")
    table.add_column("RS 63d", style="green")
    
    for analysis in results:
        if "error" in analysis:
            continue
            
        symbol = analysis["symbol"]
        price = f"${analysis['current_price']:.2f}"
        trend_score = f"{analysis['trend_analysis'].get('trend_strength', 0)}/100"
        rsi = f"{analysis.get('rsi', 0):.1f}" if analysis.get('rsi') else "N/A"
        adrp = f"{analysis.get('volatility_adrp', 0):.2f}%" if analysis.get('volatility_adrp') else "N/A"
        volume = f"{analysis.get('volume_ratio', 1):.2f}x" if analysis.get('volume_ratio') else "N/A"
        
        # Get primary pattern
        patterns = analysis["pattern_analysis"]["patterns"]
        pattern = patterns[0]["type"] if patterns else "None"
        
        # Get 63-day RS
        rs_63d = "N/A"
        if "RS_63d" in analysis["relative_strength"]:
            rs_score = analysis["relative_strength"]["RS_63d"]["score"]
            rs_63d = f"{rs_score:.1f}"
        
        table.add_row(symbol, price, trend_score, rsi, adrp, volume, pattern, rs_63d)
    
    console.print(table)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="MCP-Style Comprehensive Technical Analyzer")
    parser.add_argument("symbols", nargs="*", help="Stock symbols to analyze (e.g., AAPL NVDA TSLA)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark for relative strength (default: SPY)")
    parser.add_argument("--account-size", type=float, default=100000, help="Account size for position sizing")
    parser.add_argument("--risk-per-trade", type=float, default=1000, help="Risk per trade for position sizing")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis for each stock")
    
    # === NEW: DISCOVERY MODE ===
    parser.add_argument("--discovery", action="store_true", help="Discovery mode: scan major indices for high-scoring stocks")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum technical score threshold (0-1, default: 0.5)")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top stocks to show (default: 20)")
    parser.add_argument("--indices", nargs="+", default=["sp500"], 
                       help="Indices to scan: sp500, nasdaq, russell1000, all")
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveTechnicalAnalysis()
    
    if args.discovery:
        # Discovery mode: scan indices for high-scoring stocks
        discover_high_scoring_stocks(
            analyzer, args.indices, args.min_score, args.top_n,
            args.benchmark, args.account_size, args.risk_per_trade, args.detailed
        )
    elif args.symbols:
        if args.detailed:
            # Detailed analysis for each stock
            for symbol in args.symbols:
                analysis = analyzer.comprehensive_stock_analysis(
                    symbol, args.benchmark, args.account_size, args.risk_per_trade
                )
                print_comprehensive_analysis(analysis)
        else:
            # Quick comparison table
            analyze_multiple_stocks(args.symbols, args.benchmark)
    else:
        parser.print_help()

def discover_high_scoring_stocks(
    analyzer: ComprehensiveTechnicalAnalysis,
    indices: List[str] = ["sp500"],
    min_score: float = 0.5,
    top_n: int = 20,
    benchmark: str = "SPY",
    account_size: float = 100000,
    risk_per_trade: float = 1000,
    detailed: bool = False
):
    """
    Discovery mode: Scan major stock indices and filter high-scoring opportunities
    """
    console.print(f"\n🔍 [bold blue]MCP-STYLE STOCK DISCOVERY MODE[/bold blue]")
    console.print(f"Scanning indices: {', '.join(indices).upper()}")
    console.print(f"Filter: Technical Score ≥ {min_score*100:.0f}% | Showing Top {top_n}")
    
    # Get stock symbols from indices
    all_symbols = []
    for index in indices:
        symbols = get_index_symbols(index)
        all_symbols.extend(symbols)
        console.print(f"   📊 {index.upper()}: {len(symbols)} stocks")
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    console.print(f"\n🎯 Total unique stocks to analyze: {len(all_symbols)}")
    
    # Analyze and filter stocks
    qualifying_stocks = []
    analyzed_count = 0
    
    console.print(f"\n⚡ Analyzing stocks for technical score...")
    
    for i, symbol in enumerate(all_symbols):
        try:
            console.print(f"   [{i+1}/{len(all_symbols)}] {symbol}...", end="")
            
            # Perform comprehensive analysis
            analysis = analyzer.comprehensive_stock_analysis(symbol, benchmark, account_size, risk_per_trade)
            analyzed_count += 1
            
            if "error" in analysis:
                console.print(" ❌")
                continue
            
            # Apply simple technical score filter
            technical_score = analysis.get("technical_score", 0)
            
            # Check if stock meets minimum score
            meets_criteria = technical_score >= min_score
            
            if meets_criteria:
                qualifying_stocks.append(analysis)
                console.print(" ✅")
            else:
                console.print(" ⚠️")
                
        except Exception as e:
            console.print(" ❌")
            continue
    
    # Sort by technical score (descending)
    qualifying_stocks.sort(key=lambda x: x.get("technical_score", 0), reverse=True)
    
    # Display results
    console.print(f"\n🏆 [bold green]DISCOVERY RESULTS[/bold green]")
    console.print(f"Analyzed: {analyzed_count}/{len(all_symbols)} stocks")
    console.print(f"Qualifying: {len(qualifying_stocks)} stocks (score ≥ {min_score*100:.0f}%)")
    
    if qualifying_stocks:
        # Create results table
        table = Table(title="🔥 High-Scoring Stock Opportunities (Ranked by Technical Score)", show_header=True)
        table.add_column("Rank", style="white", width=6)
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Price", style="green", width=8)
        table.add_column("Tech Score", style="yellow", width=10)
        table.add_column("RS 63d", style="magenta", width=8)
        table.add_column("ADRP", style="blue", width=8)
        table.add_column("Volume", style="white", width=8)
        table.add_column("Top Pattern", style="red", width=15)
        table.add_column("Trend", style="green", width=10)
        
        for rank, analysis in enumerate(qualifying_stocks[:top_n], 1):  # Top N
            symbol = analysis["symbol"]
            price = f"${analysis['current_price']:.2f}"
            tech_score = f"{analysis.get('technical_score', 0)*100:.0f}%"
            rs_63d = f"{analysis.get('relative_strength', {}).get('RS_63d', {}).get('score', 0):.1f}"
            adrp = f"{analysis.get('volatility_adrp', 0):.1f}%"
            volume = f"{analysis.get('volume_ratio', 1):.1f}x"
            
            # Get primary pattern
            patterns = analysis["pattern_analysis"]["patterns"]
            pattern = patterns[0]["type"] if patterns else "None"
            
            # Get trend status
            trend_strength = analysis["trend_analysis"].get("trend_strength", 0)
            trend = f"{trend_strength}/100"
            
            table.add_row(str(rank), symbol, price, tech_score, rs_63d, adrp, volume, pattern, trend)
        
        console.print(table)
        
        if detailed:
            console.print(f"\n📋 [bold yellow]DETAILED ANALYSIS OF TOP 5 CANDIDATES[/bold yellow]")
            for analysis in qualifying_stocks[:5]:
                print_comprehensive_analysis(analysis)
                console.print("\n" + "="*80 + "\n")
    else:
        console.print(f"❌ No stocks met the minimum technical score of {min_score*100:.0f}%. Try lowering --min-score.")

def get_index_symbols(index_name: str) -> List[str]:
    """
    Get stock symbols from major indices using dynamic fetching
    """
    try:
        if index_name.lower() == "sp500":
            return get_sp500_tickers()
        elif index_name.lower() == "nasdaq":
            return get_nasdaq_tickers()
        elif index_name.lower() == "russell1000":
            return get_russell1000_tickers()
        elif index_name.lower() == "all":
            return get_all_tickers()
        else:
            console.print(f"⚠️ Unknown index: {index_name}, using S&P 500")
            return get_sp500_tickers()
            
    except Exception as e:
        console.print(f"❌ Error fetching {index_name} symbols: {e}")
        # Fallback to smaller sample
        fallback_symbols = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "XOM"
        ]
        console.print(f"⚠️ Using fallback symbols for {index_name}")
        return fallback_symbols

if __name__ == "__main__":
    main() 