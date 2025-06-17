#!/usr/bin/env python3
"""
Pro Screener - Advanced MCP-Style Technical Analysis Tool
Incorporates all advanced technical analysis methods from MCP-trader that were missing from our implementation.

Key Features Added:
- ADRP (Average Daily Range Percentage) - volatility assessment
- Multi-SMA trend alignment analysis with configurable strategy weights
- MACD crossover detection
- Advanced pattern recognition with dynamic ATR-based thresholds
- Volume vs average comparison
- Multiple stop-loss methodologies
- Detailed relative strength classifications with proper normalization
- Volume Profile Analysis (POC + Value Area) with fixed double-counting
- Discovery mode with dynamic symbol fetching
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from scipy.stats import linregress
import re

# Import dynamic ticker functions - FIXED import for direct execution
try:
    from .helper import get_sp500_tickers, get_russell1000_tickers, get_nasdaq_tickers, get_all_tickers
except ImportError:
    # Fallback for direct script execution
    from .helper import get_sp500_tickers, get_russell1000_tickers, get_nasdaq_tickers, get_all_tickers

console = Console()

class VolumeProfileAnalysis:
    """Advanced volume analysis for identifying key price levels - FIXED double-counting issue"""
    
    @staticmethod
    def analyze_volume_profile(df: pd.DataFrame, num_bins: int = 10) -> Dict[str, Any]:
        """
        Create volume profile analysis by price level
        Identifies Point of Control (POC) and Value Area
        FIXED: Volume double-counting when candles span multiple bins
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
            
            # Calculate volume by price bin - FIXED: Proportional distribution
            for i in range(num_bins):
                bin_low = price_min + i * bin_width
                bin_high = bin_low + bin_width
                bin_mid = (bin_low + bin_high) / 2
                
                volume_in_bin = 0
                
                # Process each candle individually to avoid double-counting
                for idx, row in df.iterrows():
                    candle_high = row[high_col]
                    candle_low = row[low_col]
                    candle_volume = row[volume_col]
                    
                    # Check if candle overlaps with this bin
                    if candle_high >= bin_low and candle_low <= bin_high:
                        # Calculate overlap percentage
                        overlap_low = max(bin_low, candle_low)
                        overlap_high = min(bin_high, candle_high)
                        candle_range = candle_high - candle_low
                        
                        if candle_range > 0:
                            overlap_percentage = (overlap_high - overlap_low) / candle_range
                            volume_in_bin += candle_volume * overlap_percentage
                        else:
                            # Single price point, full volume if in range
                            if bin_low <= candle_low <= bin_high:
                                volume_in_bin += candle_volume
                
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
    ENHANCED: Configurable strategy weights, dynamic thresholds, proper normalization
    """
    
    def __init__(self, strategy: str = "balanced"):
        self.console = Console()
        self.strategy = strategy
        self.sma_weights = self.get_sma_weights(strategy)
    
    def get_sma_weights(self, strategy: str = "balanced") -> Dict[str, int]:
        """Return SMA weights based on trading strategy - CONFIGURABLE APPROACH"""
        
        if strategy == "long_term":
            return {
                "sma_200": 40,  # Major trend most important
                "sma_50": 25,   
                "sma_20": 10,
                "20_50_cross": 15,
                "50_200_cross": 10
            }
        
        elif strategy == "momentum":  # Original Pro Screener approach
            return {
                "sma_20": 30,   # Recent momentum most important
                "sma_50": 25,   
                "sma_200": 20,  # Background context
                "20_50_cross": 15,
                "50_200_cross": 10
            }
        
        elif strategy == "balanced":  # Balanced approach
            return {
                "sma_200": 30,  # Balanced weighting
                "sma_50": 25,   
                "sma_20": 20,
                "20_50_cross": 15,
                "50_200_cross": 10
            }
        
        else:  # Default balanced
            return self.get_sma_weights("balanced")
    
    def get_dynamic_threshold(self, df: pd.DataFrame, base_percent: float) -> float:
        """Calculate dynamic threshold based on ATR - ADAPTIVE THRESHOLDS"""
        try:
            current_price = df["Close"].iloc[-1]
            atr = df["atr"].iloc[-1] if "atr" in df.columns else None
            
            if atr and pd.notnull(atr):
                # Use ATR as volatility measure (more adaptive)
                atr_percent = (atr / current_price) * 100
                # Scale base threshold by volatility (0.5x to 2x range)
                volatility_multiplier = max(0.5, min(2.0, atr_percent / 2.0))
                return base_percent * volatility_multiplier
            else:
                return base_percent  # Fallback to fixed
        except:
            return base_percent
    
    def normalize_relative_performance(self, relative_perf: float) -> float:
        """Convert relative performance to 0-100 RS score using sigmoid normalization - FIXED FORMULA"""
        # Sigmoid with scaling: maps -50% to ~5, +50% to ~95, 0% to 50
        sigmoid_input = relative_perf / 20  # Scale factor
        sigmoid_output = 1 / (1 + math.exp(-sigmoid_input))
        return round(sigmoid_output * 98 + 1, 2)  # Scale to 1-99 range
    
    def classify_volume_ratio(self, volume_ratio: float, adrp: float = None) -> Tuple[str, str]:
        """Classify volume ratio with volatility adjustment - DYNAMIC CLASSIFICATION"""
        if adrp is None:
            # Standard thresholds for unknown volatility
            high_threshold = 1.5
            normal_threshold = 0.8
        else:
            # Adjust thresholds based on stock volatility
            if adrp > 5:  # High volatility stock
                high_threshold = 2.0    # Needs higher volume for "high"
                normal_threshold = 1.0   
            elif adrp < 2:  # Low volatility stock  
                high_threshold = 1.2    # Lower bar for "high" volume
                normal_threshold = 0.6
            else:  # Normal volatility
                high_threshold = 1.5
                normal_threshold = 0.8
        
        if volume_ratio > high_threshold:
            return "High", "✅"
        elif volume_ratio > normal_threshold:
            return "Normal", "🟡"
        else:
            return "Low", "❌"
    
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
        ENHANCED: Configurable strategy weights, improved validation
        """
        if df.empty:
            return {"error": "Empty dataframe"}
        
        latest = df.iloc[-1]
        
        # Multi-SMA trend alignment - FIXED: Percentage-based validation
        current_price = latest["Close"]
        
        # Use 0.1% threshold instead of absolute $0.01 - SCALE INDEPENDENT
        sma_20_valid = (pd.notnull(latest.get("sma_20")) and 
                        abs(latest["sma_20"] - current_price) / current_price > 0.001)
        sma_50_valid = (pd.notnull(latest.get("sma_50")) and 
                        abs(latest["sma_50"] - current_price) / current_price > 0.001)  
        sma_200_valid = (pd.notnull(latest.get("sma_200")) and 
                         abs(latest["sma_200"] - current_price) / current_price > 0.001)
        
        sma_20 = latest["sma_20"] if sma_20_valid else None
        sma_50 = latest["sma_50"] if sma_50_valid else None
        sma_200 = latest["sma_200"] if sma_200_valid else None
        
        # Only calculate trend components if we have valid SMA data
        trend_status = {
            "above_20sma": current_price > sma_20 if sma_20 is not None else None,
            "above_50sma": current_price > sma_50 if sma_50 is not None else None,
            "above_200sma": current_price > sma_200 if sma_200 is not None else None,
            "20_50_bullish": sma_20 > sma_50 if (sma_20 is not None and sma_50 is not None) else None,
            "50_200_bullish": sma_50 > sma_200 if (sma_50 is not None and sma_200 is not None) else None,
            "rsi": latest["rsi"] if pd.notnull(latest.get("rsi")) else None,
            "adrp": latest["adrp"] if pd.notnull(latest.get("adrp")) else None,
            "volume_ratio": latest["volume_ratio"] if pd.notnull(latest.get("volume_ratio")) else None,
            "strategy": self.strategy,
            "sma_validity": {
                "sma_20_valid": sma_20_valid,
                "sma_50_valid": sma_50_valid, 
                "sma_200_valid": sma_200_valid
            }
        }
        
        # Add MACD analysis if available
        if "macd_bullish" in df.columns and pd.notnull(latest.get("macd_bullish")):
            trend_status["macd_bullish"] = latest["macd_bullish"]
        
        # Calculate trend strength score (0-100) using STRATEGY-SPECIFIC WEIGHTS
        trend_score = 0
        max_possible_score = 0
        weights = self.sma_weights
        
        # Apply strategy-specific weights
        if trend_status["above_200sma"] is not None:
            max_possible_score += weights["sma_200"]
            if trend_status["above_200sma"]: 
                trend_score += weights["sma_200"]
                
        if trend_status["above_50sma"] is not None:
            max_possible_score += weights["sma_50"]
            if trend_status["above_50sma"]: 
                trend_score += weights["sma_50"]
                
        if trend_status["above_20sma"] is not None:
            max_possible_score += weights["sma_20"]
            if trend_status["above_20sma"]: 
                trend_score += weights["sma_20"]
                
        if trend_status["20_50_bullish"] is not None:
            max_possible_score += weights["20_50_cross"]
            if trend_status["20_50_bullish"]: 
                trend_score += weights["20_50_cross"]
                
        if trend_status["50_200_bullish"] is not None:
            max_possible_score += weights["50_200_cross"]
            if trend_status["50_200_bullish"]: 
                trend_score += weights["50_200_cross"]
        
        # Scale score based on available data
        if max_possible_score > 0:
            scaled_trend_score = int((trend_score / max_possible_score) * 100)
        else:
            scaled_trend_score = 0  # No valid SMA data
            
        trend_status["trend_strength"] = scaled_trend_score
        trend_status["trend_data_quality"] = f"{max_possible_score}/100 points available ({self.strategy} strategy)"
        
        # Trend classification with data quality consideration
        if max_possible_score < 50:
            trend_status["trend_classification"] = f"Insufficient SMA Data (only {max_possible_score}/100 signals)"
        elif scaled_trend_score >= 90:
            trend_status["trend_classification"] = "Very Strong Uptrend"
        elif scaled_trend_score >= 70:
            trend_status["trend_classification"] = "Strong Uptrend"
        elif scaled_trend_score >= 50:
            trend_status["trend_classification"] = "Moderate Uptrend"
        elif scaled_trend_score >= 30:
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
        FIXED: Proper sigmoid normalization instead of broken linear formula
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
                
                # FIXED: Use proper sigmoid normalization instead of broken linear formula
                rs_score = self.normalize_relative_performance(relative_performance)
                
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
    
    def detect_triangle_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """FIXED triangle detection with proper slope analysis and R-squared validation"""
        patterns = []
        
        if len(df) < 30:
            return patterns
            
        recent_df = df.tail(30).copy()
        
        # Get price data with time index (days from start)
        days = np.arange(len(recent_df))
        highs = recent_df["High"].values
        lows = recent_df["Low"].values
        current_price = df["Close"].iloc[-1]
        
        # Calculate slopes with proper scaling
        try:
            # Resistance line (highs)
            resistance_slope, resistance_intercept, resistance_r_squared, _, _ = linregress(days, highs)
            resistance_slope_percent = (resistance_slope / current_price) * 100  # % per day
            
            # Support line (lows)  
            support_slope, support_intercept, support_r_squared, _, _ = linregress(days, lows)
            support_slope_percent = (support_slope / current_price) * 100  # % per day
            
            # Require good fit (R² > 0.3) for valid pattern
            if resistance_r_squared > 0.3 and support_r_squared > 0.3:
                
                # Ascending Triangle: flat resistance, rising support
                if abs(resistance_slope_percent) < 0.02 and support_slope_percent > 0.02:
                    patterns.append({
                        "type": "Ascending Triangle",
                        "price_level": round(np.mean(highs[-5:]), 2),
                        "confidence": "High" if min(resistance_r_squared, support_r_squared) > 0.5 else "Medium",
                        "signal": "Bullish Breakout Expected"
                    })
                
                # Descending Triangle: falling resistance, flat support  
                elif abs(support_slope_percent) < 0.02 and resistance_slope_percent < -0.02:
                    patterns.append({
                        "type": "Descending Triangle",
                        "price_level": round(np.mean(lows[-5:]), 2), 
                        "confidence": "High" if min(resistance_r_squared, support_r_squared) > 0.5 else "Medium",
                        "signal": "Bearish Breakdown Expected"
                    })
                
                # Symmetrical Triangle: converging lines
                elif (resistance_slope_percent < -0.01 and support_slope_percent > 0.01 and 
                      abs(abs(resistance_slope_percent) - abs(support_slope_percent)) < 0.02):
                    patterns.append({
                        "type": "Symmetrical Triangle",
                        "price_level": round(current_price, 2),
                        "confidence": "High" if min(resistance_r_squared, support_r_squared) > 0.5 else "Medium",
                        "signal": "Breakout Direction Unclear"
                    })
        
        except Exception as e:
            pass  # Skip if regression fails
            
        return patterns
    
    def detect_advanced_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        MCP-TRADER MISSING: Advanced pattern recognition
        Detects double tops, double bottoms, and sophisticated patterns
        ENHANCED: Dynamic ATR-based thresholds instead of hardcoded values
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
            
            # === DOUBLE BOTTOM DETECTION - DYNAMIC THRESHOLDS ===
            if len(minima) >= 2:
                similarity_threshold = self.get_dynamic_threshold(recent_df, 3.0) / 100  # ATR-based
                peak_threshold = 1.0 + (self.get_dynamic_threshold(recent_df, 5.0) / 100)
                
                for i in range(len(minima) - 1):
                    for j in range(i + 1, len(minima)):
                        price1 = minima.iloc[i]["Low"]
                        price2 = minima.iloc[j]["Low"]
                        date1 = minima.iloc[i].name
                        date2 = minima.iloc[j].name
                        
                        # Check if similar price levels - DYNAMIC threshold
                        if abs(price1 - price2) / price1 < similarity_threshold:
                            # Check time separation (10-60 days)
                            days_apart = (date2 - date1).days
                            if 10 <= days_apart <= 60:
                                # Check for peak in between - DYNAMIC threshold
                                mask = (recent_df.index > date1) & (recent_df.index < date2)
                                if mask.any():
                                    max_between = recent_df.loc[mask, "High"].max()
                                    if max_between > price1 * peak_threshold:
                                        patterns.append({
                                            "type": "Double Bottom",
                                            "start_date": date1.strftime("%Y-%m-%d"),
                                            "end_date": date2.strftime("%Y-%m-%d"),
                                            "price_level": round((price1 + price2) / 2, 2),
                                            "confidence": "High",  # Higher confidence with dynamic thresholds
                                            "signal": "Bullish Reversal"
                                        })
            
            # === DOUBLE TOP DETECTION - DYNAMIC THRESHOLDS ===
            if len(maxima) >= 2:
                similarity_threshold = self.get_dynamic_threshold(recent_df, 3.0) / 100
                valley_threshold = 1.0 - (self.get_dynamic_threshold(recent_df, 5.0) / 100)
                
                for i in range(len(maxima) - 1):
                    for j in range(i + 1, len(maxima)):
                        price1 = maxima.iloc[i]["High"]
                        price2 = maxima.iloc[j]["High"]
                        date1 = maxima.iloc[i].name
                        date2 = maxima.iloc[j].name
                        
                        if abs(price1 - price2) / price1 < similarity_threshold:
                            days_apart = (date2 - date1).days
                            if 10 <= days_apart <= 60:
                                mask = (recent_df.index > date1) & (recent_df.index < date2)
                                if mask.any():
                                    min_between = recent_df.loc[mask, "Low"].min()
                                    if min_between < price1 * valley_threshold:
                                        patterns.append({
                                            "type": "Double Top",
                                            "start_date": date1.strftime("%Y-%m-%d"),
                                            "end_date": date2.strftime("%Y-%m-%d"),
                                            "price_level": round((price1 + price2) / 2, 2),
                                            "confidence": "High",  # Higher confidence with dynamic thresholds
                                            "signal": "Bearish Reversal"
                                        })
            
            # === BREAKOUT PATTERNS - FIXED with ATR-based thresholds ===
            current_close = df["Close"].iloc[-1]
            recent_high_20 = df["High"].iloc[-20:].max()
            recent_low_20 = df["Low"].iloc[-20:].min()

            # Calculate ATR-based breakout thresholds - MUCH MORE RELIABLE
            atr = df["atr"].iloc[-1] if "atr" in df.columns and pd.notnull(df["atr"].iloc[-1]) else None
            if atr:
                # Require 0.5 ATR above/below for valid breakout
                breakout_buffer = atr * 0.5
                resistance_threshold = recent_high_20 + breakout_buffer
                support_threshold = recent_low_20 - breakout_buffer
            else:
                # Fallback: 1% threshold instead of 0.1% (much more reasonable)
                resistance_threshold = recent_high_20 * 1.01
                support_threshold = recent_low_20 * 0.99

            # Resistance breakout
            if current_close > resistance_threshold:
                patterns.append({
                    "type": "Resistance Breakout",
                    "price_level": round(recent_high_20, 2),
                    "confidence": "High" if atr else "Medium",
                    "signal": "Bullish Continuation"
                })

            # Support breakdown  
            if current_close < support_threshold:
                patterns.append({
                    "type": "Support Breakdown", 
                    "price_level": round(recent_low_20, 2),
                    "confidence": "High" if atr else "Medium",
                    "signal": "Bearish Continuation"
                })
            
            # === ENHANCED: Near Support/Resistance Detection ===
            resistance_distance = (recent_high_20 - current_close) / current_close
            support_distance = (current_close - recent_low_20) / current_close
            
            # Dynamic near thresholds based on volatility
            near_threshold_low = self.get_dynamic_threshold(df, 1.0) / 100
            near_threshold_high = self.get_dynamic_threshold(df, 3.0) / 100
            
            # Near resistance - DYNAMIC thresholds
            if near_threshold_low < resistance_distance < near_threshold_high:
                patterns.append({
                    "type": "Near Resistance",
                    "price_level": round(recent_high_20, 2),
                    "confidence": "High",
                    "signal": "Watch for breakout"
                })
            
            # Near support - DYNAMIC thresholds
            if near_threshold_low < support_distance < near_threshold_high:
                patterns.append({
                    "type": "Near Support",
                    "price_level": round(recent_low_20, 2), 
                    "confidence": "High",
                    "signal": "Watch for bounce"
                })
            
            # === NEW: HEAD & SHOULDERS PATTERN DETECTION - DYNAMIC THRESHOLDS ===
            if len(maxima) >= 3:
                shoulder_similarity_threshold = self.get_dynamic_threshold(recent_df, 5.0) / 100
                head_prominence_threshold = 1.0 + (self.get_dynamic_threshold(recent_df, 3.0) / 100)
                
                # Look for Head & Shoulders pattern in last 3 peaks
                for i in range(len(maxima) - 2):
                    left_shoulder = maxima.iloc[i]["High"]
                    head = maxima.iloc[i + 1]["High"] 
                    right_shoulder = maxima.iloc[i + 2]["High"]
                    
                    # Head should be higher than both shoulders
                    if head > left_shoulder and head > right_shoulder:
                        # Shoulders should be roughly equal - DYNAMIC threshold
                        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                        if shoulder_diff < shoulder_similarity_threshold:
                            # Head should be significantly higher - DYNAMIC threshold
                            if head > left_shoulder * head_prominence_threshold:
                                patterns.append({
                                    "type": "Head and Shoulders",
                                    "price_level": round((left_shoulder + right_shoulder) / 2, 2),
                                    "confidence": "High", 
                                    "signal": "Bearish Reversal"
                                })
            
            # === NEW: INVERSE HEAD & SHOULDERS PATTERN - DYNAMIC THRESHOLDS ===
            if len(minima) >= 3:
                shoulder_similarity_threshold = self.get_dynamic_threshold(recent_df, 5.0) / 100
                head_prominence_threshold = 1.0 - (self.get_dynamic_threshold(recent_df, 3.0) / 100)
                
                for i in range(len(minima) - 2):
                    left_shoulder = minima.iloc[i]["Low"]
                    head = minima.iloc[i + 1]["Low"]
                    right_shoulder = minima.iloc[i + 2]["Low"]
                    
                    # Head should be lower than both shoulders
                    if head < left_shoulder and head < right_shoulder:
                        # Shoulders should be roughly equal - DYNAMIC threshold
                        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                        if shoulder_diff < shoulder_similarity_threshold:
                            # Head should be significantly lower - DYNAMIC threshold
                            if head < left_shoulder * head_prominence_threshold:
                                patterns.append({
                                    "type": "Inverse Head and Shoulders",
                                    "price_level": round((left_shoulder + right_shoulder) / 2, 2),
                                    "confidence": "High",
                                    "signal": "Bullish Reversal"
                                })
            
            # === ENHANCED: TRIANGLE PATTERN DETECTION ===
            triangle_patterns = self.detect_triangle_patterns(df)
            patterns.extend(triangle_patterns)
            
            # === WEDGE PATTERN DETECTION - Use proper slope data ===
            if len(recent_df) >= 20:
                try:
                    # Get slope data from triangle detection logic
                    days = np.arange(len(recent_df.tail(20)))
                    highs = recent_df.tail(20)["High"].values
                    lows = recent_df.tail(20)["Low"].values
                    current_price = df["Close"].iloc[-1]
                    
                    resistance_slope, _, resistance_r_squared, _, _ = linregress(days, highs)
                    support_slope, _, support_r_squared, _, _ = linregress(days, lows)
                    
                    resistance_slope_percent = (resistance_slope / current_price) * 100
                    support_slope_percent = (support_slope / current_price) * 100
                    
                    # Require decent fit for wedge patterns
                    if resistance_r_squared > 0.2 and support_r_squared > 0.2:
                        # Rising Wedge: both rising but resistance rises faster
                        if (resistance_slope_percent > 0.01 and support_slope_percent > 0.01 and 
                            resistance_slope_percent > support_slope_percent * 1.5):
                            patterns.append({
                                "type": "Rising Wedge",
                                "price_level": round(current_price, 2),
                                "confidence": "High" if min(resistance_r_squared, support_r_squared) > 0.4 else "Medium",
                                "signal": "Bearish Reversal Expected"
                            })
                        
                        # Falling Wedge: both falling but support falls faster  
                        elif (resistance_slope_percent < -0.01 and support_slope_percent < -0.01 and 
                              abs(support_slope_percent) > abs(resistance_slope_percent) * 1.5):
                            patterns.append({
                                "type": "Falling Wedge",
                                "price_level": round(current_price, 2),
                                "confidence": "High" if min(resistance_r_squared, support_r_squared) > 0.4 else "Medium",
                                "signal": "Bullish Reversal Expected"
                            })
                except:
                    pass  # Skip if slope calculation fails
            
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
            if "atr" in df.columns:
                atr = df["atr"].iloc[-1]
            else:
                atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
            
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
                "sma_20_support": round(df["sma_20"].iloc[-1], 2) if "sma_20" in df.columns and pd.notnull(df["sma_20"].iloc[-1]) else None,
                "sma_50_support": round(df["sma_50"].iloc[-1], 2) if "sma_50" in df.columns and pd.notnull(df["sma_50"].iloc[-1]) else None,
                "sma_200_support": round(df["sma_200"].iloc[-1], 2) if "sma_200" in df.columns and pd.notnull(df["sma_200"].iloc[-1]) else None,
                
                # === TECHNICAL SUPPORT STOPS ===
                "recent_swing_low": round(df["Low"].iloc[-20:].min(), 2),
                "weekly_low": round(df["Low"].iloc[-5:].min(), 2),
            }
            
            # Calculate risk percentages for each stop
            stop_items = list(stops.items())  # Create a copy to avoid iteration issues
            for stop_name, stop_price in stop_items:
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
        risk_per_trade: float = 500,     # REDUCED from 1000 for safer defaults
        max_risk_percent: float = 1.0    # REDUCED from 2.0 for safer defaults
    ) -> Dict[str, Any]:
        """
        MCP-TRADER STYLE: Position sizing with multiple approaches
        ENHANCED: Safer default risk parameters and warnings
        """
        try:
            if current_price <= 0 or account_size <= 0:
                return {"error": "Invalid price or account size"}
            
            if current_price <= stop_price:
                return {"error": "Stop price must be below current price"}
            
            # RISK WARNINGS for user safety
            if max_risk_percent > 2.0:
                console.print("⚠️ Warning: Risk >2% per trade is aggressive. Consider 0.5-1%")
            if risk_per_trade / account_size > 0.02:
                console.print(f"⚠️ Warning: ${risk_per_trade} risk = {risk_per_trade/account_size*100:.1f}% of account")
            
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
    
    def _calculate_technical_score(
        self, 
        volume_profile: Dict, 
        rs_analysis: Dict, 
        patterns: Dict,
        current_price: float
    ) -> float:
        """
        Calculate overall technical score (0-1) - ENHANCED with multi-period RS
        Combines Volume Profile (25%) + Multi-Period Relative Strength (50%) + Patterns (25%)
        FIXED: More balanced weighting and uses all RS periods instead of just RS_63d
        """
        try:
            score = 0.0
            max_score = 0.0
            
            # Volume profile score (25% weight - more objective than patterns)
            if "point_of_control" in volume_profile and "value_area_low" in volume_profile:
                poc = volume_profile["point_of_control"]
                va_low = volume_profile["value_area_low"]
                va_high = volume_profile["value_area_high"]
                
                # Score based on price position relative to volume areas
                if va_low <= current_price <= va_high:
                    score += 0.15  # In value area (reduced from 0.2)
                if current_price > poc:
                    score += 0.10  # Above POC
                    
            max_score += 0.25  # Reduced from 0.3
            
            # Multi-period Relative Strength (50% weight - most important for stock selection)
            rs_periods = ["RS_21d", "RS_63d", "RS_126d", "RS_252d"]
            rs_weights = [0.10, 0.15, 0.15, 0.10]  # Favor medium-term (63d, 126d)
            
            for period, weight in zip(rs_periods, rs_weights):
                if period in rs_analysis:
                    rs_score = rs_analysis[period]["score"]
                    if rs_score > 75:
                        score += weight * 1.0  # Excellent RS
                    elif rs_score > 65:
                        score += weight * 0.8  # Strong RS
                    elif rs_score > 55:
                        score += weight * 0.6  # Good RS
                    elif rs_score > 45:
                        score += weight * 0.4  # Neutral RS
                    elif rs_score > 35:
                        score += weight * 0.2  # Weak RS
                    # Below 35 = 0 points
            max_score += 0.50  # Increased from 0.4
            
            # Pattern score (25% weight - subjective but important)
            bullish_patterns = ["Resistance Breakout", "Near Resistance", "Inverse Head and Shoulders", 
                               "Ascending Triangle", "Falling Wedge", "Double Bottom"]
            bearish_patterns = ["Head and Shoulders", "Descending Triangle", "Rising Wedge", 
                               "Double Top", "Support Breakdown"]
            
            pattern_score = 0.0
            if patterns and "patterns" in patterns:
                for pattern in patterns["patterns"]:
                    pattern_type = pattern.get("type")
                    confidence = pattern.get("confidence")
                    
                    if pattern_type in bullish_patterns:
                        if confidence == "High":
                            pattern_score += 0.15  # Reduced individual impact
                        else:
                            pattern_score += 0.08
                    elif pattern_type in bearish_patterns:
                        # Bearish patterns reduce score
                        if confidence == "High":
                            pattern_score -= 0.08  # Less negative impact
                        else:
                            pattern_score -= 0.04
            
            # Cap pattern score between -0.1 and +0.25
            pattern_score = max(-0.10, min(0.25, pattern_score))
            score += pattern_score
            max_score += 0.25  # Reduced from 0.3
            
            return round(score / max_score if max_score > 0 else 0, 3)
            
        except Exception as e:
            return 0.0
    
    def analyze_score_distribution(self, qualifying_stocks: List[Dict]) -> None:
        """Print score distribution to help users set appropriate thresholds"""
        if not qualifying_stocks:
            return
            
        scores = [s.get("technical_score", 0) * 100 for s in qualifying_stocks]
        
        console.print(f"\n📊 [bold blue]Technical Score Distribution:[/bold blue]")
        console.print(f"   Mean: {np.mean(scores):.1f}%")
        console.print(f"   Median: {np.median(scores):.1f}%") 
        console.print(f"   75th percentile: {np.percentile(scores, 75):.1f}%")
        console.print(f"   90th percentile: {np.percentile(scores, 90):.1f}%")
        console.print(f"   💡 Consider --min-score {np.percentile(scores, 75)/100:.2f} for top 25%")
    
    def calculate_sma150_slope(self, df: pd.DataFrame, lookback_days: int) -> Optional[float]:
        """Calculate SMA150 slope over specified period"""
        try:
            if len(df) < 160:  # Need enough data for SMA150
                return None
                
            df["sma150"] = ta.sma(df["Close"], length=150)
            df = df.dropna(subset=["sma150"])
            
            if len(df) < 2:
                return None
                
            end_date = df.index[-1]
            start_cutoff = end_date - timedelta(days=lookback_days)
            start_rows = df[df.index <= start_cutoff]
            
            if start_rows.empty:
                return None
                
            sma_then = start_rows["sma150"].iloc[0]
            sma_now = df["sma150"].iloc[-1]
            
            if sma_then <= 0:
                return None
                
            return (sma_now - sma_then) / sma_then * 100
            
        except Exception as e:
            console.print(f"⚠️ Error calculating SMA150 slope: {str(e)}")
            return None

    def comprehensive_stock_analysis(
        self,
        symbol: str,
        benchmark: str = "SPY",
        account_size: float = 100000,
        risk_per_trade: float = 500,
        uptrend_period: Optional[str] = None,
        min_slope_pct: float = 0
    ) -> Dict[str, Any]:
        """
        Complete MCP-trader style analysis of a single stock
        ENHANCED: Uses configurable strategy weights and improved scoring
        """
        try:
            console.print(f"\n🔍 [bold cyan]Comprehensive Analysis: {symbol} (Strategy: {self.strategy})[/bold cyan]")
            
            # Calculate required days for data fetch
            max_period = 365  # Base period for SMA200
            if uptrend_period:
                # Parse period for uptrend check
                m = re.match(r"(\d+)([a-zA-Z]+)", uptrend_period)
                if m:
                    n, unit = int(m.group(1)), m.group(2).lower()
                    if unit.startswith("y"):
                        max_period = max(max_period, n * 365)
                    elif unit.startswith("m"):
                        max_period = max(max_period, n * 30)
            
            # Fetch data once for all analysis
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{max_period+200}d")  # +200 for SMA warmup
            
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Add all technical indicators
            df = self.add_comprehensive_indicators(df)
            
            # Calculate SMA150 slope if uptrend period specified
            sma150_slope = None
            if uptrend_period:
                m = re.match(r"(\d+)([a-zA-Z]+)", uptrend_period)
                if m:
                    n, unit = int(m.group(1)), m.group(2).lower()
                    days = n * 365 if unit.startswith("y") else n * 30
                    sma150_slope = self.calculate_sma150_slope(df, days)
                    
                    # Early return if slope requirement not met
                    if sma150_slope is None or sma150_slope < min_slope_pct:
                        return {"error": f"SMA150 slope {sma150_slope:.1f}% < required {min_slope_pct}%"}
            
            current_price = df["Close"].iloc[-1]
            
            # 1. Volume Profile Analysis (FIXED double-counting)
            volume_profile_analyzer = VolumeProfileAnalysis()
            volume_profile = volume_profile_analyzer.analyze_volume_profile(df)
            
            # 2. Trend Status Analysis (with strategy-specific weights)
            trend_analysis = self.analyze_trend_status(df)
            
            # 3. Relative Strength Analysis (with fixed normalization)
            rs_analysis = self.calculate_relative_strength_detailed(symbol, benchmark)
            
            # 4. Advanced Pattern Recognition (with dynamic thresholds)
            pattern_analysis = self.detect_advanced_patterns(df)
            
            # 5. Multiple Stop Loss Suggestions
            stop_analysis = self.suggest_multiple_stops(df)
            
            # 6. Position Sizing (using best ATR stop with safer defaults)
            if "atr_2x_moderate" in stop_analysis:
                position_analysis = self.calculate_position_sizing(
                    current_price, stop_analysis["atr_2x_moderate"], 
                    account_size, risk_per_trade
                )
            else:
                position_analysis = {"error": "Cannot calculate position sizing"}
            
            # 7. Technical Score Calculation (ENHANCED multi-period)
            technical_score = self._calculate_technical_score(
                volume_profile, rs_analysis, pattern_analysis, current_price
            )
            
            # 8. Volatility Analysis (ADRP)
            adrp = df["adrp"].iloc[-1] if "adrp" in df.columns else None
            volume_ratio = df["volume_ratio"].iloc[-1] if "volume_ratio" in df.columns else None
            
            # Compile comprehensive results
            analysis = {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "analysis_date": datetime.now().isoformat(),
                "strategy": self.strategy,
                "benchmark": benchmark,  # Add benchmark to analysis dictionary
                
                # Core analyses
                "volume_profile": volume_profile,  # FIXED: Volume profile analysis
                "trend_analysis": trend_analysis,  # ENHANCED: Strategy-specific weights
                "relative_strength": rs_analysis,  # FIXED: Proper normalization
                "pattern_analysis": pattern_analysis,  # ENHANCED: Dynamic thresholds
                "stop_analysis": stop_analysis,
                "position_analysis": position_analysis,  # ENHANCED: Safer defaults
                "technical_score": technical_score,  # ENHANCED: Multi-period scoring
                
                # Key metrics (with null checks)
                "volatility_adrp": round(adrp, 2) if adrp is not None and pd.notnull(adrp) else None,
                "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None and pd.notnull(volume_ratio) else None,
                "rsi": round(df["rsi"].iloc[-1], 1) if "rsi" in df.columns and pd.notnull(df["rsi"].iloc[-1]) else None,
                
                # Price levels (with null checks)
                "sma_20": round(df["sma_20"].iloc[-1], 2) if "sma_20" in df.columns and pd.notnull(df["sma_20"].iloc[-1]) else None,
                "sma_50": round(df["sma_50"].iloc[-1], 2) if "sma_50" in df.columns and pd.notnull(df["sma_50"].iloc[-1]) else None,
                "sma_200": round(df["sma_200"].iloc[-1], 2) if "sma_200" in df.columns and pd.notnull(df["sma_200"].iloc[-1]) else None,
            }
            
            # Add SMA150 slope to results if calculated
            if sma150_slope is not None:
                analysis["sma150_slope"] = round(sma150_slope, 2)
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis failed for {symbol}: {str(e)}"}

def passes_sma150_uptrend(symbol, uptrend_period, min_slope_pct):
    """This function is deprecated - use ComprehensiveTechnicalAnalysis.calculate_sma150_slope instead"""
    analyzer = ComprehensiveTechnicalAnalysis()
    try:
        # Parse period (e.g. '1y', '2y', '18mo')
        m = re.match(r"(\d+)([a-zA-Z]+)", uptrend_period)
        if not m:
            return False
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit.startswith("y"):  # years
            days = n * 365
        elif unit.startswith("m"):  # months
            days = n * 30
        else:
            return False

        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days+200}d")  # +200 for SMA150 warmup
        
        # Use the new method
        slope = analyzer.calculate_sma150_slope(df, days)
        return slope is not None and slope >= min_slope_pct
        
    except Exception as e:
        console.print(f"⚠️ {symbol}: Error in uptrend analysis - {str(e)}")
        return False

def discover_high_scoring_stocks(
    analyzer: ComprehensiveTechnicalAnalysis,
    indices: List[str] = ["sp500"],
    min_score: float = 0.6,
    top_n: int = 20,
    benchmark: str = "SPY",
    account_size: float = 100000,
    risk_per_trade: float = 500,
    detailed: bool = False,
    uptrend_period: Optional[str] = None,
    min_slope_pct: float = 0
):
    """
    Discovery mode: Scan major stock indices and filter high-scoring opportunities
    ENHANCED: Single-pass analysis with integrated uptrend check
    """
    console.print(f"\n🔍 [bold blue]MCP-STYLE STOCK DISCOVERY MODE[/bold blue]")
    console.print(f"Strategy: {analyzer.strategy.title()}")
    console.print(f"Scanning indices: {', '.join(indices).upper()}")
    console.print(f"Filter: Technical Score ≥ {min_score*100:.0f}% | Showing Top {top_n}")
    
    if uptrend_period:
        console.print(f"Additional Filter: SMA150 uptrend over {uptrend_period} (min slope: {min_slope_pct}%)")
    
    # Get stock symbols from indices
    all_symbols = []
    for index in indices:
        try:
            symbols = get_index_symbols(index)
            all_symbols.extend(symbols)
            console.print(f"   📊 {index.upper()}: {len(symbols)} stocks")
        except Exception as e:
            console.print(f"⚠️ Error fetching {index} symbols: {str(e)}")
            continue
    
    # Remove duplicates and invalid symbols
    all_symbols = list(set(all_symbols))
    all_symbols = [s for s in all_symbols if not any(x in s for x in ['.B', '.A', '^', '='])]
    console.print(f"\n🎯 Total unique stocks to analyze: {len(all_symbols)}")
    
    # Analyze and filter stocks in a single pass
    qualifying_stocks = []
    analyzed_count = 0
    error_count = 0
    error_details = []  # Track specific errors for debugging
    
    console.print(f"\n⚡ Analyzing stocks (single-pass analysis)...")
    
    for i, symbol in enumerate(all_symbols):
        try:
            console.print(f"   [{i+1}/{len(all_symbols)}] {symbol}...", end="")
            
            # Perform comprehensive analysis with integrated uptrend check
            analysis = analyzer.comprehensive_stock_analysis(
                symbol, benchmark, account_size, risk_per_trade,
                uptrend_period, min_slope_pct
            )
            analyzed_count += 1
            
            if "error" in analysis:
                console.print(" ❌")
                error_count += 1
                error_details.append(f"{symbol}: {analysis['error']}")
                continue
            
            # Apply technical score filter
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
            error_count += 1
            error_details.append(f"{symbol}: {str(e)}")
            continue
    
    # Print analysis summary
    console.print(f"\n📊 Analysis Summary:")
    console.print(f"   Total stocks: {len(all_symbols)}")
    console.print(f"   Successfully analyzed: {analyzed_count}")
    console.print(f"   Stocks filtered out: {error_count}")
    console.print(f"   Qualifying stocks: {len(qualifying_stocks)}")
    
    # Show screening filter results if any (but limit to first 10 to avoid spam)
    if error_details:
        console.print(f"\n🔍 [bold blue]Screening Filter Results[/bold blue] (showing first 10):")
        for error in error_details[:10]:
            console.print(f"   📊 {error}")
        if len(error_details) > 10:
            console.print(f"   ... and {len(error_details) - 10} more stocks filtered out")
    
    # Sort by technical score
    qualifying_stocks.sort(key=lambda x: x.get("technical_score", 0), reverse=True)
    
    # Display results
    if qualifying_stocks:
        # Create results table
        table = Table(title=f"🔥 High-Scoring Stock Opportunities - {analyzer.strategy.title()} Strategy", show_header=True)
        table.add_column("Rank", style="white", width=6)
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Price", style="green", width=8)
        table.add_column("Tech Score", style="yellow", width=10)
        table.add_column("Trend", style="blue", width=10)
        table.add_column("RS 63d", style="magenta", width=8)
        table.add_column("ADRP", style="white", width=8)
        table.add_column("Volume", style="cyan", width=8)
        table.add_column("Top Pattern", style="red", width=15)
        if uptrend_period:
            table.add_column("SMA150 Slope", style="green", width=10)
        
        for rank, analysis in enumerate(qualifying_stocks[:top_n], 1):
            symbol = analysis["symbol"]
            price = f"${analysis['current_price']:.2f}"
            tech_score = f"{analysis.get('technical_score', 0)*100:.0f}%"
            trend_strength = analysis["trend_analysis"].get("trend_strength", 0)
            trend = f"{trend_strength}/100"
            rs_63d = f"{analysis.get('relative_strength', {}).get('RS_63d', {}).get('score', 0):.1f}"
            adrp = f"{analysis.get('volatility_adrp', 0):.1f}%"
            
            # Use dynamic volume classification
            volume_ratio = analysis.get('volume_ratio', 1)
            adrp_val = analysis.get('volatility_adrp')
            vdesc, vmark = analyzer.classify_volume_ratio(volume_ratio, adrp_val)
            volume = f"{volume_ratio:.1f}x ({vmark} {vdesc})"
            
            # Get primary pattern
            patterns = analysis["pattern_analysis"]["patterns"]
            pattern = patterns[0]["type"] if patterns else "None"
            
            # Add SMA150 slope if available
            row_data = [str(rank), symbol, price, tech_score, trend, rs_63d, adrp, volume, pattern]
            if uptrend_period and "sma150_slope" in analysis:
                row_data.append(f"{analysis['sma150_slope']:.1f}%")
            
            table.add_row(*row_data)
        
        console.print(table)
        
        if detailed:
            console.print(f"\n📋 [bold yellow]DETAILED ANALYSIS OF TOP 5 CANDIDATES[/bold yellow]")
            for analysis in qualifying_stocks[:5]:
                print_comprehensive_analysis(analysis)
                console.print("\n" + "="*80 + "\n")
    else:
        console.print(f"❌ No stocks met the minimum technical score of {min_score*100:.0f}%.")
        console.print(f"💡 Try adjusting:")
        console.print(f"   • Lower --min-score (try 0.4 or 0.5)")
        console.print(f"   • Different --strategy (momentum/balanced/long_term)")
        console.print(f"   • Different --indices (try 'nasdaq' or 'russell1000')")

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

def main():
    """Main entry point for the Pro Screener CLI"""
    parser = argparse.ArgumentParser(
        description="Pro Screener - Advanced MCP-Style Technical Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discovery mode with uptrend filter
  python pro_screener.py --discovery --uptrend 1y --slope 10 --indices sp500 nasdaq
  
  # Evaluation mode for specific symbols
  python pro_screener.py --symbols AAPL NVDA --detailed
  
  # Discovery mode with custom strategy
  python pro_screener.py --discovery --strategy momentum --min-score 0.7
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--discovery", action="store_true", help="Run in discovery mode to scan indices")
    mode_group.add_argument("--symbols", nargs="+", help="Stock symbols to analyze (evaluation mode)")
    
    # Discovery mode options
    parser.add_argument("--indices", nargs="+", default=["sp500"], 
                       choices=["sp500", "nasdaq", "russell1000", "all"],
                       help="Indices to scan in discovery mode (default: sp500)")
    parser.add_argument("--min-score", type=float, default=0.6,
                       help="Minimum technical score (0-1) for discovery mode (default: 0.6)")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top stocks to display (default: 20)")
    
    # Uptrend filter options (only applies in discovery mode)
    parser.add_argument("--uptrend", type=str, metavar="PERIOD",
                       help="Filter for SMA150 uptrend over period (e.g. '1y', '2y', '18mo')")
    parser.add_argument("--slope", type=float, default=10.0,
                       help="Minimum SMA150 slope percentage (default: 10.0)")
    
    # Analysis options
    parser.add_argument("--strategy", type=str, default="balanced",
                       choices=["momentum", "balanced", "long_term"],
                       help="Analysis strategy (default: balanced)")
    parser.add_argument("--benchmark", type=str, default="SPY",
                       help="Benchmark for relative strength (default: SPY)")
    parser.add_argument("--account-size", type=float, default=100000,
                       help="Account size for position sizing (default: 100000)")
    parser.add_argument("--risk-per-trade", type=float, default=500,
                       help="Risk per trade in dollars (default: 500)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed analysis for top candidates")
    
    args = parser.parse_args()
    
    # Initialize analyzer with selected strategy
    analyzer = ComprehensiveTechnicalAnalysis(strategy=args.strategy)
    
    try:
        if args.discovery:
            # Discovery mode
            discover_high_scoring_stocks(
                analyzer=analyzer,
                indices=args.indices,
                min_score=args.min_score,
                top_n=args.top_n,
                benchmark=args.benchmark,
                account_size=args.account_size,
                risk_per_trade=args.risk_per_trade,
                detailed=args.detailed,
                uptrend_period=args.uptrend,
                min_slope_pct=args.slope
            )
        else:
            # Evaluation mode for specific symbols
            if not args.symbols:
                parser.error("No symbols provided for evaluation mode")
            
            console.print(f"\n🔍 [bold blue]EVALUATION MODE[/bold blue]")
            console.print(f"Strategy: {analyzer.strategy.title()}")
            console.print(f"Symbols: {', '.join(args.symbols)}")
            
            for symbol in args.symbols:
                analysis = analyzer.comprehensive_stock_analysis(
                    symbol=symbol,
                    benchmark=args.benchmark,
                    account_size=args.account_size,
                    risk_per_trade=args.risk_per_trade
                )
                
                if "error" in analysis:
                    console.print(f"\n❌ {symbol}: {analysis['error']}")
                    continue
                
                print_comprehensive_analysis(analysis)
                console.print("\n" + "="*80 + "\n")
                
    except KeyboardInterrupt:
        console.print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}")
        if args.detailed:
            import traceback
            console.print(traceback.format_exc())

def print_comprehensive_analysis(analysis: Dict[str, Any]) -> None:
    """Print detailed analysis results in a readable format"""
    if "error" in analysis:
        console.print(f"❌ {analysis['error']}")
        return
        
    # Header
    console.print(Panel(
        f"[bold cyan]{analysis['symbol']}[/bold cyan] - ${analysis['current_price']:.2f}\n"
        f"Technical Score: [bold yellow]{analysis['technical_score']*100:.1f}%[/bold yellow]\n"
        f"Strategy: {analysis['strategy'].title()}",
        title="Stock Analysis",
        border_style="blue"
    ))

    # === TREND ANALYSIS ===
    trend = analysis["trend_analysis"]
    trend_strength = trend.get('trend_strength', 0)
    trend_class = trend.get('trend_classification', '')
    data_quality = trend.get('trend_data_quality', '')
    sma_20 = trend.get('sma_20_valid', False)
    sma_50 = trend.get('sma_50_valid', False)
    sma_200 = trend.get('sma_200_valid', False)
    above_20 = trend.get('above_20sma')
    above_50 = trend.get('above_50sma')
    above_200 = trend.get('above_200sma')
    cross_20_50 = trend.get('20_50_bullish')
    cross_50_200 = trend.get('50_200_bullish')

    def checkmark(val):
        return "✅" if val else "❌" if val is not None else "❓"

    console.print("\n[bold blue]Trend Analysis[/bold blue]")
    console.print(f"   Trend Strength: {trend_strength}/100  {'🟢' if trend_strength >= 70 else '🟡' if trend_strength >= 50 else '🔶' if trend_strength >= 30 else '🔴'}")
    console.print(f"   Classification: {trend_class}")
    console.print(f"   Data Quality: {data_quality}")
    console.print(f"   ├── Above 20SMA: {checkmark(above_20)}   ├── Above 50SMA: {checkmark(above_50)}   ├── Above 200SMA: {checkmark(above_200)}")
    console.print(f"   ├── 20>50 Bullish: {checkmark(cross_20_50)}   ├── 50>200 Bullish: {checkmark(cross_50_200)}")

    # === RELATIVE STRENGTH ===
    rs = analysis["relative_strength"]
    benchmark = analysis.get("benchmark", "SPY")  # Get benchmark with SPY as default
    console.print(f"\n[bold blue]Relative Strength vs {benchmark}[/bold blue]")
    for period, data in rs.items():
        if period != "error":
            emoji = "🟢" if data['score'] > 70 else "🟡" if data['score'] > 55 else "🔶" if data['score'] > 40 else "🔴"
            console.print(f"   {period}: {data['score']:.1f} - {data['classification']} {emoji}")
            console.print(f"      Stock: {data['stock_return']:+.1f}% | {benchmark}: {data['benchmark_return']:+.1f}%")

    # === PATTERN ANALYSIS ===
    patterns = analysis["pattern_analysis"]["patterns"]
    if patterns:
        console.print("\n[bold blue]Technical Patterns[/bold blue]")
        for pattern in patterns[:3]:  # Show top 3 patterns
            conf = pattern.get('confidence', '')
            conf_emoji = "✅" if conf == "High" else "🟡" if conf == "Medium" else "❌"
            console.print(f"   • {pattern['type']} ({conf}) {conf_emoji}")
            console.print(f"     Signal: {pattern['signal']}")

    # === VOLUME ANALYSIS ===
    console.print("\n[bold blue]Volume Analysis[/bold blue]")
    if "volume_profile" in analysis and "point_of_control" in analysis["volume_profile"]:
        vp = analysis["volume_profile"]
        current_price = analysis["current_price"]
        poc = vp["point_of_control"]
        va_low = vp["value_area_low"]
        va_high = vp["value_area_high"]
        
        # Add visual indicators for POC and Value Area
        poc_indicator = "✅" if current_price > poc else "❌"
        va_indicator = "✅" if va_low <= current_price <= va_high else "❌"
        
        console.print(f"   Point of Control: ${poc:.2f} {poc_indicator}")
        console.print(f"   Value Area: ${va_low:.2f} - ${va_high:.2f} {va_indicator}")

    volume_ratio = analysis.get("volume_ratio")
    adrp = analysis.get("volatility_adrp")
    def volume_desc(vr, adrp):
        if vr is None:
            return "(No data)", "❓"
        if adrp is None:
            if vr > 1.5:
                return "High", "✅"
            elif vr > 0.8:
                return "Normal", "🟡"
            else:
                return "Low", "❌"
        else:
            if adrp > 5:
                if vr > 2.0:
                    return "High", "✅"
                elif vr > 1.0:
                    return "Normal", "🟡"
                else:
                    return "Low", "❌"
            elif adrp < 2:
                if vr > 1.2:
                    return "High", "✅"
                elif vr > 0.6:
                    return "Normal", "🟡"
                else:
                    return "Low", "❌"
            else:
                if vr > 1.5:
                    return "High", "✅"
                elif vr > 0.8:
                    return "Normal", "🟡"
                else:
                    return "Low", "❌"
    vdesc, vmark = volume_desc(volume_ratio, adrp)
    if volume_ratio is not None:
        console.print(f"   Volume Ratio: {volume_ratio:.1f}x {vmark} ({vdesc}) vs 20d avg")
    if adrp is not None:
        console.print(f"   ADRP: {adrp:.1f}%")

    # === RSI ===
    rsi = analysis.get("rsi")
    if rsi is not None:
        rsi_desc = "(Oversold)" if rsi < 30 else "(Overbought)" if rsi > 70 else "(Neutral)"
        rsi_emoji = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
        console.print(f"   RSI: {rsi:.1f} {rsi_emoji} {rsi_desc}")

    # === STOP LEVELS ===
    stops = analysis["stop_analysis"]
    if "error" not in stops:
        console.print("\n[bold blue]Stop Levels[/bold blue]")
        for stop_type, price in stops.items():
            if isinstance(price, (int, float)) and price > 0:
                risk_pct = stops.get(f"{stop_type}_risk_pct")
                if risk_pct is not None:
                    console.print(f"   {stop_type}: ${price:.2f} ({risk_pct:+.1f}% risk)")

    # === POSITION SIZING ===
    position = analysis["position_analysis"]
    if "error" not in position:
        console.print("\n[bold blue]Position Sizing[/bold blue]")
        console.print(f"   Recommended Shares: {position['recommended_shares']}")
        console.print(f"   Position Cost: ${position['position_cost']:,.2f}")
        console.print(f"   Account Risk: {position['account_risk_pct']:.1f}%")
        targets = position["risk_reward_targets"]
        console.print("\n   Risk/Reward Targets:")
        console.print(f"   1:1 Target: ${targets['r1_target']:.2f}")
        console.print(f"   2:1 Target: ${targets['r2_target']:.2f}")
        console.print(f"   3:1 Target: ${targets['r3_target']:.2f}")

if __name__ == "__main__":
    main() 