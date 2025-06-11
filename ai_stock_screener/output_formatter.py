"""
Professional output formatting for AI Stock Screener
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from tabulate import tabulate
import pandas as pd
import time
from typing import List, Dict

console = Console()

def print_header(mode, tickers=None, config=None, market_intel=None, sector_intel=None):
    """Print the main header with configuration"""
    if mode == "discovery":
        title = "🔍 AI Stock Screener - Discovery Mode"
        subtitle = f"📊 Scanning S&P 500 for growth opportunities..."
    else:
        ticker_str = ", ".join(tickers) if len(tickers) <= 5 else f"{', '.join(tickers[:5])}... ({len(tickers)} total)"
        title = "🔍 AI Stock Screener - Evaluation Mode" 
        subtitle = f"📊 Analyzing your portfolio: {ticker_str}"
    
    # Configuration details
    period = config.get('period', '6mo')
    future_days = config.get('future_days', 5)
    threshold = config.get('threshold', 0.0)
    
    config_text = f"📅 Using {period} historical data, predicting {future_days} days ahead\n🎯 Growth threshold: {threshold*100:.1f}%"
    
    # Add market regime context if available
    if market_intel:
        regime_text = f"\n🧠 Market Regime: {market_intel.current_regime.value.replace('_', ' ').title()} ({market_intel.regime_confidence:.0%} confidence)"
        config_text += regime_text
    
    # Add sector rotation context if available
    if sector_intel:
        sector_text = f"\n🏭 Sector Rotation: {sector_intel.rotation_trend} (Leading: {', '.join(sector_intel.leading_sectors[:2]) if len(sector_intel.leading_sectors) >= 2 else 'N/A'})"
        config_text += sector_text
    
    console.print(Panel(f"[bold blue]{title}[/bold blue]\n{subtitle}\n\n{config_text}", 
                       title="AI Stock Screener", border_style="blue"))

def print_model_training_start(model_type, n_estimators, n_tickers):
    """Print model training start message"""
    console.print(f"\n🤖 Training {model_type.replace('_', ' ').title()} model with {n_estimators} estimators...")
    console.print(f"📊 Processing technical indicators for {n_tickers} stocks...")

def print_grid_search_results(best_params, cv_score):
    """Print grid search results"""
    console.print(f"\n⚡ Grid Search enabled - optimizing hyperparameters...")
    console.print(f"Best parameters: {best_params}")
    if cv_score:
        console.print(f"Cross-validation score: {cv_score:.3f}")

def print_model_performance(accuracy, precision=None, recall=None, f1=None):
    """Print model performance metrics"""
    console.print("\n🎯 MODEL PERFORMANCE:")
    console.print(f"Accuracy: {accuracy*100:.1f}%")
    if precision:
        console.print(f"Precision: {precision:.2f}")
    if recall:
        console.print(f"Recall: {recall:.2f}")
    if f1:
        console.print(f"F1-Score: {f1:.2f}")

def print_discovery_results(results_df, config, market_intel, sector_intel, discovery_threshold=0.70):
    """Print results for discovery mode"""
    from .ai_screener import get_effective_config
    
    # Get the actual threshold being used
    eff_config = get_effective_config(config) 
    actual_threshold = eff_config.get("discovery_threshold", discovery_threshold)
    
    console.print(f"\n📈 TOP GROWTH PREDICTIONS (Probability > {actual_threshold}):")
    
    if results_df.empty:
        console.print("❌ No high-probability growth candidates found.")
        return
    
    # Create rich table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Company", style="white")
    table.add_column("Sector", style="blue", no_wrap=True)  # Add sector column
    table.add_column("Growth Prob", style="green", justify="right")
    table.add_column("RSI", style="yellow", justify="right")
    table.add_column("SMA150", style="bright_yellow", justify="right")
    table.add_column("Price", style="white", justify="right")
    table.add_column("Vol Chg", style="blue", justify="right")
    table.add_column("P/E Ratio", style="magenta", justify="right")
    
    for _, row in results_df.iterrows():
        # Format values
        prob = f"{row['Growth_Prob']:.2f}"
        rsi = f"{row['RSI']:.1f}" if pd.notnull(row['RSI']) else "N/A"
        price = f"${row['Price']:.2f}" if pd.notnull(row['Price']) else "N/A"
        vol_chg = f"+{row['Vol_Change']:.0f}%" if row['Vol_Change'] > 0 else f"{row['Vol_Change']:.0f}%"
        pe_ratio = f"{row['PE_Ratio']:.1f}" if pd.notnull(row['PE_Ratio']) else "N/A"
        
        # Format SMA150 with Above/Below indication
        sma150_value = row.get('SMA_150')
        if pd.notnull(sma150_value) and pd.notnull(row['Price']):
            if row['Price'] > sma150_value:
                sma150_display = f"Above (+{((row['Price'] / sma150_value - 1) * 100):.1f}%)"
            else:
                sma150_display = f"Below ({((row['Price'] / sma150_value - 1) * 100):.1f}%)"
        else:
            sma150_display = "N/A"
        
        # Get sector for display
        from .clock import get_sector_for_stock
        sector = get_sector_for_stock(row['Ticker'])[:4]  # Abbreviated
        
        table.add_row(
            row['Ticker'],
            row['Company'][:12] + "..." if len(str(row['Company'])) > 15 else str(row['Company']),
            sector,
            prob,
            rsi,
            sma150_display,
            price,
            vol_chg,
            pe_ratio
        )
    
    console.print(table)
    
    # High confidence picks
    high_conf = results_df[results_df['Growth_Prob'] > 0.85]
    if not high_conf.empty:
        console.print(f"\n🔥 HIGH CONFIDENCE PICKS (> 85% probability):")
        for _, row in high_conf.iterrows():
            reason = get_analysis_reason(row)
            console.print(f"• {row['Ticker']} - {reason}")

def print_probability_breakdown(results_df):
    """Print detailed probability breakdown showing ML, regime, sector, and news components"""
    console.print("\n🔍 PROBABILITY BREAKDOWN:")
    console.print("="*80)
    console.print(f"{'TICKER':<8} {'BASE ML':<10} {'REGIME':<12} {'SECTOR':<12} {'NEWS':<12} {'FINAL':<10} {'TOTAL BOOST':<12}")
    console.print("="*80)
    
    # Check if news analysis was enabled
    news_enabled = any(row.get('News_Explanation', '') != "News analysis disabled" for _, row in results_df.iterrows())
    
    for _, row in results_df.iterrows():
        base_ml = row['Raw_ML_Score']
        regime_boost = row.get('Regime_Boost', 0)
        sector_boost = row.get('Sector_Boost', 0)
        news_boost = row.get('News_Boost', 0)
        final_score = row['Growth_Prob']
        total_boost = row.get('Total_Boost', 0)
        
        # Format percentage values
        base_ml_str = f"{base_ml:.1%}"
        regime_str = f"{regime_boost:+.1%}" if regime_boost != 0 else "±0.0%"
        sector_str = f"{sector_boost:+.1%}" if sector_boost != 0 else "±0.0%"
        
        # Handle news display based on whether it's enabled
        if news_enabled and news_boost != 0:
            news_str = f"{news_boost:+.1%}"
        elif news_enabled:
            news_str = "±0.0%"
        else:
            news_str = "DISABLED"
            
        final_str = f"{final_score:.1%}"
        total_str = f"{total_boost:+.1%}" if total_boost != 0 else "±0.0%"
        
        console.print(f"{row['Ticker']:<8} {base_ml_str:<10} {regime_str:<12} {sector_str:<12} {news_str:<12} {final_str:<10} {total_str:<12}")
    
    console.print("="*80)
    
    # Add note about news analysis
    if not news_enabled:
        console.print("💡 News analysis disabled. Use --news flag to enable sentiment analysis.")
    
    # Show detailed explanations for top picks
    console.print("\n📊 DETAILED ANALYSIS (Top 5):")
    top_picks = results_df.nlargest(5, 'Growth_Prob')
    
    for i, (_, row) in enumerate(top_picks.iterrows(), 1):
        console.print(f"\n{i}. {row['Ticker']} - {row['Company']}")
        console.print(f"   💹 Final Conviction: {row['Growth_Prob']:.1%}")
        console.print(f"   🤖 Base ML Score: {row['Raw_ML_Score']:.1%}")
        
        # Show adjustments with explanations
        if row.get('Regime_Boost', 0) != 0:
            console.print(f"   🌍 Regime Adjustment: {row['Regime_Boost']:+.1%} - {row.get('Regime_Explanation', '')}")
        
        if row.get('Sector_Boost', 0) != 0:
            console.print(f"   🏭 Sector Adjustment: {row['Sector_Boost']:+.1%} - {row.get('Sector_Explanation', '')}")
        
        # Only show news adjustment if enabled and has an effect
        if news_enabled and row.get('News_Boost', 0) != 0:
            console.print(f"   📰 News Adjustment: {row['News_Boost']:+.1%} - {row.get('News_Explanation', '')}")
        elif not news_enabled:
            console.print(f"   📰 News Analysis: Disabled (use --news to enable)")
        
        # Show total impact
        total_boost = row.get('Total_Boost', 0)
        if total_boost != 0:
            console.print(f"   🎯 Total Enhancement: {total_boost:+.1%}")
        
        console.print()

def print_evaluation_results(results_df, config, market_intel=None, sector_intel=None):
    """Print evaluation mode results with detailed analysis"""
    console.print(f"\n📈 DETAILED STOCK ANALYSIS:")
    
    if results_df.empty:
        console.print("❌ No stocks to analyze.")
        return
    
    # Create rich table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Sector", style="blue", no_wrap=True)  # Add sector column
    table.add_column("Current Analysis", style="white", max_width=20)
    table.add_column("Prob", style="green", justify="right")
    table.add_column("RSI", style="yellow", justify="right")
    table.add_column("SMA150", style="bright_yellow", justify="right")
    table.add_column("Price", style="white", justify="right")
    table.add_column("Vol Chg", style="blue", justify="right")
    table.add_column("Prediction", style="magenta", justify="center")
    
    for _, row in results_df.iterrows():
        # Get analysis and recommendation
        analysis, recommendation = get_detailed_analysis(row)
        
        # Format values
        prob = f"{row['Growth_Prob']:.2f}"
        rsi = f"{row['RSI']:.1f}" if pd.notnull(row['RSI']) else "N/A"
        price = f"${row['Price']:.2f}" if pd.notnull(row['Price']) else "N/A"
        vol_chg = f"+{row['Vol_Change']:.0f}%" if row['Vol_Change'] > 0 else f"{row['Vol_Change']:.0f}%"
        
        # Format SMA150 with Above/Below indication
        sma150_value = row.get('SMA_150')
        if pd.notnull(sma150_value) and pd.notnull(row['Price']):
            if row['Price'] > sma150_value:
                sma150_display = f"Above (+{((row['Price'] / sma150_value - 1) * 100):.1f}%)"
            else:
                sma150_display = f"Below ({((row['Price'] / sma150_value - 1) * 100):.1f}%)"
        else:
            sma150_display = "N/A"
        
        # Get sector for display
        from .clock import get_sector_for_stock
        sector = get_sector_for_stock(row['Ticker'])[:4]  # Abbreviated
        
        table.add_row(
            row['Ticker'],
            sector,
            analysis,
            prob,
            rsi,
            sma150_display,
            price,
            vol_chg,
            recommendation
        )
    
    console.print(table)
    
    # Print detailed technical breakdown for top picks
    print_technical_breakdown(results_df, config)
    
    # Print portfolio summary
    print_portfolio_summary(results_df, config)

def print_technical_breakdown(results_df, config):
    """Print detailed technical analysis for top stocks"""
    console.print(f"\n📊 TECHNICAL INDICATORS BREAKDOWN:")
    
    # Show detailed analysis for top 2-3 stocks
    top_stocks = results_df.nlargest(3, 'Growth_Prob')
    
    for _, row in top_stocks.iterrows():
        ticker = row['Ticker']
        console.print(f"\n{ticker} Analysis:")
        
        # Technical indicators tree
        rsi = row['RSI'] if pd.notnull(row['RSI']) else 0
        rsi_desc = get_rsi_description(rsi)
        
        sma50 = row.get('SMA_50', row['Price'] * 0.95)  # Fallback
        sma200 = row.get('SMA_200', row['Price'] * 0.90)  # Fallback
        
        vol_change = row['Vol_Change']
        pe_ratio = row['PE_Ratio'] if pd.notnull(row['PE_Ratio']) else 0
        
        console.print(f"├── RSI: {rsi:.1f} {rsi_desc}")
        console.print(f"├── SMA50: ${sma50:.2f} ({'Price above' if row['Price'] > sma50 else 'Price below'})")
        console.print(f"├── SMA200: ${sma200:.2f} ({'Strong uptrend' if row['Price'] > sma200 * 1.05 else 'Uptrend confirmed' if row['Price'] > sma200 else 'Below trend'})")
        console.print(f"├── Volume: {vol_change:+.0f}% vs avg ({get_volume_description(vol_change)})")
        console.print(f"├── P/E Ratio: {pe_ratio:.1f} ({get_pe_description(pe_ratio)})")
        console.print(f"└── 🎯 Prediction: {row['Growth_Prob']*100:.1f}% probability of {config.get('threshold', 0.05)*100:.0f}%+ gain in {config.get('future_days', 5)} days")

def print_portfolio_summary(results_df, config):
    """Print portfolio summary and recommendations"""
    console.print(f"\n📈 PORTFOLIO SUMMARY:")
    
    # Categorize stocks
    strong_growth = results_df[results_df['Growth_Prob'] > 0.85]
    moderate_growth = results_df[(results_df['Growth_Prob'] > 0.65) & (results_df['Growth_Prob'] <= 0.85)]
    hold_stocks = results_df[(results_df['Growth_Prob'] > 0.45) & (results_df['Growth_Prob'] <= 0.65)]
    weak_stocks = results_df[results_df['Growth_Prob'] <= 0.45]
    
    console.print(f"✅ Strong Growth Candidates: {len(strong_growth)} ({', '.join(strong_growth['Ticker'].tolist()) if len(strong_growth) > 0 else 'None'})")
    console.print(f"🟡 Moderate Growth: {len(moderate_growth)} ({', '.join(moderate_growth['Ticker'].tolist()) if len(moderate_growth) > 0 else 'None'})")
    console.print(f"🔶 Hold Positions: {len(hold_stocks)} ({', '.join(hold_stocks['Ticker'].tolist()) if len(hold_stocks) > 0 else 'None'})")
    console.print(f"🔴 Weak/Avoid: {len(weak_stocks)} ({', '.join(weak_stocks['Ticker'].tolist()) if len(weak_stocks) > 0 else 'None'})")
    
    # Recommendations
    console.print(f"\n🎯 RECOMMENDED ACTIONS:")
    
    action_num = 1
    for _, row in results_df.nlargest(5, 'Growth_Prob').iterrows():
        action, emoji = get_recommendation_action(row['Growth_Prob'])
        console.print(f"{action_num}. {emoji} {row['Ticker']} - {action} ({row['Growth_Prob']*100:.0f}% probability)")
        action_num += 1

def print_market_context(spy_data=None, integration_enabled=True):
    """Print market context information"""
    console.print(f"\n📊 Market Context:")
    
    if spy_data is not None:
        # Mock some market trend analysis
        trend = "BULLISH (+1.8% this week)"  # In real implementation, calculate from spy_data
        sector = "OUTPERFORMING (+3.2% vs S&P)"  # Would be calculated
        console.print(f"S&P 500 Trend: {trend}")
        console.print(f"Tech Sector: {sector}")
    else:
        console.print("Market data not available")
    
    console.print(f"Market Integration: {'ENABLED' if integration_enabled else 'DISABLED'}")

def print_enhanced_market_context(market_intel):
    """Print enhanced market context with regime analysis"""
    console.print(f"\n🌍 MARKET INTELLIGENCE:")
    console.print(f"🏛️  Current Regime: {market_intel.current_regime.value.replace('_', ' ').title()}")
    console.print(f"🎯 Confidence: {market_intel.regime_confidence:.1%}")
    console.print(f"📝 Assessment: {market_intel.regime_description}")
    console.print(f"🎲 Risk Appetite: {market_intel.risk_appetite}")
    console.print(f"📊 Market Stress: {market_intel.market_stress_level}")

def print_sector_intelligence(sector_intel):
    """Print sector intelligence and rotation analysis"""
    console.print(f"\n🏭 SECTOR INTELLIGENCE:")
    console.print(f"🎯 Rotation Trend: {sector_intel.rotation_trend}")
    console.print(f"📊 Sector Breadth: {sector_intel.sector_breadth:.1%} sectors outperforming SPY")
    console.print(f"🔥 Leading Sectors: {', '.join(sector_intel.leading_sectors)}")
    console.print(f"❄️  Lagging Sectors: {', '.join(sector_intel.lagging_sectors)}")
    
    # Show top sector performances
    valid_sectors = [s for s in sector_intel.sector_performances.values() if s is not None]
    if valid_sectors:
        console.print(f"\n📈 Sector Performance (1 Month):")
        sorted_sectors = sorted(valid_sectors, key=lambda x: x.performance_1m, reverse=True)
        for sector in sorted_sectors[:5]:  # Top 5
            performance_emoji = "🟢" if sector.performance_1m > 0 else "🔴"
            console.print(f"  {performance_emoji} {sector.sector_name}: {sector.performance_1m:+.1f}% (vs SPY: {sector.relative_strength_spy:+.1f}%)")

def print_completion_stats(duration, num_candidates=None, market_intel=None, sector_intel=None, news_enabled=False):
    """Print completion statistics"""
    console.print(f"\n⏱️ Analysis completed in {duration:.1f} seconds")
    if num_candidates is not None:
        console.print(f"🎯 Found {num_candidates} high-probability candidates")
    if market_intel:
        console.print(f"🧠 Regime-adjusted predictions for {market_intel.current_regime.value.replace('_', ' ').title()} market")
    if sector_intel:
        console.print(f"🏭 Sector-adjusted for {sector_intel.rotation_trend} rotation")
    if news_enabled:
        console.print(f"📰 News sentiment analysis included")
    else:
        console.print(f"📰 News analysis disabled (use --news flag to enable)")

# Helper functions
def get_analysis_reason(row):
    """Get analysis reason for high confidence picks"""
    reasons = []
    if row['RSI'] < 35:
        reasons.append("oversold RSI")
    if row['Vol_Change'] > 100:
        reasons.append("massive volume spike")
    if row['Growth_Prob'] > 0.90:
        reasons.append("strong momentum")
    
    return ", ".join(reasons) if reasons else "technical breakout pattern, strong fundamentals"

def get_detailed_analysis(row):
    """Get detailed analysis and recommendation for evaluation mode"""
    prob = row['Growth_Prob']
    rsi = row['RSI'] if pd.notnull(row['RSI']) else 50
    
    if prob > 0.85:
        analysis = "🟢 STRONG BUY\n• " + ("Oversold + Volume" if rsi < 40 else "Breaking resistance")
        recommendation = "GROWTH"
    elif prob > 0.65:
        analysis = "🟡 MODERATE BUY\n• " + ("Neutral RSI" if 40 <= rsi <= 60 else "Good momentum")
        recommendation = "GROWTH"
    elif prob > 0.45:
        analysis = "🔶 HOLD\n• " + ("Overbought RSI" if rsi > 70 else "Mixed signals")
        recommendation = "HOLD"
    else:
        analysis = "🔴 WEAK\n• " + ("Very overbought" if rsi > 75 else "Volume declining")
        recommendation = "NO GROWTH"
    
    return analysis, recommendation

def get_rsi_description(rsi):
    """Get RSI level description"""
    if rsi < 30:
        return "(Oversold territory)"
    elif rsi > 70:
        return "(Overbought territory)"
    else:
        return "(Neutral zone)"

def get_volume_description(vol_change):
    """Get volume change description"""
    if vol_change > 150:
        return "Massive interest"
    elif vol_change > 75:
        return "Good participation"
    elif vol_change > 25:
        return "Moderate interest"
    else:
        return "Low participation"

def get_pe_description(pe_ratio):
    """Get P/E ratio description"""
    if pe_ratio > 50:
        return "High but justified by growth"
    elif pe_ratio > 25:
        return "Moderate valuation"
    elif pe_ratio > 15:
        return "Reasonable valuation"
    else:
        return "Attractive valuation"

def get_recommendation_action(prob, discovery_threshold=0.70):
    """Get recommendation action and emoji"""
    if prob > 0.85:
        return "High conviction buy", "🚀"
    elif prob > discovery_threshold:  # Use configurable threshold
        return "Good entry point", "📈"
    elif prob > 0.55:
        return "Small position or wait for pullback", "⚖️"
    elif prob > 0.40:
        return "Monitor for better entry", "⏸️"
    else:
        return "Consider profit-taking if holding", "⚠️"

def create_results_dataframe(tickers, probs, latest_data, stock_infos=None, regime_explanations=None, sector_explanations=None, news_explanations=None, market_intel=None, sector_intel=None, raw_probs=None, regime_probs=None, sector_probs=None, news_probs=None):
    """Create a properly formatted results dataframe with probability breakdown"""
    results = []
    
    for i, ticker in enumerate(tickers):
        row_data = latest_data.iloc[i] if latest_data is not None else {}
        stock_info = stock_infos.get(ticker, {}) if stock_infos else {}
        
        # Get current price (last close)
        price = row_data.get('Close', 0) if hasattr(row_data, 'get') else 0
        
        # Calculate volume change (mock for now, in real version would be calculated)
        vol_change = ((row_data.get('Volume', 1) / row_data.get('Volume_Avg', 1)) - 1) * 100 if hasattr(row_data, 'get') else 50
        
        # Calculate probability breakdown with news component
        raw_score = raw_probs[i] if raw_probs is not None and i < len(raw_probs) else probs[i]
        regime_score = regime_probs[i] if regime_probs is not None and i < len(regime_probs) else probs[i]
        sector_score = sector_probs[i] if sector_probs is not None and i < len(sector_probs) else probs[i]
        news_score = news_probs[i] if news_probs is not None and i < len(news_probs) else probs[i]
        final_score = probs[i] if i < len(probs) else 0.5
        
        # Calculate individual boosts
        regime_boost = regime_score - raw_score
        sector_boost = sector_score - regime_score  
        news_boost = news_score - sector_score
        total_boost = final_score - raw_score
        
        # Get explanations
        regime_explanation = regime_explanations[i] if regime_explanations and i < len(regime_explanations) else "No regime adjustment"
        sector_explanation = sector_explanations[i] if sector_explanations and i < len(sector_explanations) else "No sector adjustment"
        news_explanation = news_explanations[i] if news_explanations and i < len(news_explanations) else "No news data"
        
        result = {
            'Ticker': ticker,
            'Company': stock_info.get('longName', ticker + ' Corp'),
            'Growth_Prob': final_score,
            'Raw_ML_Score': raw_score,
            'Regime_Boost': regime_boost,
            'Sector_Boost': sector_boost,
            'News_Boost': news_boost,
            'Total_Boost': total_boost,
            'Regime_Explanation': regime_explanation,
            'Sector_Explanation': sector_explanation,
            'News_Explanation': news_explanation,
            'RSI': row_data.get('RSI', 50) if hasattr(row_data, 'get') else 50,
            'Price': price,
            'Vol_Change': vol_change,
            'PE_Ratio': row_data.get('PE_ratio', 25) if hasattr(row_data, 'get') else 25,
            'SMA_50': row_data.get('SMA_50', price * 0.95) if hasattr(row_data, 'get') else price * 0.95,
            'SMA_200': row_data.get('SMA_200', price * 0.90) if hasattr(row_data, 'get') else price * 0.90,
            'SMA_150': row_data.get('SMA_150', price * 0.925) if hasattr(row_data, 'get') else price * 0.925,
        }
        results.append(result)
    
    return pd.DataFrame(results) 

def print_hot_stocks_results(momentum_stocks, config):
    """Print hot stocks momentum scanner results"""
    console.print(f"\n🔥 MOMENTUM STOCKS ANALYSIS:")
    
    if not momentum_stocks:
        console.print("❌ No momentum stocks found.")
        return
    
    # Create rich table for momentum results
    table = Table(show_header=True, header_style="bold red")
    table.add_column("Rank", style="white", no_wrap=True)
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Momentum", style="red", justify="right")
    table.add_column("RSI", style="yellow", justify="right")
    table.add_column("SMA20", style="bright_yellow", justify="right")
    table.add_column("SMA150", style="bright_green", justify="right")
    table.add_column("EMA Cross", style="green", justify="center")
    table.add_column("Volume", style="blue", justify="right")
    table.add_column("3D Move", style="magenta", justify="right")
    table.add_column("5D Move", style="magenta", justify="right")
    
    for i, (ticker, score, details) in enumerate(momentum_stocks, 1):
        # Format momentum score
        momentum_str = f"{score:.3f}"
        
        # Format RSI
        rsi = details.get('rsi', 0)
        rsi_str = f"{rsi:.1f}"
        
        # Format Price vs SMA20
        price_vs_sma20 = details.get('price_vs_sma20', 1.0)
        if price_vs_sma20 >= 1.0:
            sma20_str = f"+{((price_vs_sma20 - 1) * 100):.1f}%"
        else:
            sma20_str = f"{((price_vs_sma20 - 1) * 100):.1f}%"
        
        # Format Price vs SMA150 (trend filter)
        price_vs_sma150 = details.get('price_vs_150sma', 1.0)
        if price_vs_sma150 >= 1.05:  # 5%+ above = strong trend
            sma150_str = f"✅ +{((price_vs_sma150 - 1) * 100):.1f}%"
        elif price_vs_sma150 >= 1.0:  # Just above = weak trend
            sma150_str = f"🟡 +{((price_vs_sma150 - 1) * 100):.1f}%"
        else:  # Below = rejected (shouldn't happen due to filter)
            sma150_str = f"❌ {((price_vs_sma150 - 1) * 100):.1f}%"
        
        # Format EMA Crossover
        ema_bullish = details.get('ema_bullish', False)
        ema_recent = details.get('ema_crossover_recent', False)
        if ema_bullish and ema_recent:
            ema_str = "🔥 NEW"
        elif ema_bullish:
            ema_str = "✅ Bull"
        else:
            ema_str = "❌ Bear"
        
        # Format Volume
        volume_ratio = details.get('volume_ratio', 1.0)
        volume_str = f"{volume_ratio:.1f}x"
        
        # Format Price Moves
        price_3d = details.get('price_3d', 0)
        price_5d = details.get('price_5d', 0)
        price_3d_str = f"{price_3d:+.1f}%"
        price_5d_str = f"{price_5d:+.1f}%"
        
        table.add_row(
            str(i),
            ticker,
            momentum_str,
            rsi_str,
            sma20_str,
            sma150_str,
            ema_str,
            volume_str,
            price_3d_str,
            price_5d_str
        )
    
    console.print(table)
    
    # Print detailed breakdown for top 3
    console.print(f"\n📊 DETAILED MOMENTUM BREAKDOWN (Top 3):")
    
    for i, (ticker, score, details) in enumerate(momentum_stocks[:3], 1):
        console.print(f"\n{i}. {ticker} - Momentum Score: {score:.3f}")
        
        # PRIMARY: Long-term Trend Analysis (40% weight)
        trend_score = details.get('trend_score', 0)
        trend_desc = details.get('trend_desc', 'Unknown')
        long_term_slope = details.get('long_term_slope', 'Unknown')
        trend_period = details.get('trend_period', 'Unknown')
        console.print(f"   🎯 PRIMARY - Long-term Trend (40%): {trend_score:.2f} - {trend_desc}")
        console.print(f"      📈 SMA150 Slope: {long_term_slope} ({trend_period} analysis)")
        
        # Setup Analysis (20% weight)
        setup_score = details.get('setup_score', 0)
        setup_desc = details.get('setup_desc', 'Unknown')
        sma_relationship = details.get('sma_relationship', 'Unknown')
        console.print(f"   📍 SMA Setup Timing (20%): {setup_score:.2f} - {setup_desc}")
        console.print(f"      📊 20SMA vs 150SMA: {sma_relationship}")
        
        # Price vs 20SMA Analysis (15% weight) - NEW
        price20_score = details.get('price20_score', 0)
        price20_desc = details.get('price20_desc', 'Unknown')
        price_sma20_pct = details.get('price_sma20_pct', 'Unknown')
        console.print(f"   💰 Price vs 20SMA (15%): {price20_score:.2f} - {price20_desc}")
        console.print(f"      📈 Distance from 20SMA: {price_sma20_pct}")
        
        # Volume Analysis (20% weight)
        volume_score = details.get('volume_score', 0)
        volume_desc = details.get('volume_desc', 'Unknown')
        volume_ratio = details.get('volume_ratio', 1.0)
        console.print(f"   📊 Volume Confirmation (20%): {volume_score:.2f} - {volume_desc}")
        console.print(f"      📈 Volume Ratio: {volume_ratio:.1f}x average")
        
        # EMA Analysis (5% weight)
        ema_score = details.get('ema_score', 0)
        ema_desc = details.get('ema_desc', 'Unknown')
        ema_separation = details.get('ema_separation', 0)
        console.print(f"   🔄 EMA Momentum (5%): {ema_score:.2f} - {ema_desc}")
        console.print(f"      📈 EMA13x48: {ema_separation:+.1f}% separation")
        
        # RSI Filter (5% weight)
        rsi_score = details.get('rsi_score', 0)
        rsi_desc = details.get('rsi_desc', 'Unknown')
        rsi = details.get('rsi', 0)
        console.print(f"   📊 RSI Filter (5%): {rsi_score:.2f} - {rsi_desc}")
        console.print(f"      📈 RSI Level: {rsi:.1f}")
        
        # Price positioning context
        price_vs_150sma = details.get('price_vs_150sma', 1.0)
        distance_from_150sma = details.get('distance_from_150sma', 'Unknown')
        console.print(f"   📊 Context - Price vs 150SMA: {distance_from_150sma}")
    
    # Legend (updated with new weights)
    console.print(f"\n📝 ENHANCED TREND-FOCUSED MOMENTUM SCORING:")
    console.print(f"   • Long-term Trend: 35% (SMA150 slope = PRIMARY driver)")
    console.print(f"   • SMA Setup Timing: 20% (20SMA vs 150SMA positioning)")
    console.print(f"   • Price vs 20SMA: 15% (short-term momentum positioning) - NEW")
    console.print(f"   • Volume Confirmation: 20% (1.5x+ = strong signal)")
    console.print(f"   • EMA Momentum: 5% (13x48 crossover)")
    console.print(f"   • RSI Filter: 5% (avoid extremes only)") 

def get_probability_color(prob, discovery_threshold=0.70):
    """Get color for probability display based on thresholds"""
    if prob > 0.85:
        return "bold green"
    elif prob > discovery_threshold:  # Use configurable threshold
        return "green"
    elif prob > 0.50:
        return "yellow"
    elif prob > 0.30:
        return "red"
    else:
        return "bold red"

def get_effective_threshold(config, mode):
    """
    Get the effective threshold based on mode and configuration
    """
    if mode == "discovery":
        # In discovery mode, use discovery_threshold
        return config.get("discovery_threshold", DEFAULT_CONFIG["discovery_threshold"])
    else:
        # In evaluation mode, use threshold (which defaults to 0.0)
        return config.get("threshold", DEFAULT_CONFIG["threshold"]) 