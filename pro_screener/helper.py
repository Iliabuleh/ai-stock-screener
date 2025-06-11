#!/usr/bin/env python3
"""
Helper functions for Pro Screener
Contains ticker fetching functions for major stock indices
"""

import pandas as pd
from rich.console import Console

console = Console()

def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia"""
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
        
        # Option 2: Use S&P 500 as approximation
        console.print("📊 Using S&P 500 as Russell 1000 approximation (most overlap)")
        console.print("💡 For true Russell 1000, consider upgrading to a premium data provider")
        return sp500_tickers
        
    except Exception as e:
        console.print(f"❌ Error building Russell 1000 index: {e}")
        return get_sp500_tickers()

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