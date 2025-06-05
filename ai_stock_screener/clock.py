import yfinance as yf
import requests
import fear_and_greed as fg
import pandas as pd


import yfinance as yf
import pandas as pd

def detect_recent_crossover(short_sma, long_sma, label, df, days_lookback=20):
    """
    Looks back `days_lookback` periods to detect recent crossovers.
    Returns a string describing the event and how many days ago it occurred.
    """
    for i in range(1, days_lookback + 1):
        if len(df) < i + 1:
            break
        prev = df.iloc[-i - 1]
        curr = df.iloc[-i]

        # Bullish crossover
        if prev[short_sma] < prev[long_sma] and curr[short_sma] > curr[long_sma]:
            return f"📈 {label} Bullish crossover {i} days ago"
        # Bearish crossover
        elif prev[short_sma] > prev[long_sma] and curr[short_sma] < curr[long_sma]:
            return f"📉 {label} Bearish crossover {i} days ago"

    return f"⏸️ No {label} crossover in last {days_lookback} days"

def get_spy_trend():
    # Fetch SPY historical data
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="18mo")

    if df.empty or "Close" not in df.columns:
        return {
            "trend": "❌ SPY data unavailable.",
            "sma_values": {},
            "crossovers": []
        }

    # Compute SMAs
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    df = df.dropna(subset=["SMA20", "SMA50", "SMA200"])
    if df.empty:
        return {
            "trend": "❌ Not enough data for SMA calculation.",
            "sma_values": {},
            "crossovers": []
        }

    latest = df.iloc[-1]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    
    # Trend based on SMA20 vs SMA50
    if sma20 > sma50:
        trend = "🟢 Bullish (SMA20 > SMA50)"
    elif sma20 < sma50:
        trend = "🔴 Bearish (SMA20 < SMA50)"
    else:
        trend = "🟡 Neutral (SMA20 ≈ SMA50)"

    # Detect crossovers
    crossover_20_50 = detect_recent_crossover("SMA20", "SMA50", "SMA20/50", df)
    crossover_50_200 = detect_recent_crossover("SMA50", "SMA200", "SMA50/200", df)

    return {
        "trend": trend,
        "sma_values": {
            "SMA20": round(sma20, 2),
            "SMA50": round(sma50, 2),
            "SMA200": round(latest["SMA200"], 2)
        },
        "crossovers": [crossover_20_50, crossover_50_200]
    }


def get_vix_level():
    vix = yf.download("^VIX", period="1mo", interval="1d", progress=False, auto_adjust=True)
    if vix.empty or "Close" not in vix.columns:
        return "Unknown"

    try:
        latest_close = vix["Close"].dropna().iloc[-1].item()
    except Exception:
        return "Unknown"

    if latest_close < 15:
        return "🟢 Low (<15)"
    elif latest_close < 25:
        return "🟡 Medium (15–25)"
    else:
        return "🔴 High (>25)"


def get_fear_and_greed_level():
    try:
        import fear_and_greed
        index = fear_and_greed.get()
        value = index.value

        if value >= 75:
            return "🟢 Extreme Greed (75+)"
        elif value >= 55:
            return "🟢 Greed (55–74)"
        elif value >= 45:
            return "🟡 Neutral (45–54)"
        elif value >= 25:
            return "🟠 Fear (25–44)"
        else:
            return "🔴 Extreme Fear (<25)"
    except Exception as e:
        return f"⚠️ Fear & Greed unavailable ({str(e).splitlines()[0]})"


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
            return "🔴 Inverted (Recession signal)"
        elif spread < 1:
            return "🟡 Flat (Caution)"
        else:
            return "🟢 Normal (>1pt spread)"

    except Exception as e:
        return f"⚠️ Yield curve data unavailable ({str(e).splitlines()[0]})"



# def market_clock():
#     print("\n🕒 Market Clock Summary")
#     print("-----------------------")
#     print(f"📈 SPY Trend:            {get_spy_trend()}")
#     print(f"⚡ VIX Volatility:       {get_vix_level()}")
#     print(f"😬 Fear & Greed Index:   {get_fear_and_greed_level()}")
#     print(f"📉 Yield Curve (10Y-2Y): {get_yield_curve_level()}")
#     print()

def market_clock():
    result = get_spy_trend()
    print("📈 SPY Trend:",          result["trend"])
    print("SMA Values:",            result["sma_values"])
    print("Crossover Signal:",      result["crossovers"])
    print(f"⚡ VIX Volatility:       {get_vix_level()}")
    print(f"😬 Fear & Greed Index:   {get_fear_and_greed_level()}")
    print(f"📉 Yield Curve (10Y-2Y): {get_yield_curve_level()}")
    print()
