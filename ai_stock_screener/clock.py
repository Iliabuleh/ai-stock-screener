import yfinance as yf
import fear_and_greed as fg

def get_spy_trend():
    # Load 18 months of SPY data for long-term SMA calculation
    df = yf.Ticker("SPY").history(period="18mo")

    if df.empty or "Close" not in df.columns:
        return {
            "trend": "❌ SPY data unavailable.",
            "sma_values": {},
            "crossovers": [],
            "overextension": "N/A"
        }

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df = df.dropna()

    if df.empty:
        return {
            "trend": "❌ Not enough data after SMA calculation.",
            "sma_values": {},
            "crossovers": [],
            "overextension": "N/A"
        }

    latest = df.iloc[-1]
    close_price = latest["Close"]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    sma200 = latest["SMA200"]

    # 🔹 Determine Trend
    if sma20 > sma50:
        trend = "🟢 Bullish (SMA20 > SMA50)"
    elif sma20 < sma50:
        trend = "🔴 Bearish (SMA20 < SMA50)"
    else:
        trend = "🟡 Neutral (SMA20 ≈ SMA50)"

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
        "sma_values": {
            "SMA20": round(sma20, 2),
            "SMA50": round(sma50, 2),
            "SMA200": round(sma200, 2),
            "Close": round(close_price, 2)
        },
        "crossovers": crossovers or ["No recent crossovers."],
        "overextension": overext
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
        index = fg.get()
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
    print("\n🕒 Market Clock Summary")
    print("-----------------------")
    print("📈 SPY Trend:",     result["trend"])
    print("SMA Values:",       result["sma_values"])
    print("📐 Overextension:", result["overextension"])
    print("🔀 Crossovers:")
    for c in result["crossovers"]:
        print("   -", c)
    print(f"⚡ VIX Volatility:       {get_vix_level()}")
    print(f"😬 Fear & Greed Index:   {get_fear_and_greed_level()}")
    print(f"📉 Yield Curve (10Y-2Y): {get_yield_curve_level()}")
    print()
