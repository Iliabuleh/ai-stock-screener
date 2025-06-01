import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

# 🔧 Centralized list of features used in training & prediction
FEATURE_COLUMNS = [
    "PE_ratio", "RSI", "SMA_50", "SMA_200", "Volume", "Volume_Avg",
    "MACD", "MACD_signal", "BB_upper", "BB_lower",
    "Stoch_K", "Stoch_D", "ATR"
]

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url)
        df = table[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def fetch_data(ticker, config):
    print(f"🔍 Fetching {ticker}...")
    stock = yf.Ticker(ticker)
    info = stock.info
    pe_ratio = info.get("trailingPE", None)

    if pe_ratio is None or pe_ratio > 1000:
        print(f"❌ Skipped {ticker} due to missing or extreme P/E ratio.")
        return None

    data = stock.history(period=config["period"])
    if data.empty:
        print(f"❌ Skipped {ticker} (no price data available)")
        return None

    try:
        # 📈 Technical indicators
        data["RSI"] = ta.rsi(data["Close"], length=14)
        data["SMA_50"] = ta.sma(data["Close"], length=50)
        data["SMA_200"] = ta.sma(data["Close"], length=200)
        data["Volume_Avg"] = data["Volume"].rolling(window=20).mean()
        data["PE_ratio"] = pe_ratio

        macd = ta.macd(data["Close"])
        data["MACD"] = macd["MACD_12_26_9"]
        data["MACD_signal"] = macd["MACDs_12_26_9"]

        bb = ta.bbands(data["Close"], length=20)
        data["BB_upper"] = bb["BBU_20_2.0"]
        data["BB_lower"] = bb["BBL_20_2.0"]

        stoch = ta.stoch(data["High"], data["Low"], data["Close"])
        data["Stoch_K"] = stoch["STOCHk_14_3_3"]
        data["Stoch_D"] = stoch["STOCHd_14_3_3"]

        data["ATR"] = ta.atr(data["High"], data["Low"], data["Close"], length=14)

        # 📊 Labeling for training
        data["Future_Return"] = data["Close"].shift(-config["future_days"]) / data["Close"] - 1
        data["Label"] = (data["Future_Return"] > config["threshold"]).astype(int)


        label_counts = data["Label"].value_counts()
        print(f"{ticker} — Label value counts:")
        print(label_counts.to_string())
        print(f"{ticker} — Final usable rows after dropna: {len(data.dropna())}")

        return data.dropna()
    except Exception as e:
        print(f"⚠️ Error computing indicators for {ticker}: {e}")
        return None

def train_model(df, config):
    X = df[FEATURE_COLUMNS]
    y = df["Label"]
    label_counts = y.value_counts().to_dict()

    print("\n🧠 Training label distribution:")
    print(f"   - 1 (high-growth): {label_counts.get(1, 0)} samples")
    print(f"   - 0 (no significant growth): {label_counts.get(0, 0)} samples")

    if y.sum() == 0:
        print("⚠️ Model training skipped — no high-growth (label=1) samples present.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(
        n_estimators=config["n_estimators"], n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"\n✅ Model accuracy on held-out test set: {acc:.2f}")
    return clf

def run_screening(tickers, config):
    print("\n📥 Starting screening process...\n")
    all_data = []

    for ticker in tickers:
        try:
            df = fetch_data(ticker, config)
            if df is not None:
                print(f"✅ Using {ticker} for training.\n")
                df["Ticker"] = ticker
                all_data.append(df)
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")

    if not all_data:
        print("\n❌ No usable data collected from tickers. Exiting.")
        sys.exit(1)

    combined = pd.concat(all_data)
    print(f"📦 Total training samples: {len(combined)} from {len(all_data)} tickers")
    print(combined.groupby('Ticker')["Label"].value_counts().to_string())

    clf = train_model(combined, config)
    if clf is None:
        print("⚠️ Skipping prediction due to insufficient positive training data.")
        return

    print("\n📊 Predicted High-Growth Stocks Based on Latest Data:\n")
    latest = combined.groupby("Ticker").tail(1)
    X_pred = latest[FEATURE_COLUMNS]
    probs = clf.predict_proba(X_pred)
    preds = clf.predict(X_pred)

    results = []
    for i, pred in enumerate(preds):
        ticker = latest.iloc[i]["Ticker"]
        prob = probs[i][1]
        if pred == 1:
            print(f"📈 {ticker}: Predicted to GROW > {config['threshold']*100:.1f}% in {config['future_days']} days — Confidence: {prob:.2f}")
            results.append((ticker, prob))
        else:
            print(f"📉 {ticker}: Predicted to NOT grow significantly — Confidence: {(1 - prob):.2f}")

    if not results:
        print("\n❌ No high-growth candidates identified.")
        return

    print("\n🎯 High-confidence candidates sorted by prediction certainty:")
    results = sorted(results, key=lambda x: x[1], reverse=True)
    for ticker, prob in results:
        print(f"✅ {ticker} — Confidence: {prob:.2f}")
