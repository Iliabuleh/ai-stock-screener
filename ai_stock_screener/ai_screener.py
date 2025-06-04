import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import joblib
import numpy as np
import sys

# 🔧 Centralized list of features used in training & prediction
BASE_FEATURE_COLUMNS = [
    "PE_ratio", "RSI", "Volume", "Volume_Avg",
    "MACD", "MACD_signal", "BB_upper", "BB_lower",
    "Stoch_K", "Stoch_D", "ATR",
    "Return_1d", "Return_5d", "Volatility_20d",
    "Price_vs_SMA50", "Price_vs_SMA200", "Rel_Strength_SPY"
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS.copy()

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url)
        df = table[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

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
    print(f"🔍 Fetching {ticker}... {label_info}")
    stock = yf.Ticker(ticker)
    info = stock.info
    pe_ratio = info.get("trailingPE", None)

    if not is_market and (pe_ratio is None or pe_ratio > 1000):
        print(f"❌ Skipped {ticker} due to missing or extreme P/E ratio.")
        return None

    data = stock.history(period=config["period"])
    if data.empty:
        print(f"❌ Skipped {ticker} (no price data available)")
        return None

    try:
        # 📈 Technical indicators
        data["RSI"] = ta.rsi(data["Close"], length=14)
        data["Volume_Avg"] = data["Volume"].rolling(window=20).mean()
        data["PE_ratio"] = pe_ratio if not is_market else 15  # dummy if market

        for col in FEATURE_COLUMNS:
            if col.startswith("SMA_"):
                length = int(col.split("_")[1])
                data[col] = ta.sma(data["Close"], length=length)

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

        # 📉 Price Action Features
        data["Return_1d"] = data["Close"].pct_change()
        data["Return_5d"] = data["Close"].pct_change(5)
        data["Volatility_20d"] = data["Return_1d"].rolling(window=20).std()
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

                valid_sharpe = data["Sharpe_Like"].dropna()
                if not valid_sharpe.empty:
                    total = len(valid_sharpe)
                    accepted = (valid_sharpe > sharpe_threshold).sum()
                    rejected = total - accepted
                    print(f"{ticker} — return-volatility summary (threshold {sharpe_threshold}):")
                    print(f"   • Accepted: {accepted} / {total} rows")
                    print(f"   • Rejected: {rejected} rows")
                    print(f"   • Mean: {valid_sharpe.mean():.2f}, Median: {valid_sharpe.median():.2f}, Max: {valid_sharpe.max():.2f}, Min: {valid_sharpe.min():.2f}")

            label_counts = data["Label"].value_counts()
            print(f"{ticker} — Label value counts:")
            print(label_counts.to_string())
        else:
            print(f"{ticker} — Market indicators (no labels applied)")

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
        X, y, test_size=0.2, random_state=config.get("seed", 42)
    )

    model_type = config.get("model", "random_forest")
    grid_search = config.get("grid_search", 0)
    ensemble_runs = config.get("ensemble_runs", 1)

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
            print(f"\n🔍 Performing grid search for {model_type}...")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            search = GridSearchCV(base_model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
            search.fit(X_train, y_train)
            print(f"✅ Best parameters found: {search.best_params_}")
            return search.best_estimator_
        else:
            base_model.set_params(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", None),
                learning_rate=config.get("learning_rate", 0.1) if model_type == "xgboost" else None
            )
            base_model.fit(X_train, y_train)
            return base_model

    if ensemble_runs > 1:
        print(f"\n🤖 Performing ensemble training with {ensemble_runs} runs...")
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
        print(f"\n✅ ENSEMBLE {model_type.upper()} accuracy on held-out test set: {acc:.2f}")
        final_model = models[0]  # return one of the models for later use
    else:
        clf = build_model(config.get("seed", 42))
        acc = clf.score(X_test, y_test)
        print(f"\n✅ {model_type.upper()} accuracy on held-out test set: {acc:.2f}")
        final_model = clf

    if config.get("save_model_path"):
        joblib.dump(final_model, config["save_model_path"])
        print(f"💾 Saved model to {config['save_model_path']}")

    return final_model

def run_screening(tickers, config):
    print("\n📥 Starting screening process...\n")

    initialize_feature_columns(tickers, config)

    print("🧾 Feature summary before training:")
    print(f"   ➤ Using {len(FEATURE_COLUMNS)} technical indicators:")
    for feat in FEATURE_COLUMNS:
        print(f"     • {feat}")
    print()

    all_data = []

    # Fetch SPY market context first (print-only or also for training)
    spy_df = fetch_data("SPY", config, is_market=True)
    spy_close_series = spy_df["Close"] if spy_df is not None else None
    if spy_df is not None:
        print(f"📊 SPY Market overview: RSI={spy_df.iloc[-1]['RSI']:.2f}, MACD={spy_df.iloc[-1]['MACD']:.2f}, ATR={spy_df.iloc[-1]['ATR']:.2f}")
        if config.get("integrate_market"):
            spy_df["Ticker"] = "SPY_MARKET"
            spy_df["Label"] = 0  # Dummy label to keep shape
            all_data.append(spy_df)

    for ticker in tickers:
        try:
            df = fetch_data(ticker, config, spy_close=spy_close_series)
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
    combined = combined[combined["Ticker"] != "SPY_MARKET"] if not config.get("integrate_market") else combined
    print(f"📦 Total training samples: {len(combined)} from {len(all_data)} sources")
    print(combined.groupby('Ticker')["Label"].value_counts().to_string())

    clf = train_model(combined, config)
    if clf is None:
        print("⚠️ Skipping prediction due to insufficient positive training data.")
        return

    print("\n📊 Predicted High-Growth Stocks Based on Latest Data:\n")
    latest = combined[combined["Ticker"] != "SPY_MARKET"].groupby("Ticker").tail(1)
    X_pred = latest[FEATURE_COLUMNS]
    probs = clf.predict_proba(X_pred)
    preds = clf.predict(X_pred)

    results = []

    for i, pred in enumerate(preds):
        row = latest.iloc[i]
        ticker = row["Ticker"]
        prob = probs[i][1]

        future_return = row.get("Future_Return", None)
        volatility = row.get("Volatility_Future", None)
        rv_score = (
            future_return / (volatility + 1e-6)
            if pd.notnull(future_return) and pd.notnull(volatility)
            else None
        )

        summary = f"— Confidence: {prob:.2f}"
        if pd.notnull(future_return) and pd.notnull(volatility):
            summary += f" — return-volatility: {rv_score:.2f} (Return: {future_return*100:.2f}%, Volatility: {volatility*100:.2f}%)"

        if pred == 1:
            print(f"📈 {ticker}: Predicted to GROW > {config['threshold']*100:.1f}% in {config['future_days']} days {summary}")
            results.append((ticker, prob, rv_score))
        else:
            print(f"📉 {ticker}: Predicted to NOT grow significantly {summary}")

    if not results:
        print("\n❌ No high-growth candidates identified.")
        return

    print("\n🎯 High-confidence candidates sorted by prediction certainty:")
    for ticker, prob, rv_score in sorted(results, key=lambda x: x[1], reverse=True):
        line = f"✅ {ticker} — Confidence: {prob:.2f}"
        if rv_score is not None:
            line += f", return-volatility: {rv_score:.2f}"
        print(line)
