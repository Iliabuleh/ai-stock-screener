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
from .clock import get_market_intelligence, MarketIntelligence

# 🔧 Centralized list of features used in training & prediction
BASE_FEATURE_COLUMNS = [
    "PE_ratio", "RSI", "Volume", "Volume_Avg",
    "MACD", "MACD_signal", "BB_upper", "BB_lower",
    "Stoch_K", "Stoch_D", "ATR",
    "Return_1d", "Return_5d", "Volatility_20d",
    "Price_vs_SMA50", "Price_vs_SMA200", "Rel_Strength_SPY"
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS.copy()

# 🧠 Market regime adjustments for prediction confidence
REGIME_MULTIPLIERS = {
    "bull_low_vol": 1.15,      # Favorable for growth stocks
    "bull_normal_vol": 1.05,   # Slightly favorable
    "bull_high_vol": 0.95,     # Reduce confidence in volatile bull
    "bear_low_vol": 0.85,      # Potential bottoming, but still cautious
    "bear_normal_vol": 0.75,   # Bearish conditions
    "bear_high_vol": 0.65,     # High risk environment
    "sideways_low_vol": 0.90,  # Range-bound, moderate confidence
    "sideways_normal_vol": 0.85, # Neutral conditions
    "sideways_high_vol": 0.80  # Uncertain, volatile sideways
}

def apply_regime_adjustment(probability: float, market_intel: MarketIntelligence) -> tuple[float, str]:
    """
    Apply market regime adjustment to prediction probability
    Returns: (adjusted_probability, explanation)
    """
    regime_key = market_intel.current_regime.value
    multiplier = REGIME_MULTIPLIERS.get(regime_key, 1.0)
    
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
    
    stock = yf.Ticker(ticker)
    info = stock.info
    pe_ratio = info.get("trailingPE", None)

    if not is_market and (pe_ratio is None or pe_ratio > 1000):
        console.print(f"❌ Skipped {ticker} due to missing or extreme P/E ratio.")
        return None

    data = stock.history(period=config["period"])
    if data.empty:
        console.print(f"❌ Skipped {ticker} (no price data available)")
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

def run_screening(tickers, config, mode="eval"):
    start_time = time.time()
    
    # 🧠 Get market intelligence first
    console.print("\n🧠 Gathering Market Intelligence...")
    market_intel = get_market_intelligence()
    
    # Print beautiful header with market context
    print_header(mode, tickers, config, market_intel)
    
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
    
    adjusted_probs = []
    regime_explanations = []
    
    for raw_prob in raw_probs:
        adj_prob, explanation = apply_regime_adjustment(raw_prob, market_intel)
        adjusted_probs.append(adj_prob)
        regime_explanations.append(explanation)
    
    # Create results dataframe with regime-adjusted probabilities
    results_df = create_results_dataframe(
        latest["Ticker"].tolist(), 
        adjusted_probs,  # Use regime-adjusted probabilities
        latest,
        stock_infos,
        regime_explanations=regime_explanations,  # Pass regime explanations
        market_intel=market_intel  # Pass market intelligence
    )
    
    # Filter for high probability results in discovery mode
    if mode == "discovery":
        results_df = results_df[results_df['Growth_Prob'] > 0.70].sort_values('Growth_Prob', ascending=False)
        print_discovery_results(results_df, config, market_intel)
    else:
        results_df = results_df.sort_values('Growth_Prob', ascending=False)
        print_evaluation_results(results_df, config, market_intel)
    
    # Print enhanced market context
    print_enhanced_market_context(market_intel)
    
    # Print completion stats
    duration = time.time() - start_time
    num_candidates = len(results_df[results_df['Growth_Prob'] > 0.70]) if mode == "discovery" else None
    print_completion_stats(duration, num_candidates, market_intel)
