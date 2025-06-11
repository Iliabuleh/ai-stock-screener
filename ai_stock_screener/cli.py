# ai_screener/cli.py

import argparse
import sys
from ai_stock_screener.ai_screener import get_sp500_tickers, get_russell1000_tickers, get_nasdaq_tickers, get_all_tickers, run_screening, run_hot_stocks_scanner

def get_combined_tickers(index_list):
    """Get combined ticker list from multiple indexes"""
    all_tickers = set()
    
    for index in index_list:
        index = index.strip().lower()
        if index == "sp500":
            tickers = get_sp500_tickers()
            all_tickers.update(tickers)
        elif index == "russell1000":
            tickers = get_russell1000_tickers()
            all_tickers.update(tickers)
        elif index == "nasdaq":
            tickers = get_nasdaq_tickers()
            all_tickers.update(tickers)
        elif index == "all":
            tickers = get_all_tickers()
            all_tickers.update(tickers)
        else:
            from ai_stock_screener.output_formatter import console
            console.print(f"❌ Unknown index: {index}")
            console.print("💡 Valid indexes: sp500, russell1000, nasdaq, all")
            sys.exit(1)
    
    return list(all_tickers)

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Stock Screener CLI")
    parser.add_argument("--mode", choices=["eval", "discovery"], 
                        help="Run mode: eval (use provided tickers) or discovery (scan stocks)")
    parser.add_argument("--tickers", type=str,
                        help="Comma-separated list of tickers for eval mode (e.g., AAPL,NVDA,MSFT)")
    parser.add_argument("--index", type=str, default="sp500",
                        help="Stock index(es): sp500, russell1000, nasdaq, all, or comma-separated combination (e.g., sp500,nasdaq)")
    parser.add_argument("--period", default="1y", help="Historical data period (default: 1y)")
    parser.add_argument("--future_days", type=int, default=5, help="Days ahead to evaluate returns (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Label threshold (default: 0.0 for any growth)")
    parser.add_argument("--n_estimators", type=int, default=300, help="RandomForest trees (default: 300)")
    parser.add_argument("--use_sharpe_labeling", type=float, default=1.0, help="Enable return-volatility labeling with the given threshold (default: 1.0)")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "xgboost"],
                        help="Which model to train: 'random_forest' or 'xgboost'. Default is random_forest.")
    parser.add_argument("--grid_search", type=int, default=0, 
                        help="Enable grid search over model hyperparameters (1 = enabled, 0 = disabled)")
    parser.add_argument("--ensemble_runs", type=int, default=1, help="Number of ensemble runs (default: 1)")
    parser.add_argument("--no_integrate_market", action="store_true",
                    help="Disable integration of SPY market data into training (default: enabled)")
    parser.add_argument("--news", action="store_true",
                    help="Enable news sentiment analysis (adds processing time)")
    parser.add_argument("--sector", type=str, metavar="SECTOR_NAME",
                    help="Filter stocks by sector. Available sectors: Technology, Healthcare, Financials, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy, Utilities, Real Estate, Materials")
    parser.add_argument("--hot-stocks", type=int, metavar="COUNT", default=0,
                    help="Pure momentum scanner: find top N trending stocks (e.g., --hot-stocks 20). Independent of discovery/eval modes.")
    
    # === ADVANCED CONFIGURATION OPTIONS ===
    parser.add_argument("--ml-probability-threshold", type=float, default=0.70,
                    help="Minimum ML probability threshold for showing results (default: 0.70)")
    parser.add_argument("--momentum-data-period", type=str, default="18mo",
                    help="Data period for momentum analysis (default: 18mo)")
    parser.add_argument("--trend-weight", type=float, default=0.35,
                    help="Weight for long-term trend component in momentum scoring (default: 0.35)")
    parser.add_argument("--setup-weight", type=float, default=0.20,
                    help="Weight for SMA crossover setup component in momentum scoring (default: 0.20)")
    parser.add_argument("--volume-weight", type=float, default=0.20,
                    help="Weight for volume confirmation component in momentum scoring (default: 0.20)")
    parser.add_argument("--rsi-period", type=int, default=14,
                    help="RSI calculation period (default: 14)")
    parser.add_argument("--ema-short", type=int, default=13,
                    help="Short EMA period for momentum analysis (default: 13)")
    parser.add_argument("--ema-long", type=int, default=48,
                    help="Long EMA period for momentum analysis (default: 48)")

    args = parser.parse_args()

    config = {
        "period": args.period,
        "future_days": args.future_days,
        "threshold": args.threshold,
        "n_estimators": args.n_estimators,
        "use_sharpe_labeling": args.use_sharpe_labeling,
        "model": args.model,
        "grid_search": args.grid_search,
        "ensemble_runs": args.ensemble_runs,
        "integrate_market": not args.no_integrate_market,
        "sector_filter": args.sector,
        "hot_stocks_count": args.hot_stocks,
        
        # === ADVANCED CONFIGURATION OVERRIDES ===
        "discovery_threshold": args.ml_probability_threshold,
        "momentum_data_period": args.momentum_data_period,
        "momentum_weights": {
            "trend": args.trend_weight,
            "setup": args.setup_weight,
            "volume": args.volume_weight,
            # Calculate remaining weights proportionally
            "price_sma20": 0.15,  # Keep default
            "ema": 0.05,          # Keep default  
            "rsi": 0.05           # Keep default
        },
        "indicators": {
            "rsi_period": args.rsi_period,
            "ema_short": args.ema_short,
            "ema_long": args.ema_long
        }
    }
    
    # Normalize momentum weights to ensure they sum to 1.0
    total_weight = sum(config["momentum_weights"].values())
    if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
        from ai_stock_screener.output_formatter import console
        console.print(f"⚠️ Warning: Momentum weights sum to {total_weight:.3f}, normalizing to 1.0")
        for key in config["momentum_weights"]:
            config["momentum_weights"][key] /= total_weight

    # Parse and get ticker index(es) based on selection
    index_list = [idx.strip() for idx in args.index.split(",") if idx.strip()]
    
    if not index_list:
        from ai_stock_screener.output_formatter import console
        console.print("❌ No valid indexes specified")
        sys.exit(1)
    
    # Show what indexes are being used
    if len(index_list) > 1:
        from ai_stock_screener.output_formatter import console
        console.print(f"📊 Combining indexes: {', '.join(index_list)}")
        
    tickers = get_combined_tickers(index_list)
    
    # Show final index size
    if len(index_list) > 1:
        from ai_stock_screener.output_formatter import console
        console.print(f"📈 Combined index: {len(tickers)} unique stocks")

    # Hot-stocks mode - standalone momentum scanner
    if args.hot_stocks > 0:
        run_hot_stocks_scanner(tickers, config, args.hot_stocks)
        return
        
    # Regular ML modes - require explicit mode selection
    if args.mode == "eval":
        if not args.tickers:
            from ai_stock_screener.output_formatter import console
            console.print("❌ Please provide --tickers for eval mode.")
            sys.exit(1)
        ticker_list = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        run_screening(ticker_list, config, mode="eval", news_analysis=args.news)
    elif args.mode == "discovery":
        run_screening(tickers, config, mode="discovery", news_analysis=args.news)
    else:
        from ai_stock_screener.output_formatter import console
        console.print("❌ Please specify either --mode (discovery/eval) or --hot-stocks COUNT")
        console.print("💡 Examples:")
        console.print("   python -m ai_stock_screener.cli --mode discovery")
        console.print("   python -m ai_stock_screener.cli --mode eval --tickers AAPL,NVDA")
        console.print("   python -m ai_stock_screener.cli --hot-stocks 20")
        console.print("   python -m ai_stock_screener.cli --mode discovery --ml-probability-threshold 0.80 --trend-weight 0.40")
        sys.exit(1)

if __name__ == "__main__":
    main()
