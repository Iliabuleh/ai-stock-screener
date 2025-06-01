# ai_screener/cli.py

import argparse
import sys
from ai_screener import get_sp500_tickers, run_screening

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Stock Screener CLI")
    parser.add_argument("--mode", choices=["eval", "discovery"], default="eval",
                        help="Run mode: eval (use provided tickers) or discovery (scan S&P 500)")
    parser.add_argument("--tickers", type=str,
                        help="Comma-separated list of tickers for eval mode (e.g., AAPL,NVDA,MSFT)")
    parser.add_argument("--period", default="6mo", help="Historical data period (default: 6mo)")
    parser.add_argument("--future_days", type=int, default=5, help="Days ahead to evaluate returns (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Label threshold (default: 0.0 for any growth)")
    parser.add_argument("--n_estimators", type=int, default=300, help="RandomForest trees (default: 300)")

    args = parser.parse_args()

    config = {
        "period": args.period,
        "future_days": args.future_days,
        "threshold": args.threshold,
        "n_estimators": args.n_estimators
    }

    print("\n🛠️  Configuration Parameters (overrides or defaults):")
    for k, v in config.items():
        print(f"   {k}: {v}")
    print("-" * 45)

    if args.mode == "eval":
        if not args.tickers:
            print("❌ Please provide --tickers for eval mode.")
            sys.exit(1)
        ticker_list = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        run_screening(ticker_list, config)
    else:
        tickers = get_sp500_tickers()
        print(f"🔎 Discovery Mode: Screening {len(tickers)} S&P 500 tickers...\n")
        run_screening(tickers, config)

if __name__ == "__main__":
    main()
