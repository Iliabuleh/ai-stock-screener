import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    """Main CLI entry point for covered call ETF analyzer."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze covered call ETF dividend returns')
    parser.add_argument('--underlying', default='NVDA', help='Underlying stock ticker (default: NVDA)')
    parser.add_argument('--etf', default='NVDY', help='Covered call ETF ticker (default: NVDY)')
    parser.add_argument('--investment', type=float, default=1000, help='Total investment amount in dollars (default: 1000)')
    parser.add_argument('--period', default='2y', help='Time period for analysis (default: 2y)')
    parser.add_argument('--start-date', help='Start date in YYYY-MM-DD format (overrides period)')
    args = parser.parse_args()

    # Define tickers from arguments
    underlying = args.underlying
    etf_ticker = args.etf
    total_investment = args.investment
    period = args.period
    start_date = getattr(args, 'start_date', None)

    print(f"📊 Analyzing {etf_ticker} (tracking {underlying}) with ${total_investment:,.2f} investment over {period}")

    # Download historical data
    if start_date:
        nvda = yf.download(underlying, start=start_date, interval="1d", auto_adjust=False)
        etf_data = yf.download(etf_ticker, start=start_date, interval="1d", auto_adjust=False)
    else:
        nvda = yf.download(underlying, period=period, interval="1d", auto_adjust=False)
        etf_data = yf.download(etf_ticker, period=period, interval="1d", auto_adjust=False)

    dividends = yf.Ticker(etf_ticker).dividends

    # Show actual date range
    if not etf_data.empty:
        print(f"📅 Data range: {etf_data.index[0].strftime('%Y-%m-%d')} to {etf_data.index[-1].strftime('%Y-%m-%d')}")
        initial_etf_price = etf_data['Close'].iloc[0].item()  # Use .item() to extract scalar
        shares_bought = total_investment / initial_etf_price
    else:
        print(f"⚠️  Warning: Could not get {etf_ticker} price data. Using normalized pricing.")
        initial_etf_price = 10  # Fallback to normalized price
        shares_bought = total_investment / initial_etf_price

    print(f"💰 Initial {etf_ticker} price: ${initial_etf_price:.2f}")
    print(f"📈 Shares bought: {shares_bought:.2f} shares")

    # Prepare ETF NAV table
    etf_df = nvda[['Close']].copy()
    etf_df.rename(columns={'Close': f'{underlying}_Close'}, inplace=True)

    initial_underlying_price = etf_df[f'{underlying}_Close'].iloc[0]
    etf_df['ETF_NAV'] = (etf_df[f'{underlying}_Close'] / initial_underlying_price) * initial_etf_price
    etf_df['Dividend_Per_Share'] = 0.0
    etf_df['Cumulative_Dividend_Per_Share'] = 0.0
    etf_df['ETF_NAV_PostDiv'] = etf_df['ETF_NAV'].copy()

    # Map dividends into ETF table on the closest trading date
    for div_date, amount in dividends.items():
        div_date = pd.to_datetime(div_date).tz_localize(None)  # Remove timezone info
        if div_date in etf_df.index:
            etf_df.loc[div_date, 'Dividend_Per_Share'] = amount
        else:
            # Find next valid trading day in index
            future_dates = etf_df.index[etf_df.index > div_date]
            if not future_dates.empty:
                etf_df.loc[future_dates[0], 'Dividend_Per_Share'] = amount

    # Simulate ETF_NAV_PostDiv changes over time
    for i in range(1, len(etf_df)):
        curr_dividend = etf_df['Dividend_Per_Share'].iloc[i]
        if curr_dividend > 0:
            prev_nav = etf_df['ETF_NAV_PostDiv'].iloc[i-1]
            etf_df.iloc[i, etf_df.columns.get_loc('ETF_NAV_PostDiv')] = prev_nav - curr_dividend
        else:
            etf_df.iloc[i, etf_df.columns.get_loc('ETF_NAV_PostDiv')] = etf_df['ETF_NAV_PostDiv'].iloc[i-1]

    # Calculate actual dollar amounts based on shares owned
    etf_df['Cumulative_Dividend_Per_Share'] = etf_df['Dividend_Per_Share'].cumsum()
    etf_df['Total_Dividend_Payment'] = etf_df['Dividend_Per_Share'] * shares_bought
    etf_df['Total_Cumulative_Dividends'] = etf_df['Cumulative_Dividend_Per_Share'] * shares_bought

    # Add current ETF price for comparison
    if not etf_data.empty:
        # Align ETF data with our DataFrame dates using forward fill for missing dates
        etf_df[f'Current_{etf_ticker}_Price'] = etf_data['Close'].reindex(etf_df.index, method='ffill')
    else:
        etf_df[f'Current_{etf_ticker}_Price'] = etf_df['ETF_NAV']  # Fallback

    # Calculate portfolio value using actual ETF market price, not the theoretical NAV
    etf_df['Portfolio_Value'] = etf_df[f'Current_{etf_ticker}_Price'] * shares_bought
    etf_df['Total_Portfolio_Value'] = etf_df['Portfolio_Value'] + etf_df['Total_Cumulative_Dividends']

    # Keep the theoretical NAV calculation for educational purposes
    etf_df['Theoretical_NAV_PostDiv'] = etf_df['ETF_NAV_PostDiv'] * shares_bought

    # Display dividend-paying rows
    dividend_rows = etf_df[etf_df['Dividend_Per_Share'] > 0]
    print("📦 All Dividend Payment Rows:")
    print(dividend_rows[['ETF_NAV', f'Current_{etf_ticker}_Price', 'Dividend_Per_Share', 'Total_Dividend_Payment', 'Total_Cumulative_Dividends', 'Portfolio_Value']])

    print(f"\n📊 Summary:")
    print(f"Total dividends per share: ${dividend_rows['Dividend_Per_Share'].sum():.3f}")
    print(f"Total dividends received: ${dividend_rows['Total_Dividend_Payment'].sum():.2f}")
    print(f"Number of dividend payments: {len(dividend_rows)}")
    print(f"Average dividend per share: ${dividend_rows['Dividend_Per_Share'].mean():.3f}")
    print(f"Average dividend payment: ${dividend_rows['Total_Dividend_Payment'].mean():.2f}")

    # Also show last few rows for context
    print(f"\n📈 Last few trading days:")
    print(etf_df[['ETF_NAV', f'Current_{etf_ticker}_Price', 'Dividend_Per_Share', 'Total_Cumulative_Dividends', 'Portfolio_Value']].tail())

    # Show price analysis
    current_etf_price = etf_df[f'Current_{etf_ticker}_Price'].iloc[-1]
    price_change = current_etf_price - initial_etf_price
    price_change_pct = (price_change / initial_etf_price) * 100

    print(f"\n📊 {etf_ticker} Price Analysis:")
    print(f"💰 Initial {etf_ticker} price: ${initial_etf_price:.2f}")
    print(f"📈 Current {etf_ticker} price: ${current_etf_price:.2f}")
    print(f"📉 Price change: ${price_change:.2f} ({price_change_pct:+.1f}%)")
    print(f"⚠️  Note: If initial price seems low, it might be due to stock splits or data availability.")

    # 🧾 Final Evaluation
    final_cumulative_div_total = etf_df['Total_Cumulative_Dividends'].iloc[-1]  # Total dollars received
    final_portfolio_value = etf_df['Portfolio_Value'].iloc[-1]  # Current portfolio value

    print("\n📌 Final Investment Analysis:")
    print(f"💵 Initial investment: ${total_investment:,.2f}")
    print(f"💰 Total dividends received: ${final_cumulative_div_total:.2f}")
    print(f"📊 Current portfolio value: ${final_portfolio_value:.2f}")
    print(f"🏆 Total portfolio value: ${final_portfolio_value + final_cumulative_div_total:.2f}")

    # Calculate time to capital recovery
    capital_recovery_date = None
    if final_cumulative_div_total >= total_investment:
        # Find when dividends first exceeded initial investment
        recovery_mask = etf_df['Total_Cumulative_Dividends'] >= total_investment
        if recovery_mask.any():
            capital_recovery_date = etf_df[recovery_mask].index[0]
            
    print("\n📌 Capital Recovery Analysis:")
    if final_cumulative_div_total >= total_investment:
        print(f"✅ You recovered your initial investment (${total_investment:,.2f}) in dividends alone!")
        excess = final_cumulative_div_total - total_investment
        print(f"🎉 Excess dividend return: ${excess:.2f} ({excess/total_investment*100:.1f}% above initial investment)")
        
        if capital_recovery_date:
            start_date = etf_df.index[0]
            days_to_recovery = (capital_recovery_date - start_date).days
            months_to_recovery = days_to_recovery / 30.44  # Average month length
            years_to_recovery = days_to_recovery / 365.25   # Average year length
            
            print(f"⏱️  Time to capital recovery:")
            print(f"   📅 Recovery date: {capital_recovery_date.strftime('%Y-%m-%d')}")
            print(f"   📊 Time taken: {days_to_recovery} days ({months_to_recovery:.1f} months, {years_to_recovery:.1f} years)")
    else:
        print(f"❌ Dividends alone have not yet recovered your initial investment.")
        shortfall = total_investment - final_cumulative_div_total
        print(f"💸 Dividend shortfall: ${shortfall:.2f} ({shortfall/total_investment*100:.1f}% of initial investment)")

    # Total return analysis
    total_current_value = final_portfolio_value + final_cumulative_div_total
    total_return = total_current_value - total_investment
    return_pct = (total_return / total_investment) * 100

    print(f"\n📈 Total Return Analysis:")
    print(f"🎯 Total return: ${total_return:.2f} ({return_pct:+.1f}%)")
    if total_return > 0:
        print(f"✅ Profitable investment!")
    else:
        print(f"❌ Currently at a loss.")


if __name__ == "__main__":
    main() 