import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from typing import List, Tuple

def main():
    """Main CLI entry point for covered call ETF analyzer."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze covered call ETF dividend returns with support for multiple investments')
    parser.add_argument('--underlying', default='NVDA', help='Underlying stock ticker (default: NVDA)')
    parser.add_argument('--etf', default='NVDY', help='Covered call ETF ticker (default: NVDY)')
    parser.add_argument('--investment', type=float, action='append', help='Investment amount in dollars. Can be used multiple times for multiple investments.')
    parser.add_argument('--start-date', action='append', help='Start date in YYYY-MM-DD format. Must match number of --investment flags.')
    parser.add_argument('--period', default='2y', help='Time period for analysis (default: 2y, used for single investment without start-date)')
    args = parser.parse_args()

    # Handle investment inputs
    investments = []
    
    if args.investment and args.start_date:
        # Multiple investments mode with explicit dates
        if len(args.investment) != len(args.start_date):
            raise ValueError(f"Number of --investment flags ({len(args.investment)}) must match number of --start-date flags ({len(args.start_date)})")
        
        # Validate dates and pair them with investments
        for amount, date_str in zip(args.investment, args.start_date):
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                investments.append((amount, date_str))
            except ValueError:
                raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD format.")
        
        print(f"📊 Analyzing {args.etf} (tracking {args.underlying}) with {len(investments)} investments")
        for i, (amount, date) in enumerate(investments, 1):
            print(f"   Investment #{i}: ${amount:,.2f} on {date}")
            
    elif args.investment and not args.start_date:
        # Multiple investments without explicit dates - use period to calculate dates
        print(f"⚠️  Warning: Multiple --investment flags provided without --start-date. Using default 2-year period for the first investment.")
        amount = args.investment[0]  # Use only the first investment
        end_date = datetime.now()
        if args.period == '1y':
            start_date = (end_date - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        elif args.period == '2y':
            start_date = (end_date - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        elif args.period == '6mo':
            start_date = (end_date - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
        else:
            start_date = (end_date - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        investments = [(amount, start_date)]
        print(f"📊 Analyzing {args.etf} (tracking {args.underlying}) with ${amount:,.2f} investment")
        
    elif not args.investment and args.start_date:
        raise ValueError("--start-date provided without --investment. Please provide both flags.")
        
    else:
        # Default investment (no flags provided)
        end_date = datetime.now()
        start_date = (end_date - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        investments = [(1000.0, start_date)]
        print(f"📊 Analyzing {args.etf} (tracking {args.underlying}) with $1,000.00 default investment")

    # Sort investments by date
    investments.sort(key=lambda x: x[1])
    
    # Determine data range
    earliest_date = investments[0][1]
    latest_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download historical data for the full range
    underlying = args.underlying
    etf_ticker = args.etf
    
    print(f"\n📅 Downloading data from {earliest_date} to {latest_date}")
    underlying_data = yf.download(underlying, start=earliest_date, interval="1d", auto_adjust=False)
    etf_data = yf.download(etf_ticker, start=earliest_date, interval="1d", auto_adjust=False)
    dividends = yf.Ticker(etf_ticker).dividends

    if etf_data.empty:
        print(f"⚠️  Error: Could not get {etf_ticker} price data.")
        return

    # Initialize tracking DataFrames
    etf_df = underlying_data[['Close']].copy()
    etf_df.rename(columns={'Close': f'{underlying}_Close'}, inplace=True)
    
    # Add ETF price data
    etf_df[f'Current_{etf_ticker}_Price'] = etf_data['Close'].reindex(etf_df.index, method='ffill')
    
    # Initialize columns for tracking investments
    etf_df['Total_Shares'] = 0.0
    etf_df['Total_Investment'] = 0.0
    etf_df['Dividend_Per_Share'] = 0.0
    etf_df['Total_Dividend_Payment'] = 0.0
    etf_df['Total_Cumulative_Dividends'] = 0.0
    etf_df['Portfolio_Value'] = 0.0
    etf_df['Total_Portfolio_Value'] = 0.0

    # Process each investment
    investment_summary = []
    running_shares = 0.0
    running_investment = 0.0

    for inv_num, (amount, date) in enumerate(investments, 1):
        investment_date = pd.to_datetime(date).tz_localize(None)
        
        # Find the closest trading date
        if investment_date in etf_df.index:
            actual_date = investment_date
        else:
            # Find next available trading date
            future_dates = etf_df.index[etf_df.index >= investment_date]
            if future_dates.empty:
                print(f"⚠️  Warning: Investment date {date} is after available data range. Skipping.")
                continue
            actual_date = future_dates[0]
        
        # Get price on investment date
        investment_price = etf_df.loc[actual_date, f'Current_{etf_ticker}_Price']
        if hasattr(investment_price, 'item'):
            investment_price = investment_price.item()  # Convert pandas scalar to Python scalar
        shares_bought = amount / investment_price
        
        # Update running totals from this date forward
        mask = etf_df.index >= actual_date
        etf_df.loc[mask, 'Total_Shares'] = etf_df.loc[mask, 'Total_Shares'] + shares_bought
        etf_df.loc[mask, 'Total_Investment'] = etf_df.loc[mask, 'Total_Investment'] + amount
        
        running_shares += shares_bought
        running_investment += amount
        
        investment_summary.append({
            'Investment': inv_num,
            'Date': actual_date.strftime('%Y-%m-%d'),
            'Amount': amount,
            'Price': investment_price,
            'Shares': shares_bought,
            'Running_Shares': running_shares,
            'Running_Investment': running_investment
        })

    # Map dividends into the DataFrame
    for div_date, div_amount in dividends.items():
        div_date = pd.to_datetime(div_date).tz_localize(None)
        if div_date in etf_df.index:
            etf_df.loc[div_date, 'Dividend_Per_Share'] = div_amount
        else:
            # Find next valid trading day
            future_dates = etf_df.index[etf_df.index > div_date]
            if not future_dates.empty:
                etf_df.loc[future_dates[0], 'Dividend_Per_Share'] = div_amount

    # Calculate dividend payments and portfolio values
    cumulative_dividends = 0.0
    for i in range(len(etf_df)):
        date = etf_df.index[i]
        shares_owned = etf_df['Total_Shares'].iloc[i]
        dividend_per_share = etf_df['Dividend_Per_Share'].iloc[i]
        current_price = etf_df[f'Current_{etf_ticker}_Price'].iloc[i]
        
        # Calculate dividend payment for this date
        dividend_payment = dividend_per_share * shares_owned
        cumulative_dividends += dividend_payment
        
        etf_df.iloc[i, etf_df.columns.get_loc('Total_Dividend_Payment')] = dividend_payment
        etf_df.iloc[i, etf_df.columns.get_loc('Total_Cumulative_Dividends')] = cumulative_dividends
        etf_df.iloc[i, etf_df.columns.get_loc('Portfolio_Value')] = current_price * shares_owned
        etf_df.iloc[i, etf_df.columns.get_loc('Total_Portfolio_Value')] = (current_price * shares_owned) + cumulative_dividends

    # Display investment summary
    print(f"\n💰 Investment Summary:")
    for inv in investment_summary:
        print(f"   #{inv['Investment']}: ${inv['Amount']:,.2f} on {inv['Date']} @ ${inv['Price']:.2f} = {inv['Shares']:.2f} shares")
    
    total_invested = sum(inv['Amount'] for inv in investment_summary)
    total_shares = sum(inv['Shares'] for inv in investment_summary)
    avg_price = total_invested / total_shares
    
    print(f"\n📊 Overall Investment Stats:")
    print(f"💵 Total invested: ${total_invested:,.2f}")
    print(f"📈 Total shares: {total_shares:.2f}")
    print(f"💰 Average price per share: ${avg_price:.2f}")

    # Show dividend payments
    dividend_rows = etf_df[etf_df['Dividend_Per_Share'] > 0]
    if not dividend_rows.empty:
        print(f"\n📦 Dividend Payments Summary:")
        print(f"Total dividend payments: {len(dividend_rows)}")
        print(f"Total dividends per share: ${dividend_rows['Dividend_Per_Share'].sum():.3f}")
        print(f"Average dividend per share: ${dividend_rows['Dividend_Per_Share'].mean():.3f}")
        
        print(f"\n📦 Recent Dividend Payments:")
        recent_divs = dividend_rows[['Dividend_Per_Share', 'Total_Shares', 'Total_Dividend_Payment', 'Total_Cumulative_Dividends']].tail(5)
        print(recent_divs)

    # Final analysis
    final_shares = etf_df['Total_Shares'].iloc[-1]
    final_investment = etf_df['Total_Investment'].iloc[-1]
    final_cumulative_divs = etf_df['Total_Cumulative_Dividends'].iloc[-1]
    final_portfolio_value = etf_df['Portfolio_Value'].iloc[-1]
    final_total_value = etf_df['Total_Portfolio_Value'].iloc[-1]
    current_price = etf_df[f'Current_{etf_ticker}_Price'].iloc[-1]

    print(f"\n📊 Current Position Analysis:")
    print(f"📈 Current {etf_ticker} price: ${current_price:.2f}")
    print(f"💼 Total shares owned: {final_shares:.2f}")
    print(f"💵 Total invested: ${final_investment:,.2f}")
    print(f"💰 Total dividends received: ${final_cumulative_divs:.2f}")
    print(f"📊 Current portfolio value: ${final_portfolio_value:.2f}")
    print(f"🏆 Total portfolio value: ${final_total_value:.2f}")

    # Performance analysis
    total_return = final_total_value - final_investment
    return_pct = (total_return / final_investment) * 100
    
    capital_gain_loss = final_portfolio_value - final_investment
    capital_return_pct = (capital_gain_loss / final_investment) * 100

    print(f"\n📈 Performance Analysis:")
    print(f"🎯 Total return: ${total_return:.2f} ({return_pct:+.1f}%)")
    print(f"📉 Capital gain/loss: ${capital_gain_loss:.2f} ({capital_return_pct:+.1f}%)")
    print(f"💰 Dividend return: ${final_cumulative_divs:.2f} ({final_cumulative_divs/final_investment*100:.1f}%)")

    # Capital recovery analysis
    if final_cumulative_divs >= final_investment:
        print(f"\n✅ Capital Recovery: ACHIEVED!")
        excess = final_cumulative_divs - final_investment
        print(f"🎉 Dividend excess: ${excess:.2f} ({excess/final_investment*100:.1f}% above total investment)")
        
        # Find recovery date
        recovery_mask = etf_df['Total_Cumulative_Dividends'] >= final_investment
        if recovery_mask.any():
            recovery_date = etf_df[recovery_mask].index[0]
            first_investment_date = pd.to_datetime(investments[0][1])
            days_to_recovery = (recovery_date - first_investment_date).days
            months_to_recovery = days_to_recovery / 30.44
            years_to_recovery = days_to_recovery / 365.25
            
            print(f"⏱️  Recovery timeline:")
            print(f"   📅 Recovery date: {recovery_date.strftime('%Y-%m-%d')}")
            print(f"   📊 Time to recovery: {days_to_recovery} days ({months_to_recovery:.1f} months, {years_to_recovery:.1f} years)")
    else:
        shortfall = final_investment - final_cumulative_divs
        print(f"\n❌ Capital Recovery: NOT YET ACHIEVED")
        print(f"💸 Dividend shortfall: ${shortfall:.2f} ({shortfall/final_investment*100:.1f}% of total investment)")

    if total_return > 0:
        print(f"\n✅ Overall Result: PROFITABLE!")
    else:
        print(f"\n❌ Overall Result: CURRENTLY AT A LOSS")

if __name__ == "__main__":
    main() 