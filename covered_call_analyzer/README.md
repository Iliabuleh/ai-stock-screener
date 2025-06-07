# Covered Call ETF Analyzer 📊

A powerful CLI tool to analyze dividend returns and capital performance of covered call ETFs. This tool helps investors understand the trade-offs between high dividend yields and potential capital erosion in covered call strategies.

## 🚀 What It Does

- **Analyzes any covered call ETF** (YieldMax, Global X, JPMorgan, etc.)
- **Calculates real dividend returns** based on actual investment amounts
- **Shows capital recovery timeline** - when dividends recover your initial investment
- **Compares ETF performance** vs underlying asset
- **Uses real market prices** (not adjusted prices) for accurate analysis
- **Provides total return analysis** including both dividends and capital gains/losses

## 📈 Supported ETF Types

### YieldMax ETFs
- **NVDY** - NVIDIA covered call ETF
- **TSLY** - Tesla covered call ETF  
- **MSTY** - MicroStrategy covered call ETF
- **AMZY** - Amazon covered call ETF

### Global X ETFs
- **QYLD** - NASDAQ-100 covered call ETF
- **RYLD** - Russell 2000 covered call ETF
- **XYLD** - S&P 500 covered call ETF

### JPMorgan ETFs
- **JEPI** - Equity Premium Income ETF
- **JEPQ** - NASDAQ Equity Premium Income ETF

### And many more!

## 🛠️ Installation

The tool is already installed as part of the ai-stock-screener package. If you need to install dependencies:

```bash
poetry install
```

## 📋 Command Line Options

```bash
poetry run covered-call-div [OPTIONS]
```

### Available Flags:

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--underlying` | Underlying stock/ETF ticker | `NVDA` | `--underlying QQQ` |
| `--etf` | Covered call ETF ticker | `NVDY` | `--etf QYLD` |
| `--investment` | Investment amount in dollars | `1000` | `--investment 5000` |
| `--period` | Time period for analysis | `2y` | `--period 1y` |
| `--start-date` | Custom start date (YYYY-MM-DD) | None | `--start-date 2023-01-01` |
| `--help` | Show help message | - | `--help` |

### Time Period Options:
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

## 💡 Usage Examples

### 1. Analyze QYLD (NASDAQ-100 Covered Call ETF)
```bash
poetry run covered-call-div --underlying QQQ --etf QYLD --investment 1000
```

### 2. Analyze JEPI with Custom Investment Amount
```bash
poetry run covered-call-div --underlying SPY --etf JEPI --investment 5000 --period 1y
```

### 3. Analyze TSLY from Specific Start Date
```bash
poetry run covered-call-div --underlying TSLA --etf TSLY --investment 2000 --start-date 2023-06-01
```

### 4. Analyze MSTY with Large Investment
```bash
poetry run covered-call-div --underlying MSTR --etf MSTY --investment 10000 --period 6mo
```

## 📊 Sample Output

```
📊 Analyzing QYLD (tracking QQQ) with $1,000.00 investment over 2y
📅 Data range: 2023-06-07 to 2025-06-06
💰 Initial QYLD price: $17.73
📈 Shares bought: 56.40 shares

📦 All Dividend Payment Rows:
[Detailed dividend payment table showing dates, prices, and payments]

📊 Summary:
Total dividends per share: $4.489
Total dividends received: $253.19
Number of dividend payments: 25
Average dividend per share: $0.180
Average dividend payment: $10.13

📊 QYLD Price Analysis:
💰 Initial QYLD price: $17.73
📈 Current QYLD price: $16.57
📉 Price change: $-1.16 (-6.5%)

📌 Final Investment Analysis:
💵 Initial investment: $1,000.00
💰 Total dividends received: $253.19
📊 Current portfolio value: $934.57
🏆 Total portfolio value: $1,187.76

📌 Capital Recovery Analysis:
❌ Dividends alone have not yet recovered your initial investment.
💸 Dividend shortfall: $746.81 (74.7% of initial investment)

📈 Total Return Analysis:
🎯 Total return: $187.76 (+18.8%)
✅ Profitable investment!
```

## 🎯 Key Metrics Explained

### 📊 **Dividend Analysis**
- **Total dividends per share**: Cumulative dividend payments per share
- **Total dividends received**: Total dollar amount of dividends based on shares owned
- **Average dividend per share**: Mean dividend payment per distribution

### 💰 **Investment Performance**
- **Initial investment**: Your original dollar investment
- **Current portfolio value**: Current market value of your shares
- **Total portfolio value**: Current value + all dividends received

### ⏱️ **Capital Recovery Analysis**
- **Capital recovery**: When total dividends equal your initial investment
- **Recovery timeline**: How long it takes to recover your investment through dividends alone
- **Excess return**: Dividend income above your initial investment

### 📈 **Total Return**
- **Total return**: Complete performance including both capital gains/losses AND dividend income
- **Return percentage**: Total return as a percentage of initial investment

## 🧠 Investment Insights

### ✅ **When Covered Call ETFs Work Well:**
- High dividend income needs
- Sideways or mildly bullish markets
- Portfolio income supplementation
- "Cash flow recovery" strategy focus

### ⚠️ **Potential Drawbacks:**
- **Capital erosion**: Share prices often decline over time
- **Opportunity cost**: May underperform in strong bull markets
- **Complexity**: Multiple moving parts (dividends + price changes)

### 🎯 **Key Questions This Tool Answers:**
1. How long until dividends recover my initial investment?
2. What's my total return including both dividends and capital changes?
3. How does the ETF perform vs. just holding the underlying asset?
4. What's the trade-off between dividend income and capital preservation?

## 🔍 Understanding the Data

### **Real vs. Adjusted Prices**
The tool uses `auto_adjust=False` to show **real market prices** rather than adjusted prices. This gives you the true investor experience and shows actual capital erosion/gains.

### **Dividend Mapping**  
Dividends are mapped to the closest trading day, ensuring accurate timing of income vs. price movements.

### **Share Calculations**
All dollar amounts are calculated based on the actual number of shares you could buy with your investment amount, providing realistic scenarios.

## 🚨 Important Notes

- **Past performance** does not guarantee future results
- **Covered call strategies** inherently limit upside potential in exchange for income
- **Market conditions** greatly affect covered call ETF performance
- **Tax implications** of frequent dividend payments should be considered

## 🤝 Contributing

This tool is part of the larger ai-stock-screener project. Feel free to suggest improvements or report issues!

## 📜 License

Part of the ai-stock-screener project by Ilia Buleh.

---

**Happy analyzing! 📊✨** 