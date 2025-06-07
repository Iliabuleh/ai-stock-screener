# Covered Call ETF Analyzer 📊

A powerful CLI tool to analyze dividend returns and capital performance of covered call ETFs. This tool helps investors understand the trade-offs between high dividend yields and potential capital erosion in covered call strategies.

## 🚀 What It Does

- **Analyzes any covered call ETF** (YieldMax, Global X, JPMorgan, etc.)
- **Calculates real dividend returns** based on actual investment amounts
- **Shows capital recovery timeline** - when dividends recover your initial investment
- **Supports multiple investments** - track dollar-cost averaging and multiple purchase dates
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
| `--investment` | Investment amount in dollars (can be repeated) | `1000` | `--investment 5000` |
| `--start-date` | Investment date in YYYY-MM-DD format (can be repeated) | None | `--start-date 2023-01-01` |
| `--period` | Time period for single investment analysis | `2y` | `--period 1y` |
| `--help` | Show help message | - | `--help` |

### Time Period Options:
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

### Multiple Investment Syntax:
For multiple investments, repeat the `--investment` and `--start-date` flags:
```bash
--investment 1000 --start-date 2024-01-01 --investment 500 --start-date 2024-04-01
```
**Note**: Number of `--investment` flags must match number of `--start-date` flags.

## 💡 Usage Examples

### 1. Single Investment - QYLD Analysis
```bash
poetry run covered-call-div --underlying QQQ --etf QYLD --investment 1000 --start-date 2024-01-01
```

### 2. Single Investment with Period
```bash
poetry run covered-call-div --underlying SPY --etf JEPI --investment 5000 --period 1y
```

### 3. Multiple Investments - Dollar Cost Averaging
```bash
poetry run covered-call-div --underlying MSTR --etf MSTY \
  --investment 1000 --start-date 2024-05-01 \
  --investment 500 --start-date 2024-08-01
```

### 4. Multiple Investments - Quarterly Strategy
```bash
poetry run covered-call-div --underlying QQQ --etf QYLD \
  --investment 2000 --start-date 2024-01-01 \
  --investment 1000 --start-date 2024-04-01 \
  --investment 500 --start-date 2024-07-01
```

### 5. Large Portfolio Tracking
```bash
poetry run covered-call-div --underlying NVDA --etf NVDY \
  --investment 5000 --start-date 2023-01-01 \
  --investment 3000 --start-date 2023-06-01 \
  --investment 2000 --start-date 2024-01-01 \
  --investment 1000 --start-date 2024-06-01
```

### 6. Default Analysis (No Flags)
```bash
poetry run covered-call-div --underlying SPY --etf JEPI
```
*Uses default $1,000 investment over 2 years*

## 📊 Sample Output - Multiple Investments

```
📊 Analyzing MSTY (tracking MSTR) with 2 investments
   Investment #1: $1,000.00 on 2024-05-01
   Investment #2: $500.00 on 2024-08-01

📅 Downloading data from 2024-05-01 to 2025-06-07

💰 Investment Summary:
   #1: $1,000.00 on 2024-05-01 @ $27.07 = 36.94 shares
   #2: $500.00 on 2024-08-01 @ $27.85 = 17.95 shares

📊 Overall Investment Stats:
💵 Total invested: $1,500.00
📈 Total shares: 54.89
💰 Average price per share: $27.33

📦 Dividend Payments Summary:
Total dividend payments: 15
Total dividends per share: $38.370
Average dividend per share: $2.558

📊 Current Position Analysis:
📈 Current MSTY price: $20.59
💼 Total shares owned: 54.89
💵 Total invested: $1,500.00
💰 Total dividends received: $1,890.60
📊 Current portfolio value: $1,130.28
🏆 Total portfolio value: $3,020.88

📈 Performance Analysis:
🎯 Total return: $1,520.88 (+101.4%)
📉 Capital gain/loss: $-369.72 (-24.6%)
💰 Dividend return: $1,890.60 (126.0%)

✅ Capital Recovery: ACHIEVED!
🎉 Dividend excess: $390.60 (26.0% above total investment)
⏱️  Recovery timeline:
   📅 Recovery date: 2025-02-13
   📊 Time to recovery: 288 days (9.5 months, 0.8 years)

✅ Overall Result: PROFITABLE!
```

## 🎯 Key Metrics Explained

### 📊 **Investment Tracking (Multiple Investments)**
- **Investment Summary**: Each purchase with date, price, and shares bought
- **Overall Stats**: Total invested, total shares, average price per share
- **Running Analysis**: How your position builds over time

### 💰 **Dividend Analysis**
- **Total dividends per share**: Cumulative dividend payments per share
- **Total dividends received**: Total dollar amount based on shares owned at each payment date
- **Average dividend per share**: Mean dividend payment per distribution

### 📈 **Performance Analysis**
- **Total return**: Complete performance including both capital and dividend returns
- **Capital gain/loss**: Change in share value from total investment to current value
- **Dividend return**: Total dividend income as percentage of total investment

### ⏱️ **Capital Recovery Analysis**
- **Capital recovery**: When total dividends equal your total investment amount
- **Recovery timeline**: Time from first investment to dividend recovery
- **Excess return**: Dividend income above your total investment

## 🧠 Investment Strategies Supported

### 💼 **Single Investment Analysis**
- One-time investment analysis
- Perfect for analyzing past performance
- Compare different entry points

### 📈 **Dollar-Cost Averaging (DCA)**
- Multiple investments over time
- Analyze average price effects
- Track recovery timeline across multiple purchases

### 🎯 **Strategic Timing**
- Compare different investment dates
- Analyze how timing affects returns
- Optimize entry points

### 🔄 **Portfolio Building**
- Track building a position over time
- See how dividends compound with new investments
- Analyze total portfolio performance

## 🧠 Investment Insights

### ✅ **When Covered Call ETFs Work Well:**
- High dividend income needs
- Sideways or mildly bullish markets
- Portfolio income supplementation
- "Cash flow recovery" strategy focus
- Dollar-cost averaging into volatile positions

### ⚠️ **Potential Drawbacks:**
- **Capital erosion**: Share prices often decline over time
- **Opportunity cost**: May underperform in strong bull markets
- **Complexity**: Multiple moving parts (dividends + price changes)
- **Timing risk**: Earlier investments may underperform later ones

### 🎯 **Key Questions This Tool Answers:**
1. How does dollar-cost averaging affect my returns?
2. When do my dividends recover my total investment?
3. What's my average price across multiple purchases?
4. How do different entry dates affect performance?
5. What's the trade-off between dividend income and capital preservation?

## 🔍 Understanding the Data

### **Real vs. Adjusted Prices**
The tool uses `auto_adjust=False` to show **real market prices** rather than adjusted prices. This gives you the true investor experience and shows actual capital erosion/gains.

### **Dividend Mapping**  
Dividends are mapped to the closest trading day, ensuring accurate timing of income vs. price movements.

### **Multiple Investment Tracking**
Each investment is tracked separately with its own purchase date and price, then combined for total portfolio analysis.

### **Share Calculations**
All dollar amounts are calculated based on the actual number of shares you could buy at each investment date, providing realistic scenarios.

## 🚨 Important Notes

- **Past performance** does not guarantee future results
- **Covered call strategies** inherently limit upside potential in exchange for income
- **Market conditions** greatly affect covered call ETF performance
- **Tax implications** of frequent dividend payments should be considered
- **Multiple investments** show the power of dollar-cost averaging but also timing risks

## 🤝 Contributing

This tool is part of the larger ai-stock-screener project. Feel free to suggest improvements or report issues!

## 📜 License

Part of the ai-stock-screener project by Ilia Buleh.

---

**Happy analyzing! 📊✨** 