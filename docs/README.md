# AI Stock Screener Documentation

This documentation is organized into two main components:

## 📊 AI Screener
The AI Screener focuses on intelligent stock discovery and analysis using advanced momentum detection and regime-based filtering.

**Documentation:**
- [Momentum Scanner Architecture](ai_screener/momentum-scanner-architecture.md) - Core scanning engine design
- [Regime Calculation](ai_screener/regime-calculation.md) - Market regime detection methodology

## 🎯 Pro Screener  
The Pro Screener provides sophisticated technical analysis with a 3-pillar scoring system for institutional-grade stock evaluation.

**Documentation:**
- [Technical Scoring Algorithm](pro_screener/technical-scoring-algorithm.md) - Comprehensive algorithm documentation

## Getting Started

Choose your screening approach:

### AI Screener (Discovery Mode)
```bash
python -m ai_stock_screener.cli discover --universe sp500 --limit 10
```

### Pro Screener (Technical Analysis)
```bash
python -m ai_stock_screener.cli discover --universe sp500 --limit 10 --min-score 70
```

## Architecture Overview

Both screeners share core infrastructure while providing different analytical approaches:

- **AI Screener**: Momentum-based discovery with regime awareness
- **Pro Screener**: Technical scoring with volume profile, relative strength, and pattern analysis

For detailed implementation guides, see the respective subdirectories.

## 🚀 Contributing to Docs

When adding new documentation:
1. Create markdown files in the appropriate subdirectory (`ai_screener/` or `pro_screener/`)
2. Update this README index with links to new documentation
3. Use clear examples and code snippets
4. Include emoji for better readability 📊
5. Reference actual code locations when relevant 