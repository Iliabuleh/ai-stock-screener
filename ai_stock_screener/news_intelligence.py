"""
News & Sentiment Intelligence Module
Professional-grade news analysis for stock screening
"""

import requests
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import re
from textblob import TextBlob
import json
import time

@dataclass
class NewsItem:
    """Individual news article data structure"""
    ticker: str
    headline: str
    summary: str
    published_at: datetime
    source: str
    url: str
    sentiment_score: float  # -1 to +1 (negative to positive)
    sentiment_magnitude: float  # 0 to 1 (confidence in sentiment)
    event_type: str  # "earnings", "product", "legal", "regulatory", "analyst", "general"
    impact_level: str  # "high", "medium", "low"

@dataclass
class SentimentAnalysis:
    """Aggregated sentiment analysis for a stock"""
    ticker: str
    overall_sentiment: float  # -1 to +1
    sentiment_confidence: float  # 0 to 1
    news_volume: int  # Number of recent articles
    news_velocity: float  # Articles per day over last 7 days
    sentiment_trend: str  # "improving", "deteriorating", "stable"
    dominant_themes: List[str]  # Top news themes
    last_updated: datetime

@dataclass
class NewsIntelligence:
    """Comprehensive news intelligence for portfolio"""
    sentiment_analyses: Dict[str, SentimentAnalysis]
    market_sentiment: float  # Overall market news sentiment
    sector_sentiments: Dict[str, float]  # Sentiment by sector
    breaking_news: List[NewsItem]  # High-impact recent news
    earnings_calendar: Dict[str, datetime]  # Upcoming earnings dates
    last_update: datetime

# News source configurations
NEWS_SOURCES = {
    "yahoo_finance": {
        "base_url": "https://query1.finance.yahoo.com/v1/finance/search",
        "rate_limit": 5,  # requests per second
        "reliability": 0.8
    },
    "alpha_vantage": {
        "base_url": "https://www.alphavantage.co/query",
        "api_key_required": True,
        "rate_limit": 1,  # requests per second  
        "reliability": 0.9
    }
}

def get_yahoo_news(ticker: str, max_articles: int = 10) -> List[NewsItem]:
    """Fetch news from Yahoo Finance (free tier)"""
    try:
        # Use yfinance to get news - it's more reliable than direct API calls
        stock = yf.Ticker(ticker)
        news_data = stock.news
        
        news_items = []
        for article in news_data[:max_articles]:
            try:
                # Parse Yahoo Finance news format
                headline = article.get('title', 'No headline')
                summary = article.get('summary', headline)  # Fallback to headline
                published_timestamp = article.get('providerPublishTime', int(time.time()))
                published_at = datetime.fromtimestamp(published_timestamp)
                source = article.get('publisher', 'Yahoo Finance')
                url = article.get('link', '')
                
                # Basic sentiment analysis using TextBlob
                sentiment_score, sentiment_magnitude = analyze_text_sentiment(headline + " " + summary)
                
                # Classify event type based on headline keywords
                event_type = classify_news_event(headline + " " + summary)
                
                # Determine impact level
                impact_level = determine_impact_level(headline, event_type)
                
                news_item = NewsItem(
                    ticker=ticker,
                    headline=headline,
                    summary=summary,
                    published_at=published_at,
                    source=source,
                    url=url,
                    sentiment_score=sentiment_score,
                    sentiment_magnitude=sentiment_magnitude,
                    event_type=event_type,
                    impact_level=impact_level
                )
                
                news_items.append(news_item)
                
            except Exception as e:
                print(f"⚠️ Error parsing news item for {ticker}: {e}")
                continue
                
        return news_items
        
    except Exception as e:
        print(f"⚠️ Error fetching news for {ticker}: {e}")
        return []

def analyze_text_sentiment(text: str) -> Tuple[float, float]:
    """Analyze sentiment of text using TextBlob"""
    try:
        blob = TextBlob(text)
        
        # TextBlob returns polarity (-1 to 1) and subjectivity (0 to 1)
        sentiment_score = blob.sentiment.polarity
        sentiment_magnitude = blob.sentiment.subjectivity
        
        return sentiment_score, sentiment_magnitude
        
    except Exception as e:
        print(f"⚠️ Error in sentiment analysis: {e}")
        return 0.0, 0.0

def classify_news_event(text: str) -> str:
    """Classify news event type based on keywords"""
    text_lower = text.lower()
    
    # Earnings-related keywords
    if any(keyword in text_lower for keyword in ['earnings', 'revenue', 'profit', 'eps', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
        return "earnings"
    
    # Product/business keywords  
    elif any(keyword in text_lower for keyword in ['product', 'launch', 'release', 'innovation', 'partnership', 'acquisition', 'merger']):
        return "product"
    
    # Legal/regulatory keywords
    elif any(keyword in text_lower for keyword in ['lawsuit', 'sec', 'fda', 'regulation', 'investigation', 'fine', 'settlement']):
        return "legal"
    
    # Analyst keywords
    elif any(keyword in text_lower for keyword in ['upgrade', 'downgrade', 'target', 'rating', 'analyst', 'price target']):
        return "analyst"
    
    # Regulatory keywords
    elif any(keyword in text_lower for keyword in ['regulatory', 'approval', 'fda', 'patent', 'license']):
        return "regulatory"
    
    else:
        return "general"

def determine_impact_level(headline: str, event_type: str) -> str:
    """Determine the potential market impact level of news"""
    headline_lower = headline.lower()
    
    # High impact indicators
    high_impact_words = ['breakthrough', 'major', 'significant', 'massive', 'record', 'historic', 'unprecedented']
    if any(word in headline_lower for word in high_impact_words):
        return "high"
    
    # Event type based impact
    if event_type in ["earnings", "legal", "regulatory"]:
        return "high"
    elif event_type in ["analyst", "product"]:
        return "medium"
    else:
        return "low"

def calculate_sentiment_analysis(ticker: str, news_items: List[NewsItem]) -> SentimentAnalysis:
    """Calculate aggregated sentiment analysis for a stock"""
    
    if not news_items:
        return SentimentAnalysis(
            ticker=ticker,
            overall_sentiment=0.0,
            sentiment_confidence=0.0,
            news_volume=0,
            news_velocity=0.0,
            sentiment_trend="stable",
            dominant_themes=[],
            last_updated=datetime.now()
        )
    
    # Calculate weighted sentiment (more recent news has higher weight)
    now = datetime.now()
    weighted_sentiment = 0.0
    total_weight = 0.0
    
    for news in news_items:
        # Time decay: news loses impact over time
        hours_old = (now - news.published_at).total_seconds() / 3600
        time_weight = max(0.1, 1.0 - (hours_old / 168))  # 7-day decay to 10%
        
        # Impact weight: high impact news matters more
        impact_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}[news.impact_level]
        
        # Confidence weight: higher magnitude sentiment matters more
        confidence_weight = news.sentiment_magnitude
        
        final_weight = time_weight * impact_weight * confidence_weight
        weighted_sentiment += news.sentiment_score * final_weight
        total_weight += final_weight
    
    overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    # Calculate sentiment confidence based on consistency and volume
    sentiment_scores = [n.sentiment_score for n in news_items]
    sentiment_std = pd.Series(sentiment_scores).std() if len(sentiment_scores) > 1 else 1.0
    sentiment_confidence = min(1.0, len(news_items) / 10) * (1.0 - min(1.0, sentiment_std))
    
    # Calculate news velocity (articles per day over last 7 days)
    recent_news = [n for n in news_items if (now - n.published_at).days <= 7]
    news_velocity = len(recent_news) / 7.0
    
    # Determine sentiment trend
    if len(news_items) >= 3:
        recent_sentiment = sum(n.sentiment_score for n in news_items[:3]) / 3
        older_sentiment = sum(n.sentiment_score for n in news_items[3:6]) / max(1, len(news_items[3:6]))
        
        if recent_sentiment > older_sentiment + 0.1:
            sentiment_trend = "improving"
        elif recent_sentiment < older_sentiment - 0.1:
            sentiment_trend = "deteriorating"
        else:
            sentiment_trend = "stable"
    else:
        sentiment_trend = "stable"
    
    # Extract dominant themes
    event_types = [n.event_type for n in news_items]
    dominant_themes = list(pd.Series(event_types).value_counts().head(3).index)
    
    return SentimentAnalysis(
        ticker=ticker,
        overall_sentiment=overall_sentiment,
        sentiment_confidence=sentiment_confidence,
        news_volume=len(news_items),
        news_velocity=news_velocity,
        sentiment_trend=sentiment_trend,
        dominant_themes=dominant_themes,
        last_updated=now
    )

def calculate_news_multiplier(sentiment_analysis: SentimentAnalysis) -> float:
    """Calculate news-based prediction multiplier"""
    
    if sentiment_analysis.news_volume == 0:
        return 1.0  # No news, no adjustment
    
    base_multiplier = 1.0
    
    # Sentiment impact (stronger for higher confidence)
    sentiment_impact = sentiment_analysis.overall_sentiment * sentiment_analysis.sentiment_confidence
    sentiment_multiplier = 1.0 + (sentiment_impact * 0.15)  # Max ±15% from sentiment
    
    # News velocity impact (more news = more volatility)
    velocity_multiplier = 1.0 + min(0.05, sentiment_analysis.news_velocity * 0.01)  # Max +5% from high news volume
    
    # Trend impact
    trend_multiplier = 1.0
    if sentiment_analysis.sentiment_trend == "improving":
        trend_multiplier = 1.03  # +3% for improving sentiment
    elif sentiment_analysis.sentiment_trend == "deteriorating":
        trend_multiplier = 0.97  # -3% for deteriorating sentiment
    
    final_multiplier = base_multiplier * sentiment_multiplier * velocity_multiplier * trend_multiplier
    
    # Cap the multiplier to prevent extreme adjustments
    return max(0.7, min(1.4, final_multiplier))  # Cap between 70%-140%

def get_news_intelligence(tickers: List[str]) -> NewsIntelligence:
    """Get comprehensive news intelligence for a list of tickers"""
    
    sentiment_analyses = {}
    all_news = []
    
    for ticker in tickers:
        print(f"📰 Fetching news for {ticker}...")
        news_items = get_yahoo_news(ticker)
        
        if news_items:
            sentiment_analysis = calculate_sentiment_analysis(ticker, news_items)
            sentiment_analyses[ticker] = sentiment_analysis
            all_news.extend(news_items)
        
        # Rate limiting to be respectful to APIs
        time.sleep(0.2)  # 200ms delay between requests
    
    # Calculate market-wide sentiment
    if sentiment_analyses:
        market_sentiment = sum(s.overall_sentiment for s in sentiment_analyses.values()) / len(sentiment_analyses)
    else:
        market_sentiment = 0.0
    
    # Find breaking news (high impact news from last 24 hours)
    now = datetime.now()
    breaking_news = [
        news for news in all_news 
        if news.impact_level == "high" and (now - news.published_at).hours <= 24
    ]
    
    # Sort by recency
    breaking_news.sort(key=lambda x: x.published_at, reverse=True)
    
    return NewsIntelligence(
        sentiment_analyses=sentiment_analyses,
        market_sentiment=market_sentiment,
        sector_sentiments={},  # TODO: Implement sector-level sentiment
        breaking_news=breaking_news[:10],  # Top 10 breaking news
        earnings_calendar={},  # TODO: Implement earnings calendar
        last_update=now
    ) 