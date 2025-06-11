"""
News & Sentiment Intelligence Module
Professional-grade news analysis for stock screening
"""

import requests
import yfinance as yf
import pandas as pd
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import json

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

# Enhanced financial keywords and sentiment modifiers
ENHANCED_KEYWORDS = {
    "earnings": [
        'earnings', 'revenue', 'profit', 'eps', 'quarterly', 'q1', 'q2', 'q3', 'q4',
        'guidance', 'outlook', 'consensus', 'estimates', 'forecast', 'results',
        'beat', 'miss', 'exceed', 'surprise', 'report', 'announced'
    ],
    "product": [
        'product', 'launch', 'release', 'innovation', 'partnership', 'acquisition', 'merger',
        'deal', 'agreement', 'contract', 'expansion', 'growth', 'development'
    ],
    "legal": [
        'lawsuit', 'sec', 'fda', 'regulation', 'investigation', 'fine', 'settlement',
        'compliance', 'violation', 'penalty', 'court', 'legal', 'ruling'
    ],
    "analyst": [
        'upgrade', 'downgrade', 'target', 'rating', 'analyst', 'price target',
        'recommendation', 'coverage', 'initiate', 'maintain', 'reiterate'
    ],
    "regulatory": [
        'regulatory', 'approval', 'fda', 'patent', 'license', 'authorization',
        'clearance', 'permit', 'compliance', 'standard'
    ]
}

# Financial sentiment modifiers
SENTIMENT_MODIFIERS = {
    # Positive modifiers
    'beat': +0.3, 'exceed': +0.25, 'strong': +0.2, 'robust': +0.2,
    'growth': +0.15, 'gain': +0.15, 'rise': +0.15, 'surge': +0.2,
    'outperform': +0.25, 'upgrade': +0.2, 'bullish': +0.3,
    
    # Negative modifiers  
    'miss': -0.3, 'disappoint': -0.25, 'weak': -0.2, 'decline': -0.2,
    'fall': -0.15, 'drop': -0.15, 'plunge': -0.25, 'crash': -0.3,
    'underperform': -0.25, 'downgrade': -0.2, 'bearish': -0.3,
    
    # Neutral but contextual
    'stable': 0.05, 'maintain': 0.0, 'hold': 0.0
}

# Alpha Vantage configuration
ALPHA_VANTAGE_CONFIG = {
    "base_url": "https://www.alphavantage.co/query",
    "api_key": os.getenv('ALPHA_VANTAGE_API_KEY'),  # Only check environment variable
    "timeout": 10
}

# News source fallback configuration  
NEWS_FALLBACK_CONFIG = {
    "delay_between_calls": 1,  # Short delay to be respectful to APIs
}

def get_yahoo_news(ticker: str, max_articles: int = 10) -> List[NewsItem]:
    """Fetch news from Yahoo Finance with enhanced processing"""
    try:
        # Use yfinance to get news - it's more reliable than direct API calls
        stock = yf.Ticker(ticker)
        news_data = stock.news
        
        news_items = []
        for article in news_data[:max_articles]:
            try:
                # Parse Yahoo Finance news format - handle new nested structure
                content = article.get('content', {})
                headline = content.get('title') or article.get('title', 'No headline')
                summary = content.get('description') or article.get('summary', headline)  # Fallback to headline
                published_timestamp = content.get('publishedAt') or article.get('providerPublishTime', int(time.time()))
                
                # Handle different timestamp formats
                if isinstance(published_timestamp, str):
                    try:
                        # Try ISO format first
                        published_at = datetime.fromisoformat(published_timestamp.replace('Z', '+00:00'))
                    except:
                        # Fallback to current time
                        published_at = datetime.now()
                else:
                    published_at = datetime.fromtimestamp(published_timestamp)
                
                source = content.get('source') or article.get('publisher', 'Yahoo Finance')
                url = content.get('clickThroughUrl') or article.get('link', '')
                
                # Enhanced sentiment analysis using our improved method
                sentiment_score, sentiment_magnitude = analyze_text_sentiment(headline + " " + summary)
                
                # Calculate relevance score
                relevance_score = calculate_relevance_score(headline, summary, ticker)
                
                # Classify event type using enhanced keywords
                event_type = classify_news_event(headline + " " + summary)
                
                # Determine impact level
                impact_level = determine_impact_level(headline, event_type)
                
                # Apply backend-agnostic normalization for consistent scoring
                # This ensures same sentiment scale regardless of data source
                normalized_sentiment = normalize_sentiment_score(sentiment_score, source)
                weighted_sentiment = normalized_sentiment * relevance_score
                
                news_item = NewsItem(
                    ticker=ticker,
                    headline=headline,
                    summary=summary,
                    published_at=published_at,
                    source=source,
                    url=url,
                    sentiment_score=weighted_sentiment,
                    sentiment_magnitude=sentiment_magnitude * relevance_score,  # Adjust magnitude by relevance
                    event_type=event_type,
                    impact_level=impact_level
                )
                
                # Only include reasonably relevant news
                if relevance_score > 0.2:
                    news_items.append(news_item)
                
            except Exception as e:
                print(f"⚠️ Error parsing news item for {ticker}: {e}")
                continue
                
        return news_items
        
    except Exception as e:
        print(f"⚠️ Error fetching news for {ticker}: {e}")
        return []

def analyze_text_sentiment(text: str) -> Tuple[float, float]:
    """Enhanced sentiment analysis using TextBlob + financial context"""
    try:
        blob = TextBlob(text)
        
        # Base TextBlob sentiment
        base_sentiment = blob.sentiment.polarity
        base_magnitude = blob.sentiment.subjectivity
        
        # Apply financial context modifiers
        text_lower = text.lower()
        sentiment_adjustment = 0.0
        
        for modifier, adjustment in SENTIMENT_MODIFIERS.items():
            if modifier in text_lower:
                sentiment_adjustment += adjustment
        
        # Enhanced sentiment score (capped between -1 and 1)
        enhanced_sentiment = max(-1.0, min(1.0, base_sentiment + sentiment_adjustment))
        
        # Increase magnitude if we found financial keywords
        if sentiment_adjustment != 0:
            base_magnitude = min(1.0, base_magnitude + 0.2)
        
        return enhanced_sentiment, base_magnitude
        
    except Exception as e:
        print(f"⚠️ Error in enhanced sentiment analysis: {e}")
        return 0.0, 0.0

def classify_news_event(text: str) -> str:
    """Enhanced news event classification using expanded keywords"""
    text_lower = text.lower()
    
    # Use enhanced keyword sets
    for event_type, keywords in ENHANCED_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return event_type
    
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

def calculate_relevance_score(headline: str, summary: str, ticker: str) -> float:
    """Calculate how relevant news is to specific ticker"""
    try:
        text = (headline + " " + summary).lower()
        ticker_lower = ticker.lower()
        
        # Direct ticker mentions
        ticker_mentions = text.count(ticker_lower)
        
        # TODO: Add company name lookup for better relevance
        # For now, use ticker mentions as primary relevance indicator
        
        # Base relevance score
        if ticker_mentions >= 2:
            relevance = 1.0
        elif ticker_mentions == 1:
            relevance = 0.8
        else:
            # Check if ticker appears in headline (more important)
            if ticker_lower in headline.lower():
                relevance = 0.9
            else:
                relevance = 0.3  # Low relevance if ticker barely mentioned
        
        return min(1.0, relevance)
        
    except Exception as e:
        print(f"⚠️ Error calculating relevance: {e}")
        return 0.5  # Default moderate relevance

def get_news_intelligence(tickers: List[str]) -> NewsIntelligence:
    """Get comprehensive news intelligence using simple hybrid approach"""
    
    sentiment_analyses = {}
    all_news = []
    
    for i, ticker in enumerate(tickers):
        # Clean progress indicator instead of individual ticker spam
        if (i + 1) % 25 == 0 or i == 0:  # Show progress every 25 tickers, plus first one
            print(f"📰 Processing news... {i + 1}/{len(tickers)} stocks")
        
        # Simple approach: try to get news, let fallback handle any issues
        news_items = get_enhanced_news(ticker, quiet=True)  # Add quiet mode
        
        if news_items:
            sentiment_analysis = calculate_sentiment_analysis(ticker, news_items)
            sentiment_analyses[ticker] = sentiment_analysis
            all_news.extend(news_items)
        
        # Small delay to be respectful to APIs
        time.sleep(NEWS_FALLBACK_CONFIG["delay_between_calls"])
    
    # Calculate market-wide sentiment
    if sentiment_analyses:
        market_sentiment = sum(s.overall_sentiment for s in sentiment_analyses.values()) / len(sentiment_analyses)
    else:
        market_sentiment = 0.0
    
    # Find breaking news (high impact news from last 24 hours)
    now = datetime.now()
    breaking_news = [
        news for news in all_news 
        if news.impact_level == "high" and (now - news.published_at).total_seconds() <= 86400  # 24 hours
    ]
    
    # Sort by recency
    breaking_news.sort(key=lambda x: x.published_at, reverse=True)
    
    # Summary of sources used
    total_articles = len(all_news)
    av_articles = len([news for news in all_news if news.source != "Yahoo Finance"])
    yahoo_articles = total_articles - av_articles
    
    print(f"📊 Market sentiment: {market_sentiment:.3f} | Breaking news: {len(breaking_news)} items")
    print(f"📰 Sources: {av_articles} Alpha Vantage, {yahoo_articles} Yahoo Finance articles")
    
    return NewsIntelligence(
        sentiment_analyses=sentiment_analyses,
        market_sentiment=market_sentiment,
        sector_sentiments={},  # TODO: Implement sector-level sentiment
        breaking_news=breaking_news[:10],  # Top 10 breaking news
        earnings_calendar={},  # TODO: Implement earnings calendar
        last_update=now
    )

def get_alpha_vantage_news(ticker: str, limit: int = 50) -> Optional[Dict]:
    """Fetch news sentiment from Alpha Vantage API - simple approach"""
    
    # Check if API key is set in environment variable
    api_key = ALPHA_VANTAGE_CONFIG['api_key']
    if not api_key:
        return None  # No API key = immediate fallback
    
    try:
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'limit': limit,
            'apikey': api_key
        }
        
        response = requests.get(
            ALPHA_VANTAGE_CONFIG['base_url'], 
            params=params,
            timeout=ALPHA_VANTAGE_CONFIG['timeout']
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for any API errors or rate limits
            if any(key in data for key in ['Error Message', 'Note', 'Information']):
                return None
            
            # Check if we got actual news data
            if 'feed' in data:
                return data
                
        return None
        
    except Exception:
        # Any error = fallback to Yahoo Finance
        return None

def parse_alpha_vantage_news(av_data: Dict, ticker: str) -> List[NewsItem]:
    """Parse Alpha Vantage news data into our NewsItem format"""
    try:
        if 'feed' not in av_data:
            return []
        
        news_items = []
        
        for article in av_data['feed']:
            try:
                # Extract basic info
                headline = article.get('title', 'No headline')
                summary = article.get('summary', headline)
                url = article.get('url', '')
                source = article.get('source', 'Alpha Vantage')
                
                # Parse timestamp
                time_published = article.get('time_published', '')
                try:
                    # Format: 20250605T170025
                    published_at = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
                except:
                    published_at = datetime.now()
                
                # Get ticker-specific sentiment
                ticker_sentiment = 0.0
                relevance_score = 0.0
                
                for ticker_data in article.get('ticker_sentiment', []):
                    if ticker_data.get('ticker') == ticker:
                        ticker_sentiment = float(ticker_data.get('ticker_sentiment_score', 0))
                        relevance_score = float(ticker_data.get('relevance_score', 0))
                        break
                
                # Use overall sentiment if no ticker-specific found
                if ticker_sentiment == 0.0:
                    ticker_sentiment = float(article.get('overall_sentiment_score', 0))
                    relevance_score = 0.5  # Default moderate relevance
                
                # Use normalization function for consistent backend-agnostic scoring
                sentiment_score = normalize_sentiment_score(ticker_sentiment, source)
                
                # Classify event type from topics
                event_type = "general"
                topics = article.get('topics', [])
                
                for topic in topics:
                    topic_name = topic.get('topic', '').lower()
                    if any(keyword in topic_name for keyword in ['earning', 'financial']):
                        event_type = "earnings"
                        break
                    elif 'ipo' in topic_name:
                        event_type = "product"
                        break
                    elif any(keyword in topic_name for keyword in ['legal', 'regulatory']):
                        event_type = "legal"
                        break
                
                # Determine impact level
                impact_level = determine_impact_level(headline, event_type)
                
                news_item = NewsItem(
                    ticker=ticker,
                    headline=headline,
                    summary=summary,
                    published_at=published_at,
                    source=source,
                    url=url,
                    sentiment_score=sentiment_score,  # Already normalized above
                    sentiment_magnitude=relevance_score,  # Use relevance as magnitude proxy
                    event_type=event_type,
                    impact_level=impact_level
                )
                
                # Only include relevant news (relevance > 0.1)
                if relevance_score > 0.1:
                    news_items.append(news_item)
                
            except Exception as e:
                print(f"⚠️ Error parsing Alpha Vantage article: {e}")
                continue
        
        return news_items
        
    except Exception as e:
        print(f"⚠️ Error parsing Alpha Vantage data: {e}")
        return []

def get_enhanced_news(ticker: str, max_articles: int = 10, quiet: bool = False) -> List[NewsItem]:
    """Get news using simple hybrid approach: Alpha Vantage first, Yahoo Finance fallback"""
    
    # Step 1: Check if API key is set
    if not ALPHA_VANTAGE_CONFIG["api_key"]:
        if not quiet:
            print(f"📰 No Alpha Vantage API key, using Yahoo Finance for {ticker}")
        return get_yahoo_news(ticker, max_articles)
    
    # Step 2: Try Alpha Vantage
    if not quiet:
        print(f"📰 Trying Alpha Vantage for {ticker}...")
    av_data = get_alpha_vantage_news(ticker, limit=max_articles)
    
    if av_data:
        av_news = parse_alpha_vantage_news(av_data, ticker)
        if av_news:
            if not quiet:
                print(f"✅ Got {len(av_news)} articles from Alpha Vantage for {ticker}")
            return av_news[:max_articles]
    
    # Step 3: Fallback to Yahoo Finance
    if not quiet:
        print(f"📰 Falling back to Yahoo Finance for {ticker}...")
    yahoo_news = get_yahoo_news(ticker, max_articles)
    
    if yahoo_news:
        if not quiet:
            print(f"✅ Got {len(yahoo_news)} articles from Yahoo Finance for {ticker}")
        return yahoo_news
    
    if not quiet:
        print(f"⚠️ No news found for {ticker}")
    return [] 

def normalize_sentiment_score(sentiment_score: float, source: str) -> float:
    """
    Normalize sentiment scores to be backend-agnostic and consistent
    
    Target range: -0.4 to +0.4 (conservative, realistic for financial sentiment)
    All sources mapped to this consistent range regardless of backend
    """
    
    if source in ["Alpha Vantage", "Benzinga", "CNBC", "Motley Fool"]:
        # Alpha Vantage and professional sources: typically -0.35 to +0.35
        # Keep as-is since they're already in our target range
        normalized = sentiment_score
        
    else:
        # Yahoo Finance and other sources: our enhanced TextBlob (can be -1 to +1)
        # Scale down to conservative range to match professional sources
        normalized = sentiment_score * 0.4  # Map -1,+1 to -0.4,+0.4
    
    # Ensure all scores fit in our standard conservative range
    return max(-0.4, min(0.4, normalized)) 