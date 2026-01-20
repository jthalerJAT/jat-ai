"""
Trading Analysis Agent
Full-featured trading assistant with news, social sentiment, screening, and multi-LLM support.

SETUP:
pip install streamlit groq yfinance pandas pandas-ta plotly requests praw openai anthropic python-dotenv

Create a .env file with your API keys:
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
FINNHUB_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret

Run: streamlit run trading_agent.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from groq import Groq
from openai import OpenAI
from anthropic import Anthropic
import requests
import praw
import os
import re
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# ============== PAGE SETUP ==============
st.set_page_config(page_title="Trading Analysis Agent", page_icon="📈", layout="wide")
st.title("📈 Trading Analysis Agent")
st.caption("Portfolio monitoring, trade ideas, and due diligence powered by AI")

# ============== SIDEBAR ==============
st.sidebar.header("🔧 Data Sources")
use_general_llm = st.sidebar.checkbox("🌐 General LLM Knowledge", value=True)
use_market_data = st.sidebar.checkbox("📊 Market Data (Yahoo Finance)", value=True)
use_news = st.sidebar.checkbox("📰 News (Finnhub + Alpha Vantage)", value=True)
use_social = st.sidebar.checkbox("💬 Social Sentiment (StockTwits + Reddit)", value=True)
use_email = st.sidebar.checkbox("📧 Outlook Inbox", value=False, help="[Phase 3] Coming soon")

st.sidebar.divider()
st.sidebar.header("🔑 API Keys")
llm_choice = st.sidebar.selectbox("LLM Provider", ["Claude Opus (Anthropic)", "OpenAI (GPT-4o)", "Groq (Free)"])
anthropic_key = st.sidebar.text_input("Anthropic API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password")
groq_key = st.sidebar.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
openai_key = st.sidebar.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
finnhub_key = st.sidebar.text_input("Finnhub API Key", value=os.getenv("FINNHUB_API_KEY", ""), type="password")
alpha_key = st.sidebar.text_input("Alpha Vantage API Key", value=os.getenv("ALPHA_VANTAGE_API_KEY", ""), type="password")

st.sidebar.divider()
st.sidebar.header("🔑 Reddit Credentials")
reddit_client_id = st.sidebar.text_input("Reddit Client ID", value=os.getenv("REDDIT_CLIENT_ID", ""), type="password")
reddit_client_secret = st.sidebar.text_input("Reddit Client Secret", value=os.getenv("REDDIT_CLIENT_SECRET", ""), type="password")

st.sidebar.divider()
st.sidebar.header("🔑 X/Twitter Credentials")
x_bearer_token = st.sidebar.text_input("X Bearer Token", value=os.getenv("X_BEARER_TOKEN", ""), type="password")
rapidapi_key = st.sidebar.text_input("RapidAPI Key", value=os.getenv("RAPIDAPI_KEY", ""), type="password")

# ============== DATA FUNCTIONS ==============

def get_stock_data(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or df.empty:
            return None
        df = df.copy()
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        }
    except:
        return {}

def calculate_technicals(df):
    if df is None or df.empty:
        return None
    df = df.copy()
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['EMA_26'] = ta.ema(df['Close'], length=26)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
    return df

def generate_technical_summary(df, ticker):
    if df is None or df.empty:
        return "No data available."
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    price = float(latest['Close'])
    prev_price = float(prev['Close'])
    change_pct = ((price - prev_price) / prev_price) * 100
    
    def fmt_price(val):
        if pd.notna(val):
            return f"${float(val):.2f}"
        return "N/A"
    
    def fmt_num(val, decimals=2):
        if pd.notna(val):
            return f"{float(val):.{decimals}f}"
        return "N/A"
    
    sma_20 = fmt_price(latest.get('SMA_20'))
    sma_50 = fmt_price(latest.get('SMA_50'))
    sma_200 = fmt_price(latest.get('SMA_200'))
    rsi_val = fmt_num(latest.get('RSI'), 1)
    macd_val = fmt_num(latest.get('MACD_12_26_9'), 3)
    macd_sig = fmt_num(latest.get('MACDs_12_26_9'), 3)
    
    rsi_status = ''
    if pd.notna(latest.get('RSI')):
        rsi = float(latest['RSI'])
        if rsi > 70:
            rsi_status = '(Overbought)'
        elif rsi < 30:
            rsi_status = '(Oversold)'
    
    trend_signals = []
    if pd.notna(latest.get('SMA_20')) and pd.notna(latest.get('SMA_50')):
        if float(latest['SMA_20']) > float(latest['SMA_50']):
            trend_signals.append("[TA-1] Short-term: BULLISH (20 SMA > 50 SMA)")
        else:
            trend_signals.append("[TA-1] Short-term: BEARISH (20 SMA < 50 SMA)")
    
    if pd.notna(latest.get('SMA_50')) and pd.notna(latest.get('SMA_200')):
        if float(latest['SMA_50']) > float(latest['SMA_200']):
            trend_signals.append("[TA-2] Long-term: BULLISH (Golden Cross)")
        else:
            trend_signals.append("[TA-2] Long-term: BEARISH (Death Cross)")
    
    trends = '\n'.join(f"- {t}" for t in trend_signals) if trend_signals else "- Insufficient data"
    
    return f"""**Technical Analysis for {ticker}** (Source: Yahoo Finance)

[PRICE] Current Price: ${price:.2f} (Change: {change_pct:+.2f}%)
[MA] Moving Averages: 20-SMA: {sma_20} | 50-SMA: {sma_50} | 200-SMA: {sma_200}
[RSI] RSI (14): {rsi_val} {rsi_status}
[MACD] MACD: {macd_val} | Signal: {macd_sig}

**Trend Signals:**
{trends}"""

# ============== NEWS FUNCTIONS ==============

def get_finnhub_news(ticker, api_key):
    if not api_key:
        return []
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={week_ago}&to={today}&token={api_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            news = resp.json()[:10]
            return [{"source": "Finnhub", "headline": n.get("headline", ""), 
                     "summary": n.get("summary", "")[:200], "date": n.get("datetime", ""),
                     "url": n.get("url", "")} for n in news]
    except:
        pass
    return []

def get_alpha_vantage_sentiment(ticker, api_key):
    if not api_key:
        return []
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            feed = data.get("feed", [])[:10]
            results = []
            for item in feed:
                sentiment = "Neutral"
                score = item.get("overall_sentiment_score", 0)
                if score > 0.15:
                    sentiment = "Bullish"
                elif score < -0.15:
                    sentiment = "Bearish"
                results.append({
                    "source": "Alpha Vantage",
                    "headline": item.get("title", ""),
                    "sentiment": sentiment,
                    "score": score,
                    "url": item.get("url", "")
                })
            return results
    except:
        pass
    return []

def format_news_for_context(finnhub_news, alpha_news):
    context = "\n**Recent News:**\n"
    for i, n in enumerate(finnhub_news[:5], 1):
        context += f"[NEWS-{i}] {n['headline']} (Source: Finnhub)\n"
    if alpha_news:
        context += "\n**News Sentiment (Alpha Vantage):**\n"
        for i, n in enumerate(alpha_news[:5], 1):
            context += f"[SENT-{i}] {n['headline']} [Sentiment: {n['sentiment']}, Score: {n['score']:.2f}] (Source: Alpha Vantage)\n"
    return context

# ============== SOCIAL FUNCTIONS ==============

def get_stocktwits(ticker):
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            messages = data.get("messages", [])[:15]
            results = []
            for msg in messages:
                sentiment = msg.get("entities", {}).get("sentiment", {})
                sent_label = sentiment.get("basic", "Neutral") if sentiment else "Neutral"
                msg_id = msg.get("id", "")
                results.append({
                    "source": "StockTwits",
                    "body": msg.get("body", "")[:200],
                    "sentiment": sent_label,
                    "user": msg.get("user", {}).get("username", "unknown"),
                    "created": msg.get("created_at", ""),
                    "url": f"https://stocktwits.com/{msg.get('user', {}).get('username', 'unknown')}/message/{msg_id}" if msg_id else ""
                })
            return results
    except:
        pass
    return []

def get_reddit_posts(ticker, client_id, client_secret):
    if not client_id or not client_secret:
        return []
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="TradingAgent/1.0"
        )
        results = []
        subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket"]
        for sub_name in subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.search(ticker, limit=5, time_filter="week"):
                    results.append({
                        "source": f"r/{sub_name}",
                        "title": post.title[:150],
                        "score": post.score,
                        "comments": post.num_comments,
                        "url": f"https://reddit.com{post.permalink}"
                    })
            except:
                continue
        return sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    except:
        return []

# ============== X/TWITTER FUNCTIONS ==============

def get_x_posts_official(ticker, bearer_token):
    """Get tweets using official X API v2 (free tier)"""
    if not bearer_token:
        return []
    try:
        # Search for cashtag
        query = f"${ticker} -is:retweet lang:en"
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        params = {
            "query": query,
            "max_results": 10,
            "tweet.fields": "created_at,public_metrics,author_id",
        }
        
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            tweets = data.get("data", [])
            results = []
            for tweet in tweets:
                metrics = tweet.get("public_metrics", {})
                results.append({
                    "source": "X (Official)",
                    "text": tweet.get("text", "")[:280],
                    "likes": metrics.get("like_count", 0),
                    "retweets": metrics.get("retweet_count", 0),
                    "replies": metrics.get("reply_count", 0),
                    "created": tweet.get("created_at", ""),
                    "url": f"https://twitter.com/i/web/status/{tweet.get('id', '')}"
                })
            return results
        elif resp.status_code == 429:
            return [{"source": "X (Official)", "text": "Rate limit reached. Try again later.", "error": True}]
    except Exception as e:
        pass
    return []

def get_x_posts_rapidapi(ticker, rapidapi_key):
    """Get tweets using RapidAPI as backup"""
    if not rapidapi_key:
        return []
    try:
        # Using Twitter135 API on RapidAPI (one of the popular ones)
        url = "https://twitter135.p.rapidapi.com/v2/Search/"
        headers = {
            "X-RapidAPI-Key": rapidapi_key,
            "X-RapidAPI-Host": "twitter135.p.rapidapi.com"
        }
        params = {"q": f"${ticker}", "count": "10"}
        
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            tweets = data.get("globalObjects", {}).get("tweets", {})
            results = []
            for tweet_id, tweet in list(tweets.items())[:10]:
                results.append({
                    "source": "X (RapidAPI)",
                    "text": tweet.get("full_text", tweet.get("text", ""))[:280],
                    "likes": tweet.get("favorite_count", 0),
                    "retweets": tweet.get("retweet_count", 0),
                    "replies": tweet.get("reply_count", 0),
                    "created": tweet.get("created_at", ""),
                    "url": f"https://twitter.com/i/web/status/{tweet_id}"
                })
            return results
    except:
        pass
    return []

def get_x_posts(ticker, bearer_token, rapidapi_key):
    """Try official API first, fall back to RapidAPI"""
    results = get_x_posts_official(ticker, bearer_token)
    
    # If official API failed or rate limited, try RapidAPI
    if not results or (results and results[0].get("error")):
        rapidapi_results = get_x_posts_rapidapi(ticker, rapidapi_key)
        if rapidapi_results:
            return rapidapi_results
    
    return results

def analyze_x_sentiment(posts):
    """Simple sentiment analysis based on engagement"""
    if not posts:
        return "No X/Twitter data available"
    
    total_likes = sum(p.get('likes', 0) for p in posts)
    total_retweets = sum(p.get('retweets', 0) for p in posts)
    total_posts = len(posts)
    
    # High engagement typically indicates strong interest
    avg_engagement = (total_likes + total_retweets) / total_posts if total_posts > 0 else 0
    
    if avg_engagement > 100:
        buzz = "HIGH"
    elif avg_engagement > 20:
        buzz = "MODERATE"
    else:
        buzz = "LOW"
    
    return f"**X/Twitter Buzz: {buzz}** ({total_posts} posts, {total_likes} likes, {total_retweets} retweets)"

def format_x_for_context(posts):
    """Format X posts for LLM context"""
    if not posts:
        return ""
    
    context = "\n**X/Twitter Posts:**\n"
    context += analyze_x_sentiment(posts) + "\n"
    
    for i, p in enumerate(posts[:5], 1):
        if not p.get("error"):
            context += f"[X-{i}] \"{p['text'][:150]}...\" (Likes: {p.get('likes', 0)}, RTs: {p.get('retweets', 0)}) (Source: {p['source']})\n"
    
    return context

def format_social_for_context(stocktwits, reddit):
    context = "\n**Social Sentiment:**\n"
    
    if stocktwits:
        bulls = sum(1 for s in stocktwits if s['sentiment'] == 'Bullish')
        bears = sum(1 for s in stocktwits if s['sentiment'] == 'Bearish')
        context += f"\nStockTwits ({len(stocktwits)} posts): {bulls} Bullish, {bears} Bearish\n"
        for i, s in enumerate(stocktwits[:3], 1):
            context += f"[ST-{i}] @{s['user']}: \"{s['body'][:100]}...\" [Sentiment: {s['sentiment']}] (Source: StockTwits)\n"
    
    if reddit:
        context += f"\nReddit ({len(reddit)} posts):\n"
        for i, r in enumerate(reddit[:5], 1):
            context += f"[RD-{i}] \"{r['title']}\" (Up:{r['score']}, Comments:{r['comments']}) (Source: {r['source']})\n"
    
    return context

def analyze_social_sentiment(stocktwits, reddit):
    total_bull, total_bear, total_neutral = 0, 0, 0
    
    for s in stocktwits:
        if s['sentiment'] == 'Bullish':
            total_bull += 1
        elif s['sentiment'] == 'Bearish':
            total_bear += 1
        else:
            total_neutral += 1
    
    total = total_bull + total_bear + total_neutral
    if total == 0:
        return "No social data available"
    
    bull_pct = (total_bull / total) * 100
    bear_pct = (total_bear / total) * 100
    
    if bull_pct > 60:
        overall = "BULLISH"
    elif bear_pct > 60:
        overall = "BEARISH"
    else:
        overall = "MIXED"
    
    return f"**Social Sentiment: {overall}** (Bullish: {bull_pct:.0f}%, Bearish: {bear_pct:.0f}%)"

def detect_screening_request(prompt):
    """Detect if user is asking for a stock screen and extract parameters"""
    prompt_lower = prompt.lower()
    
    screening_keywords = ['find', 'screen', 'scan', 'list', 'which stocks', 'what stocks', 
                          'show me stocks', 'looking for stocks', 'stocks that are', 'tickers that',
                          'find me', 'give me a list', 'what are some']
    is_screening = any(kw in prompt_lower for kw in screening_keywords)
    
    if not is_screening:
        return None
    
    sector = None
    sector_mapping = {
        'ai': 'AI & Semiconductors',
        'artificial intelligence': 'AI & Semiconductors', 
        'semiconductor': 'AI & Semiconductors',
        'chip': 'AI & Semiconductors',
        'tech': 'Mega Cap Tech',
        'mega cap': 'Mega Cap Tech',
        'big tech': 'Mega Cap Tech',
        'biotech': 'Biotech',
        'pharma': 'Biotech',
        'healthcare': 'Biotech',
        'financial': 'Financials',
        'bank': 'Financials',
        'energy': 'Energy',
        'oil': 'Energy',
        'ev': 'EV & Clean Energy',
        'electric vehicle': 'EV & Clean Energy',
        'clean energy': 'EV & Clean Energy',
        'solar': 'EV & Clean Energy',
        'retail': 'Retail & Consumer',
        'consumer': 'Retail & Consumer',
        'cannabis': 'Cannabis',
        'weed': 'Cannabis',
        'meme': 'Meme Stocks',
    }
    
    for keyword, sector_name in sector_mapping.items():
        if keyword in prompt_lower:
            sector = sector_name
            break
    
    if not sector:
        sector = 'AI & Semiconductors'
    
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', prompt_lower)
    threshold = -float(pct_match.group(1)) if pct_match else -3.0
    
    if any(word in prompt_lower for word in ['up', 'gain', 'rising', 'climbing', 'green']):
        threshold = abs(threshold)
    elif any(word in prompt_lower for word in ['down', 'drop', 'fall', 'loss', 'red', 'declining']):
        threshold = -abs(threshold)
    
    days = 5
    if 'today' in prompt_lower or '1 day' in prompt_lower:
        days = 1
    elif 'week' in prompt_lower or '5 day' in prompt_lower or '7 day' in prompt_lower:
        days = 5
    elif 'month' in prompt_lower or '30 day' in prompt_lower:
        days = 30
    elif '2 week' in prompt_lower or '14 day' in prompt_lower:
        days = 14
    elif '3 day' in prompt_lower:
        days = 3
    
    day_match = re.search(r'(\d+)\s*days?', prompt_lower)
    if day_match:
        days = int(day_match.group(1))
    
    return {
        'sector': sector,
        'threshold': threshold,
        'days': days
    }

# ============== SECTOR WATCHLISTS ==============

WATCHLISTS = {
    "AI & Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "MRVL", "ARM", "SMCI", "AI", "PLTR", "PATH", "SNOW", "MDB", "DDOG", "NET", "CRWD", "ZS", "PANW", "FTNT", "SYM", "BBAI", "SOUN", "UPST", "AEHR", "ACLS", "ONTO", "LSCC", "MPWR", "WOLF", "ON", "ADI", "TXN", "NXPI", "MCHP", "SWKS", "QRVO", "CRUS", "SLAB"],
    "Mega Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "CRM", "ORCL", "ADBE", "NOW", "IBM", "CSCO", "ACN"],
    "Biotech": ["MRNA", "BNTX", "REGN", "VRTX", "GILD", "BIIB", "ILMN", "DXCM", "ALGN", "ISRG", "TMO", "DHR", "ABT", "BMY", "LLY", "PFE", "JNJ", "MRK", "ABBV", "AMGN"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA", "PYPL", "SQ", "COIN", "HOOD", "SOFI"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "MPC", "VLO", "PSX", "HAL", "DVN", "FANG", "HES"],
    "EV & Clean Energy": ["TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM", "PLUG", "FCEL", "BE", "ENPH", "SEDG", "RUN", "NOVA", "CHPT", "BLNK", "EVGO", "QS", "MVST"],
    "Retail & Consumer": ["WMT", "COST", "TGT", "HD", "LOW", "AMZN", "BABA", "JD", "PDD", "MELI", "ETSY", "W", "CHWY", "DG", "DLTR"],
    "Cannabis": ["TLRY", "CGC", "ACB", "CRON", "SNDL", "OGI", "HEXO", "VFF", "GRWG"],
    "Meme Stocks": ["GME", "AMC", "BB", "BBBY", "KOSS", "EXPR", "NAKD", "WISH", "CLOV", "SPCE", "PLTR", "SOFI"],
}

def screen_stocks(watchlist_name, days=5, threshold=-3.0):
    if watchlist_name not in WATCHLISTS:
        return []
    
    tickers = WATCHLISTS[watchlist_name]
    results = []
    
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period="1mo")
            if df is None or len(df) < days:
                continue
            
            current_price = df['Close'].iloc[-1]
            past_price = df['Close'].iloc[-days]
            pct_change = ((current_price - past_price) / past_price) * 100
            
            if threshold < 0 and pct_change <= threshold:
                results.append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "pct_change": pct_change,
                    "days": days
                })
            elif threshold > 0 and pct_change >= threshold:
                results.append({
                    "ticker": ticker,
                    "current_price": current_price,
                    "pct_change": pct_change,
                    "days": days
                })
        except:
            continue
    
    return sorted(results, key=lambda x: x['pct_change'], reverse=(threshold > 0))

def get_batch_sentiment(tickers):
    all_sentiment = {}
    for ticker in tickers[:5]:
        st_data = get_stocktwits(ticker)
        if st_data:
            bulls = sum(1 for s in st_data if s['sentiment'] == 'Bullish')
            bears = sum(1 for s in st_data if s['sentiment'] == 'Bearish')
            total = len(st_data)
            sample_posts = [s['body'][:100] for s in st_data[:3]]
            all_sentiment[ticker] = {
                "bullish": bulls,
                "bearish": bears,
                "total": total,
                "sample_posts": sample_posts
            }
    return all_sentiment

# ============== CHART FUNCTION ==============

def create_chart(df, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2], subplot_titles=(f'{ticker} Price', 'Volume', 'RSI'))
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                                  line=dict(color='orange', width=1)), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                                  line=dict(color='blue', width=1)), row=1, col=1)
    
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                  line=dict(color='purple', width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, template='plotly_dark', showlegend=True)
    return fig

# ============== LLM FUNCTION ==============

def chat_with_llm(message, context="", groq_key="", openai_key="", anthropic_key="", provider="Claude Opus (Anthropic)"):
    system = """You are a senior equity research analyst at Goldman Sachs with 20 years of experience. You provide institutional-quality, detailed analysis that would be found in a professional research report.

YOUR ANALYSIS MUST BE COMPREHENSIVE AND DETAILED. Never give surface-level answers.

CITATION REQUIREMENTS:
- The data below contains citation tags like [NEWS-1], [ST-2], [TA-1], [PRICE], [RSI], etc.
- You MUST reference these citations when making claims
- Example: "The stock is showing bearish momentum [RSI] with RSI at 28, indicating oversold conditions"
- Example: "Recent headlines suggest regulatory concerns [NEWS-1][NEWS-3]"
- Example: "Retail sentiment on StockTwits is mixed, with one user noting 'loading up on this dip' [ST-1]"
- At the end of your response, include a "Sources" section listing all citations used

ANALYSIS FRAMEWORK - Structure every response with depth:

## 1. EXECUTIVE SUMMARY
- Clear bullish/bearish/neutral stance with conviction level (high/medium/low)
- One-paragraph thesis explaining WHY
- Price target context if relevant

## 2. TECHNICAL ANALYSIS (when price data available)
- **Trend Structure**: Primary trend, secondary trend, current phase
- **Key Levels**: Immediate support/resistance, major levels, psychological levels
- **Moving Averages**: Price vs 20/50/200 SMA, MA slope, cross status
- **Momentum**: RSI interpretation, MACD histogram, volume confirmation
- **Pattern Recognition**: Any chart patterns forming

## 3. CATALYST ANALYSIS (when news available)
- **Near-term catalysts** (next 1-4 weeks)
- **Medium-term catalysts** (1-3 months)
- **News sentiment**: What's the narrative?

## 4. SENTIMENT DEEP DIVE (when social data available)
- **Retail positioning**: Bullish or bearish?
- **Narrative analysis**: Bull and bear cases being discussed
- **Key quotes**: 2-3 actual quotes from social posts

## 5. RISK ASSESSMENT
- **Bull case risks**: What could go wrong for longs?
- **Bear case risks**: What could go wrong for shorts?
- **Binary events**: Upcoming events that could cause >10% moves

## 6. ACTIONABLE TRADE IDEAS
- **Entry strategy**: Specific levels or conditions
- **Stop loss**: Where to cut losses
- **Profit targets**: T1, T2, T3 levels
- **Risk/reward ratio**

## 7. BOTTOM LINE
- 2-3 sentence summary
- Clear "If X happens, do Y" guidance

## 8. SOURCES
- List all citation tags used

CRITICAL RULES:
- NEVER give vague answers. Be SPECIFIC with numbers, levels, percentages
- ALWAYS reference the actual data provided
- TAKE A STANCE - don't sit on the fence
- Write at least 500 words for any stock analysis
- Disclaimer: "This is analysis for educational purposes, not financial advice."
"""

    if context:
        system += f"\n\n===== MARKET DATA (ANALYZE THIS IN DETAIL) =====\n{context}\n===== END DATA ====="
    else:
        system += "\n\nNo data loaded. Ask user to specify a ticker clearly, e.g., 'Analyze TSLA' or use the Screener tab."
    
    if provider == "Claude Opus (Anthropic)":
        if not anthropic_key:
            return "Please enter your Anthropic API key in the sidebar."
        try:
            client = Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": message}]
            )
            return resp.content[0].text
        except Exception as e:
            return f"Anthropic Error: {str(e)}"
    elif provider == "OpenAI (GPT-4o)":
        if not openai_key:
            return "Please enter your OpenAI API key in the sidebar."
        try:
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
                temperature=0.7, max_tokens=4000
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
    else:
        if not groq_key:
            return "Please enter your Groq API key in the sidebar."
        try:
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
                temperature=0.7, max_tokens=4000
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Groq Error: {str(e)}"

# ============== SESSION STATE ==============
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_context" not in st.session_state:
    st.session_state.current_context = ""
if "screener_results" not in st.session_state:
    st.session_state.screener_results = None
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []

# ============== MAIN TABS ==============
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📊 Stock Analysis", "🔍 Screener", "📰 News Feed", "💬 Social Sentiment"])

# ============== TAB 1: CHAT ==============
with tab1:
    st.subheader("Chat with your Trading Agent")
    
    sources_list = []
    if use_general_llm: sources_list.append("🌐 LLM")
    if use_market_data: sources_list.append("📊 Market")
    if use_news: sources_list.append("📰 News")
    if use_social: sources_list.append("💬 Social")
    st.caption(f"Active: {' | '.join(sources_list) if sources_list else 'None'}")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about stocks, markets, or your portfolio..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        context = st.session_state.current_context
        sources = []
        
        # Check if this is a screening request
        screen_params = detect_screening_request(prompt)
        
        if screen_params:
            st.info(f"Screening {screen_params['sector']} for stocks {'up' if screen_params['threshold'] > 0 else 'down'} {abs(screen_params['threshold'])}% in {screen_params['days']} days...")
            
            with st.spinner(f"Scanning {screen_params['sector']}..."):
                results = screen_stocks(screen_params['sector'], screen_params['days'], screen_params['threshold'])
                
                if results:
                    context = f"**Screening Results: {screen_params['sector']}**\n"
                    context += f"Filter: Stocks {'up' if screen_params['threshold'] > 0 else 'down'} more than {abs(screen_params['threshold'])}% in {screen_params['days']} days\n\n"
                    context += "**Matching Stocks:**\n"
                    
                    for r in results:
                        context += f"- {r['ticker']}: ${r['current_price']:.2f} ({r['pct_change']:+.2f}% over {r['days']} days)\n"
                        sources.append({"id": f"[{r['ticker']}]", "name": f"{r['ticker']}: ${r['current_price']:.2f} ({r['pct_change']:+.2f}%)", "url": f"https://finance.yahoo.com/quote/{r['ticker']}", "type": "Technical"})
                    
                    if use_social:
                        context += "\n**Social Sentiment:**\n"
                        ticker_list = [r['ticker'] for r in results[:5]]
                        sentiment_data = get_batch_sentiment(ticker_list)
                        
                        for ticker, data in sentiment_data.items():
                            pct_bull = (data['bullish'] / data['total'] * 100) if data['total'] > 0 else 0
                            context += f"- {ticker}: {pct_bull:.0f}% bullish ({data['bullish']} bull, {data['bearish']} bear)\n"
                            if data['sample_posts']:
                                for i, post in enumerate(data['sample_posts'][:2], 1):
                                    context += f"  [ST-{ticker}-{i}] \"{post}...\"\n"
                    
                    st.session_state.current_sources = sources
                else:
                    context = f"No stocks found in {screen_params['sector']} matching the criteria."
        
        else:
            # Normal ticker-based analysis
            prompt_upper = prompt.upper()
            tickers = []
            
            dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', prompt_upper)
            tickers.extend(dollar_tickers)
            
            keyword_patterns = [
                r'\bANALYZE\s+([A-Z]{1,5})\b',
                r'\bANALYSIS\s+(?:OF\s+)?([A-Z]{1,5})\b',
                r'\bON\s+([A-Z]{1,5})\b',
                r'\bFOR\s+([A-Z]{1,5})\b',
                r'\bABOUT\s+([A-Z]{1,5})\b',
                r'\b([A-Z]{1,5})\s+STOCK\b',
                r'\b([A-Z]{1,5})\s+PRICE\b',
                r'\b([A-Z]{1,5})\s+NEWS\b',
            ]
            
            for pattern in keyword_patterns:
                matches = re.findall(pattern, prompt_upper)
                tickers.extend(matches)
            
            common_words = {'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHO', 'WHY', 'HOW', 'THE', 'AND', 'FOR', 
                            'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 
                            'OUT', 'HAS', 'HAVE', 'BEEN', 'WILL', 'MORE', 'WITH', 'THEY', 'THIS',
                            'THAT', 'FROM', 'ABOUT', 'INTO', 'OVER', 'SUCH', 'NEWS', 'TELL', 'GIVE',
                            'SHOW', 'FIND', 'LOOK', 'GET', 'MAKE', 'KNOW', 'TAKE', 'SEE', 'COME',
                            'COULD', 'NOW', 'THAN', 'LIKE', 'OTHER', 'THEN', 'ITS', 'ALSO', 'AFTER',
                            'USE', 'TWO', 'SOME', 'WELL', 'WAY', 'EVEN', 'NEW', 'WANT', 'ANY', 'THESE',
                            'MOST', 'SAY', 'SHE', 'HIM', 'HIS', 'DOES', 'DID', 'GOT', 'LET', 'PUT',
                            'STOCK', 'PRICE', 'NEWS', 'SENTIMENT', 'ANALYSIS', 'ANALYZE', 'THINK',
                            'SHOULD', 'BUY', 'SELL', 'HOLD', 'LONG', 'SHORT', 'BULL', 'BEAR', 'MARKET',
                            'PULL', 'PUSH', 'DOWN', 'GOING', 'BEEN', 'BEING', 'WERE', 'WOULD', 'COULD',
                            'THEIR', 'THERE', 'HERE', 'JUST', 'ONLY', 'VERY', 'MUCH', 'SUCH', 'EACH',
                            'SAME', 'BACK', 'GOOD', 'BEST', 'BOTH', 'FULL', 'LAST', 'YEAR', 'WEEK',
                            'DAYS', 'TIME', 'OVER', 'UNDER', 'HIGH', 'LOWS', 'MOVE', 'CALL', 'PUTS',
                            'PLAY', 'TERM', 'NEXT', 'PLUS', 'LESS', 'MANY', 'REAL', 'WORK', 'LIST',
                            'NEED', 'HELP', 'KEEP', 'FEEL', 'SEEM', 'MEAN', 'SURE', 'MUST', 'STILL',
                            'DEEP', 'DIVE', 'INTO', 'GIVE', 'TOLD', 'TELL', 'TALK', 'SAID', 'SAYS',
                            'DATA', 'INFO', 'TECH', 'SECTOR', 'SPACE', 'AREA', 'ZONE', 'LEVEL',
                            'ME', 'MY', 'IT', 'IS', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'IF', 'IN',
                            'NO', 'OF', 'OK', 'OR', 'SO', 'TO', 'UP', 'US', 'WE'}
            
            tickers = [t for t in tickers if t not in common_words]
            
            seen = set()
            tickers = [t for t in tickers if not (t in seen or seen.add(t))]
            
            if tickers:
                st.caption(f"Detected ticker(s): {', '.join(tickers)}")
            
            for ticker in tickers:
                found_data = False
                
                if use_market_data:
                    df = get_stock_data(ticker, "3mo")
                    if df is not None and not df.empty:
                        df = calculate_technicals(df)
                        tech_summary = generate_technical_summary(df, ticker)
                        context += "\n" + tech_summary
                        latest_price = float(df['Close'].iloc[-1])
                        sources.append({"id": "[PRICE]", "name": f"{ticker} Price: ${latest_price:.2f}", "url": f"https://finance.yahoo.com/quote/{ticker}", "type": "Technical"})
                        sources.append({"id": "[TA]", "name": f"{ticker} Technical Analysis", "url": f"https://finance.yahoo.com/quote/{ticker}/chart", "type": "Technical"})
                        found_data = True
                
                if use_news:
                    fn = get_finnhub_news(ticker, finnhub_key)
                    av = get_alpha_vantage_sentiment(ticker, alpha_key)
                    if fn or av:
                        context += format_news_for_context(fn, av)
                        for i, n in enumerate(fn[:5], 1):
                            if n.get('url'):
                                sources.append({"id": f"[NEWS-{i}]", "name": n['headline'][:60] + "...", "url": n['url'], "type": "News"})
                        for i, n in enumerate(av[:5], 1):
                            if n.get('url'):
                                sources.append({"id": f"[SENT-{i}]", "name": n['headline'][:60] + "...", "url": n['url'], "type": "News"})
                        found_data = True
                
                if use_social:
                    st_data = get_stocktwits(ticker)
                    rd_data = get_reddit_posts(ticker, reddit_client_id, reddit_client_secret)
                    x_data = get_x_posts(ticker, x_bearer_token, rapidapi_key)
                    if st_data or rd_data or x_data:
                        context += format_social_for_context(st_data, rd_data)
                        if x_data:
                            context += format_x_for_context(x_data)
                        context += "\n" + analyze_social_sentiment(st_data, rd_data)
                        for i, s in enumerate(st_data[:3], 1):
                            if s.get('url'):
                                sources.append({"id": f"[ST-{i}]", "name": f"@{s['user']}: {s['body'][:40]}...", "url": s['url'], "type": "Social"})
                        for i, r in enumerate(rd_data[:5], 1):
                            if r.get('url'):
                                sources.append({"id": f"[RD-{i}]", "name": r['title'][:50] + "...", "url": r['url'], "type": "Social"})
                        for i, x in enumerate(x_data[:5], 1):
                            if x.get('url') and not x.get('error'):
                                sources.append({"id": f"[X-{i}]", "name": x['text'][:50] + "...", "url": x['url'], "type": "Social"})
                        found_data = True
                
                if found_data:
                    st.session_state.current_sources = sources
                    break
        
        with st.chat_message("assistant"):
            if use_general_llm:
                if context:
                    with st.expander("📋 Data gathered for this query"):
                        st.text(context)
                response = chat_with_llm(prompt, context, groq_key, openai_key, anthropic_key, llm_choice)
            else:
                response = "Enable 'General LLM Knowledge' to chat."
            st.markdown(response)
            
            if st.session_state.current_sources:
                with st.expander("🔗 Click to view sources"):
                    tech_sources = [s for s in st.session_state.current_sources if s['type'] == 'Technical']
                    news_sources = [s for s in st.session_state.current_sources if s['type'] == 'News']
                    social_sources = [s for s in st.session_state.current_sources if s['type'] == 'Social']
                    
                    if tech_sources:
                        st.markdown("**📊 Technical Data:**")
                        for s in tech_sources:
                            st.markdown(f"- {s['id']} <a href='{s['url']}' target='_blank'>{s['name']}</a>", unsafe_allow_html=True)
                    
                    if news_sources:
                        st.markdown("**📰 News:**")
                        for s in news_sources:
                            st.markdown(f"- {s['id']} <a href='{s['url']}' target='_blank'>{s['name']}</a>", unsafe_allow_html=True)
                    
                    if social_sources:
                        st.markdown("**💬 Social:**")
                        for s in social_sources:
                            st.markdown(f"- {s['id']} <a href='{s['url']}' target='_blank'>{s['name']}</a>", unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# ============== TAB 2: STOCK ANALYSIS ==============
with tab2:
    st.subheader("Stock Technical Analysis")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        ticker = st.text_input("Ticker", value="AAPL").upper()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {ticker}..."):
                df = get_stock_data(ticker, period)
                if df is not None:
                    df = calculate_technicals(df)
                    info = get_stock_info(ticker)
                    
                    st.session_state.current_context = generate_technical_summary(df, ticker)
                    
                    with col2:
                        st.markdown(f"### {info.get('name', ticker)}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Sector", info.get('sector', 'N/A'))
                        c2.metric("P/E", f"{info.get('pe_ratio', 0):.2f}" if isinstance(info.get('pe_ratio'), (int,float)) else 'N/A')
                        c3.metric("52W High", f"${info.get('52w_high', 0):.2f}" if isinstance(info.get('52w_high'), (int,float)) else 'N/A')
                        c4.metric("52W Low", f"${info.get('52w_low', 0):.2f}" if isinstance(info.get('52w_low'), (int,float)) else 'N/A')
                        
                        st.plotly_chart(create_chart(df, ticker), use_container_width=True)
                        st.markdown(generate_technical_summary(df, ticker))
                        
                        if anthropic_key or openai_key or groq_key:
                            with st.spinner("AI analysis..."):
                                ai = chat_with_llm(f"Give a trading outlook for {ticker}", 
                                                   st.session_state.current_context, groq_key, openai_key, anthropic_key, llm_choice)
                                st.markdown("### 🤖 AI Interpretation")
                                st.markdown(ai)
                else:
                    st.error(f"Could not fetch data for {ticker}")

# ============== TAB 3: SCREENER ==============
with tab3:
    st.subheader("🔍 Stock Screener")
    st.caption("Find stocks by sector and price performance, then get sentiment analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sector = st.selectbox("Select Sector", list(WATCHLISTS.keys()))
    with col2:
        days = st.slider("Lookback Period (days)", 1, 30, 5)
    with col3:
        threshold = st.slider("Price Change Threshold (%)", -20.0, 20.0, -3.0, 0.5)
    
    if st.button("🔍 Run Screener", type="primary"):
        with st.spinner(f"Scanning {len(WATCHLISTS[sector])} stocks in {sector}..."):
            results = screen_stocks(sector, days, threshold)
            st.session_state.screener_results = results
            
            if results:
                st.success(f"Found {len(results)} stocks matching criteria")
                
                st.markdown("### Matching Stocks")
                for r in results:
                    st.markdown(f"**{r['ticker']}**: ${r['current_price']:.2f} ({r['pct_change']:+.2f}% over {r['days']} days)")
                
                st.markdown("### Sentiment Analysis")
                with st.spinner("Fetching social sentiment..."):
                    ticker_list = [r['ticker'] for r in results]
                    sentiment = get_batch_sentiment(ticker_list)
                    
                    if sentiment:
                        for sticker, data in sentiment.items():
                            pct_bull = (data['bullish'] / data['total'] * 100) if data['total'] > 0 else 0
                            emoji = "🟢" if pct_bull > 60 else "🔴" if pct_bull < 40 else "⚪"
                            st.markdown(f"{emoji} **{sticker}**: {data['bullish']} bullish, {data['bearish']} bearish ({pct_bull:.0f}% bullish)")
                            if data['sample_posts']:
                                with st.expander(f"Sample posts for {sticker}"):
                                    for post in data['sample_posts']:
                                        st.caption(f"* {post}...")
                        
                        screen_context = f"**Screener Results: {sector} stocks {'up' if threshold > 0 else 'down'} more than {abs(threshold)}% in {days} days**\n\n"
                        for r in results:
                            screen_context += f"- {r['ticker']}: ${r['current_price']:.2f} ({r['pct_change']:+.2f}%)\n"
                        screen_context += f"\n**Social Sentiment:**\n"
                        for sticker, data in sentiment.items():
                            pct_bull = (data['bullish'] / data['total'] * 100) if data['total'] > 0 else 0
                            screen_context += f"- {sticker}: {pct_bull:.0f}% bullish, sample: {data['sample_posts'][0] if data['sample_posts'] else 'N/A'}\n"
                        
                        st.session_state.current_context = screen_context
                        
                        st.markdown("### 🤖 AI Analysis")
                        with st.spinner("Generating analysis..."):
                            ai_prompt = f"Analyze these {sector} stocks that are {'up' if threshold > 0 else 'down'}. Which ones look {'overbought' if threshold > 0 else 'oversold'}? Which have the worst sentiment? Any potential opportunities or value traps?"
                            ai_response = chat_with_llm(ai_prompt, screen_context, groq_key, openai_key, anthropic_key, llm_choice)
                            st.markdown(ai_response)
                    else:
                        st.info("No sentiment data available for these tickers")
            else:
                st.warning(f"No stocks found matching criteria")

# ============== TAB 4: NEWS ==============
with tab4:
    st.subheader("📰 News Feed")
    news_ticker = st.text_input("Enter ticker for news", value="AAPL", key="news_ticker").upper()
    
    if st.button("🔍 Get News", type="primary"):
        if not finnhub_key and not alpha_key:
            st.warning("Please enter Finnhub and/or Alpha Vantage API keys in the sidebar.")
        else:
            with st.spinner("Fetching news..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Finnhub News")
                    fn_news = get_finnhub_news(news_ticker, finnhub_key)
                    if fn_news:
                        for n in fn_news:
                            st.markdown(f"**{n['headline']}**")
                            st.caption(n['summary'])
                            if n.get('url'):
                                st.markdown(f"<a href='{n['url']}' target='_blank'>Read more</a>", unsafe_allow_html=True)
                            st.divider()
                    else:
                        st.info("No Finnhub news found")
                
                with col2:
                    st.markdown("### Alpha Vantage Sentiment")
                    av_news = get_alpha_vantage_sentiment(news_ticker, alpha_key)
                    if av_news:
                        for n in av_news:
                            emoji = "🟢" if n['sentiment'] == "Bullish" else "🔴" if n['sentiment'] == "Bearish" else "⚪"
                            st.markdown(f"{emoji} **{n['headline']}**")
                            st.caption(f"Sentiment: {n['sentiment']} (Score: {n['score']:.2f})")
                            if n.get('url'):
                                st.markdown(f"<a href='{n['url']}' target='_blank'>Read more</a>", unsafe_allow_html=True)
                            st.divider()
                    else:
                        st.info("No Alpha Vantage news found")

# ============== TAB 5: SOCIAL ==============
with tab5:
    st.subheader("💬 Social Sentiment")
    social_ticker = st.text_input("Enter ticker for social", value="AAPL", key="social_ticker").upper()
    
    if st.button("🔍 Get Social Sentiment", type="primary"):
        with st.spinner("Fetching social data..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### StockTwits")
                st_data = get_stocktwits(social_ticker)
                if st_data:
                    st.markdown(analyze_social_sentiment(st_data, []))
                    st.divider()
                    for s in st_data[:8]:
                        emoji = "🟢" if s['sentiment'] == "Bullish" else "🔴" if s['sentiment'] == "Bearish" else "⚪"
                        st.markdown(f"{emoji} **@{s['user']}**")
                        st.caption(s['body'])
                        if s.get('url'):
                            st.markdown(f"<a href='{s['url']}' target='_blank'>View post</a>", unsafe_allow_html=True)
                        st.divider()
                else:
                    st.info("No StockTwits data found")
            
            with col2:
                st.markdown("### X/Twitter")
                x_data = get_x_posts(social_ticker, x_bearer_token, rapidapi_key)
                if x_data and not x_data[0].get("error"):
                    st.markdown(analyze_x_sentiment(x_data))
                    st.divider()
                    for x in x_data[:8]:
                        st.markdown(f"**{x['text'][:150]}...**")
                        st.caption(f"❤️ {x.get('likes', 0)} | 🔁 {x.get('retweets', 0)} | 💬 {x.get('replies', 0)}")
                        if x.get('url'):
                            st.markdown(f"<a href='{x['url']}' target='_blank'>View tweet</a>", unsafe_allow_html=True)
                        st.divider()
                elif x_data and x_data[0].get("error"):
                    st.warning(x_data[0]['text'])
                else:
                    if not x_bearer_token and not rapidapi_key:
                        st.warning("Enter X Bearer Token or RapidAPI Key in sidebar")
                    else:
                        st.info("No X/Twitter data found")
            
            with col3:
                st.markdown("### Reddit")
                if not reddit_client_id or not reddit_client_secret:
                    st.warning("Enter Reddit credentials in sidebar")
                else:
                    rd_data = get_reddit_posts(social_ticker, reddit_client_id, reddit_client_secret)
                    if rd_data:
                        for r in rd_data[:8]:
                            st.markdown(f"**{r['title']}**")
                            st.caption(f"{r['source']} | Up: {r['score']} | Comments: {r['comments']}")
                            if r.get('url'):
                                st.markdown(f"<a href='{r['url']}' target='_blank'>View post</a>", unsafe_allow_html=True)
                            st.divider()
                    else:
                        st.info("No Reddit posts found")

# ============== FOOTER ==============
st.divider()
st.caption("This tool is for informational purposes only and does not constitute financial advice. Always do your own research.")