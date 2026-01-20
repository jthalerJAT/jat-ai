# 📈 Trading Analysis Agent

An AI-powered trading assistant that provides institutional-quality analysis by combining real-time market data, news, and social sentiment.

## Features

- **Multi-LLM Support** - Choose between Claude Opus, GPT-4o, or Groq (free)
- **Technical Analysis** - Price data, moving averages, RSI, MACD from Yahoo Finance
- **News Aggregation** - Headlines and sentiment from Finnhub & Alpha Vantage
- **Social Sentiment** - Posts from StockTwits, Reddit, and X/Twitter
- **Stock Screener** - Find stocks by sector and price performance
- **Natural Language Queries** - Ask questions like "Find me AI stocks down 5% this week"
- **Source Citations** - Every insight is backed by clickable source links

## Screenshots

*Chat interface with AI analysis*

*Stock screener with sentiment*

## Setup

### 1. Install Python
Download Python 3.11 or 3.12 from [python.org](https://python.org). During installation, check "Add Python to PATH".

### 2. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/trading-agent.git
cd trading-agent
```

### 3. Install dependencies
```bash
pip install streamlit groq yfinance pandas pandas-ta plotly requests praw openai anthropic python-dotenv
```

### 4. Set up API keys
Copy `env_example.txt` to `.env` and fill in your API keys:

```bash
cp env_example.txt .env
```

Then edit `.env` with your keys:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here
X_BEARER_TOKEN=your_token_here
RAPIDAPI_KEY=your_key_here
```

### 5. Run the app
```bash
streamlit run trading_agent.py
```

The app will open in your browser at `http://localhost:8501`

## Getting API Keys

| Service | Link | Cost |
|---------|------|------|
| Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com) | Pay per use |
| OpenAI (GPT-4) | [platform.openai.com](https://platform.openai.com) | Pay per use |
| Groq | [console.groq.com](https://console.groq.com) | Free tier |
| Finnhub | [finnhub.io](https://finnhub.io) | Free tier |
| Alpha Vantage | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Free tier |
| Reddit | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Free |
| X/Twitter | [developer.twitter.com](https://developer.twitter.com) | Free tier |
| RapidAPI | [rapidapi.com](https://rapidapi.com) | Free tier |

## Usage

### Chat Tab
Ask natural language questions:
- "Analyze NVDA"
- "What's the sentiment on TSLA?"
- "Find me biotech stocks down 10% this month"

### Stock Analysis Tab
Enter a ticker to see:
- Interactive candlestick chart
- Technical indicators (SMA, RSI, MACD)
- AI-generated analysis

### Screener Tab
Filter stocks by:
- Sector (AI, Tech, Biotech, Energy, etc.)
- Price change percentage
- Time period

### News & Social Tabs
View raw data from:
- Finnhub news
- Alpha Vantage sentiment
- StockTwits posts
- Reddit discussions
- X/Twitter posts

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Always do your own research before making investment decisions.

## License

MIT License - feel free to use and modify as you like.
