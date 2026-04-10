# ── INDIAN MARKET (NSE) ──────────────────────────────────────────
INDIA_TICKERS = {
    "RELIANCE":    "RELIANCE.NS",
    "TCS":         "TCS.NS",
    "HDFCBANK":    "HDFCBANK.NS",
    "INFY":        "INFY.NS",
    "ICICIBANK":   "ICICIBANK.NS",
    "HINDUNILVR":  "HINDUNILVR.NS",
    "ITC":         "ITC.NS",
    "SBIN":        "SBIN.NS",
    "BAJFINANCE":  "BAJFINANCE.NS",
    "ASIANPAINT":  "ASIANPAINT.NS",
}

# ── US MARKET (NYSE / NASDAQ) ────────────────────────────────────
US_TICKERS = {
    "APPLE":       "AAPL",
    "MICROSOFT":   "MSFT",
    "GOOGLE":      "GOOGL",
    "AMAZON":      "AMZN",
    "NVIDIA":      "NVDA",
    "TESLA":       "TSLA",
    "META":        "META",
    "BERKSHIRE":   "BRK-B",
}

# ── EUROPEAN MARKET ──────────────────────────────────────────────
EU_TICKERS = {
    "NESTLE":      "NESN.SW",    # Swiss Exchange
    "LVMH":        "MC.PA",      # Paris
    "ASML":        "ASML.AS",    # Amsterdam
    "SAP":         "SAP.DE",     # Frankfurt
}

# ── ASIAN MARKET ─────────────────────────────────────────────────
ASIA_TICKERS = {
    "SAMSUNG":     "005930.KS",  # Korea
    "ALIBABA":     "9988.HK",    # Hong Kong
    "SONY":        "6758.T",     # Tokyo
    "TSMC":        "TSM",        # NYSE-listed ADR
}

# ── COMBINE ALL ──────────────────────────────────────────────────
ALL_TICKERS = {
    **INDIA_TICKERS,
    **US_TICKERS,
    **EU_TICKERS,
    **ASIA_TICKERS,
}

# Markets and their currencies (for display)
MARKET_META = {
    "RELIANCE":   {"market": "India",  "currency": "INR", "sector": "Energy"},
    "TCS":        {"market": "India",  "currency": "INR", "sector": "IT"},
    "HDFCBANK":   {"market": "India",  "currency": "INR", "sector": "Finance"},
    "INFY":       {"market": "India",  "currency": "INR", "sector": "IT"},
    "ICICIBANK":  {"market": "India",  "currency": "INR", "sector": "Finance"},
    "HINDUNILVR": {"market": "India",  "currency": "INR", "sector": "FMCG"},
    "ITC":        {"market": "India",  "currency": "INR", "sector": "FMCG"},
    "SBIN":       {"market": "India",  "currency": "INR", "sector": "Finance"},
    "BAJFINANCE": {"market": "India",  "currency": "INR", "sector": "Finance"},
    "ASIANPAINT": {"market": "India",  "currency": "INR", "sector": "Materials"},
    "APPLE":      {"market": "US",     "currency": "USD", "sector": "Technology"},
    "MICROSOFT":  {"market": "US",     "currency": "USD", "sector": "Technology"},
    "GOOGLE":     {"market": "US",     "currency": "USD", "sector": "Technology"},
    "AMAZON":     {"market": "US",     "currency": "USD", "sector": "Consumer"},
    "NVIDIA":     {"market": "US",     "currency": "USD", "sector": "Semiconductors"},
    "TESLA":      {"market": "US",     "currency": "USD", "sector": "Automotive"},
    "META":       {"market": "US",     "currency": "USD", "sector": "Technology"},
    "BERKSHIRE":  {"market": "US",     "currency": "USD", "sector": "Finance"},
    "NESTLE":     {"market": "Europe", "currency": "CHF", "sector": "FMCG"},
    "LVMH":       {"market": "Europe", "currency": "EUR", "sector": "Luxury"},
    "ASML":       {"market": "Europe", "currency": "EUR", "sector": "Semiconductors"},
    "SAP":        {"market": "Europe", "currency": "EUR", "sector": "Technology"},
    "SAMSUNG":    {"market": "Asia",   "currency": "KRW", "sector": "Technology"},
    "ALIBABA":    {"market": "Asia",   "currency": "HKD", "sector": "Consumer"},
    "SONY":       {"market": "Asia",   "currency": "JPY", "sector": "Consumer"},
    "TSMC":       {"market": "Asia",   "currency": "USD", "sector": "Semiconductors"},
}

# Historical range for backtesting
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# Base currency to normalize everything into
BASE_CURRENCY = "USD"
# ── NEWS API ──────────────────────────────────
NEWS_API_KEY = "06df601f06884d598fba471cad8dd732"