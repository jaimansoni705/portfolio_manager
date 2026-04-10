import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import logging
from config import NEWS_API_KEY, ALL_TICKERS, MARKET_META

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

os.makedirs("data/sentiment", exist_ok=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def load_models():
    """
    Load both sentiment models:
    1. VADER  — fast rule-based
    2. FinBERT — ML transformer trained on financial text
    """
    print("Loading sentiment models...")

    # VADER
    vader = SentimentIntensityAnalyzer()
    print("  ✓ VADER loaded")

    # FinBERT
    print("  Loading FinBERT (first time may take 2-3 mins)...")
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        max_length=512,
        truncation=True,
    )
    print("  ✓ FinBERT loaded")

    return vader, finbert


# ─────────────────────────────────────────────
# FETCH NEWS
# ─────────────────────────────────────────────
def fetch_news(stock_name: str, days_back: int = 7) -> list:
    """
    Fetch recent news articles for a stock
    using NewsAPI.
    """
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # build search query
    # use common name for better results
    search_map = {
        "RELIANCE":   "Reliance Industries",
        "TCS":        "Tata Consultancy Services",
        "HDFCBANK":   "HDFC Bank",
        "INFY":       "Infosys",
        "ICICIBANK":  "ICICI Bank",
        "HINDUNILVR": "Hindustan Unilever",
        "ITC":        "ITC Limited",
        "SBIN":       "State Bank India",
        "BAJFINANCE": "Bajaj Finance",
        "ASIANPAINT": "Asian Paints",
        "APPLE":      "Apple Inc",
        "MICROSOFT":  "Microsoft",
        "GOOGLE":     "Google Alphabet",
        "AMAZON":     "Amazon",
        "NVIDIA":     "NVIDIA",
        "TESLA":      "Tesla",
        "META":       "Meta Facebook",
        "BERKSHIRE":  "Berkshire Hathaway",
        "NESTLE":     "Nestle",
        "LVMH":       "LVMH",
        "ASML":       "ASML",
        "SAP":        "SAP",
        "SAMSUNG":    "Samsung",
        "ALIBABA":    "Alibaba",
        "SONY":       "Sony",
        "TSMC":       "TSMC Taiwan",
    }

    query = search_map.get(stock_name, stock_name)

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date   = datetime.now().strftime("%Y-%m-%d")

    try:
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="relevancy",
            page_size=10,
        )
        articles = response.get("articles", [])
        logger.info(f"  {stock_name:<15} → {len(articles)} articles")
        return articles

    except Exception as e:
        logger.error(f"  ✗ {stock_name}: {e}")
        return []


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalize article text
    before sentiment scoring.
    """
    if not text:
        return ""

    # lowercase
    text = text.lower()

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    tokens = [
        t for t in tokens
        if t.isalpha() and t not in stop_words
    ]

    return " ".join(tokens)


# ─────────────────────────────────────────────
# VADER SENTIMENT
# ─────────────────────────────────────────────
def vader_sentiment(text: str, vader) -> dict:
    """
    Score text using VADER.
    Returns compound score between -1 and +1.
    """
    if not text:
        return {"compound": 0, "pos": 0, "neg": 0, "neu": 1}

    scores = vader.polarity_scores(text)
    return scores


# ─────────────────────────────────────────────
# FINBERT SENTIMENT
# ─────────────────────────────────────────────
def finbert_sentiment(text: str, finbert) -> dict:
    """
    Score text using FinBERT.
    Returns label (positive/negative/neutral)
    and confidence score.
    """
    if not text or len(text.strip()) < 10:
        return {"label": "neutral", "score": 0.0, "compound": 0.0}

    try:
        # truncate to 512 chars for BERT
        result = finbert(text[:512])[0]
        label  = result["label"].lower()
        score  = result["score"]

        # convert to compound score (-1 to +1)
        if label == "positive":
            compound = score
        elif label == "negative":
            compound = -score
        else:
            compound = 0.0

        return {
            "label":    label,
            "score":    round(score, 4),
            "compound": round(compound, 4),
        }

    except Exception as e:
        return {"label": "neutral", "score": 0.0, "compound": 0.0}


# ─────────────────────────────────────────────
# ANALYZE ONE STOCK
# ─────────────────────────────────────────────
def analyze_stock_sentiment(stock_name: str, vader, finbert) -> dict:
    """
    Fetch news and score sentiment for one stock.
    Returns aggregated scores from both models.
    """
    articles = fetch_news(stock_name)

    if not articles:
        return {
            "name":              stock_name,
            "articles_found":    0,
            "vader_compound":    0.0,
            "finbert_compound":  0.0,
            "combined_score":    0.0,
            "sentiment_label":   "neutral",
            "confidence":        0.0,
            "headlines":         [],
        }

    vader_scores   = []
    finbert_scores = []
    headlines      = []

    for article in articles:
        # combine title + description
        title       = article.get("title", "") or ""
        description = article.get("description", "") or ""
        text        = f"{title}. {description}"

        if not text.strip():
            continue

        headlines.append(title)

        # preprocess
        clean_text = preprocess_text(text)

        # score with both models
        v_score = vader_sentiment(text, vader)
        f_score = finbert_sentiment(text, finbert)

        vader_scores.append(v_score["compound"])
        finbert_scores.append(f_score["compound"])

    # aggregate scores
    avg_vader   = np.mean(vader_scores)   if vader_scores   else 0.0
    avg_finbert = np.mean(finbert_scores) if finbert_scores else 0.0

    # combined score (weighted — FinBERT is more accurate)
    combined = (avg_vader * 0.3) + (avg_finbert * 0.7)

    # label
    if combined > 0.15:
        label = "positive"
    elif combined < -0.15:
        label = "negative"
    else:
        label = "neutral"

    # confidence
    confidence = abs(combined)

    return {
        "name":             stock_name,
        "articles_found":   len(articles),
        "vader_compound":   round(avg_vader,   4),
        "finbert_compound": round(avg_finbert, 4),
        "combined_score":   round(combined,    4),
        "sentiment_label":  label,
        "confidence":       round(confidence,  4),
        "headlines":        headlines[:5],
    }


# ─────────────────────────────────────────────
# ANALYZE ALL STOCKS
# ─────────────────────────────────────────────
def analyze_all_stocks(vader, finbert) -> pd.DataFrame:
    """
    Run sentiment analysis for all stocks
    in the portfolio universe.
    """
    print("\nAnalyzing sentiment for all stocks...")
    print("=" * 55)

    results = []

    for i, stock_name in enumerate(ALL_TICKERS.keys()):
        print(f"\n[{i+1}/{len(ALL_TICKERS)}] {stock_name}")
        result = analyze_stock_sentiment(stock_name, vader, finbert)
        results.append(result)

        # print summary
        label = result["sentiment_label"]
        score = result["combined_score"]
        icon  = "📈" if label == "positive" else "📉" if label == "negative" else "➡️"
        print(f"  {icon} {label.upper():<10} score: {score:>6.3f}  "
              f"articles: {result['articles_found']}")

        # rate limit — 100 requests/day free
        time.sleep(1)

    df = pd.DataFrame(results)
    return df


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_sentiment_results(df: pd.DataFrame):
    # save main results
    save_df = df.drop(columns=["headlines"])
    save_df.to_csv("data/sentiment/sentiment_scores.csv", index=False)
    print("\n  ✓ data/sentiment/sentiment_scores.csv")

    # save headlines separately
    headlines_data = []
    for _, row in df.iterrows():
        for headline in row["headlines"]:
            headlines_data.append({
                "stock":    row["name"],
                "headline": headline,
                "sentiment": row["sentiment_label"],
                "score":    row["combined_score"],
            })

    pd.DataFrame(headlines_data).to_csv(
        "data/sentiment/headlines.csv", index=False
    )
    print("  ✓ data/sentiment/headlines.csv")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  SENTIMENT ANALYSIS SUMMARY")
    print("=" * 55)

    positive = df[df["sentiment_label"] == "positive"]
    negative = df[df["sentiment_label"] == "negative"]
    neutral  = df[df["sentiment_label"] == "neutral"]

    print(f"\n  📈 Positive stocks : {len(positive)}")
    print(f"  📉 Negative stocks : {len(negative)}")
    print(f"  ➡️  Neutral stocks  : {len(neutral)}")

    print(f"\n  Top 5 Most Positive:")
    top5 = df.nlargest(5, "combined_score")[
        ["name", "sentiment_label", "combined_score"]
    ]
    print(top5.to_string(index=False))

    print(f"\n  Top 5 Most Negative:")
    bot5 = df.nsmallest(5, "combined_score")[
        ["name", "sentiment_label", "combined_score"]
    ]
    print(bot5.to_string(index=False))

    print(f"\n  Overall Market Sentiment: ", end="")
    avg = df["combined_score"].mean()
    if avg > 0.1:
        print(f"📈 BULLISH ({avg:.3f})")
    elif avg < -0.1:
        print(f"📉 BEARISH ({avg:.3f})")
    else:
        print(f"➡️  NEUTRAL ({avg:.3f})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  SENTIMENT ANALYSIS ENGINE")
    print("  VADER + FinBERT (ML Transformer)")
    print("=" * 55)

    # load models
    vader, finbert = load_models()

    # analyze all stocks
    df = analyze_all_stocks(vader, finbert)

    # save
    print("\nSaving results...")
    save_sentiment_results(df)

    # summary
    print_summary(df)