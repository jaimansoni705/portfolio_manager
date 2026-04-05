import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime
from config import ALL_TICKERS, MARKET_META, START_DATE, END_DATE


# ─────────────────────────────────────────────
# 1. LIVE PRICES  (15-min delayed, free)
# ─────────────────────────────────────────────
def fetch_live_prices(tickers: dict) -> pd.DataFrame:
    records = []
    print("Fetching live prices...\n")

    for name, symbol in tickers.items():
        try:
            ticker_obj = yf.Ticker(symbol)

            # ── Method 1: try fast_info first ──
            price    = None
            prev     = None

            try:
                fi    = ticker_obj.fast_info
                price = fi.get("last_price") or fi.get("regularMarketPrice")
                prev  = fi.get("previous_close") or fi.get("regularMarketPreviousClose")
            except Exception:
                pass

            # ── Method 2: fallback to .info (more reliable for global stocks) ──
            if not price:
                try:
                    info  = ticker_obj.info
                    price = (info.get("currentPrice")
                          or info.get("regularMarketPrice")
                          or info.get("navPrice"))
                    prev  = (info.get("previousClose")
                          or info.get("regularMarketPreviousClose"))
                except Exception:
                    pass

            # ── Method 3: fallback to latest 1-day history ──
            if not price:
                try:
                    hist  = ticker_obj.history(period="2d")
                    if not hist.empty:
                        price = round(hist["Close"].iloc[-1], 2)
                        prev  = round(hist["Close"].iloc[-2], 2) if len(hist) > 1 else None
                except Exception:
                    pass

            # ── Calculate change ──
            change   = round(price - prev, 4)        if price and prev else None
            change_p = round((change / prev) * 100, 2) if change and prev else None

            meta = MARKET_META.get(name, {})

            records.append({
                "name":       name,
                "symbol":     symbol,
                "market":     meta.get("market",   "Unknown"),
                "sector":     meta.get("sector",   "Unknown"),
                "currency":   meta.get("currency", "USD"),
                "price":      round(price, 2) if price else None,
                "prev_close": round(prev, 2)  if prev  else None,
                "change":     change,
                "change_pct": change_p,
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            # pretty print
            p_str = f"{price:>10.2f} {meta.get('currency',''):<5}" if price else f"{'N/A':>10}"
            c_str = f"+{change_p}%" if change_p and change_p > 0 else (f"{change_p}%" if change_p else "N/A")
            print(f"  {name:<15} {symbol:<15} {p_str}  {c_str}")

        except Exception as e:
            print(f"  ✗ {name} ({symbol}): {e}")

        time.sleep(0.5)   # slightly longer delay — helps with global tickers

    return pd.DataFrame(records)

# ─────────────────────────────────────────────
# 2. HISTORICAL PRICES  (for optimizer)
# ─────────────────────────────────────────────
def fetch_historical_prices(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """
    Downloads full OHLCV history for all tickers.
    Used for returns calculation, correlation matrix, optimizer.
    """
    all_data = []
    print(f"\nFetching historical data ({start} → {end})...\n")

    for name, symbol in tickers.items():
        try:
            df = yf.download(symbol, start=start, end=end,
                             progress=False, auto_adjust=True)

            if df.empty:
                print(f"  ⚠ No data: {name} ({symbol})")
                continue

            # flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]

            meta = MARKET_META.get(name, {})
            df["name"]     = name
            df["symbol"]   = symbol
            df["market"]   = meta.get("market", "Unknown")
            df["sector"]   = meta.get("sector", "Unknown")
            df["currency"] = meta.get("currency", "USD")
            df["date"]     = df.index
            df.reset_index(drop=True, inplace=True)

            all_data.append(df)
            print(f"  ✓ {name:<15} {len(df):>5} rows")

        except Exception as e:
            print(f"  ✗ {name}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# ─────────────────────────────────────────────
# 3. SAVE TO CSV
# ─────────────────────────────────────────────
def save_to_csv(df: pd.DataFrame, filename: str):
    path = f"data/raw/{filename}"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  Saved → {path}  ({len(df):,} rows)")


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  GLOBAL PORTFOLIO — DATA COLLECTION")
    print("=" * 55)

    # Live snapshot
    live_df = fetch_live_prices(ALL_TICKERS)
    save_to_csv(live_df, "live_prices.csv")

    print("\nLive Snapshot:")
    print(live_df[["name", "market", "price", "currency",
                   "change_pct"]].to_string(index=False))

    # Historical data
    hist_df = fetch_historical_prices(ALL_TICKERS, START_DATE, END_DATE)
    save_to_csv(hist_df, "historical_prices.csv")

    print(f"\nTotal historical rows: {len(hist_df):,}")
    print(f"Markets covered: {hist_df['market'].unique()}")
    print(f"Date range: {hist_df['date'].min()} → {hist_df['date'].max()}")