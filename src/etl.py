import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import psycopg2
from database import get_connection

# ─────────────────────────────────────────────
# EXTRACT — pull from PostgreSQL
# ─────────────────────────────────────────────
def extract_prices():
    conn = get_connection()

    print("Extracting prices from database...")

    query = """
        SELECT
            s.name,
            s.symbol,
            s.market,
            s.sector,
            s.currency,
            p.date,
            p.close
        FROM prices p
        JOIN stocks s ON s.id = p.stock_id
        ORDER BY s.name, p.date;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)

    print(f"  ✓ Extracted {len(df):,} rows across {df['name'].nunique()} stocks")
    return df


# ─────────────────────────────────────────────
# TRANSFORM 1 — normalize currencies to USD
# ─────────────────────────────────────────────
def normalize_to_usd(df):
    """
    Convert all closing prices to USD using
    approximate average exchange rates.
    """
    print("\nNormalizing currencies to USD...")

    fx_rates = {
        "USD": 1.0,
        "INR": 0.012,    # 1 INR = 0.012 USD
        "EUR": 1.08,     # 1 EUR = 1.08 USD
        "CHF": 1.12,     # 1 CHF = 1.12 USD
        "JPY": 0.0067,   # 1 JPY = 0.0067 USD
        "KRW": 0.00075,  # 1 KRW = 0.00075 USD
        "HKD": 0.128,    # 1 HKD = 0.128 USD
    }

    df["fx_rate"]    = df["currency"].map(fx_rates).fillna(1.0)
    df["close_usd"]  = df["close"] * df["fx_rate"]

    for ccy, rate in fx_rates.items():
        count = len(df[df["currency"] == ccy])
        if count > 0:
            print(f"  {ccy} → USD (rate: {rate})  |  {count:,} rows converted")

    return df


# ─────────────────────────────────────────────
# TRANSFORM 2 — calculate daily returns
# ─────────────────────────────────────────────
def calculate_daily_returns(df):
    """
    Daily return = (today's close - yesterday's close) / yesterday's close
    Calculated per stock using USD-normalized prices.
    """
    print("\nCalculating daily returns...")

    df = df.sort_values(["name", "date"])
    df["daily_return"] = df.groupby("name")["close_usd"].pct_change()

    # drop first row per stock (NaN return)
    df = df.dropna(subset=["daily_return"])

    print(f"  ✓ {len(df):,} daily return rows calculated")
    return df


# ─────────────────────────────────────────────
# TRANSFORM 3 — annualised stats per stock
# ─────────────────────────────────────────────
def calculate_stock_stats(df):
    """
    Per stock:
      - Annualised return
      - Annualised risk (std deviation)
      - Sharpe ratio (assume risk-free rate = 5%)
    """
    print("\nCalculating annualised statistics...")

    TRADING_DAYS  = 252
    RISK_FREE_RATE = 0.05

    stats = []

    for name, group in df.groupby("name"):
        daily_ret  = group["daily_return"]

        ann_return = daily_ret.mean() * TRADING_DAYS
        ann_risk   = daily_ret.std()  * np.sqrt(TRADING_DAYS)
        sharpe     = (ann_return - RISK_FREE_RATE) / ann_risk if ann_risk > 0 else 0

        market = group["market"].iloc[0]
        sector = group["sector"].iloc[0]

        stats.append({
            "name":       name,
            "market":     market,
            "sector":     sector,
            "ann_return": round(ann_return * 100, 2),   # in %
            "ann_risk":   round(ann_risk   * 100, 2),   # in %
            "sharpe":     round(sharpe, 4),
            "data_points": len(daily_ret),
        })

        print(f"  {name:<15} return: {ann_return*100:>6.1f}%  "
              f"risk: {ann_risk*100:>6.1f}%  sharpe: {sharpe:>5.2f}")

    stats_df = pd.DataFrame(stats).sort_values("sharpe", ascending=False)
    return stats_df


# ─────────────────────────────────────────────
# TRANSFORM 4 — correlation matrix
# ─────────────────────────────────────────────
def calculate_correlation_matrix(df):
    """
    Build a pivot table of daily returns per stock
    then compute pairwise correlation.
    """
    print("\nCalculating correlation matrix...")

    pivot = df.pivot_table(
        index="date",
        columns="name",
        values="daily_return"
    )

    # only keep dates where we have data for most stocks
    pivot = pivot.dropna(thresh=int(pivot.shape[1] * 0.7))

    corr_matrix = pivot.corr()

    print(f"  ✓ Correlation matrix: {corr_matrix.shape[0]} x {corr_matrix.shape[1]}")
    return corr_matrix, pivot


# ─────────────────────────────────────────────
# LOAD — push returns back to PostgreSQL
# ─────────────────────────────────────────────
def load_returns(df):
    """
    Load calculated daily returns into the returns table.
    """
    conn = get_connection()
    cur  = conn.cursor()

    print("\nLoading returns into database...")

    inserted = 0
    errors   = 0

    for _, row in df.iterrows():
        try:
            # get stock_id
            cur.execute("SELECT id FROM stocks WHERE name = %s", (row["name"],))
            result = cur.fetchone()
            if not result:
                continue
            stock_id = result[0]

            cur.execute("""
                INSERT INTO returns (stock_id, date, daily_return)
                VALUES (%s, %s, %s)
                ON CONFLICT (stock_id, date) DO NOTHING;
            """, (
                stock_id,
                row["date"].date(),
                round(float(row["daily_return"]), 6)
            ))

            inserted += 1

        except Exception as e:
            errors += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"  ✓ Returns → inserted: {inserted:,}, errors: {errors}")


# ─────────────────────────────────────────────
# SAVE ANALYSIS FILES
# ─────────────────────────────────────────────
def save_analysis_files(stats_df, corr_matrix, pivot):
    os.makedirs("data/processed", exist_ok=True)

    stats_df.to_csv("data/processed/stock_stats.csv", index=False)
    corr_matrix.to_csv("data/processed/correlation_matrix.csv")
    pivot.to_csv("data/processed/returns_pivot.csv")

    print("\nSaved analysis files:")
    print("  ✓ data/processed/stock_stats.csv")
    print("  ✓ data/processed/correlation_matrix.csv")
    print("  ✓ data/processed/returns_pivot.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("  PORTFOLIO ETL PIPELINE")
    print("=" * 50)

    # EXTRACT
    df = extract_prices()

    # TRANSFORM
    df         = normalize_to_usd(df)
    df         = calculate_daily_returns(df)
    stats_df   = calculate_stock_stats(df)
    corr_matrix, pivot = calculate_correlation_matrix(df)

    # LOAD
    load_returns(df)

    # SAVE
    save_analysis_files(stats_df, corr_matrix, pivot)

    # SUMMARY
    print("\n" + "=" * 50)
    print("  ETL COMPLETE — SUMMARY")
    print("=" * 50)
    print(f"\n  Stocks processed : {stats_df.shape[0]}")
    print(f"\n  Top 5 by Sharpe Ratio:")
    print(stats_df[["name", "market", "ann_return",
                     "ann_risk", "sharpe"]].head(5).to_string(index=False))
    print(f"\n  Bottom 5 by Sharpe Ratio:")
    print(stats_df[["name", "market", "ann_return",
                     "ann_risk", "sharpe"]].tail(5).to_string(index=False))