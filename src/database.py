import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import pandas as pd
from config import MARKET_META

# ─────────────────────────────────────────────
# CONNECTION CONFIG
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "portfolio_manager_db",   # ← your db name
    "user":     "postgres",
    "password": "Jaiman-2005",     # ← your pgAdmin password
}

def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("  ✓ Connected to PostgreSQL")
        return conn
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
# CREATE ALL TABLES
# ─────────────────────────────────────────────
def create_tables():
    conn = get_connection()
    cur  = conn.cursor()

    print("\nCreating tables...")

    # 1. STOCKS — master list
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            id          SERIAL PRIMARY KEY,
            name        VARCHAR(50)  NOT NULL UNIQUE,
            symbol      VARCHAR(20)  NOT NULL,
            market      VARCHAR(20)  NOT NULL,
            sector      VARCHAR(50),
            currency    VARCHAR(10),
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("  ✓ stocks table ready")

    # 2. PRICES — historical OHLCV
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id          SERIAL PRIMARY KEY,
            stock_id    INT REFERENCES stocks(id) ON DELETE CASCADE,
            date        DATE        NOT NULL,
            open        NUMERIC(12,4),
            high        NUMERIC(12,4),
            low         NUMERIC(12,4),
            close       NUMERIC(12,4),
            volume      BIGINT,
            UNIQUE(stock_id, date)
        );
    """)
    print("  ✓ prices table ready")

    # 3. LIVE PRICES — latest snapshot
    cur.execute("""
        CREATE TABLE IF NOT EXISTS live_prices (
            id          SERIAL PRIMARY KEY,
            stock_id    INT REFERENCES stocks(id) ON DELETE CASCADE,
            price       NUMERIC(12,4),
            prev_close  NUMERIC(12,4),
            change      NUMERIC(10,4),
            change_pct  NUMERIC(8,4),
            fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("  ✓ live_prices table ready")

    # 4. RETURNS — calculated in Step 3 ETL
    cur.execute("""
        CREATE TABLE IF NOT EXISTS returns (
            id            SERIAL PRIMARY KEY,
            stock_id      INT REFERENCES stocks(id) ON DELETE CASCADE,
            date          DATE        NOT NULL,
            daily_return  NUMERIC(10,6),
            UNIQUE(stock_id, date)
        );
    """)
    print("  ✓ returns table ready")

    conn.commit()
    cur.close()
    conn.close()
    print("\nAll tables created successfully!")


# ─────────────────────────────────────────────
# LOAD STOCKS MASTER
# ─────────────────────────────────────────────
def load_stocks_master(live_df):
    conn = get_connection()
    cur  = conn.cursor()

    inserted = 0
    skipped  = 0

    print("\nLoading stocks master table...")

    for _, row in live_df.iterrows():
        try:
            cur.execute("""
                INSERT INTO stocks (name, symbol, market, sector, currency)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING;
            """, (
                row["name"],
                row["symbol"],
                row["market"],
                row["sector"],
                row["currency"]
            ))

            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"  ✗ {row['name']}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Stocks → inserted: {inserted}, skipped: {skipped}")


# ─────────────────────────────────────────────
# LOAD HISTORICAL PRICES
# ─────────────────────────────────────────────
def load_historical_prices(hist_df):
    conn = get_connection()
    cur  = conn.cursor()

    inserted = 0
    errors   = 0

    print("\nLoading historical prices (this may take a minute)...")

    for _, row in hist_df.iterrows():
        try:
            # get stock_id
            cur.execute("SELECT id FROM stocks WHERE name = %s", (row["name"],))
            result = cur.fetchone()
            if not result:
                continue
            stock_id = result[0]

            cur.execute("""
                INSERT INTO prices (stock_id, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (stock_id, date) DO NOTHING;
            """, (
                stock_id,
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"]
            ))

            inserted += 1

        except Exception as e:
            errors += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Historical prices → inserted: {inserted:,}, errors: {errors}")


# ─────────────────────────────────────────────
# LOAD LIVE PRICES
# ─────────────────────────────────────────────
def load_live_prices(live_df):
    conn = get_connection()
    cur  = conn.cursor()

    inserted = 0
    errors   = 0

    print("\nLoading live prices...")

    for _, row in live_df.iterrows():
        try:
            cur.execute("SELECT id FROM stocks WHERE name = %s", (row["name"],))
            result = cur.fetchone()
            if not result:
                continue
            stock_id = result[0]

            cur.execute("""
                INSERT INTO live_prices (stock_id, price, prev_close, change, change_pct)
                VALUES (%s, %s, %s, %s, %s);
            """, (
                stock_id,
                row["price"]      if pd.notna(row["price"])      else None,
                row["prev_close"] if pd.notna(row["prev_close"]) else None,
                row["change"]     if pd.notna(row["change"])     else None,
                row["change_pct"] if pd.notna(row["change_pct"]) else None,
            ))

            inserted += 1

        except Exception as e:
            errors += 1
            print(f"  ✗ {row['name']}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Live prices → inserted: {inserted}, errors: {errors}")


# ─────────────────────────────────────────────
# VERIFY DATA
# ─────────────────────────────────────────────
def verify_data():
    conn = get_connection()
    cur  = conn.cursor()

    print("\n" + "=" * 40)
    print("  DATABASE VERIFICATION")
    print("=" * 40)

    tables = ["stocks", "prices", "live_prices", "returns"]
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        print(f"  {table:<20} {count:>8,} rows")

    print("\nSample — latest prices:")
    cur.execute("""
        SELECT s.name, s.market, p.date, p.close, s.currency
        FROM prices p
        JOIN stocks s ON s.id = p.stock_id
        ORDER BY p.date DESC, s.name
        LIMIT 8;
    """)
    rows = cur.fetchall()
    print(f"  {'Stock':<15} {'Market':<10} {'Date':<12} {'Close':>10} {'CCY'}")
    print("  " + "-" * 55)
    for r in rows:
        print(f"  {r[0]:<15} {r[1]:<10} {str(r[2]):<12} {float(r[3]):>10.2f} {r[4]}")

    cur.close()
    conn.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 40)
    print("  PORTFOLIO DB — SETUP & LOAD")
    print("=" * 40)

    # load CSVs
    live_df = pd.read_csv("data/raw/live_prices.csv")
    hist_df = pd.read_csv("data/raw/historical_prices.csv")

    print(f"\nCSVs loaded:")
    print(f"  live_prices.csv      → {len(live_df)} rows")
    print(f"  historical_prices.csv → {len(hist_df):,} rows")

    # run pipeline
    create_tables()
    load_stocks_master(live_df)
    load_historical_prices(hist_df)
    load_live_prices(live_df)

    # verify
    verify_data()