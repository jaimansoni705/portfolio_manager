import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schedule
import time
import logging
from datetime import datetime

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),  # saves to file
        logging.StreamHandler(),                    # also prints to terminal
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PIPELINE TASKS
# ─────────────────────────────────────────────
def task_collect():
    logger.info("─" * 45)
    logger.info("TASK 1: Collecting live + historical prices")
    try:
        import pandas as pd
        from src.collect import fetch_live_prices, fetch_historical_prices, save_to_csv
        from config import ALL_TICKERS
        from datetime import date, timedelta

        # live prices
        live_df = fetch_live_prices(ALL_TICKERS)
        save_to_csv(live_df, "live_prices.csv")
        logger.info(f"  ✓ Live prices: {len(live_df)} stocks")

        # last 7 days historical
        end   = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        new_df = fetch_historical_prices(ALL_TICKERS, start, end)

        # append to existing
        existing_path = "data/raw/historical_prices.csv"
        if os.path.exists(existing_path):
            existing = pd.read_csv(existing_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["name","date"], keep="last")
            combined.to_csv(existing_path, index=False)
            logger.info(f"  ✓ Historical: +{len(new_df)} rows (total {len(combined):,})")
        else:
            new_df.to_csv(existing_path, index=False)

        logger.info("TASK 1: Complete ✓")
        return True

    except Exception as e:
        logger.error(f"TASK 1 FAILED: {e}")
        return False


def task_update_database():
    logger.info("TASK 2: Updating PostgreSQL database")
    try:
        import pandas as pd
        from src.database import (
            load_stocks_master,
            load_historical_prices,
            load_live_prices,
        )

        live_df = pd.read_csv("data/raw/live_prices.csv")
        hist_df = pd.read_csv("data/raw/historical_prices.csv")

        load_stocks_master(live_df)
        load_historical_prices(hist_df)
        load_live_prices(live_df)

        logger.info("TASK 2: Complete ✓")
        return True

    except Exception as e:
        logger.error(f"TASK 2 FAILED: {e}")
        return False


def task_run_etl():
    logger.info("TASK 3: Running ETL pipeline")
    try:
        from src.etl import (
            extract_prices,
            normalize_to_usd,
            calculate_daily_returns,
            calculate_stock_stats,
            calculate_correlation_matrix,
            load_returns,
            save_analysis_files,
        )

        df                 = extract_prices()
        df                 = normalize_to_usd(df)
        df                 = calculate_daily_returns(df)
        stats_df           = calculate_stock_stats(df)
        corr_matrix, pivot = calculate_correlation_matrix(df)

        load_returns(df)
        save_analysis_files(stats_df, corr_matrix, pivot)

        logger.info(f"  ✓ Processed {stats_df.shape[0]} stocks")
        logger.info("TASK 3: Complete ✓")
        return True

    except Exception as e:
        logger.error(f"TASK 3 FAILED: {e}")
        return False


def task_run_optimizer():
    logger.info("TASK 4: Running portfolio optimizer")
    try:
        import pandas as pd
        import numpy as np
        from src.optimizer import (
            maximize_sharpe,
            minimize_variance,
            generate_efficient_frontier,
            build_risk_profiles,
            portfolio_metrics,
            save_results,
        )

        returns_pivot = pd.read_csv(
            "data/processed/returns_pivot.csv",
            index_col=0, parse_dates=True
        )
        returns_pivot = returns_pivot.dropna(
            axis=1, thresh=int(len(returns_pivot) * 0.8)
        ).fillna(0)

        tickers      = list(returns_pivot.columns)
        mean_returns = returns_pivot.mean()
        cov_matrix   = returns_pivot.cov()

        ms_result = maximize_sharpe(mean_returns, cov_matrix)
        mv_result = minimize_variance(mean_returns, cov_matrix)

        ms_weights = ms_result.x if ms_result.success else np.array(
            [1 / len(tickers)] * len(tickers)
        )
        mv_weights = mv_result.x if mv_result.success else np.array(
            [1 / len(tickers)] * len(tickers)
        )

        frontier_df, _  = generate_efficient_frontier(mean_returns, cov_matrix)
        profile_results = build_risk_profiles(mean_returns, cov_matrix, tickers)

        save_results(profile_results, frontier_df, tickers,
                     ms_weights, mv_weights)

        ret, risk, sharpe = portfolio_metrics(ms_weights, mean_returns, cov_matrix)
        logger.info(f"  ✓ Max Sharpe → Return: {ret*100:.1f}%  "
                    f"Risk: {risk*100:.1f}%  Sharpe: {sharpe:.2f}")
        logger.info("TASK 4: Complete ✓")
        return True

    except Exception as e:
        logger.error(f"TASK 4 FAILED: {e}")
        return False


def task_health_check():
    logger.info("TASK 5: Running health check")
    try:
        import pandas as pd

        expected_files = [
            "data/raw/live_prices.csv",
            "data/raw/historical_prices.csv",
            "data/processed/stock_stats.csv",
            "data/processed/efficient_frontier.csv",
            "data/processed/returns_pivot.csv",
            "data/processed/portfolio_conservative.csv",
            "data/processed/portfolio_moderate.csv",
            "data/processed/portfolio_aggressive.csv",
        ]

        missing = [f for f in expected_files if not os.path.exists(f)]
        if missing:
            logger.error(f"  ✗ Missing files: {missing}")
            return False

        live_df = pd.read_csv("data/raw/live_prices.csv")
        missing_prices = live_df["price"].isna().sum()
        if missing_prices > 5:
            logger.warning(f"  ⚠ {missing_prices} stocks missing prices")

        logger.info("  ✓ All files present")
        logger.info("  ✓ Data looks healthy")
        logger.info("TASK 5: Complete ✓")
        return True

    except Exception as e:
        logger.error(f"TASK 5 FAILED: {e}")
        return False


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    start = datetime.now()

    logger.info("=" * 45)
    logger.info(f"PIPELINE STARTED — {start.strftime('%d %b %Y %H:%M')}")
    logger.info("=" * 45)

    steps = [
        ("Data Collection",   task_collect),
        ("Database Update",   task_update_database),
        ("ETL Pipeline",      task_run_etl),
        ("Optimizer",         task_run_optimizer),
        ("Health Check",      task_health_check),
    ]

    results = {}
    for name, task in steps:
        results[name] = task()
        if not results[name]:
            logger.error(f"Pipeline stopped at: {name}")
            break

    # summary
    duration = (datetime.now() - start).seconds
    passed   = sum(results.values())
    total    = len(results)

    logger.info("=" * 45)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 45)
    for name, status in results.items():
        icon = "✓" if status else "✗"
        logger.info(f"  {icon}  {name}")
    logger.info(f"\n  {passed}/{total} tasks passed")
    logger.info(f"  Duration: {duration}s")
    logger.info(f"  Next run: tomorrow at 6:00 AM")
    logger.info("=" * 45)


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # install schedule if needed
    try:
        import schedule
    except ImportError:
        os.system("pip install schedule")
        import schedule

    print("=" * 45)
    print("  PORTFOLIO PIPELINE SCHEDULER")
    print("  Running on Windows Task Scheduler")
    print("=" * 45)
    print(f"  Started : {datetime.now().strftime('%d %b %Y %H:%M')}")
    print(f"  Schedule: Mon–Fri at 6:00 AM")
    print(f"  Logs    : logs/pipeline.log")
    print("=" * 45)
    print("\n  Press Ctrl+C to stop\n")

    # ── schedule jobs ──
    schedule.every().monday.at("06:00").do(run_pipeline)
    schedule.every().tuesday.at("06:00").do(run_pipeline)
    schedule.every().wednesday.at("06:00").do(run_pipeline)
    schedule.every().thursday.at("06:00").do(run_pipeline)
    schedule.every().friday.at("06:00").do(run_pipeline)

    # run once immediately on start to test
    logger.info("Running pipeline once on startup to verify...")
    run_pipeline()

    # keep running
    while True:
        schedule.run_pending()
        time.sleep(60)   # check every minute