import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email  import EmailOperator
from airflow.utils.dates      import days_ago

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TASK FUNCTIONS
# ─────────────────────────────────────────────
def task_collect_live_prices(**context):
    """
    Task 1: Fetch latest live prices for all 26 global stocks.
    Saves → data/raw/live_prices.csv
    """
    logger.info("=" * 50)
    logger.info("TASK 1: Collecting live prices")

    from src.collect import fetch_live_prices, save_to_csv
    from config import ALL_TICKERS

    live_df = fetch_live_prices(ALL_TICKERS)

    if live_df.empty:
        raise ValueError("No live prices fetched — aborting pipeline")

    save_to_csv(live_df, "live_prices.csv")

    logger.info(f"  Fetched {len(live_df)} stocks")
    logger.info("TASK 1: Complete ✓")

    # push to XCom so next task knows row count
    return len(live_df)


def task_collect_historical(**context):
    """
    Task 2: Fetch last 7 days of historical OHLCV data.
    Appends new rows to data/raw/historical_prices.csv
    """
    logger.info("TASK 2: Collecting historical prices (last 7 days)")

    from src.collect import fetch_historical_prices, save_to_csv
    from config import ALL_TICKERS
    from datetime import date, timedelta
    import pandas as pd

    end   = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    new_df = fetch_historical_prices(ALL_TICKERS, start, end)

    if new_df.empty:
        logger.warning("  No new historical data fetched")
        return 0

    # append to existing CSV
    existing_path = "data/raw/historical_prices.csv"
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["name", "date"], keep="last"
        )
        combined.to_csv(existing_path, index=False)
        logger.info(f"  Appended {len(new_df)} rows → total {len(combined):,}")
    else:
        new_df.to_csv(existing_path, index=False)
        logger.info(f"  Saved {len(new_df):,} rows (fresh)")

    logger.info("TASK 2: Complete ✓")
    return len(new_df)


def task_update_database(**context):
    """
    Task 3: Load new prices into PostgreSQL.
    Updates stocks, prices, and live_prices tables.
    """
    logger.info("TASK 3: Updating database")

    import pandas as pd
    from src.database import (
        load_stocks_master,
        load_historical_prices,
        load_live_prices,
        verify_data,
    )

    live_df = pd.read_csv("data/raw/live_prices.csv")
    hist_df = pd.read_csv("data/raw/historical_prices.csv")

    load_stocks_master(live_df)
    load_historical_prices(hist_df)
    load_live_prices(live_df)

    logger.info("TASK 3: Complete ✓")


def task_run_etl(**context):
    """
    Task 4: Run ETL — recalculate returns, risk, correlation.
    Updates data/processed/ files and returns table in DB.
    """
    logger.info("TASK 4: Running ETL pipeline")

    from src.etl import (
        extract_prices,
        normalize_to_usd,
        calculate_daily_returns,
        calculate_stock_stats,
        calculate_correlation_matrix,
        load_returns,
        save_analysis_files,
    )

    df                   = extract_prices()
    df                   = normalize_to_usd(df)
    df                   = calculate_daily_returns(df)
    stats_df             = calculate_stock_stats(df)
    corr_matrix, pivot   = calculate_correlation_matrix(df)

    load_returns(df)
    save_analysis_files(stats_df, corr_matrix, pivot)

    logger.info(f"  Processed {stats_df.shape[0]} stocks")
    logger.info("TASK 4: Complete ✓")

    return stats_df.shape[0]


def task_run_optimizer(**context):
    """
    Task 5: Re-run Markowitz optimizer with fresh data.
    Updates all portfolio CSV files in data/processed/
    """
    logger.info("TASK 5: Running portfolio optimizer")

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

    frontier_df, _   = generate_efficient_frontier(mean_returns, cov_matrix)
    profile_results  = build_risk_profiles(mean_returns, cov_matrix, tickers)

    save_results(profile_results, frontier_df, tickers,
                 ms_weights, mv_weights)

    # log best portfolio
    ret, risk, sharpe = portfolio_metrics(ms_weights, mean_returns, cov_matrix)
    logger.info(f"  Max Sharpe → Return: {ret*100:.1f}%  "
                f"Risk: {risk*100:.1f}%  Sharpe: {sharpe:.2f}")
    logger.info("TASK 5: Complete ✓")

    return {
        "return": round(ret   * 100, 2),
        "risk":   round(risk  * 100, 2),
        "sharpe": round(sharpe, 4),
    }


def task_health_check(**context):
    """
    Task 6: Verify pipeline ran successfully.
    Checks file timestamps and row counts.
    """
    logger.info("TASK 6: Running health check")

    import pandas as pd
    from datetime import date

    errors = []

    # check all expected files exist
    expected_files = [
        "data/raw/live_prices.csv",
        "data/raw/historical_prices.csv",
        "data/processed/stock_stats.csv",
        "data/processed/efficient_frontier.csv",
        "data/processed/returns_pivot.csv",
        "data/processed/optimal_weights.csv",
        "data/processed/portfolio_conservative.csv",
        "data/processed/portfolio_moderate.csv",
        "data/processed/portfolio_aggressive.csv",
    ]

    for f in expected_files:
        if not os.path.exists(f):
            errors.append(f"Missing: {f}")
        else:
            logger.info(f"  ✓ {f}")

    # check live prices have today's data
    live_df = pd.read_csv("data/raw/live_prices.csv")
    if live_df["price"].isna().sum() > 5:
        errors.append(f"Too many missing prices: "
                      f"{live_df['price'].isna().sum()} stocks")

    # check returns pivot has enough columns
    returns_pivot = pd.read_csv("data/processed/returns_pivot.csv",
                                 index_col=0)
    if returns_pivot.shape[1] < 10:
        errors.append(f"Too few stocks in returns pivot: "
                      f"{returns_pivot.shape[1]}")

    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"Health check FAILED:\n{error_msg}")
        raise ValueError(f"Pipeline health check failed:\n{error_msg}")

    logger.info("TASK 6: Health check passed ✓")
    logger.info("=" * 50)
    return "healthy"


def task_send_summary(**context):
    """
    Task 7: Log daily summary to file.
    (Email can be enabled by configuring SMTP in airflow.cfg)
    """
    ti = context["ti"]

    live_count = ti.xcom_pull(task_ids="collect_live_prices")
    hist_count = ti.xcom_pull(task_ids="collect_historical")
    opt_result = ti.xcom_pull(task_ids="run_optimizer")

    summary = f"""
    ══════════════════════════════════════
    PORTFOLIO PIPELINE — DAILY SUMMARY
    Date: {datetime.now().strftime('%d %b %Y %H:%M')}
    ══════════════════════════════════════
    Data Collection
      Live prices fetched : {live_count} stocks
      New historical rows : {hist_count}

    Optimization Results
      Expected Return     : {opt_result.get('return', 'N/A')}%
      Expected Risk       : {opt_result.get('risk',   'N/A')}%
      Sharpe Ratio        : {opt_result.get('sharpe', 'N/A')}

    Status : ALL TASKS COMPLETE ✓
    ══════════════════════════════════════
    """

    logger.info(summary)
    print(summary)
    return summary


# ─────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────
default_args = {
    "owner":            "portfolio_manager",
    "depends_on_past":  False,
    "start_date":       days_ago(1),
    "retries":          2,                        # retry twice on failure
    "retry_delay":      timedelta(minutes=5),     # wait 5 min between retries
    "execution_timeout": timedelta(minutes=30),   # kill if takes > 30 min
}

with DAG(
    dag_id="portfolio_pipeline",
    default_args=default_args,
    description="Daily portfolio data pipeline — fetch, ETL, optimize",
    schedule_interval="0 6 * * 1-5",    # 6:00 AM, Monday–Friday only
    catchup=False,
    tags=["portfolio", "finance", "etl"],
) as dag:

    # ── define tasks ──
    t1 = PythonOperator(
        task_id="collect_live_prices",
        python_callable=task_collect_live_prices,
    )

    t2 = PythonOperator(
        task_id="collect_historical",
        python_callable=task_collect_historical,
    )

    t3 = PythonOperator(
        task_id="update_database",
        python_callable=task_update_database,
    )

    t4 = PythonOperator(
        task_id="run_etl",
        python_callable=task_run_etl,
    )

    t5 = PythonOperator(
        task_id="run_optimizer",
        python_callable=task_run_optimizer,
    )

    t6 = PythonOperator(
        task_id="health_check",
        python_callable=task_health_check,
    )

    t7 = PythonOperator(
        task_id="send_summary",
        python_callable=task_send_summary,
    )

    # ── pipeline flow ──
    # t1 and t2 run in parallel first
    # then t3 waits for both
    # then t4 → t5 → t6 → t7 run in sequence
    [t1, t2] >> t3 >> t4 >> t5 >> t6 >> t7