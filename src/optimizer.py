import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("Loading processed data...")

    stats_df     = pd.read_csv("data/processed/stock_stats.csv")
    returns_pivot = pd.read_csv("data/processed/returns_pivot.csv",
                                index_col=0, parse_dates=True)

    # drop columns with too many NaNs
    returns_pivot = returns_pivot.dropna(axis=1, thresh=int(len(returns_pivot) * 0.8))
    returns_pivot = returns_pivot.fillna(0)

    print(f"  ✓ {len(stats_df)} stocks loaded")
    print(f"  ✓ Returns matrix: {returns_pivot.shape}")

    return stats_df, returns_pivot


# ─────────────────────────────────────────────
# PORTFOLIO METRICS
# ─────────────────────────────────────────────
def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.05):
    """
    Given weights, return:
      - annualised portfolio return
      - annualised portfolio risk
      - Sharpe ratio
    """
    TRADING_DAYS = 252

    port_return = np.dot(weights, mean_returns) * TRADING_DAYS
    port_risk   = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix * TRADING_DAYS, weights))
    )
    sharpe = (port_return - risk_free_rate) / port_risk if port_risk > 0 else 0

    return port_return, port_risk, sharpe


# ─────────────────────────────────────────────
# OPTIMIZER 1 — Maximum Sharpe Ratio
# ─────────────────────────────────────────────
def maximize_sharpe(mean_returns, cov_matrix, risk_free_rate=0.05,
                    max_weight=0.20, min_stocks=8):
    """
    Find weights that maximize the Sharpe Ratio.
    Constraints:
      - weights sum to 1
      - each weight between 0 and max_weight (default 20%)
      - at least min_stocks stocks in portfolio
    """
    n = len(mean_returns)

    def neg_sharpe(weights):
        _, _, sharpe = portfolio_metrics(weights, mean_returns,
                                         cov_matrix, risk_free_rate)
        return -sharpe   # minimize negative sharpe = maximize sharpe

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},   # weights sum to 1
    ]

    bounds = tuple((0.01, max_weight) for _ in range(n))

    # equal weight starting point
    init_weights = np.array([1 / n] * n)

    result = minimize(
        neg_sharpe,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    return result


# ─────────────────────────────────────────────
# OPTIMIZER 2 — Minimum Variance
# ─────────────────────────────────────────────
def minimize_variance(mean_returns, cov_matrix, max_weight=0.20):
    """
    Find weights that minimize portfolio variance (risk).
    """
    n = len(mean_returns)

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix * 252, weights))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]

    bounds = tuple((0.01, max_weight) for _ in range(n))
    init_weights = np.array([1 / n] * n)

    result = minimize(
        portfolio_variance,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )

    return result


# ─────────────────────────────────────────────
# EFFICIENT FRONTIER — Monte Carlo simulation
# ─────────────────────────────────────────────
def generate_efficient_frontier(mean_returns, cov_matrix,
                                 n_portfolios=10000, risk_free_rate=0.05):
    """
    Simulate 10,000 random portfolios to plot the efficient frontier.
    """
    print(f"\nSimulating {n_portfolios:,} random portfolios...")

    n        = len(mean_returns)
    results  = np.zeros((3, n_portfolios))   # [return, risk, sharpe]
    weights_record = []

    for i in range(n_portfolios):
        w = np.random.random(n)
        w = w / np.sum(w)                    # normalize to sum = 1

        ret, risk, sharpe = portfolio_metrics(w, mean_returns,
                                               cov_matrix, risk_free_rate)
        results[0, i] = ret
        results[1, i] = risk
        results[2, i] = sharpe
        weights_record.append(w)

    frontier_df = pd.DataFrame({
        "return": results[0],
        "risk":   results[1],
        "sharpe": results[2],
    })

    print(f"  ✓ Frontier generated")
    print(f"  Best Sharpe in simulation : {frontier_df['sharpe'].max():.4f}")
    print(f"  Return range              : {frontier_df['return'].min()*100:.1f}% "
          f"→ {frontier_df['return'].max()*100:.1f}%")
    print(f"  Risk range                : {frontier_df['risk'].min()*100:.1f}% "
          f"→ {frontier_df['risk'].max()*100:.1f}%")

    return frontier_df, weights_record


# ─────────────────────────────────────────────
# RISK PROFILE PORTFOLIOS
# ─────────────────────────────────────────────
def build_risk_profiles(mean_returns, cov_matrix, tickers):
    """
    Build 3 portfolios for different investor risk appetites.
    Conservative → max 10% per stock
    Moderate     → max 20% per stock
    Aggressive   → max 35% per stock
    """
    profiles = {
        "Conservative": 0.10,
        "Moderate":     0.20,
        "Aggressive":   0.35,
    }

    profile_results = {}

    for profile, max_wt in profiles.items():
        result = maximize_sharpe(mean_returns, cov_matrix, max_weight=max_wt)

        if result.success:
            weights = result.x
            ret, risk, sharpe = portfolio_metrics(weights, mean_returns, cov_matrix)

            # build allocation table
            alloc = pd.DataFrame({
                "stock":  tickers,
                "weight": np.round(weights * 100, 2)
            })
            alloc = alloc[alloc["weight"] > 0.5].sort_values("weight", ascending=False)

            profile_results[profile] = {
                "weights":       weights,
                "return":        round(ret   * 100, 2),
                "risk":          round(risk  * 100, 2),
                "sharpe":        round(sharpe, 4),
                "allocation":    alloc,
                "n_stocks":      len(alloc),
            }

    return profile_results


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
def save_results(profile_results, frontier_df, tickers,
                 max_sharpe_weights, min_var_weights):

    os.makedirs("data/processed", exist_ok=True)

    # save efficient frontier
    frontier_df.to_csv("data/processed/efficient_frontier.csv", index=False)
    print("\n  ✓ data/processed/efficient_frontier.csv")

    # save optimal weights
    for profile, res in profile_results.items():
        fname = f"data/processed/portfolio_{profile.lower()}.csv"
        res["allocation"].to_csv(fname, index=False)
        print(f"  ✓ {fname}")

    # save max sharpe + min variance
    pd.DataFrame({
        "stock":           tickers,
        "max_sharpe_wt":   np.round(max_sharpe_weights * 100, 2),
        "min_var_wt":      np.round(min_var_weights    * 100, 2),
    }).to_csv("data/processed/optimal_weights.csv", index=False)
    print("  ✓ data/processed/optimal_weights.csv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  PORTFOLIO OPTIMIZER — MARKOWITZ MODEL")
    print("=" * 55)

    # load
    stats_df, returns_pivot = load_data()

    tickers      = list(returns_pivot.columns)
    mean_returns = returns_pivot.mean()
    cov_matrix   = returns_pivot.cov()

    # ── Max Sharpe ──
    print("\nOptimizing Max Sharpe Ratio portfolio...")
    ms_result = maximize_sharpe(mean_returns, cov_matrix)

    if ms_result.success:
        ms_weights = ms_result.x
        ms_ret, ms_risk, ms_sharpe = portfolio_metrics(ms_weights,
                                                        mean_returns, cov_matrix)
        print(f"  ✓ Max Sharpe Portfolio:")
        print(f"     Return : {ms_ret  * 100:.2f}%")
        print(f"     Risk   : {ms_risk * 100:.2f}%")
        print(f"     Sharpe : {ms_sharpe:.4f}")
    else:
        print("  ✗ Max Sharpe optimization failed")
        ms_weights = np.array([1 / len(tickers)] * len(tickers))

    # ── Min Variance ──
    print("\nOptimizing Minimum Variance portfolio...")
    mv_result = minimize_variance(mean_returns, cov_matrix)

    if mv_result.success:
        mv_weights = mv_result.x
        mv_ret, mv_risk, mv_sharpe = portfolio_metrics(mv_weights,
                                                        mean_returns, cov_matrix)
        print(f"  ✓ Minimum Variance Portfolio:")
        print(f"     Return : {mv_ret  * 100:.2f}%")
        print(f"     Risk   : {mv_risk * 100:.2f}%")
        print(f"     Sharpe : {mv_sharpe:.4f}")
    else:
        print("  ✗ Min Variance optimization failed")
        mv_weights = np.array([1 / len(tickers)] * len(tickers))

    # ── Efficient Frontier ──
    frontier_df, _ = generate_efficient_frontier(mean_returns, cov_matrix)

    # ── Risk Profiles ──
    print("\nBuilding risk profile portfolios...")
    profile_results = build_risk_profiles(mean_returns, cov_matrix, tickers)

    for profile, res in profile_results.items():
        print(f"\n  {profile} Portfolio:")
        print(f"     Return   : {res['return']}%")
        print(f"     Risk     : {res['risk']}%")
        print(f"     Sharpe   : {res['sharpe']}")
        print(f"     Stocks   : {res['n_stocks']}")
        print(f"     Top holdings:")
        print(res["allocation"].head(5).to_string(index=False))

    # ── Save ──
    print("\nSaving results...")
    save_results(profile_results, frontier_df, tickers,
                 ms_weights, mv_weights)

    print("\n" + "=" * 55)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 55)