import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("Loading data for visualization...")

    frontier_df   = pd.read_csv("data/processed/efficient_frontier.csv")
    stats_df      = pd.read_csv("data/processed/stock_stats.csv")
    corr_matrix   = pd.read_csv("data/processed/correlation_matrix.csv",
                                 index_col=0)
    optimal_df    = pd.read_csv("data/processed/optimal_weights.csv")
    returns_pivot = pd.read_csv("data/processed/returns_pivot.csv",
                                 index_col=0, parse_dates=True)

    profiles = {}
    for p in ["conservative", "moderate", "aggressive"]:
        path = f"data/processed/portfolio_{p}.csv"
        if os.path.exists(path):
            profiles[p.capitalize()] = pd.read_csv(path)

    print(f"  ✓ All data loaded")
    return frontier_df, stats_df, corr_matrix, optimal_df, returns_pivot, profiles


# ─────────────────────────────────────────────
# CHART 1 — Efficient Frontier
# ─────────────────────────────────────────────
def plot_efficient_frontier(frontier_df, stats_df):
    print("\nBuilding Chart 1: Efficient Frontier...")

    # find max sharpe and min variance points
    max_sharpe_idx = frontier_df["sharpe"].idxmax()
    min_var_idx    = frontier_df["risk"].idxmin()

    fig = go.Figure()

    # all simulated portfolios
    fig.add_trace(go.Scatter(
        x=frontier_df["risk"]   * 100,
        y=frontier_df["return"] * 100,
        mode="markers",
        marker=dict(
            color=frontier_df["sharpe"],
            colorscale="Viridis",
            size=3,
            opacity=0.5,
            colorbar=dict(title="Sharpe Ratio"),
        ),
        name="Simulated Portfolios",
        hovertemplate="Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # max sharpe point
    fig.add_trace(go.Scatter(
        x=[frontier_df.loc[max_sharpe_idx, "risk"]   * 100],
        y=[frontier_df.loc[max_sharpe_idx, "return"] * 100],
        mode="markers",
        marker=dict(color="red", size=14, symbol="star"),
        name="Max Sharpe",
        hovertemplate="Max Sharpe<br>Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # min variance point
    fig.add_trace(go.Scatter(
        x=[frontier_df.loc[min_var_idx, "risk"]   * 100],
        y=[frontier_df.loc[min_var_idx, "return"] * 100],
        mode="markers",
        marker=dict(color="green", size=14, symbol="diamond"),
        name="Min Variance",
        hovertemplate="Min Variance<br>Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    # individual stocks
    fig.add_trace(go.Scatter(
        x=stats_df["ann_risk"],
        y=stats_df["ann_return"],
        mode="markers+text",
        marker=dict(color="orange", size=10, symbol="circle"),
        text=stats_df["name"],
        textposition="top center",
        textfont=dict(size=9),
        name="Individual Stocks",
        hovertemplate="%{text}<br>Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Efficient Frontier — 10,000 Simulated Portfolios",
        xaxis_title="Annualised Risk (Std Dev %)",
        yaxis_title="Annualised Return (%)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600,
        template="plotly_white",
    )

    fig.write_html("data/processed/chart_efficient_frontier.html")
    print("  ✓ Saved → chart_efficient_frontier.html")
    return fig


# ─────────────────────────────────────────────
# CHART 2 — Risk vs Return Bubble Chart
# ─────────────────────────────────────────────
def plot_risk_return(stats_df):
    print("Building Chart 2: Risk vs Return...")

    MARKET_COLORS = {
        "India":  "#1D9E75",
        "US":     "#185FA5",
        "Europe": "#D85A30",
        "Asia":   "#BA7517",
    }

    fig = go.Figure()

    for market in stats_df["market"].unique():
        df_m = stats_df[stats_df["market"] == market]

        fig.add_trace(go.Scatter(
            x=df_m["ann_risk"],
            y=df_m["ann_return"],
            mode="markers+text",
            marker=dict(
                size=df_m["sharpe"].apply(lambda s: max(10, s * 20)),
                color=MARKET_COLORS.get(market, "#888"),
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            text=df_m["name"],
            textposition="top center",
            textfont=dict(size=9),
            name=market,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Return: %{y:.1f}%<br>"
                "Risk: %{x:.1f}%<br>"
                "<extra></extra>"
            ),
        ))

    # add risk-free rate line
    fig.add_hline(
        y=5, line_dash="dash",
        line_color="gray",
        annotation_text="Risk-Free Rate (5%)",
        annotation_position="right",
    )

    fig.update_layout(
        title="Risk vs Return — Global Stock Universe",
        xaxis_title="Annualised Risk (Std Dev %)",
        yaxis_title="Annualised Return (%)",
        height=550,
        template="plotly_white",
        legend_title="Market",
    )

    fig.write_html("data/processed/chart_risk_return.html")
    print("  ✓ Saved → chart_risk_return.html")
    return fig


# ─────────────────────────────────────────────
# CHART 3 — Portfolio Allocation Donut Charts
# ─────────────────────────────────────────────
def plot_allocations(profiles):
    print("Building Chart 3: Portfolio Allocations...")

    PROFILE_COLORS = {
        "Conservative": ["#1D9E75","#0F6E56","#085041","#5DCAA5","#9FE1CB",
                         "#E1F5EE","#04342C","#3B6D11","#639922","#97C459"],
        "Moderate":     ["#185FA5","#0C447C","#042C53","#378ADD","#85B7EB",
                         "#B5D4F4","#E6F1FB","#534AB7","#7F77DD","#AFA9EC"],
        "Aggressive":   ["#D85A30","#993C1D","#712B13","#F0997B","#F5C4B3",
                         "#FAECE7","#BA7517","#EF9F27","#FAC775","#FAEEDA"],
    }

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Conservative", "Moderate", "Aggressive"],
        horizontal_spacing=0.05,
    )

    for i, (profile, alloc_df) in enumerate(profiles.items()):

        # ── only keep stocks with weight > 2%, group rest as "Others" ──
        significant = alloc_df[alloc_df["weight"] >= 2].copy()
        others_wt   = alloc_df[alloc_df["weight"] <  2]["weight"].sum()

        if others_wt > 0:
            others_row = pd.DataFrame([{"stock": "Others", "weight": round(others_wt, 2)}])
            significant = pd.concat([significant, others_row], ignore_index=True)

        significant = significant.sort_values("weight", ascending=False)

        colors = PROFILE_COLORS.get(profile, px.colors.qualitative.Set3)

        fig.add_trace(
            go.Pie(
                labels=significant["stock"],
                values=significant["weight"],
                hole=0.5,
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=11),
                marker=dict(
                    colors=colors[:len(significant)],
                    line=dict(color="white", width=2),
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Weight: %{value:.1f}%<br>"
                    "Share: %{percent}<extra></extra>"
                ),
                pull=[0.05 if j == 0 else 0 for j in range(len(significant))],
                name=profile,
                direction="clockwise",
                sort=False,
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(
        title=dict(
            text="Portfolio Allocations by Risk Profile",
            font=dict(size=18),
            x=0.5,
        ),
        height=520,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(t=80, b=100, l=40, r=40),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        annotations=[
            dict(text="Conservative", x=0.115, y=0.5, font_size=13,
                 font_color="#1D9E75", showarrow=False, xref="paper", yref="paper"),
            dict(text="Moderate",     x=0.5,   y=0.5, font_size=13,
                 font_color="#185FA5", showarrow=False, xref="paper", yref="paper"),
            dict(text="Aggressive",   x=0.885, y=0.5, font_size=13,
                 font_color="#D85A30", showarrow=False, xref="paper", yref="paper"),
        ],
    )

    fig.write_html(
        "data/processed/chart_allocations.html",
        config={
            "scrollZoom":     True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["downloadSVG"],
        }
    )
    print("  ✓ Saved → chart_allocations.html")
    return fig

# ─────────────────────────────────────────────
# CHART 4 — Correlation Heatmap
# ─────────────────────────────────────────────
def plot_correlation_heatmap(corr_matrix):
    print("Building Chart 4: Correlation Heatmap...")

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=8),
        hoverongaps=False,
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    ))

    fig.update_layout(
        title="Correlation Matrix — Global Portfolio",
        height=650,
        template="plotly_white",
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )

    fig.write_html("data/processed/chart_correlation.html")
    print("  ✓ Saved → chart_correlation.html")
    return fig


# ─────────────────────────────────────────────
# CHART 5 — Cumulative Returns
# ─────────────────────────────────────────────
def plot_cumulative_returns(returns_pivot, profiles, stats_df):
    print("Building Chart 5: Cumulative Returns...")

    fig = go.Figure()

    PROFILE_COLORS = {
        "Conservative": "#1D9E75",
        "Moderate":     "#185FA5",
        "Aggressive":   "#D85A30",
    }

    for profile, alloc_df in profiles.items():
        # get stocks that exist in returns pivot
        stocks    = alloc_df["stock"].tolist()
        available = [s for s in stocks if s in returns_pivot.columns]

        if not available:
            continue

        weights = alloc_df.set_index("stock").loc[available, "weight"] / 100
        weights = weights / weights.sum()   # renormalize

        port_returns   = returns_pivot[available].fillna(0).dot(weights)
        cumulative_ret = (1 + port_returns).cumprod() - 1

        fig.add_trace(go.Scatter(
            x=returns_pivot.index,
            y=cumulative_ret * 100,
            mode="lines",
            name=profile,
            line=dict(color=PROFILE_COLORS[profile], width=2),
            hovertemplate=f"{profile}<br>%{{x}}<br>Return: %{{y:.1f}}%<extra></extra>",
        ))

    # equal weight benchmark
    eq_returns    = returns_pivot.fillna(0).mean(axis=1)
    eq_cumulative = (1 + eq_returns).cumprod() - 1

    fig.add_trace(go.Scatter(
        x=returns_pivot.index,
        y=eq_cumulative * 100,
        mode="lines",
        name="Equal Weight Benchmark",
        line=dict(color="gray", width=1.5, dash="dash"),
        hovertemplate="Benchmark<br>%{x}<br>Return: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Cumulative Portfolio Returns vs Benchmark (2020–2024)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=500,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    fig.write_html("data/processed/chart_cumulative_returns.html")
    print("  ✓ Saved → chart_cumulative_returns.html")
    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    frontier_df, stats_df, corr_matrix, \
    optimal_df, returns_pivot, profiles = load_data()

    # regenerate just the allocation chart
    plot_allocations(profiles)
    print("Done! Open data/processed/chart_allocations.html")

    print("=" * 50)
    print("  PORTFOLIO VISUALIZATION")
    print("=" * 50)

    # load
    frontier_df, stats_df, corr_matrix, \
    optimal_df, returns_pivot, profiles = load_data()

    # build all charts
    plot_efficient_frontier(frontier_df, stats_df)
    plot_risk_return(stats_df)
    plot_allocations(profiles)
    plot_correlation_heatmap(corr_matrix)
    plot_cumulative_returns(returns_pivot, profiles, stats_df)

    print("\n" + "=" * 50)
    print("  ALL CHARTS SAVED")
    print("=" * 50)
    print("""
  Open these files in your browser:
  → data/processed/chart_efficient_frontier.html
  → data/processed/chart_risk_return.html
  → data/processed/chart_allocations.html
  → data/processed/chart_correlation.html
  → data/processed/chart_cumulative_returns.html
    """)