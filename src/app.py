import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf
import time

from optimizer import (
    load_data, portfolio_metrics,
    maximize_sharpe, minimize_variance,
    generate_efficient_frontier, build_risk_profiles
)
from config import ALL_TICKERS, MARKET_META

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Global Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        color: #e2e8f0;
    }
    .metric-sub {
        font-size: 12px;
        margin-top: 4px;
    }
    .pos { color: #1D9E75; }
    .neg { color: #D85A30; }
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 0.5rem;
    }
    div[data-testid="stTabs"] button {
        font-size: 14px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_data():
    stats_df      = pd.read_csv("data/processed/stock_stats.csv")
    frontier_df   = pd.read_csv("data/processed/efficient_frontier.csv")
    corr_matrix   = pd.read_csv("data/processed/correlation_matrix.csv",
                                 index_col=0)
    returns_pivot = pd.read_csv("data/processed/returns_pivot.csv",
                                 index_col=0, parse_dates=True)
    optimal_df    = pd.read_csv("data/processed/optimal_weights.csv")

    profiles = {}
    for p in ["conservative", "moderate", "aggressive"]:
        path = f"data/processed/portfolio_{p}.csv"
        if os.path.exists(path):
            profiles[p.capitalize()] = pd.read_csv(path)

    return stats_df, frontier_df, corr_matrix, returns_pivot, optimal_df, profiles


@st.cache_data(ttl=300)
def fetch_live_prices_cached():
    records = []
    for name, symbol in ALL_TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period="2d")
            if not hist.empty:
                price = round(float(hist["Close"].iloc[-1]), 2)
                prev  = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else price
                chg   = round(price - prev, 2)
                chgp  = round((chg / prev) * 100, 2) if prev else 0
                meta  = MARKET_META.get(name, {})
                records.append({
                    "name":       name,
                    "symbol":     symbol,
                    "market":     meta.get("market",   "Unknown"),
                    "sector":     meta.get("sector",   "Unknown"),
                    "currency":   meta.get("currency", "USD"),
                    "price":      price,
                    "change":     chg,
                    "change_pct": chgp,
                })
        except:
            pass
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/2331/2331970.png",
        width=60,
    )
    st.sidebar.title("Portfolio Manager")
    st.sidebar.markdown("*Global Multi-Market Optimizer*")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### ⚙️ Settings")

    investment = st.sidebar.number_input(
        "Investment Amount (USD)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=5000,
        format="%d",
    )

    risk_profile = st.sidebar.selectbox(
        "Risk Profile",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
    )

    max_weight = st.sidebar.slider(
        "Max weight per stock (%)",
        min_value=5,
        max_value=40,
        value=20,
        step=5,
    )

    risk_free = st.sidebar.slider(
        "Risk-Free Rate (%)",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Markets")
    markets = st.sidebar.multiselect(
        "Include Markets",
        ["India", "US", "Europe", "Asia"],
        default=["India", "US", "Europe", "Asia"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<small>Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}</small>",
        unsafe_allow_html=True,
    )

    return investment, risk_profile, max_weight, risk_free / 100, markets


# ─────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────
def tab_overview(stats_df, profiles, risk_profile, investment):
    st.markdown("### 📊 Portfolio Overview")

    profile_data = profiles.get(risk_profile, list(profiles.values())[0])

    # compute portfolio metrics
    total_wt  = profile_data["weight"].sum()
    top_stock = profile_data.iloc[0]["stock"]
    n_stocks  = len(profile_data[profile_data["weight"] >= 2])

    merged = profile_data.merge(
        stats_df[["name", "ann_return", "ann_risk", "sharpe"]],
        left_on="stock", right_on="name", how="left"
    )
    merged["contrib"] = merged["weight"] / 100 * merged["ann_return"]
    port_return = merged["contrib"].sum()
    port_risk   = merged["weight"].apply(
        lambda w: w / 100
    ).values @ np.diag(
        merged["ann_risk"].fillna(15).values
    ) @ (merged["weight"] / 100).values
    sharpe = (port_return - 5) / port_risk if port_risk > 0 else 0

    # ── metric cards ──
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Portfolio Value",  f"${investment:,.0f}",
         f"+${investment*port_return/100:,.0f} est.", "pos"),
        (c2, "Expected Return",  f"{port_return:.1f}%",
         "Annualised", "pos"),
        (c3, "Expected Risk",    f"{port_risk:.1f}%",
         "Std Deviation", ""),
        (c4, "Sharpe Ratio",     f"{sharpe:.2f}",
         "Risk-adjusted", "pos" if sharpe > 1 else ""),
        (c5, "No. of Stocks",    f"{n_stocks}",
         f"Top: {top_stock}", ""),
    ]
    for col, label, val, sub, cls in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub {cls}">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── allocation table + donut ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Stock Allocations</div>',
                    unsafe_allow_html=True)

        significant = profile_data[profile_data["weight"] >= 1].copy()
        significant["amount"] = (significant["weight"] / 100 * investment).round(2)
        significant = significant.sort_values("weight", ascending=False)

        significant_display = significant[["stock", "weight", "amount"]].copy()
        significant_display.columns = ["Stock", "Weight (%)", "Amount (USD)"]
        significant_display["Weight (%)"] = significant_display["Weight (%)"].apply(
            lambda x: f"{x:.1f}%"
        )
        significant_display["Amount (USD)"] = significant_display["Amount (USD)"].apply(
            lambda x: f"${x:,.0f}"
        )
        st.dataframe(significant_display, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">Allocation Breakdown</div>',
                    unsafe_allow_html=True)

        significant2 = profile_data[profile_data["weight"] >= 2].copy()
        others_wt    = profile_data[profile_data["weight"] < 2]["weight"].sum()
        if others_wt > 0:
            others_row = pd.DataFrame([{"stock": "Others", "weight": round(others_wt, 2)}])
            significant2 = pd.concat([significant2, others_row], ignore_index=True)

        COLORS = ["#1D9E75","#185FA5","#D85A30","#BA7517","#534AB7",
                  "#0F6E56","#0C447C","#993C1D","#854F0B","#3C3489"]

        fig = go.Figure(go.Pie(
            labels=significant2["stock"],
            values=significant2["weight"],
            hole=0.55,
            marker=dict(colors=COLORS, line=dict(color="#0e1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── market breakdown ──
    st.markdown('<div class="section-title">Market Exposure</div>',
                unsafe_allow_html=True)

    merged2 = profile_data.merge(
        stats_df[["name", "market", "sector"]],
        left_on="stock", right_on="name", how="left"
    )
    market_exp = merged2.groupby("market")["weight"].sum().reset_index()
    sector_exp = merged2.groupby("sector")["weight"].sum().reset_index()

    mc1, mc2 = st.columns(2)
    for col, df, label, color in [
        (mc1, market_exp, "market", "#185FA5"),
        (mc2, sector_exp, "sector", "#1D9E75"),
    ]:
        fig = go.Figure(go.Bar(
            x=df[label],
            y=df["weight"],
            marker_color=color,
            text=df["weight"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            height=280,
            margin=dict(t=20, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            yaxis=dict(showgrid=False, visible=False),
            xaxis=dict(showgrid=False),
        )
        col.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2 — OPTIMIZER
# ─────────────────────────────────────────────
def tab_optimizer(returns_pivot, stats_df, max_weight, risk_free):
    st.markdown("### 🧠 Portfolio Optimizer")

    mean_returns = returns_pivot.mean()
    cov_matrix   = returns_pivot.cov()
    tickers      = list(returns_pivot.columns)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Max Sharpe Portfolio</div>',
                    unsafe_allow_html=True)
        with st.spinner("Optimizing..."):
            ms = maximize_sharpe(mean_returns, cov_matrix,
                                  risk_free_rate=risk_free,
                                  max_weight=max_weight / 100)
            if ms.success:
                ret, risk, sharpe = portfolio_metrics(
                    ms.x, mean_returns, cov_matrix, risk_free
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Return",  f"{ret*100:.1f}%")
                m2.metric("Risk",    f"{risk*100:.1f}%")
                m3.metric("Sharpe",  f"{sharpe:.2f}")

                alloc = pd.DataFrame({
                    "Stock":  tickers,
                    "Weight": np.round(ms.x * 100, 2)
                })
                alloc = alloc[alloc["Weight"] >= 1].sort_values(
                    "Weight", ascending=False
                )
                fig = go.Figure(go.Bar(
                    x=alloc["Stock"],
                    y=alloc["Weight"],
                    marker_color="#1D9E75",
                    text=alloc["Weight"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    yaxis=dict(showgrid=False, visible=False),
                    xaxis=dict(showgrid=False, tickangle=-30),
                    title="Weight per Stock (%)",
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Min Variance Portfolio</div>',
                    unsafe_allow_html=True)
        with st.spinner("Optimizing..."):
            mv = minimize_variance(mean_returns, cov_matrix,
                                    max_weight=max_weight / 100)
            if mv.success:
                ret, risk, sharpe = portfolio_metrics(
                    mv.x, mean_returns, cov_matrix, risk_free
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Return",  f"{ret*100:.1f}%")
                m2.metric("Risk",    f"{risk*100:.1f}%")
                m3.metric("Sharpe",  f"{sharpe:.2f}")

                alloc = pd.DataFrame({
                    "Stock":  tickers,
                    "Weight": np.round(mv.x * 100, 2)
                })
                alloc = alloc[alloc["Weight"] >= 1].sort_values(
                    "Weight", ascending=False
                )
                fig = go.Figure(go.Bar(
                    x=alloc["Stock"],
                    y=alloc["Weight"],
                    marker_color="#185FA5",
                    text=alloc["Weight"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    yaxis=dict(showgrid=False, visible=False),
                    xaxis=dict(showgrid=False, tickangle=-30),
                    title="Weight per Stock (%)",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── efficient frontier ──
    st.markdown('<div class="section-title">Efficient Frontier</div>',
                unsafe_allow_html=True)

    frontier_df = pd.read_csv("data/processed/efficient_frontier.csv")
    max_s_idx   = frontier_df["sharpe"].idxmax()
    min_v_idx   = frontier_df["risk"].idxmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["risk"]   * 100,
        y=frontier_df["return"] * 100,
        mode="markers",
        marker=dict(
            color=frontier_df["sharpe"],
            colorscale="Viridis",
            size=3, opacity=0.4,
            colorbar=dict(title="Sharpe"),
        ),
        name="Simulated Portfolios",
        hovertemplate="Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[frontier_df.loc[max_s_idx, "risk"]   * 100],
        y=[frontier_df.loc[max_s_idx, "return"] * 100],
        mode="markers", marker=dict(color="red", size=14, symbol="star"),
        name="Max Sharpe",
    ))
    fig.add_trace(go.Scatter(
        x=[frontier_df.loc[min_v_idx, "risk"]   * 100],
        y=[frontier_df.loc[min_v_idx, "return"] * 100],
        mode="markers", marker=dict(color="#1D9E75", size=14, symbol="diamond"),
        name="Min Variance",
    ))
    fig.add_trace(go.Scatter(
        x=stats_df["ann_risk"],
        y=stats_df["ann_return"],
        mode="markers+text",
        marker=dict(color="orange", size=8),
        text=stats_df["name"],
        textposition="top center",
        textfont=dict(size=8),
        name="Stocks",
    ))
    fig.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="Risk (%)", showgrid=True,
                   gridcolor="#2d3748"),
        yaxis=dict(title="Return (%)", showgrid=True,
                   gridcolor="#2d3748"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3 — CHARTS
# ─────────────────────────────────────────────
def tab_charts(stats_df, corr_matrix, returns_pivot, profiles):
    st.markdown("### 📈 Analytics Charts")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "Risk vs Return", "Correlation Matrix", "Cumulative Returns"
    ])

    MARKET_COLORS = {
        "India": "#1D9E75", "US": "#185FA5",
        "Europe": "#D85A30", "Asia": "#BA7517",
    }

    with chart_tab1:
        fig = go.Figure()
        for market in stats_df["market"].unique():
            df_m = stats_df[stats_df["market"] == market]
            fig.add_trace(go.Scatter(
                x=df_m["ann_risk"],
                y=df_m["ann_return"],
                mode="markers+text",
                marker=dict(
                    size=df_m["sharpe"].apply(lambda s: max(10, s * 18)),
                    color=MARKET_COLORS.get(market, "#888"),
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                text=df_m["name"],
                textposition="top center",
                textfont=dict(size=9),
                name=market,
                hovertemplate=(
                    "<b>%{text}</b><br>Return: %{y:.1f}%"
                    "<br>Risk: %{x:.1f}%<extra></extra>"
                ),
            ))
        fig.add_hline(y=5, line_dash="dash", line_color="#718096",
                      annotation_text="Risk-Free Rate 5%")
        fig.update_layout(
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="Risk (%)", gridcolor="#2d3748"),
            yaxis=dict(title="Return (%)", gridcolor="#2d3748"),
            legend_title="Market",
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_tab2:
        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=8),
            hovertemplate="%{y} vs %{x}<br>Corr: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_tab3:
        PROFILE_COLORS = {
            "Conservative": "#1D9E75",
            "Moderate":     "#185FA5",
            "Aggressive":   "#D85A30",
        }
        fig = go.Figure()
        for profile, alloc_df in profiles.items():
            stocks    = alloc_df["stock"].tolist()
            available = [s for s in stocks if s in returns_pivot.columns]
            if not available:
                continue
            weights = alloc_df.set_index("stock").loc[available, "weight"] / 100
            weights = weights / weights.sum()
            port_ret = returns_pivot[available].fillna(0).dot(weights)
            cum_ret  = (1 + port_ret).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=returns_pivot.index,
                y=cum_ret * 100,
                mode="lines",
                name=profile,
                line=dict(color=PROFILE_COLORS[profile], width=2),
            ))

        eq_ret = returns_pivot.fillna(0).mean(axis=1)
        eq_cum = (1 + eq_ret).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=returns_pivot.index,
            y=eq_cum * 100,
            mode="lines",
            name="Equal Weight Benchmark",
            line=dict(color="#718096", width=1.5, dash="dash"),
        ))
        fig.update_layout(
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="Date", gridcolor="#2d3748"),
            yaxis=dict(title="Cumulative Return (%)", gridcolor="#2d3748"),
            hovermode="x unified",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4 — LIVE PRICES
# ─────────────────────────────────────────────
def tab_live_prices(markets):
    st.markdown("### 🌐 Live Market Prices")
    st.caption("Prices are 15-min delayed · Refreshes every 5 minutes")

    with st.spinner("Fetching live prices..."):
        live_df = fetch_live_prices_cached()

    if live_df.empty:
        st.error("Could not fetch live prices. Check your internet connection.")
        return

    # filter by selected markets
    live_df = live_df[live_df["market"].isin(markets)]

    # ── market summary cards ──
    cols = st.columns(len(markets))
    for i, market in enumerate(markets):
        mdf     = live_df[live_df["market"] == market]
        gainers = (mdf["change_pct"] > 0).sum()
        losers  = (mdf["change_pct"] < 0).sum()
        cols[i].markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{market}</div>
            <div class="metric-value">{len(mdf)} stocks</div>
            <div class="metric-sub">
                <span class="pos">▲ {gainers}</span> &nbsp;
                <span class="neg">▼ {losers}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── live price table ──
    display_df = live_df.copy()
    display_df["change_pct"] = display_df["change_pct"].apply(
        lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
    )
    display_df["price"] = display_df.apply(
        lambda r: f"{r['price']:,.2f} {r['currency']}", axis=1
    )
    display_df["change"] = display_df["change"].apply(
        lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}"
    )

    st.dataframe(
        display_df[["name", "symbol", "market", "sector",
                    "price", "change", "change_pct"]].rename(columns={
            "name":       "Stock",
            "symbol":     "Symbol",
            "market":     "Market",
            "sector":     "Sector",
            "price":      "Price",
            "change":     "Change",
            "change_pct": "Change %",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── gainers vs losers bar chart ──
    st.markdown('<div class="section-title">Gainers & Losers</div>',
                unsafe_allow_html=True)

    live_sorted = live_df.sort_values("change_pct", ascending=True)
    colors      = ["#D85A30" if x < 0 else "#1D9E75"
                   for x in live_sorted["change_pct"]]

    fig = go.Figure(go.Bar(
        x=live_sorted["change_pct"],
        y=live_sorted["name"],
        orientation="h",
        marker_color=colors,
        text=live_sorted["change_pct"].apply(
            lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
        ),
        textposition="outside",
        hovertemplate="%{y}<br>Change: %{x:.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#718096", line_width=1)
    fig.update_layout(
        height=max(400, len(live_df) * 22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="Change (%)", gridcolor="#2d3748",
                   zeroline=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=100, r=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Refresh Prices"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # sidebar
    investment, risk_profile, max_weight, risk_free, markets = render_sidebar()

    # header
    st.markdown("""
    <h1 style='font-size:28px; font-weight:700; margin-bottom:0;'>
        📈 Global Portfolio Manager
    </h1>
    <p style='color:#718096; margin-top:4px;'>
        Multi-market · Markowitz Optimizer · Live Prices
    </p>
    <hr style='border-color:#2d3748; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    # load data
    try:
        (stats_df, frontier_df, corr_matrix,
         returns_pivot, optimal_df, profiles) = load_all_data()
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e}. Run ETL and Optimizer first.")
        st.stop()

    # filter by selected markets
    stats_df = stats_df[stats_df["market"].isin(markets)]

    # tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🧠 Optimizer",
        "📈 Charts",
        "🌐 Live Prices",
    ])

    with tab1:
        tab_overview(stats_df, profiles, risk_profile, investment)

    with tab2:
        filtered_pivot = returns_pivot[
            [c for c in returns_pivot.columns
             if c in stats_df["name"].values]
        ]
        tab_optimizer(filtered_pivot, stats_df, max_weight, risk_free)

    with tab3:
        filtered_corr = corr_matrix.loc[
            corr_matrix.index.isin(stats_df["name"]),
            corr_matrix.columns.isin(stats_df["name"])
        ]
        tab_charts(stats_df, filtered_corr, returns_pivot, profiles)

    with tab4:
        tab_live_prices(markets)


if __name__ == "__main__":
    main()