# 📈 Global Portfolio Manager

A production-grade portfolio management system built with Python,
PostgreSQL, and Streamlit — covering the full data engineering pipeline
from raw market data to an interactive web dashboard.

---

## 🏗️ Architecture
Data Collection → PostgreSQL → ETL Pipeline → Optimizer → Streamlit App
↑                                                          ↓
Yahoo Finance API                                        Interactive Dashboard
(26 Global Stocks)                                      (Live on Streamlit Cloud)
---

## 🌍 Markets Covered
- 🇮🇳 India (NIFTY 50 — RELIANCE, TCS, HDFC, INFY...)
- 🇺🇸 US (AAPL, MSFT, GOOGL, NVDA, TSLA...)
- 🇪🇺 Europe (NESTLE, LVMH, ASML, SAP...)
- 🌏 Asia (SAMSUNG, ALIBABA, SONY, TSMC...)

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data Collection | Python, yfinance, Yahoo Finance API |
| Storage | PostgreSQL, psycopg2 |
| Processing | Pandas, NumPy |
| Modeling | SciPy, Markowitz Optimization |
| Visualization | Plotly |
| Frontend | Streamlit |
| Automation | Windows Task Scheduler |

---

## 📊 Features

- ✅ Live stock prices (15-min delayed, free)
- ✅ Historical OHLCV data (2020–2024)
- ✅ Currency normalization (all → USD)
- ✅ Daily returns & risk calculation
- ✅ 26x26 correlation matrix
- ✅ Markowitz Mean-Variance Optimizer
- ✅ Max Sharpe Ratio portfolio
- ✅ Minimum Variance portfolio
- ✅ Efficient Frontier (10,000 simulations)
- ✅ Conservative / Moderate / Aggressive profiles
- ✅ Interactive Streamlit dashboard
- ✅ Automated daily pipeline (Mon–Fri 6 AM)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/portfolio_manager.git
cd portfolio_manager
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up PostgreSQL
- Create database: `portfolio_manager_db`
- Update credentials in `src/database.py`

### 4. Run the pipeline
```bash
# Collect data
python src/collect.py

# Load into database
python src/database.py

# Run ETL
python src/etl.py

# Run optimizer
python src/optimizer.py

# Generate charts
python src/visualization.py
```

### 5. Launch the app
```bash
streamlit run src/app.py
```

---

## 📁 Project Structure
portfolio_manager/
├── config.py                 # Stock universe & settings
├── requirements.txt
├── README.md
├── src/
│   ├── collect.py            # Data collection
│   ├── database.py           # PostgreSQL setup & loading
│   ├── etl.py                # ETL pipeline
│   ├── optimizer.py          # Markowitz optimizer
│   ├── visualization.py      # Plotly charts
│   ├── app.py                # Streamlit dashboard
│   └── scheduler.py          # Automation scheduler
├── data/
│   ├── raw/                  # Raw CSV data
│   └── processed/            # Analysis outputs
└── logs/
└── pipeline.log          # Pipeline run logs
---

## 🧠 Financial Models

### Markowitz Mean-Variance Optimization
- Maximizes Sharpe Ratio subject to weight constraints
- Minimum Variance portfolio for risk-averse investors
- Efficient Frontier via Monte Carlo simulation (10,000 portfolios)

### Risk Profiles
| Profile | Max per Stock | Target |
|---------|-------------|--------|
| Conservative | 10% | Stability |
| Moderate | 20% | Balanced |
| Aggressive | 35% | Growth |

---

## 👨‍💻 Author
**Jaiman**
Built as a full-stack data engineering + finance project.

## 🧠 Sentiment Analysis (ML)

- **VADER** — Rule-based NLP sentiment scoring
- **FinBERT** — Finance-specific BERT transformer model
- **Ensemble** — 70% FinBERT + 30% VADER weighted scoring
- **NewsAPI** — Live news fetched for all 26 global stocks
- Sentiment scored from **-1 (Bearish) to +1 (Bullish)**
- Market and sector level sentiment aggregation
- Color-coded news headlines in dashboard