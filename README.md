
# QuantFolio: Algorithmic Trading & Risk Dashboard

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Focus](https://img.shields.io/badge/Focus-Quantitative_Finance-orange)

## 📖 Executive Summary
**QuantFolio** is an end-to-end quantitative finance platform designed to simulate the workflow of a modern hedge fund analyst. It bridges the gap between raw data and actionable intelligence by integrating **Machine Learning (XGBoost)** for trend classification and **Stochastic Calculus (Geometric Brownian Motion)** for risk modeling.

Unlike black-box AI tools, QuantFolio emphasizes **interpretability**—allowing users to see exactly *why* a model predicts a move and *what* the mathematical risks are.

---

## ☁️ Live Demonstration
The application is deployed on a serverless architecture using Streamlit Community Cloud.
### [👉 Launch QuantFolio Dashboard](https://share.streamlit.io/)
*(Please allow 10-30 seconds for the container to wake up from sleep mode)*

---

## 🏗️ Technical Architecture
The system follows a modular pipeline architecture, separating concerns between data ingestion, computation, and presentation.

### 1. Data Ingestion Layer (`src/data_loader.py`)
-   **Source**: S&P 500 Historical Dataset (`data/all_stocks_5yr.csv`).
-   **ETL Process**: 
    -   *Extraction*: Loads raw CSV data via Pandas.
    -   *Transformation*: Casts dates, sorts temporally, and filters by ticker.
    -   *Validation*: Ensures OHLCV (Open, High, Low, Close, Volume) consistency.

### 2. Feature Engineering Layer (`src/features.py`)
Raw prices are transformed into stationary features suitable for supervised learning:
-   **Momentum Indicators**: Relative Strength Index (RSI 14) to identify overbought/oversold conditions.
-   **Trend Indicators**: SMA 50/200 crossover logic (Golden Cross detection).
-   **Volatility Metrics**: Annualized Log-Return Standard Deviation (Rolling 20-day).
-   **Lagged Features**: $R_{t-1}, R_{t-5}$ time-shifted returns to capture serial autocorrelation.

### 3. Machine Learning Engine (`src/models.py`)
-   **Algorithm**: **XGBoost Classifier** (Extreme Gradient Boosting).
-   **Objective**: Binary Classification ($Target = 1$ if $Price_{t+1} > Price_t$ else $0$).
-   **Validation**: Time-Series Cross-Validation (80% Train, 20% Test, strictly chronological to prevent look-ahead bias).
-   **Explainability**: Feature Importance extraction to quantify the weighted contribution of each signal.

### 4. Stochastic Risk Engine (`src/models.py`)
-   **Model**: **Geometric Brownian Motion (GBM)**.
-   **Equation**: $d S_t = \mu S_t d t + \sigma S_t d W_t$
    -   $\mu$: Drift (expected return)
    -   $\sigma$: Volatility (risk)
    -   $dW_t$: Wiener Process (Random Walk)
-   **Simulation**: 1,000 parallel Monte Carlo paths over a 30-day horizon.
-   **Metric**: **Value at Risk (VaR 95%)** calculated from the 5th percentile of the terminal distribution.

---

## � Tech Stack & Libraries
| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Core** | `Python 3.13` | Primary logic and orchestration. |
| **DataFrames** | `Pandas` / `NumPy` | Vectorized operations and time-series manipulation. |
| **ML/AI** | `XGBoost` | Gradient Boosting for non-linear pattern recognition. |
| **Evaluation** | `Scikit-Learn` | Metric reporting (Precision/Recall) and data splitting. |
| **Visualization**| `Plotly` | Interactive, WebGL-accelerated financial charts. |
| **Frontend** | `Streamlit` | Reactive web framework for data apps. |

---

## 🚀 Installation & Local Deployment

If you wish to run the engine locally or contribute:

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/QuantFolio.git
cd QuantFolio
```

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the Application**
```bash
streamlit run app.py
```
The dashboard will open automatically at `http://localhost:8501`.

---

## 🔮 Future Roadmap (Professional Scope)
-   [ ] **Sentiment Analysis**: Integrate NLP (BERT/Transformers) to parse financial news sentiment as a feature.
-   [ ] **Portfolio Optimization**: Implement Markowitz Mean-Variance Optimization for multi-asset portfolios.
-   [ ] **Live Data Feed**: Connect to Yahoo Finance (`yfinance`) or Alpaca API for real-time streaming data.

---

## 👨‍💻 Project Philosophy
I built QuantFolio to demonstrate that complex financial concepts doesn't have to be opaque. By combining rigorous backend math with a human-centric UI, we can make institutional-grade analysis accessible.

*Developed by a Data Scientist & Quantitative Analyst.*
