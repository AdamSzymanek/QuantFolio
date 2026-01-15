# QuantFolio: Algorithmic Trading & Risk Dashboard

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Author](https://img.shields.io/badge/Author-Adam_Szymanek-blueviolet)

## 📋 Project Overview
**QuantFolio** is a quantitative analysis platform engineered to simulate institutional-grade risk assessment workflows. Unlike standard stock trackers, this tool implements rigorous mathematical models to evaluate market conditions.

The project integrates **Machine Learning (XGBoost)** for directional trend prediction and **Stochastic Calculus (Monte Carlo Simulations)** to quantify portfolio risk (VaR), providing a transparent look into the math behind the metrics.

---

## ☁️ Live Demo
The application is deployed via Streamlit Community Cloud and connects to a static dataset for demonstration purposes.

### [👉 Launch QuantFolio Dashboard](https://share.streamlit.io/TWOJ_LINK_DO_APKI)
*(Note: The container may take 10-30 seconds to wake up)*

---

## 🏗️ Technical Architecture

I designed the system using a modular pipeline approach to separate data processing from inference and visualization.

### 1. Data Pipeline (`src/data_loader.py`)
-   **Source**: S&P 500 Historical Data (`data/all_stocks_5yr.csv`).
-   **ETL**: Automated extraction, type casting, and temporal sorting. The pipeline ensures strictly chronological data splitting to prevent **look-ahead bias** during model training.

### 2. Feature Engineering (`src/features.py`)
Transformation of raw OHLCV data into stationary features for supervised learning:
-   **Momentum**: RSI (14-day) and Rate of Change (ROC).
-   **Trend**: SMA Crossovers (Golden/Death Cross logic).
-   **Volatility**: Rolling annualized standard deviation (20-day window).
-   **Lagged Returns**: Time-shifted features ($R_{t-1}, R_{t-5}$) to capture serial autocorrelation.

### 3. Predictive Engine (`src/models.py`)
-   **Model**: **XGBoost Classifier** (Gradient Boosted Decision Trees).
-   **Target**: Binary classification of next-day price direction.
-   **Optimization**: The model provides a **Feature Importance** map, explaining which technical indicators (e.g., Volume vs. RSI) drive the prediction logic.

### 4. Risk Modeling (`src/models.py`)
-   **Method**: **Geometric Brownian Motion (GBM)** via Monte Carlo Simulation.
-   **Formula**: $d S_t = \mu S_t d t + \sigma S_t d W_t$
-   **Execution**: Runs 1,000 parallel simulation paths over a 30-day horizon.
-   **Metric**: **Value at Risk (VaR 95%)**, calculated dynamically from the terminal distribution of simulated paths.

---

## 🛠 Tech Stack

| Component | Technology | Use Case |
| :--- | :--- | :--- |
| **Core Logic** | `Python 3.13` | Backend orchestration and mathematical modeling. |
| **Data Ops** | `Pandas` / `NumPy` | Vectorized calculations and time-series manipulation. |
| **ML** | `XGBoost` | Non-linear classification and pattern recognition. |
| **Viz**| `Plotly` | Interactive, WebGL-accelerated financial charting. |
| **UI** | `Streamlit` | Frontend framework for rapid data prototyping. |

---

## 🚀 Local Installation

To run the engine locally for development or testing:

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/AdamSzymanek/QuantFolio-Trading-Dashboard.git](https://github.com/AdamSzymanek/QuantFolio-Trading-Dashboard.git)
    cd QuantFolio-Trading-Dashboard
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run app.py
    ```

---

## 👨‍💻 Author & Contact

**Adam Szymanek**
*Computer Science Student @ AGH UST*

This project was built to demonstrate proficiency in financial modeling and full-stack data science.

[<img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin">](www.linkedin.com/in/adamszymanek)
[<img src="https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github">](https://github.com/AdamSzymanek)
