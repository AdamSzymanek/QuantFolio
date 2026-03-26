# Configuration for QuantFolio

import os

# Data Paths
# Using relative paths for Streamlit Community Cloud compatibility
DATA_PATH = "data/all_stocks_5yr.csv"

# Risk Parameters
RISK_FREE_RATE = 0.02  # Approximate 10-year Treasury yield
CONFIDENCE_LEVEL = 0.95

# Simulation Parameters
SIMULATION_ITERATIONS = 200
FORECAST_DAYS = 30

# Model Parameters
XGB_PARAMS = {
    'n_estimators': 10,
    'learning_rate': 0.1,
    'max_depth': 2,
    'random_state': 42,
    'eval_metric': 'logloss',
    'n_jobs': 1
}

# Technical Indicators
SMA_WINDOW_SHORT = 50
SMA_WINDOW_LONG = 200
RSI_WINDOW = 14
VOLATILITY_WINDOW = 20

# Trading Strategy
INITIAL_CAPITAL = 10000
