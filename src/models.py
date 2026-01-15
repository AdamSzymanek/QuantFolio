
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import config
import streamlit as st

class TrendPredictor:
    """
    Predictive model for market trend analysis.
    """
    def __init__(self):
        self.model = XGBClassifier(**config.XGB_PARAMS)
        self.feature_importance = None

    def train(self, df):
        """
        Trains the XGBoost Classifier.
        Expects a DataFrame with features and a 'Target' column.
        """
        # Define Features (exclude date, target, and non-numeric columns if any)
        # We assume 'Target' and 'date' are in the dataframe, and 'Name' if present
        exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return'] 
        # Note: We keep computed features like SMA, RSI, Lags, etc.
        # Let's filter strictly for features we created in FinancialFeatures
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols]
        y = df['Target']
        
        # Split Data (80% Train, 20% Test) - Shuffle=False for Time Series!
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        self.feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return accuracy, report, X_test, y_test, predictions

    def predict(self, df):
        """
        Makes predictions on new data.
        """
        exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return self.model.predict(df[feature_cols])


class RiskSimulator:
    """
    Monte Carlo Simulation for Risk Analysis.
    """
    def __init__(self):
        pass

    def run_simulation(self, current_price, log_returns, days=30, iterations=1000):
        """
        Runs Monte Carlo simulation using Geometric Brownian Motion.
        
        S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
        """
        mu = log_returns.mean()
        var = log_returns.var()
        sigma = log_returns.std()
        
        # Drift and Diffusion
        drift = mu - (0.5 * var)
        
        # Random component: Z score
        daily_volatility = sigma
        
        # Simulation
        # multiple paths: iterations x days
        daily_returns = np.exp(drift + daily_volatility * np.random.normal(0, 1, (days, iterations)))
        
        price_paths = np.zeros((days, iterations))
        price_paths[0] = current_price
        
        for t in range(1, days):
            price_paths[t] = price_paths[t-1] * daily_returns[t]
            
        return price_paths

    def calculate_var(self, final_prices, initial_price, confidence_level=0.95):
        """
        Calculates Value at Risk (VaR).
        """
        returns = (final_prices - initial_price) / initial_price
        # VaR is the quantile at (1 - confidence_level)
        # e.g., for 95% confidence, we look at the 5th percentile of worst outcomes
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        
        # VaR in Dollar amount for a hypothetical portfolio (e.g., $10k investment)
        # But here we return the percentage VaR
        return var_percentile
