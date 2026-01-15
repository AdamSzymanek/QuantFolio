import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import config
import streamlit as st


@st.cache_resource(show_spinner="Training AI Model (XGBoost)... please wait")
def _train_xgboost_cached(df):
    """
    Internal function that performs the heavy lifting of training.
    Cached by Streamlit.
    """
    # 1. Define Features
    exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Target']
    
    # 2. Split Data (Shuffle=False for Time Series is crucial!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. Train Model
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train, y_train)
    
    # 4. Generate Predictions & Metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    # 5. Calculate Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Return everything needed by the class
    return model, feature_importance, accuracy, report, X_test, y_test, predictions, feature_cols



class TrendPredictor:
    """
    Predictive model for market trend analysis.
    Acts as a wrapper around the cached training function.
    """
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.feature_cols = None

    def train(self, df):
        """
        Trains the XGBoost Classifier (uses Cache).
        """
        # Call the cached function instead of training manually
        results = _train_xgboost_cached(df)
        
        # Unpack results into class attributes
        self.model = results[0]
        self.feature_importance = results[1]
        accuracy = results[2]
        report = results[3]
        X_test = results[4]
        y_test = results[5]
        predictions = results[6]
        self.feature_cols = results[7] # Save feature names for prediction safety
        
        return accuracy, report, X_test, y_test, predictions

    def predict(self, df):
        """
        Makes predictions on new data.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Ensure we use exactly the same features as in training
        # If feature_cols is not set yet, try to deduce 
        if self.feature_cols is None:
             exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
             self.feature_cols = [c for c in df.columns if c not in exclude_cols]
             
        return self.model.predict(df[self.feature_cols])


class RiskSimulator:
    """
    Monte Carlo Simulation for Risk Analysis.
    """
    def __init__(self):
        pass


    @st.cache_data(show_spinner="Running Monte Carlo Simulation...") 
    def run_simulation(_self, current_price, log_returns, days=30, iterations=1000):
        """
        Runs Monte Carlo simulation using Geometric Brownian Motion.
        """
        mu = log_returns.mean()
        var = log_returns.var()
        sigma = log_returns.std()
        
        drift = mu - (0.5 * var)
        daily_volatility = sigma
        
   
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
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return var_percentile
