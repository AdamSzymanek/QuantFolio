import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import config
import streamlit as st

# --- MAGIC HAPPENS HERE ---
# I changed the argument name to '_df'.
# The underscore (_) tells Streamlit: "Do not waste time hashing this object".
# The cache relies NOW only on 'ticker_name'.

@st.cache_resource(show_spinner="Training AI Model (Fast)...", max_entries=100)
def _train_xgboost_cached(_df: pd.DataFrame, ticker_name: str):
    """
    Cached training function.
    Prefixing argument with '_' (_df) prevents Streamlit from hashing the DataFrame.
    Returns MINIMAL data to memory usage: just the model and feature names.
    Calculations are done on the fly.
    """
    # 1. Define Features
    exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
    feature_cols = [c for c in _df.columns if c not in exclude_cols]
    
    X = _df[feature_cols]
    y = _df['Target']
    
    # 2. Split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. Train
    # HARDCODED n_jobs=1 to prevent freezing on Free Tier
    # This overrides global config just in case.
    params = config.XGB_PARAMS.copy()
    params['n_jobs'] = 1
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # ONLY return the model and metadata. Do NOT return large arrays (X_test, predictions)
    # to avoid RAM eviction.
    return model, feature_importance, feature_cols


class TrendPredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.feature_cols = None

    def train(self, df):
        # Explicitly extract the ticker name for the cache key
        if 'Name' in df.columns:
            ticker_name = str(df['Name'].iloc[0])
        else:
            ticker_name = "UNKNOWN_TICKER"

        # Call the cached function to get the MODEL
        results = _train_xgboost_cached(_df=df, ticker_name=ticker_name)
        
        self.model = results[0]
        self.feature_importance = results[1]
        self.feature_cols = results[2]
        
        # --- RE-RUN PREDICTION ON THE FLY ---
        # This is fast (ms) and saves RAM in the cache.
        exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
        X = df[self.feature_cols]
        y = df['Target']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return accuracy, report, X_test, y_test, predictions

    def predict(self, df):
        if self.model is None:
             raise ValueError("Model not trained! Call train() first.")
             
        # Ensure we only use the features the model was trained on
        if self.feature_cols is None:
             raise ValueError("Model trained but feature columns missing.")

        return self.model.predict(df[self.feature_cols])


class RiskSimulator:
    def __init__(self):
        pass

    @st.cache_data(show_spinner=False)
    def run_simulation(_self, current_price, log_returns, days=30, iterations=1000):
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
        returns = (final_prices - initial_price) / initial_price
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return var_percentile
