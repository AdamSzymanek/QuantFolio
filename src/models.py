import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import config
import streamlit as st

class TrendPredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.feature_cols = None

    def train(self, df):

        if 'trained_models' not in st.session_state:
            st.session_state['trained_models'] = {}

        if 'Name' in df.columns:
            ticker_name = str(df['Name'].iloc[0])
        else:
            ticker_name = "UNKNOWN_TICKER"

        if ticker_name in st.session_state['trained_models']:
            cached_data = st.session_state['trained_models'][ticker_name]
            self.model = cached_data['model']
            self.feature_importance = cached_data['feature_importance']
            self.feature_cols = cached_data['feature_cols']
        else:
    
            params = config.XGB_PARAMS.copy()
            params['n_jobs'] = 1

            # Use full dataset for training, but respect time series split
            exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            X = df[feature_cols]
            y = df['Target']
            
            # Strict Time Series Split: Train on first 80%, Test on last 20%
            split_idx = int(len(df) * 0.8)
            
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            
            # Test set for evaluation (Unseen future data)
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
     
            st.session_state['trained_models'][ticker_name] = {
                'model': model,
                'feature_importance': feature_importance,
                'feature_cols': feature_cols,
                'test_data': (X_test, y_test) # Cache test data to ensure consistency
            }
            
            self.model = model
            self.feature_importance = feature_importance
            self.feature_cols = feature_cols

        # Retrieve test data from cache or split again if needed (should match above)
        if 'test_data' in st.session_state['trained_models'][ticker_name]:
             X_test, y_test = st.session_state['trained_models'][ticker_name]['test_data']
        else:
             # Fallback if cached model doesn't have test_data (legacy cache support)
             # Re-create split
             exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
             feature_cols = [c for c in df.columns if c not in exclude_cols]
             split_idx = int(len(df) * 0.8)
             X_test = df[feature_cols].iloc[split_idx:]
             y_test = df['Target'].iloc[split_idx:]

        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return accuracy, report, X_test, y_test, predictions

    def predict(self, df):
        if self.model is None:
             raise ValueError("Model not trained! Call train() first.")
             
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
        
        daily_returns[0] = 1.0 
        
        price_paths = current_price * np.cumprod(daily_returns, axis=0)
        
        return price_paths

    def calculate_var(self, final_prices, initial_price, confidence_level=0.95):
        returns = (final_prices - initial_price) / initial_price
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return var_percentile
