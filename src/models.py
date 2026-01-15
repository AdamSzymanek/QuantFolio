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

    def train(self, df):
        # 1. Initialize session state for models if not present
        if 'trained_models' not in st.session_state:
            st.session_state['trained_models'] = {}

        # 2. Extract ticker name
        if 'Name' in df.columns:
            ticker_name = str(df['Name'].iloc[0])
        else:
            ticker_name = "UNKNOWN_TICKER"

        # 3. Check Cache (st.session_state)
        # If model exists for this ticker, load it instantly.
        if ticker_name in st.session_state['trained_models']:
            cached_data = st.session_state['trained_models'][ticker_name]
            self.model = cached_data['model']
            self.feature_importance = cached_data['feature_importance']
            self.feature_cols = cached_data['feature_cols']
        else:
            # 4. Train New Model (only if not in cache)
            params = config.XGB_PARAMS.copy()
            params['n_jobs'] = 1 # Force single thread
            
            # --- OPTIMIZATION: Limit to recent history (last 1 year) ---
            # Training on full history is unnecessary for short-term prediction.
            # 252 rows (1 year) is fastest and sufficient for demo.
            train_df = df.tail(252).copy()
            
            exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
            feature_cols = [c for c in train_df.columns if c not in exclude_cols]
            
            X = train_df[feature_cols]
            y = train_df['Target']
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            # 5. Save to Cache
            st.session_state['trained_models'][ticker_name] = {
                'model': model,
                'feature_importance': feature_importance,
                'feature_cols': feature_cols
            }
            
            self.model = model
            self.feature_importance = feature_importance
            self.feature_cols = feature_cols

        # 6. Predict (Always runs fast)
        # Re-create X_test for current DataFrame
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
        
        # Vectorized path generation using cumprod (Cumulative Product)
        # This eliminates the Python loop completely.
        # price_paths = current_price * np.cumprod(daily_returns, axis=0) 
        
        # Careful: daily_returns[0] should not be applied to t=0, 
        # but to keep it simple and consistent with previous logic:
        # We start at current_price.
        
        price_paths = np.zeros((days, iterations))
        price_paths[0] = current_price
        
        # Calculate cumulative returns from t=1 to end
        # We need to explicitly set the first row of returns to 1 (neutral) so cumprod works from start
        daily_returns[0] = 1.0 
        
        price_paths = current_price * np.cumprod(daily_returns, axis=0)
        
        return price_paths

    def calculate_var(self, final_prices, initial_price, confidence_level=0.95):
        returns = (final_prices - initial_price) / initial_price
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return var_percentile
