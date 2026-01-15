import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import config
import streamlit as st

class TrendPredictor:
    """
    Predictive model wrapper with HARD SESSION STATE CACHING.
    """
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.feature_cols = None

    def train(self, df):
        """
        Trains the XGBoost Classifier but checks Session State first.
        If model for this specific ticker exists in RAM, loads it instantly.
        """
        # 1. Sprawdzamy, jakiej spółki dotyczy ten DataFrame
        # Zakładamy, że w df jest kolumna 'Name' (z Twojego data_loadera)
        if 'Name' in df.columns:
            ticker = df['Name'].iloc[0]
        else:
            ticker = "UNKNOWN"

        # 2. Tworzymy klucz do pamięci RAM
        cache_key = f"trained_model_{ticker}"

        # 3. SPRAWDZAMY CZY JUŻ TO MAMY (Instant Load)
        if cache_key in st.session_state:
            cached_data = st.session_state[cache_key]
            self.model = cached_data['model']
            self.feature_importance = cached_data['feat_imp']
            self.feature_cols = cached_data['feat_cols']
            return cached_data['acc'], cached_data['report'], cached_data['X_test'], cached_data['y_test'], cached_data['preds']

        # ==========================================================
        # JEŚLI NIE MAMY W PAMIĘCI -> DOPIERO WTEDY TRENUJEMY
        # ==========================================================
        
        # Define Features
        exclude_cols = ['date', 'Name', 'Target', 'close', 'open', 'high', 'low', 'volume', 'Daily_Return']
        # Bezpiecznik: bierzemy tylko kolumny numeryczne
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols]
        y = df['Target']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train
        model = XGBClassifier(**config.XGB_PARAMS)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Zapisujemy wyniki do obiektu klasy
        self.model = model
        self.feature_importance = feature_importance
        self.feature_cols = feature_cols

        # 4. ZAPISUJEMY DO SESSION STATE (RAM) NA PRZYSZŁOŚĆ
        st.session_state[cache_key] = {
            'model': model,
            'feat_imp': feature_importance,
            'feat_cols': feature_cols,
            'acc': accuracy,
            'report': report,
            'X_test': X_test,
            'y_test': y_test,
            'preds': predictions
        }
        
        return accuracy, report, X_test, y_test, predictions

    def predict(self, df):
        if self.model is None:
            # Próba ratunku z session state, jeśli obiekt został stworzony na nowo
            if 'Name' in df.columns:
                ticker = df['Name'].iloc[0]
                cache_key = f"trained_model_{ticker}"
                if cache_key in st.session_state:
                    self.model = st.session_state[cache_key]['model']
                    self.feature_cols = st.session_state[cache_key]['feat_cols']
                else:
                    raise ValueError("Model not trained yet!")
            else:
                 raise ValueError("Model not trained yet!")

        # Ensure features match
        return self.model.predict(df[self.feature_cols])


class RiskSimulator:
    def __init__(self):
        pass

    # Tutaj zwykły cache jest OK, bo to czysta matematyka
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
