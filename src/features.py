
import pandas as pd
import numpy as np

class FinancialFeatures:
    
    @staticmethod
    def add_technical_indicators(df):
        df = df.copy()
        
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        
        df['Daily_Return'] = df['close'].pct_change()
        
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def add_ml_features(df):
        df = df.copy()
        
        df['Return_Lag_1'] = df['Daily_Return'].shift(1)
        df['Return_Lag_2'] = df['Daily_Return'].shift(2)
        df['Return_Lag_5'] = df['Daily_Return'].shift(5)
        
        df['Volume_Change'] = df['volume'].pct_change()
        df['Volume_Change_Lag_1'] = df['Volume_Change'].shift(1)
        
        df['Momentum_1d'] = df['close'] / df['close'].shift(1) - 1
        

        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        

        df = df.dropna()
        
        return df
