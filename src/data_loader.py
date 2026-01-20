import pandas as pd
import os
import streamlit as st
from datetime import datetime

@st.cache_data(show_spinner="Loading S&P 500 Data...")
def _load_csv_cached(file_path):
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
    return df

class MarketData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = _load_csv_cached(self.file_path)

        if self.data is None:
             raise FileNotFoundError(f"Data file not found at: {self.file_path}")

    def get_stock(self, ticker):
        if self.data is None:
            self.load_data()
            
        stock_df = self.data[self.data['Name'] == ticker].copy()
        
        if stock_df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
            
        return stock_df.sort_values('date')

    def get_all_tickers(self):
        if self.data is None:
            self.load_data()
        return sorted(self.data['Name'].unique().tolist())
