
import pandas as pd
import os
from datetime import datetime

class MarketData:
    """
    Class to handle loading and processing of stock market data.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Loads the CSV data, converts dates, and performs basic cleaning.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at: {self.file_path}")

        try:
            self.data = pd.read_csv(self.file_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            # Sort by date just in case
            self.data = self.data.sort_values(by='date')
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def get_stock(self, ticker):
        """
        Retrieves data for a specific stock ticker.
        """
        if self.data is None:
            self.load_data()
            
        stock_df = self.data[self.data['Name'] == ticker].copy()
        
        if stock_df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
            
        return stock_df.sort_values('date')

    def get_all_tickers(self):
        """
        Returns a list of all unique tickers in the dataset.
        """
        if self.data is None:
            self.load_data()
        return sorted(self.data['Name'].unique().tolist())
