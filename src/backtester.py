
import pandas as pd
import numpy as np
import config

class StrategyBacktest:
    """
    Engine for simulating trading strategy performance methods.
    """
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def run_strategy(self, df, predictions):
        """
        Simulates the trading strategy:
        Buy (Hold) if Prediction is 1 (UP), Cash if Prediction is 0 (DOWN).
        
        Assumes 'predictions' aligns with the test set portion of 'df'.
        """
        # We need to align the dataframe with the predictions equal to the test size
        # The predictions correspond to the *next day's* movement.
        # But in our feature engine, 'Target' is shifted -1. 
        # So prediction at index i is for price movement from i to i+1.
        
        # Slice df to match test set size
        test_size = len(predictions)
        strategy_df = df.iloc[-test_size:].copy()
        
        strategy_df['Predicted_Signal'] = predictions
        
        # Strategy Returns: If Signal is 1, we get Daily_Return. If 0, we get 0 (Risk Free Rate assumed 0 for simplicity here or handled separately)
        # Note: The prediction at time t tells us to hold for t+1. 
        # The 'Daily_Return' at time t is (Close_t - Close_t-1) / Close_t-1.
        # So we need to shift the signal to align with the return it generates.
        # Signal calculated at t-1 (based on data up to t-1) decides position for t.
        
        strategy_df['Strategy_Return'] = strategy_df['Predicted_Signal'].shift(1) * strategy_df['Daily_Return']
        
        # Cumulative Returns
        strategy_df['Cumulative_Market_Return'] = (1 + strategy_df['Daily_Return']).cumprod()
        strategy_df['Cumulative_Strategy_Return'] = (1 + strategy_df['Strategy_Return'].fillna(0)).cumprod()
        
        # Portfolio Values
        strategy_df['Market_Value'] = self.initial_capital * strategy_df['Cumulative_Market_Return']
        strategy_df['Strategy_Value'] = self.initial_capital * strategy_df['Cumulative_Strategy_Return']
        
        return strategy_df

    def calculate_metrics(self, strategy_df):
        """
        Calculates performance metrics: Total Return, Sharpe Ratio, Max Drawdown.
        """
        # Total Return
        total_return = (strategy_df['Strategy_Value'].iloc[-1] / self.initial_capital) - 1
        
        # Sharpe Ratio (assuming roughly 252 trading days)
        # Daily excess return
        excess_returns = strategy_df['Strategy_Return'].fillna(0) - (config.RISK_FREE_RATE / 252)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0
        
        # Max Drawdown
        cum_returns = strategy_df['Cumulative_Strategy_Return']
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'Total_Return': total_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }
