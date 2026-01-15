
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

        test_size = len(predictions)
        strategy_df = df.iloc[-test_size:].copy()
        
        strategy_df['Predicted_Signal'] = predictions
        

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
        
        # Sharpe Ratio
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
