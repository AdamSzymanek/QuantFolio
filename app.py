
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import config
from src.data_loader import MarketData
from src.features import FinancialFeatures
from src.models import TrendPredictor, RiskSimulator
from src.backtester import StrategyBacktest

st.set_page_config(
    page_title="QuantFolio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Custom CSS 
st.markdown("""
<style>
    /* Hide specific Streamlit menu items */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide the top right 'Deploy' and 'Running' status but keep sidebar toggle */
    [data-testid="stHeaderAction"] {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    
    /* Main Background & Font */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global Text Color */
    .stMarkdown, .stText, h1, h2, h3, p {
        color: #E0E0E0 !important;
    }

    /* Cards */
    .metric-card {
        background-color: #16161D; /* Slightly darker/sleeker */
        border: 1px solid #2D2D3A;
        border-radius: 8px; /* Less rounded, more professional */
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
        min-height: 160px; /* Enforce equal height */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4CAF50;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #888899;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sub {
        font-size: 14px;
        font-weight: 500;
    }
    .positive-val { color: #4CAF50; }
    .negative-val { color: #FF5252; }
    
    /* Sidebar Cleanup */
    .css-1d391kg {
        background-color: #0E1117;
        border-right: 1px solid #2D2D3A;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #2D2D3A;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        border-radius: 0px;
        color: #888899;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #4CAF50 !important;
        border-bottom-color: #4CAF50 !important;
    }
    
    /* Hide Header Anchors (Link Icons) - Aggressive Selector */
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a {
        display: none !important;
        pointer-events: none;
        cursor: default;
        visibility: hidden;
        width: 0px !important;
        height: 0px !important;
        opacity: 0 !important;
        color: transparent !important;
    }
    
    /* Target Streamlit's specific anchor link class if visible */
    a.anchor-link, [data-testid="stHeaderAnchor"], .st-emotion-cache-1629p8f a {
        display: none !important;
        visibility: hidden !important;
    }
    
    [data-testid="stHeaderAction"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

#Helpers
def render_metric_card(label, value, sub_value=None, help_text=None, is_negative=False):
    color_class = "negative-val" if is_negative else "positive-val"
    sub_html = f"<div class='metric-sub {color_class}'>{sub_value}</div>" if sub_value else ""
    tooltip = f"title='{help_text}'" if help_text else ""
    st.markdown(f"""
    <div class="metric-card" {tooltip}>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

#Sidebar
st.md = st.markdown

st.sidebar.markdown("### Configuration")

@st.cache_data
def load_market_data():
    return MarketData(config.DATA_PATH)

try:
    data_loader = load_market_data()
    tickers = data_loader.get_all_tickers()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

selected_ticker = st.sidebar.selectbox("Select Asset", tickers, index=tickers.index('AAPL') if 'AAPL' in tickers else 0)
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000, step=500)

st.sidebar.markdown("---")
with st.sidebar.expander("Usage Guide"):
    st.markdown("""
    **Workflow:**
    1. Select an asset to analyze.
    2. Review **Market Overview** for technicals.
    3. Check **Quant Signals** for model forecasts.
    4. Run **Risk Analysis** for stress testing.
    """)

st.sidebar.markdown("---")
st.sidebar.caption(f"Asset: **{selected_ticker}**")

#Data Loading
@st.cache_data
def get_processed_data(ticker):
    df = data_loader.get_stock(ticker)
    
    # --- PERFORMANCE OPTIMIZATION ---
    # We limit the data for technical indicator calculation to recent history 
    # to ensure responsiveness on cloud instances.
    df = df.tail(1000).copy() 
    
    df = FinancialFeatures.add_technical_indicators(df)
    df = FinancialFeatures.add_ml_features(df)
    return df

with st.spinner("Analyzing market data..."):
    df = get_processed_data(selected_ticker)

if df.empty:
    st.error(f"Insufficient data for {selected_ticker}.")
    st.stop()
    
#Calculate durations for display
start_date = df['date'].iloc[0].strftime('%Y-%m-%d')
end_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
duration_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25

#Common Plot Config
PLOT_CONFIG = {
    'displayModeBar': 'hover',
    'scrollZoom': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'toImage', 'autoScale2d']
}

#Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Quant Signals", "Risk Analysis", "Methodology"])

#TAB 1:Market Overview
with tab1:
    st.caption(f"Historical Price Action ({start_date} to {end_date} • ~{duration_years:.1f} Years)")
    last_row = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Last Close", f"${last_row['close']:.2f}", f"{last_row['Daily_Return']*100:.2f}%")
    with c2:
        val = last_row['RSI']
        status = "Overbought" if val > 70 else "Oversold" if val < 30 else "Neutral"
        render_metric_card("RSI Index", f"{val:.1f}", status)
    with c3:
        render_metric_card("Volatility (Ann.)", f"{last_row['Volatility']*100:.2f}%")
    with c4:
        render_metric_card("Volume (M)", f"{last_row['volume']/1e6:.1f}M")

    #Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_50'], name='SMA 50', line=dict(color='#FFA726', width=1.5)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA_200'], name='SMA 200', line=dict(color='#29B6F6', width=1.5)))
    
    fig.update_layout(
        template="plotly_dark", 
        height=500, 
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    
    with st.expander("Chart Components"):
        st.markdown("""
        - **Candlesticks**: OHLC (Open, High, Low, Close) pricing.
        - **SMA 50/200**: Trend indicators (50-day and 200-day Simple Moving Averages).
        """)


#TAB 2:Quant Signals
with tab2:
    st.subheader("Model Forecast")
    st.caption("Predictive signals generated by XGBoost algorithm (Training on 80% history, Testing on recent 20%).")
    
    def train_model(data):
        # Caching is now handled internally by TrendPredictor via _train_xgboost_cached
        predictor = TrendPredictor()
        acc, rep, X_test, y_test, preds = predictor.train(data)
        return predictor, acc, rep, preds

    with st.status(f"Generating AI signals for {selected_ticker}...", expanded=True) as status:
        st.write("Fetching historical data...")
        st.write("Training XGBoost model (this may take a few seconds)...")
        predictor, accuracy, report, predictions = train_model(df)
        status.update(label="Analysis completed successfully!", state="complete", expanded=False)
    
    #Prediction Performance
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        render_metric_card("Test Accuracy", f"{accuracy*100:.1f}%", help_text="Out-of-sample accuracy.")
        st.markdown("##### Performance Metrics")
        
        class_1_prec = report['1']['precision']
        class_0_prec = report['0']['precision']
        
        st.write(f"**Long Precision:** `{class_1_prec*100:.1f}%`")
        st.caption("_Correctness of Buy signals._")
        
        st.write(f"**Short/Neutral Precision:** `{class_0_prec*100:.1f}%`")
        st.caption("_Correctness of Avoid signals._")

    with col_r:
        st.markdown(f"##### Factor Contribution")
        imp_df = predictor.feature_importance.head(8)
        
        feature_map = {
            'RSI': 'Relative Strength',
            'Volatility': 'Realized Volatility',
            'SMA_50': 'Trend (Short)',
            'SMA_200': 'Trend (Long)',
            'Volume_Change': 'Volume Delta',
            'Momentum_1d': 'Momentum (1D)',
            'Return_Lag_1': 'Return (1-Day)',
            'Return_Lag_2': 'Return (2-Day)',
            'Return_Lag_3': 'Return (3-Day)',
            'Return_Lag_5': 'Return (1-Week)'
        }
        
        imp_df['Feature'] = imp_df['Feature'].map(lambda x: feature_map.get(x, x.replace('_', ' ')))
        
        fig_imp = px.bar(
            imp_df, 
            x='Importance', 
            y='Feature', 
            orientation='h', 
            color='Importance', 
            template="plotly_dark", 
            color_continuous_scale='Tealgrn'
        )
        
        fig_imp.update_traces(
            texttemplate='%{x:.1%}', 
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.2%}<extra></extra>'
        )
        
        fig_imp.update_layout(
            yaxis={'categoryorder':'total ascending', 'title': None}, 
            xaxis={'showticklabels': False, 'title': None},
            showlegend=False, 
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            height=300
        )
        
        static_config = {'displayModeBar': False, 'staticPlot': True}
        st.plotly_chart(fig_imp, width="stretch", config=static_config)

    st.markdown("---")
    
    #Strategy Backtest
    st.subheader("Strategy Backtest")
    test_days = len(predictions)
    st.caption(f"Performance simulation on unseen data (Last {test_days} trading days). Capital: ${initial_capital:,}")
    
    backtester = StrategyBacktest(initial_capital=initial_capital)
    strategy_df = backtester.run_strategy(df, predictions)
    metrics = backtester.calculate_metrics(strategy_df)
    
    b1, b2, b3 = st.columns(3)
    with b1:
        net_profit = (metrics['Total_Return'] * initial_capital)
        is_neg = net_profit < 0
        render_metric_card("Net Return", f"${net_profit:,.2f}", f"{metrics['Total_Return']*100:.1f}%", is_negative=is_neg)
    with b2:
        render_metric_card("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.2f}", help_text="Risk-Adjusted Return Measure")
    with b3:
        render_metric_card("Max Drawdown", f"{metrics['Max_Drawdown']*100:.2f}%", help_text="Maximum Peak-to-Trough Decline", is_negative=True)
    
    #Equity Curve
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=strategy_df['date'], y=strategy_df['Market_Value'], name='Benchmark', line=dict(color='gray', dash='dash')))
    fig_eq.add_trace(go.Scatter(x=strategy_df['date'], y=strategy_df['Strategy_Value'], name='Quant Strategy', line=dict(color='#00CC96', width=2)))
    
    fig_eq.add_annotation(x=strategy_df['date'].iloc[-1], y=strategy_df['Strategy_Value'].iloc[-1],
        text=f"Final: ${strategy_df['Strategy_Value'].iloc[-1]:,.0f}", showarrow=True, arrowhead=1)
    
    fig_eq.update_layout(template="plotly_dark", title="Cumulative Performance", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_eq, width="stretch", config=PLOT_CONFIG)


#TAB 3: Risk Simulator
with tab3:
    st.subheader("Monte Carlo Simulation")
    st.caption(f"Stochastic projection of 1,000 potential price paths for the next 30 Days (GBM Process).")
    
    risk_sim = RiskSimulator()
    recent_log_returns = np.log(df['close'] / df['close'].shift(1)).dropna().tail(252)
    last_price = df['close'].iloc[-1]
    
    simulation_paths = risk_sim.run_simulation(last_price, recent_log_returns, days=config.FORECAST_DAYS, iterations=config.SIMULATION_ITERATIONS)
    
    # VaR
    final_prices = simulation_paths[-1, :]
    var_percent = risk_sim.calculate_var(final_prices, last_price, confidence_level=0.95)
    var_amount = initial_capital * var_percent
    
    # Layout for risk
    r_col1, r_col2 = st.columns([1, 2])
    
    with r_col1:
        st.markdown("#### VaR (95%)")
        st.write("Projected maximum loss at 95% confidence:")
        render_metric_card("Value at Risk", f"${abs(var_amount):,.2f}", f"{var_percent*100:.2f}%", is_negative=True)
        
        st.markdown("""
        **Statistical Note**
        
        This metric indicates that in 95% of simulated scenarios, the portfolio loss does not exceed this threshold over the 30-day horizon.
        """)
        
    with r_col2:
        # Plot
        fig_mc = go.Figure()
        # Sample paths
        for i in range(min(50, config.SIMULATION_ITERATIONS)):
            fig_mc.add_trace(go.Scatter(y=simulation_paths[:, i], mode='lines', line=dict(width=1, color='rgba(75, 192, 192, 0.2)'), showlegend=False))
        
        # Mean path
        mean_path = np.mean(simulation_paths, axis=1)
        fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', name='Average/Mean', line=dict(color='white', width=3, dash='dot')))
        
        fig_mc.update_layout(template="plotly_dark", title=f"Stochastic Price Paths", xaxis_title="Time", yaxis_title="Price ($)", height=450,
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_mc, width="stretch", config=PLOT_CONFIG)

#TAB 4: Methodology
with tab4:
    st.subheader("Methodology & Glossary")
    st.markdown("Comprehensive guide to the metrics, indicators, and models used in this dashboard.")
    
    #Technical Indicators
    with st.expander("📊 Technical Indicators (Market Overview)", expanded=True):
        st.markdown("These indicators are calculated directly from historical price and volume data to assess market state.")
        
        st.info("""
        **SMA (Simple Moving Average)**
        The average price over a specific period. It smooths out price noise to show the underlying trend.
        *   **SMA 50 (Orange Line)**: Represents the **short-term trend** (approx. 2.5 months). When price is above this, the immediate trend is up.
        *   **SMA 200 (Blue Line)**: Represents the **long-term trend** (approx. 1 year). This is a major support/resistance level for institutions.
        *   **Golden Cross**: When the SMA 50 crosses *above* the SMA 200. This is typically a very bullish (buy) signal.
        """)
        
        st.info("""
        **RSI (Relative Strength Index)**
        A momentum oscillator that measures the speed and change of price movements. range is 0 to 100.
        *   **> 70 (Overbought)**: The asset may be overvalued and due for a correction (price drop).
        *   **< 30 (Oversold)**: The asset may be undervalued and due for a bounce (price rise).
        *   **50**: The centerline. Above 50 indicates bullish momentum, below 50 indicates bearish.
        """)
        
        st.info("""
        **Volatility (Annualized)**
        A statistical measure of the dispersion of returns.
        *   **High Volatility**: Prices swing wildly. Higher risk, but potential for higher rapid returns.
        *   **Low Volatility**: Prices are stable. Lower risk, but generally lower rapid returns.
        *   *Formula*: Standard Deviation of Daily Returns * Square Root(252 trading days).
        """)

    #Algorithmic Factors
    with st.expander("🤖 Algorithmic Model (Quant Signals)", expanded=False):
        st.markdown("How the Machine Learning model (XGBoost) sees the market.")
        
        st.success("""
        **XGBoost Classifier**
        We use an algorithm called **Extreme Gradient Boosting**. It builds hundreds of small "decision trees" that check various conditions (e.g., "Is RSI > 70?" AND "Is Volume rising?"). It votes on whether the price is likely to go UP (Long) or DOWN/FLAT (Neutral) tomorrow.
        """)
        
        st.success("""
        **Lag Features (Time Travel)**
        The model looks at the past to predict the future.
        *   **Return (1-Day)**: How much did the price change yesterday?
        *   **Return (1-Week)**: How much did it change over the last 5 trading days?
        *   *Why?* Markets often exhibit "mean reversion" (bouncing back) or "momentum" (continuing trend), which these features capture.
        """)
        
        st.success("""
        **Volume Efficiency**
        A derived metric that compares price change to volume.
        *   If Price rises 5% on **High Volume**: The move is "real" and supported by big money.
        *   If Price rises 5% on **Low Volume**: The move might be a "fake out" and could reverse.
        """)
        
    #Risk Metrics
    with st.expander("🛡️ Risk Management (Simulation)", expanded=False):
        st.markdown("Metrics to help you sleep at night by understanding the worst-case scenarios.")
        
        st.warning("""
        **Sharpe Ratio**
        The gold standard for comparing investment performance. It asks: *"Is the return worth the risk?"*
        *   **Ratio < 1.0**: Suboptimal. You are taking too much risk for not enough profit.
        *   **Ratio > 1.0**: Good.
        *   **Ratio > 2.0**: Very Good.
        *   *Formula*: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility.
        """)
        
        st.error("""
        **Max Drawdown (MDD)**
        The "Pain Index". It implies the maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained.
        *   *Example*: If you bought at the absolute top and sold at the absolute bottom, how much would you have lost?
        """)
        
        st.error("""
        **Monte Carlo Simulation**
        We can't predict the future, but we can simulate it. We run **1,000 different future scenarios** based on the asset's past behavior (volatility and drift).
        *   The **Mean Projection** is the average of all 1,000 paths (the "most likely" outcome).
        *   The cloud of lines shows the range of possibilities.
        """)
        
        st.error("""
        **VaR (Value at Risk 95%)**
        A probabilistic metric used by banks.
        *   *Definition*: "We are 95% confident that the loss will NOT exceed this amount over the next 30 days."
        *   *Alternatively*: There is only a 5% chance (1 in 20) that losses will be worse than this number.
        """)