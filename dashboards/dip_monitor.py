"""
Streamlit dashboard for monitoring dip trades and predictions.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from agents.dip_trade_logger import DipTradeLogger
from agents.sentiment_analyzer import SentimentAnalyzer
from agents.model_trainer import ModelTrainer
from typing import Dict, Any, List, Optional
import asyncio
from plotly.subplots import make_subplots
import os
import joblib

# Initialize components
logger = DipTradeLogger()
sentiment_analyzer = SentimentAnalyzer({})  # Configure with API keys
model_trainer = ModelTrainer({})

def main():
    st.set_page_config(
        page_title="Dip Trading Monitor",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Dip Trading Monitor")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Live Trading", "Backtesting", "Model Performance", "Market Sentiment"]
    )
    
    # Common filters
    symbols = get_available_symbols()
    selected_symbol = st.sidebar.selectbox(
        "Select Trading Pair",
        options=symbols
    )
    
    time_windows = {
        "Last 24 Hours": 24,
        "Last 7 Days": 168,
        "Last 30 Days": 720
    }
    selected_window = st.sidebar.selectbox(
        "Time Window",
        options=list(time_windows.keys())
    )
    hours = time_windows[selected_window]
    
    # Route to appropriate page
    if page == "Live Trading":
        display_live_trading(selected_symbol, hours)
    elif page == "Backtesting":
        display_backtesting(selected_symbol)
    elif page == "Model Performance":
        display_model_performance(selected_symbol, hours)
    else:
        display_market_sentiment(selected_symbol, hours)

def display_live_trading(symbol: str, hours: int):
    """Display live trading dashboard."""
    col1, col2 = st.columns(2)
    
    with col1:
        display_performance_metrics(symbol)
        
    with col2:
        display_prediction_accuracy(symbol, hours)
        
    # Recent trades table
    st.header("Recent Dip Trades")
    display_recent_trades(symbol)
    
    # Price and volume chart with ML predictions
    st.header("Price Action & Dip Detection")
    display_price_chart(symbol, hours, with_predictions=True)
    
    # Risk metrics
    st.header("Risk Metrics")
    display_risk_metrics(symbol)
    
def display_backtesting(symbol: str):
    """Display backtesting interface and results."""
    st.header("Backtesting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30)
        )
        
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
        
    # Strategy parameters
    st.subheader("Strategy Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_dip = st.number_input(
            "Minimum Dip %",
            value=2.0,
            step=0.1
        )
        
    with col2:
        recovery_target = st.number_input(
            "Recovery Target %",
            value=1.0,
            step=0.1
        )
        
    with col3:
        stop_loss = st.number_input(
            "Stop Loss %",
            value=1.0,
            step=0.1
        )
        
    if st.button("Run Backtest"):
        results = run_backtest(
            symbol,
            start_date,
            end_date,
            {
                'min_dip': min_dip,
                'recovery_target': recovery_target,
                'stop_loss': stop_loss
            }
        )
        display_backtest_results(results)
        
def display_model_performance(symbol: str, hours: int):
    """Display model performance metrics and insights."""
    st.header("Model Performance Analysis")
    
    # Performance metrics over time
    metrics = get_model_performance_history(symbol, hours)
    if metrics is not None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "F1 Score", "Precision & Recall",
                "ROC Curve", "Confusion Matrix"
            )
        )
        
        # F1 Score trend
        fig.add_trace(
            go.Scatter(
                x=metrics['timestamp'],
                y=metrics['f1_score'],
                name="F1 Score"
            ),
            row=1, col=1
        )
        
        # Precision & Recall
        fig.add_trace(
            go.Scatter(
                x=metrics['timestamp'],
                y=metrics['precision'],
                name="Precision"
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=metrics['timestamp'],
                y=metrics['recall'],
                name="Recall"
            ),
            row=1, col=2
        )
        
        # ROC Curve
        if 'fpr' in metrics and 'tpr' in metrics:
            fig.add_trace(
                go.Scatter(
                    x=metrics['fpr'],
                    y=metrics['tpr'],
                    name="ROC Curve"
                ),
                row=2, col=1
            )
            
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            fig.add_trace(
                go.Heatmap(
                    z=metrics['confusion_matrix'],
                    x=['Predicted 0', 'Predicted 1'],
                    y=['Actual 0', 'Actual 1'],
                    colorscale='RdBu'
                ),
                row=2, col=2
            )
            
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
        
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = get_feature_importance(symbol)
    if feature_importance is not None:
        fig = px.bar(
            feature_importance,
            x='feature',
            y='importance',
            title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Training history
    st.subheader("Training History")
    training_history = get_training_history(symbol)
    if training_history:
        st.dataframe(training_history)
        
def display_market_sentiment(symbol: str, hours: int):
    """Display market sentiment analysis."""
    st.header("Market Sentiment Analysis")
    
    # Get sentiment data
    sentiment_data = asyncio.run(
        sentiment_analyzer.get_combined_sentiment(symbol)
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_sentiment_gauge(
            "Overall Sentiment",
            sentiment_data['combined_score']
        )
        
    with col2:
        display_sentiment_gauge(
            "Social Media Sentiment",
            sentiment_data['social_sentiment']['score']
        )
        
    with col3:
        display_sentiment_gauge(
            "News Sentiment",
            sentiment_data['news_sentiment']['score']
        )
        
    # Sentiment trends
    st.subheader("Sentiment Trends")
    sentiment_history = get_sentiment_history(symbol, hours)
    if sentiment_history:
        fig = go.Figure()
        
        # Add sentiment lines
        for source in ['combined', 'social', 'news', 'technical']:
            fig.add_trace(go.Scatter(
                x=sentiment_history['timestamp'],
                y=sentiment_history[f'{source}_score'],
                name=f"{source.title()} Sentiment"
            ))
            
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Time",
            yaxis_title="Sentiment Score"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # News and social media analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent News Analysis")
        display_news_analysis(
            sentiment_data['news_sentiment'].get('metadata', {})
        )
        
    with col2:
        st.subheader("Social Media Analysis")
        display_social_analysis(
            sentiment_data['social_sentiment'].get('metadata', {})
        )
        
def display_sentiment_gauge(title: str, score: float):
    """Display a sentiment gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "red"},
                {'range': [-0.3, 0.3], 'color': "gray"},
                {'range': [0.3, 1], 'color': "green"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
def display_news_analysis(metadata: Dict[str, Any]):
    """Display news analysis details."""
    if metadata:
        st.metric("Articles Analyzed", metadata.get('article_count', 0))
        st.metric(
            "Sentiment Volatility",
            f"{metadata.get('sentiment_std', 0):.2f}"
        )
        
def display_social_analysis(metadata: Dict[str, Any]):
    """Display social media analysis details."""
    if metadata:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tweets Analyzed", metadata.get('tweet_count', 0))
        with col2:
            st.metric("Reddit Posts Analyzed", metadata.get('reddit_count', 0))
            
        st.metric(
            "Sentiment Volatility",
            f"{metadata.get('sentiment_std', 0):.2f}"
        )
        
def run_backtest(symbol: str, start_date: datetime,
                 end_date: datetime, params: Dict[str, float]) -> Dict[str, Any]:
    """Run backtest simulation."""
    try:
        # Validate dates
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return None
            
        # Get historical price data
        price_data = logger.db.market_data.find(
            {
                'symbol': symbol,
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
        )
        
        # Convert price data to DataFrame
        if isinstance(price_data, list):
            df = pd.DataFrame(price_data)
        elif hasattr(price_data, 'to_dict'):  # Handle mock DataFrame
            df = pd.DataFrame(price_data.to_dict('records'))
        else:
            df = pd.DataFrame(list(price_data))
            
        if df.empty:
            st.error("No price data found")
            return None
            
        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        st.write(f"Found {len(df)} price points")
        
        # Initialize results
        trades = []
        position = None
        entry_price = 0
        
        # Simulate trading
        for i in range(1, len(df)):
            current_price = float(df.iloc[i]['price'])
            prev_price = float(df.iloc[i-1]['price'])
            
            # Calculate price change
            price_change = (prev_price - current_price) / prev_price * 100
            
            # Check for dip entry
            if not position and price_change >= params['min_dip']:
                st.write(f"Found dip entry at {df.iloc[i]['timestamp']}: {price_change:.2f}%")
                position = 'long'
                entry_price = current_price
                trades.append({
                    'type': 'entry',
                    'timestamp': df.iloc[i]['timestamp'],
                    'price': current_price
                })
                continue
                
            # Check for exit conditions
            if position == 'long':
                # Check profit target
                profit_pct = (current_price - entry_price) / entry_price * 100
                loss_pct = (entry_price - current_price) / entry_price * 100
                
                if profit_pct >= params['recovery_target'] or loss_pct >= params['stop_loss']:
                    st.write(f"Found exit at {df.iloc[i]['timestamp']}: {profit_pct:.2f}%")
                    trades.append({
                        'type': 'exit',
                        'timestamp': df.iloc[i]['timestamp'],
                        'price': current_price,
                        'pnl': profit_pct
                    })
                    position = None
                    
        st.write(f"Found {len(trades)} total trades")
        
        # Calculate performance metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            entry_exits = []
            
            for i in range(0, len(trades_df), 2):
                if i + 1 < len(trades_df):
                    entry = trades_df.iloc[i]
                    exit = trades_df.iloc[i + 1]
                    entry_exits.append({
                        'entry_time': entry['timestamp'],
                        'exit_time': exit['timestamp'],
                        'entry_price': entry['price'],
                        'exit_price': exit['price'],
                        'pnl': exit['pnl']
                    })
                    
            if entry_exits:  # Only return results if we have complete trades
                st.write(f"Found {len(entry_exits)} complete trades")
                results = {
                    'trades': entry_exits,
                    'total_trades': len(entry_exits),
                    'win_rate': len([t for t in entry_exits if t['pnl'] > 0]) / len(entry_exits),
                    'avg_profit': sum(t['pnl'] for t in entry_exits) / len(entry_exits),
                    'max_drawdown': calculate_max_drawdown(pd.Series([t['pnl'] for t in entry_exits])),
                    'sharpe_ratio': calculate_sharpe_ratio(pd.Series([t['pnl'] for t in entry_exits])),
                    'price_data': df[['timestamp', 'price']].to_dict('records')
                }
                return results
                
        st.error("No complete trades found")
        return None
        
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        return None
        
def display_backtest_results(results: Optional[Dict[str, Any]]):
    """Display backtest results."""
    if not results:
        st.error("No backtest results available")
        return
        
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", results['total_trades'])
    with col2:
        st.metric("Win Rate", f"{results['win_rate']:.2%}")
    with col3:
        st.metric("Avg Profit", f"{results['avg_profit']:.2%}")
    with col4:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
        
    # Equity curve
    trades_df = pd.DataFrame(results['trades'])
    cumulative_returns = (1 + trades_df['pnl'] / 100).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'],
        y=cumulative_returns,
        name="Equity Curve"
    ))
    
    fig.update_layout(
        title="Backtest Equity Curve",
        xaxis_title="Time",
        yaxis_title="Portfolio Value (Starting at 1.0)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade distribution
    st.subheader("Trade Distribution")
    fig = px.histogram(
        trades_df,
        x='pnl',
        nbins=50,
        title="Profit Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed trades table
    st.subheader("Trade History")
    st.dataframe(trades_df)
    
def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    try:
        if returns.empty:
            return 0.0
            
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        if cum_returns.empty:
            return 0.0
            
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdowns
        drawdowns = cum_returns / running_max - 1
        
        # Get maximum drawdown
        max_drawdown = drawdowns.min()
        
        return 0.0 if pd.isna(max_drawdown) else float(max_drawdown)
        
    except Exception:
        return 0.0
    
def calculate_sharpe_ratio(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    try:
        if returns.empty or len(returns) < 2:
            return float('nan')  # Return NaN for empty/insufficient series
            
        # Calculate mean and standard deviation
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Handle edge cases
        if std_return == 0:
            return 0.0 if mean_return == 0 else float('inf')
            
        # Calculate annualized Sharpe ratio
        return float(np.sqrt(252) * mean_return / std_return)
        
    except Exception:
        return float('nan')  # Return NaN on calculation errors
    
def get_model_performance_history(symbol: str, hours: int) -> Optional[pd.DataFrame]:
    """Get model performance history."""
    try:
        # Get performance metrics from database
        metrics = logger.db.dip_analytics.find(
            {
                'symbol': symbol,
                'type': 'model_performance',
                'timestamp': {
                    '$gte': datetime.now() - timedelta(hours=hours)
                }
            }
        )
        
        # Convert metrics to DataFrame
        if isinstance(metrics, list):
            df = pd.DataFrame(metrics)
        else:
            df = pd.DataFrame(list(metrics))
            
        if df.empty:
            return None
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure required columns exist
        required_cols = ['f1_score', 'precision', 'recall']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        return df
        
    except Exception as e:
        st.error(f"Error getting model performance history: {str(e)}")
        return None
        
def get_feature_importance(symbol: str) -> Optional[pd.DataFrame]:
    """Get feature importance scores."""
    try:
        metadata_file = os.path.join(model_trainer.model_dir, 'training_metadata.json')
        if os.path.exists(metadata_file):
            metadata = joblib.load(metadata_file)
            if symbol in metadata and 'feature_importance' in metadata[symbol]:
                return pd.DataFrame(metadata[symbol]['feature_importance'])
        return None
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return None
        
def get_training_history(symbol: str) -> Optional[pd.DataFrame]:
    """Get model training history."""
    try:
        history = list(logger.db.dip_analytics.find({
            'symbol': symbol,
            'event_type': 'model_training'
        }).sort('timestamp', -1))
        
        return pd.DataFrame(history) if history else None
        
    except Exception as e:
        logger.error(f"Error getting training history: {str(e)}")
        return None
        
def get_sentiment_history(symbol: str, hours: int) -> Optional[pd.DataFrame]:
    """Get sentiment history for a symbol."""
    try:
        # Get sentiment data from database
        sentiment_data = logger.db.dip_analytics.find(
            {
                'symbol': symbol,
                'type': 'sentiment',
                'timestamp': {
                    '$gte': datetime.now() - timedelta(hours=hours)
                }
            }
        )
        
        # Convert sentiment data to DataFrame
        if isinstance(sentiment_data, list):
            df = pd.DataFrame(sentiment_data)
        else:
            df = pd.DataFrame(list(sentiment_data))
            
        if df.empty:
            return None
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure required columns exist
        required_cols = ['combined_score', 'social_score', 'news_score', 'technical_score']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        return df
        
    except Exception as e:
        st.error(f"Error getting sentiment history: {str(e)}")
        return None

def get_available_symbols() -> List[str]:
    """Get list of available trading pairs."""
    trades = logger.get_recent_trades(limit=1000)
    return sorted(list(set(t['symbol'] for t in trades)))
    
def display_performance_metrics(symbol: str):
    """Display key performance metrics."""
    metrics = logger.get_performance_metrics(symbol)
    
    st.subheader("Performance Metrics")
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Trades", metrics['total_trades'])
    with cols[1]:
        st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
    with cols[2]:
        st.metric("Avg Profit", f"{metrics['avg_profit']:.2%}")
    with cols[3]:
        st.metric("Total Profit", f"{metrics['total_profit']:.2%}")
        
def display_prediction_accuracy(symbol: str, hours: int):
    """Display prediction accuracy metrics."""
    accuracy = logger.get_prediction_accuracy(symbol, hours)
    
    st.subheader("Prediction Accuracy")
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Total Predictions", accuracy['total_predictions'])
    with cols[1]:
        st.metric("True Positives", accuracy['true_positives'])
    with cols[2]:
        st.metric("Accuracy", f"{accuracy['accuracy']:.2%}")
        
def display_recent_trades(symbol: str):
    """Display table of recent trades."""
    trades = logger.get_recent_trades(symbol)
    if not trades:
        st.info("No trades found for selected symbol")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    df['pnl'] = df['pnl'].apply(lambda x: f"{x:.2%}" if x else "")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Format and display
    display_cols = [
        'timestamp', 'symbol', 'entry_price', 'exit_price',
        'size', 'pnl', 'status'
    ]
    st.dataframe(
        df[display_cols].sort_values('timestamp', ascending=False),
        use_container_width=True
    )
    
def display_price_chart(symbol: str, hours: int, with_predictions: bool = False):
    """Display price chart with dip annotations."""
    # Get price data
    since = datetime.now() - timedelta(hours=hours)
    price_data = logger.db.market_data.find({
        'symbol': symbol,
        'timestamp': {'$gte': since}
    }).sort('timestamp', 1)
    
    if not price_data:
        st.info("No price data available")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(price_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get dip trades
    dip_trades = logger.get_recent_trades(symbol)
    dip_df = pd.DataFrame(dip_trades)
    if not dip_df.empty:
        dip_df['timestamp'] = pd.to_datetime(dip_df['timestamp'])
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))
    
    # Add dip trade markers
    if not dip_df.empty:
        fig.add_trace(go.Scatter(
            x=dip_df['timestamp'],
            y=dip_df['entry_price'],
            mode='markers',
            name='Dip Trades',
            marker=dict(
                size=10,
                symbol='triangle-down',
                color='red'
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Action",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right"
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def display_risk_metrics(symbol: str):
    """Display risk metrics."""
    # Implementation of display_risk_metrics function
    pass

if __name__ == "__main__":
    main() 