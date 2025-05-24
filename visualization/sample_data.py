"""
Generate sample trading data for visualization testing.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_trades(num_trades=100):
    """Generate sample trade data."""
    now = datetime.now()
    trades = []
    
    # Initial parameters
    entry_price = 50000.0  # Starting BTC price
    volatility = 0.02      # 2% volatility
    win_rate = 0.6        # 60% win rate
    
    for i in range(num_trades):
        # Generate timestamps
        entry_time = now - timedelta(days=num_trades-i)
        exit_time = entry_time + timedelta(hours=random.randint(1, 24))
        
        # Generate prices with trend and volatility
        price_change = np.random.normal(0, volatility)
        if random.random() < win_rate:  # Winning trade
            exit_price = entry_price * (1 + abs(price_change))
        else:  # Losing trade
            exit_price = entry_price * (1 - abs(price_change))
        
        # Generate position size
        size = random.uniform(0.1, 1.0)
        
        # Calculate P&L
        pnl = (exit_price - entry_price) * size
        
        trade = {
            'id': i + 1,
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'timestamp': entry_time.isoformat(),
            'strategy': 'Sample_Strategy',
            'status': 'CLOSED'
        }
        trades.append(trade)
        
        # Update entry price for next trade
        entry_price = exit_price * (1 + np.random.normal(0, volatility/2))
    
    return trades

def generate_sample_metrics(trades):
    """Generate sample performance metrics."""
    df = pd.DataFrame(trades)
    
    total_trades = len(trades)
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    
    metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_profit': df['pnl'].sum(),
        'average_profit': df['pnl'].mean(),
        'max_drawdown': calculate_max_drawdown(df),
        'sharpe_ratio': calculate_sharpe_ratio(df),
        'profit_factor': calculate_profit_factor(df),
        'average_win': df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
        'average_loss': abs(df[df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0,
        'largest_win': df['pnl'].max(),
        'largest_loss': abs(df['pnl'].min()),
        'timestamp': datetime.now().isoformat()
    }
    
    return metrics

def calculate_max_drawdown(df):
    """Calculate maximum drawdown from trade data."""
    cumulative = df['pnl'].cumsum()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(float(drawdown.min())) if not drawdown.empty else 0.0

def calculate_sharpe_ratio(df):
    """Calculate Sharpe ratio from trade data."""
    returns = df['pnl'] / df['entry_price']
    if returns.empty or returns.std() == 0:
        return 0.0
    return np.sqrt(252) * (returns.mean() / returns.std())

def calculate_profit_factor(df):
    """Calculate profit factor from trade data."""
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
    return gross_profit / gross_loss if gross_loss != 0 else 0.0

if __name__ == "__main__":
    # Generate sample data
    trades = generate_sample_trades(100)
    metrics = generate_sample_metrics(trades)
    
    # Save to CSV for testing
    pd.DataFrame(trades).to_csv('sample_trades.csv', index=False)
    pd.DataFrame([metrics]).to_csv('sample_metrics.csv', index=False)
    
    print("Sample data generated successfully!")
    print(f"Number of trades: {len(trades)}")
    print(f"Total P&L: {metrics['total_profit']:.2f}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%") 