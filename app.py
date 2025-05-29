from flask import Flask, render_template, jsonify, request
import logging
from datetime import datetime, timedelta
from config.database import (
    init_collections,
    get_database,
    get_active_trades,
    get_trade_history,
    get_latest_account_info,
    get_performance_metrics,
    insert_trade
)
import os
from bson.objectid import ObjectId

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def calculate_trading_stats(trades):
    """Calculate trading statistics from trade history"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_trade': "$0.00",
            'avg_trade_color': 'danger'
        }

    wins = sum(1 for trade in trades if float(trade.get('pl', 0)) > 0)
    total = len(trades)
    
    return {
        'total_trades': total,
        'win_rate': round((wins/total) * 100, 2) if total > 0 else 0,
        'profit_factor': round(sum(float(t.get('pl', 0)) for t in trades if float(t.get('pl', 0)) > 0) / 
                              abs(sum(float(t.get('pl', 0)) for t in trades if float(t.get('pl', 0)) < 0)), 2),
        'avg_trade': f"${round(sum(float(t.get('pl', 0)) for t in trades) / total, 2)}",
        'avg_trade_color': 'success' if sum(float(t.get('pl', 0)) for t in trades) > 0 else 'danger'
    }

def calculate_performance_metrics(trades, start_date=None):
    """Calculate detailed performance metrics from trade history"""
    if not trades:
        return {
            'total_profit': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'avg_drawdown': 0,
            'recovery_time': '0d',
            'current_drawdown': 0,
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'avg_win': '$0.00',
            'avg_loss': '$0.00',
            'largest_win': '$0.00',
            'largest_loss': '$0.00',
            'avg_duration': '0h',
            'best_day': 'N/A',
            'worst_day': 'N/A',
            'active_hours': 'N/A',
            'time_between_trades': '0h',
            'monthly_stats': [],
            'equity_curve': {'dates': [], 'values': []},
            'drawdown_curve': {'dates': [], 'values': []}
        }

    # Calculate basic metrics
    wins = [t for t in trades if float(t.get('pl', 0)) > 0]
    losses = [t for t in trades if float(t.get('pl', 0)) < 0]
    
    total_profit = sum(float(t.get('pl', 0)) for t in trades)
    win_rate = (len(wins) / len(trades)) * 100 if trades else 0
    
    # Calculate profit factor
    gross_profit = sum(float(t.get('pl', 0)) for t in wins) if wins else 0
    gross_loss = abs(sum(float(t.get('pl', 0)) for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    returns = [float(t.get('pl', 0)) for t in trades]
    avg_return = sum(returns) / len(returns) if returns else 0
    std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5 if returns else 0
    sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0
    
    # Calculate drawdown metrics
    equity_curve = []
    current_equity = 0
    max_equity = 0
    drawdowns = []
    current_drawdown = 0
    
    for trade in sorted(trades, key=lambda x: x.get('timestamp')):
        current_equity += float(trade.get('pl', 0))
        max_equity = max(max_equity, current_equity)
        drawdown = ((max_equity - current_equity) / max_equity * 100) if max_equity > 0 else 0
        drawdowns.append(drawdown)
        equity_curve.append({
            'date': trade['timestamp'].strftime('%Y-%m-%d'),
            'equity': current_equity
        })
    
    max_drawdown = max(drawdowns) if drawdowns else 0
    avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0
    current_drawdown = drawdowns[-1] if drawdowns else 0
    
    # Calculate time-based metrics
    trade_durations = []
    daily_pls = {}
    
    for trade in trades:
        # Duration
        if 'open_time' in trade and 'close_time' in trade:
            duration = trade['close_time'] - trade['open_time']
            trade_durations.append(duration)
        
        # Daily P/L
        date = trade['timestamp'].strftime('%Y-%m-%d')
        daily_pls[date] = daily_pls.get(date, 0) + float(trade.get('pl', 0))
    
    avg_duration = str(sum(trade_durations, timedelta()) / len(trade_durations)) if trade_durations else '0h'
    best_day = max(daily_pls.items(), key=lambda x: x[1])[0] if daily_pls else 'N/A'
    worst_day = min(daily_pls.items(), key=lambda x: x[1])[0] if daily_pls else 'N/A'
    
    # Calculate monthly statistics
    monthly_stats = []
    monthly_trades = {}
    
    for trade in trades:
        month = trade['timestamp'].strftime('%Y-%m')
        if month not in monthly_trades:
            monthly_trades[month] = []
        monthly_trades[month].append(trade)
    
    for month, month_trades in monthly_trades.items():
        month_profit = sum(float(t.get('pl', 0)) for t in month_trades)
        month_wins = len([t for t in month_trades if float(t.get('pl', 0)) > 0])
        
        monthly_stats.append({
            'month': month,
            'return': round(month_profit, 2),
            'trades': len(month_trades),
            'win_rate': round((month_wins / len(month_trades)) * 100, 2),
            'profit_factor': calculate_profit_factor(month_trades),
            'max_drawdown': calculate_max_drawdown(month_trades)
        })
    
    # Prepare chart data
    equity_dates = [point['date'] for point in equity_curve]
    equity_values = [point['equity'] for point in equity_curve]
    
    return {
        'total_profit': round(total_profit, 2),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'avg_drawdown': round(avg_drawdown, 2),
        'recovery_time': calculate_recovery_time(drawdowns),
        'current_drawdown': round(current_drawdown, 2),
        'total_trades': len(trades),
        'profitable_trades': len(wins),
        'losing_trades': len(losses),
        'avg_win': f"${round(sum(float(t.get('pl', 0)) for t in wins) / len(wins), 2)}" if wins else '$0.00',
        'avg_loss': f"${round(sum(float(t.get('pl', 0)) for t in losses) / len(losses), 2)}" if losses else '$0.00',
        'largest_win': f"${round(max(float(t.get('pl', 0)) for t in wins), 2)}" if wins else '$0.00',
        'largest_loss': f"${round(min(float(t.get('pl', 0)) for t in losses), 2)}" if losses else '$0.00',
        'avg_duration': avg_duration,
        'best_day': best_day,
        'worst_day': worst_day,
        'active_hours': calculate_active_hours(trades),
        'time_between_trades': calculate_time_between_trades(trades),
        'monthly_stats': sorted(monthly_stats, key=lambda x: x['month'], reverse=True),
        'equity_curve': {
            'dates': equity_dates,
            'values': equity_values
        },
        'drawdown_curve': {
            'dates': equity_dates,
            'values': drawdowns
        }
    }

def calculate_profit_factor(trades):
    """Helper function to calculate profit factor"""
    wins = [t for t in trades if float(t.get('pl', 0)) > 0]
    losses = [t for t in trades if float(t.get('pl', 0)) < 0]
    
    gross_profit = sum(float(t.get('pl', 0)) for t in wins) if wins else 0
    gross_loss = abs(sum(float(t.get('pl', 0)) for t in losses)) if losses else 0
    
    return round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0

def calculate_max_drawdown(trades):
    """Helper function to calculate maximum drawdown"""
    equity = 0
    peak = 0
    max_dd = 0
    
    for trade in sorted(trades, key=lambda x: x.get('timestamp')):
        equity += float(trade.get('pl', 0))
        peak = max(peak, equity)
        dd = ((peak - equity) / peak * 100) if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return round(max_dd, 2)

def calculate_recovery_time(drawdowns):
    """Helper function to calculate recovery time"""
    if not drawdowns:
        return '0d'
    
    current_drawdown = 0
    max_recovery_time = 0
    current_recovery_time = 0
    
    for dd in drawdowns:
        if dd > 0:
            current_drawdown = dd
            current_recovery_time += 1
        else:
            if current_drawdown > 0:
                max_recovery_time = max(max_recovery_time, current_recovery_time)
            current_drawdown = 0
            current_recovery_time = 0
    
    return f"{max_recovery_time}d"

def calculate_active_hours(trades):
    """Helper function to calculate most active trading hours"""
    if not trades:
        return 'N/A'
    
    hour_counts = {}
    for trade in trades:
        hour = trade['timestamp'].hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    most_active = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return ', '.join(f"{h:02d}:00" for h, _ in most_active)

def calculate_time_between_trades(trades):
    """Helper function to calculate average time between trades"""
    if len(trades) < 2:
        return '0h'
    
    sorted_trades = sorted(trades, key=lambda x: x.get('timestamp'))
    time_diffs = []
    
    for i in range(1, len(sorted_trades)):
        diff = sorted_trades[i]['timestamp'] - sorted_trades[i-1]['timestamp']
        time_diffs.append(diff)
    
    avg_time = sum(time_diffs, timedelta()) / len(time_diffs)
    return str(avg_time)

def get_alerts():
    """Get all alerts from database"""
    try:
        db = get_database()
        return list(db.alerts.find())
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise

def get_alert_history():
    """Get alert history"""
    try:
        db = get_database()
        return list(db.alerts.find({'status': {'$in': ['triggered', 'failed']}}).sort('timestamp', -1))
    except Exception as e:
        logger.error(f"Error getting alert history: {str(e)}")
        raise

def calculate_alert_stats():
    """Calculate alert statistics"""
    try:
        db = get_database()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        active = db.alerts.count_documents({'status': 'active'})
        pending = db.alerts.count_documents({'status': 'pending'})
        triggered_today = db.alerts.count_documents({
            'status': 'triggered',
            'triggered_at': {'$gte': today}
        })
        failed = db.alerts.count_documents({'status': 'failed'})
        
        return {
            'active': active,
            'pending': pending,
            'triggered_today': triggered_today,
            'failed': failed
        }
    except Exception as e:
        logger.error(f"Error calculating alert stats: {str(e)}")
        return {
            'active': 0,
            'pending': 0,
            'triggered_today': 0,
            'failed': 0
        }

def get_strategies():
    """Get all strategies from database"""
    try:
        db = get_database()
        return list(db.strategies.find())
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        raise

def calculate_strategy_stats():
    """Calculate strategy statistics"""
    try:
        db = get_database()
        
        active = db.strategies.count_documents({'status': 'active'})
        profitable = db.strategies.count_documents({
            'status': 'active',
            'performance': {'$gt': 0}
        })
        testing = db.strategies.count_documents({'status': 'testing'})
        stopped = db.strategies.count_documents({'status': 'stopped'})
        
        return {
            'active': active,
            'profitable': profitable,
            'testing': testing,
            'stopped': stopped
        }
    except Exception as e:
        logger.error(f"Error calculating strategy stats: {str(e)}")
        return {
            'active': 0,
            'profitable': 0,
            'testing': 0,
            'stopped': 0
        }

def calculate_performance_data():
    """Calculate performance data for charts"""
    try:
        db = get_database()
        strategies = list(db.strategies.find({'status': 'active'}))
        
        # Get dates for the last 30 days
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') 
                for x in range(30, 0, -1)]
        
        # Calculate daily performance for each strategy
        performance_values = []
        benchmark_values = []
        
        for date in dates:
            daily_trades = list(db.trades.find({
                'timestamp': {
                    '$gte': datetime.strptime(date, '%Y-%m-%d'),
                    '$lt': datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
                }
            }))
            
            # Calculate strategy performance
            daily_pl = sum(float(t.get('pl', 0)) for t in daily_trades)
            daily_equity = sum(float(t.get('equity', 0)) for t in daily_trades) or 1
            daily_return = (daily_pl / daily_equity) * 100
            performance_values.append(daily_return)
            
            # Calculate benchmark (e.g., BTC price change)
            benchmark_values.append(0)  # Replace with actual benchmark calculation
        
        return {
            'dates': dates,
            'values': performance_values,
            'benchmark': benchmark_values
        }
    except Exception as e:
        logger.error(f"Error calculating performance data: {str(e)}")
        return {
            'dates': [],
            'values': [],
            'benchmark': []
        }

def calculate_risk_data():
    """Calculate risk metrics for strategies"""
    try:
        db = get_database()
        strategies = list(db.strategies.find({'status': 'active'}))
        
        strategy_names = []
        sharpe_ratios = []
        
        for strategy in strategies:
            # Get trades for this strategy
            trades = list(db.trades.find({'strategy_id': strategy['_id']}))
            
            # Calculate returns
            returns = [float(t.get('pl', 0)) for t in trades]
            
            if returns:
                # Calculate Sharpe ratio
                avg_return = sum(returns) / len(returns)
                std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                sharpe = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0
                
                strategy_names.append(strategy['name'])
                sharpe_ratios.append(round(sharpe, 2))
        
        return {
            'strategies': strategy_names,
            'sharpe_ratios': sharpe_ratios
        }
    except Exception as e:
        logger.error(f"Error calculating risk data: {str(e)}")
        return {
            'strategies': [],
            'sharpe_ratios': []
        }

def get_market_data():
    """Get market data from database"""
    try:
        db = get_database()
        return list(db.market_data.find().sort('timestamp', -1))
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        raise

def calculate_market_stats():
    """Calculate market statistics"""
    try:
        db = get_database()
        market_data = get_market_data()
        
        # Calculate total volume
        total_volume = sum(float(data.get('volume', 0)) for data in market_data)
        
        # Calculate market trend
        price_changes = [float(data.get('change', 0)) for data in market_data]
        market_trend = 'Bullish' if sum(price_changes) > 0 else 'Bearish'
        
        # Count active pairs
        active_pairs = len(set(data.get('symbol') for data in market_data))
        
        # Calculate volatility
        volatility = calculate_market_volatility(market_data)
        
        return {
            'total_volume': f"${total_volume:,.2f}",
            'market_trend': market_trend,
            'active_pairs': active_pairs,
            'volatility': round(volatility, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating market stats: {str(e)}")
        return {
            'total_volume': '$0.00',
            'market_trend': 'N/A',
            'active_pairs': 0,
            'volatility': 0
        }

def calculate_market_volatility(market_data):
    """Calculate market volatility"""
    try:
        if not market_data:
            return 0
        
        # Calculate average true range
        atr_values = []
        for data in market_data:
            high = float(data.get('high', 0))
            low = float(data.get('low', 0))
            close = float(data.get('close', 0))
            
            true_range = max(high - low,
                           abs(high - close),
                           abs(low - close))
            atr_values.append(true_range)
        
        # Calculate ATR
        atr = sum(atr_values) / len(atr_values) if atr_values else 0
        
        # Convert to percentage
        last_price = float(market_data[0].get('close', 1))
        volatility = (atr / last_price) * 100
        
        return volatility
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return 0

def get_technical_indicators(symbol):
    """Calculate technical indicators for a symbol"""
    try:
        db = get_database()
        market_data = list(db.market_data.find({
            'symbol': symbol
        }).sort('timestamp', -1).limit(100))
        
        if not market_data:
            return []
        
        # Calculate indicators
        closes = [float(data.get('close', 0)) for data in market_data]
        
        # RSI
        rsi = calculate_rsi(closes)
        rsi_signal = 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
        rsi_color = 'danger' if rsi < 30 else 'warning' if rsi > 70 else 'success'
        
        # MACD
        macd, signal = calculate_macd(closes)
        macd_signal = 'Buy' if macd > signal else 'Sell'
        macd_color = 'success' if macd > signal else 'danger'
        
        # Moving Averages
        ma20 = sum(closes[:20]) / 20 if len(closes) >= 20 else 0
        ma50 = sum(closes[:50]) / 50 if len(closes) >= 50 else 0
        ma_signal = 'Bullish' if ma20 > ma50 else 'Bearish'
        ma_color = 'success' if ma20 > ma50 else 'danger'
        
        return [
            {
                'name': 'RSI (14)',
                'value': f"{round(rsi, 2)}",
                'signal': rsi_signal,
                'signal_color': rsi_color,
                'updated_at': market_data[0]['timestamp'].strftime('%Y-%m-%d %H:%M')
            },
            {
                'name': 'MACD (12,26,9)',
                'value': f"{round(macd, 2)}",
                'signal': macd_signal,
                'signal_color': macd_color,
                'updated_at': market_data[0]['timestamp'].strftime('%Y-%m-%d %H:%M')
            },
            {
                'name': 'MA Cross (20,50)',
                'value': f"MA20: {round(ma20, 2)}, MA50: {round(ma50, 2)}",
                'signal': ma_signal,
                'signal_color': ma_color,
                'updated_at': market_data[0]['timestamp'].strftime('%Y-%m-%d %H:%M')
            }
        ]
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return []

def calculate_rsi(prices, periods=14):
    """Calculate RSI indicator"""
    if len(prices) < periods:
        return 50
    
    # Calculate price changes
    deltas = [prices[i-1] - prices[i] for i in range(1, len(prices))]
    
    # Calculate gains and losses
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    # Calculate average gains and losses
    avg_gain = sum(gains[:periods]) / periods
    avg_loss = sum(losses[:periods]) / periods
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    if len(prices) < slow:
        return 0, 0
    
    # Calculate EMAs
    ema_fast = sum(prices[:fast]) / fast
    ema_slow = sum(prices[:slow]) / slow
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = sum(prices[:signal]) / signal
    
    return macd_line, signal_line

@app.route('/')
def index():
    """Main dashboard route"""
    try:
        # Get data from MongoDB
        active_trades = get_active_trades()
        trade_history = get_trade_history(50)
        account_info = get_latest_account_info()
        performance = get_performance_metrics()

        # Ensure account_info has all required keys with default values
        if account_info is None:
            account_info = {'balance': '$0', 'available': '$0', 'equity': '$0', 'used_margin': '$0'}
        else:
            account_info.setdefault('balance', '$0')
            account_info.setdefault('available', '$0')
            account_info.setdefault('equity', '$0')
            account_info.setdefault('used_margin', '$0')

        # Calculate statistics
        trading_stats = calculate_trading_stats(trade_history)
        performance_metrics = calculate_performance_metrics(trade_history)

        # Prepare chart data
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30, 0, -1)]
        
        # Calculate daily P/L for chart
        daily_pl = {}
        for trade in trade_history:
            date = trade.get('timestamp').strftime('%Y-%m-%d')
            daily_pl[date] = daily_pl.get(date, 0) + float(trade.get('pl', 0))
        
        profit_chart_data = {
            'dates': dates,
            'values': [daily_pl.get(date, 0) for date in dates]
        }

        # Calculate daily volume
        daily_volume = {}
        for trade in trade_history:
            date = trade.get('timestamp').strftime('%Y-%m-%d')
            daily_volume[date] = daily_volume.get(date, 0) + float(trade.get('volume', 0))

        volume_chart_data = {
            'dates': dates,
            'values': [daily_volume.get(date, 0) for date in dates]
        }

        return render_template('index.html',
            trading_stats=trading_stats,
            performance_stats=performance_metrics,
            alert_stats={'active_alerts': len([t for t in active_trades if t.get('has_alert', False)]),
                        'triggered_today': len([t for t in trade_history if t.get('alert_triggered')])},
            market_stats={'active_pairs': len(set(t.get('pair') for t in active_trades)),
                        'market_trend': 'Bullish' if sum(profit_chart_data['values'][-7:]) > 0 else 'Bearish'},
            profit_chart_data=profit_chart_data,
            volume_chart_data=volume_chart_data,
            recent_activities=trade_history[:5],
            active_trades=active_trades,
            account=account_info)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        # Provide default empty values for template variables to avoid template errors
        return render_template('index.html',
                               error=str(e),
                               trading_stats={},
                               performance_stats={
                                   'total_profit': 0,
                                   'profit_change': 0,
                                   'win_rate': 0,
                                   'total_trades': 0
                               },
                               alert_stats={},
                               market_stats={},
                               profit_chart_data={},
                               volume_chart_data={},
                               recent_activities=[],
                               active_trades=[],
                               account={'balance': '$0', 'available': '$0', 'equity': '$0', 'used_margin': '$0'})

@app.route('/trading')
def trading():
    """Trading dashboard route"""
    try:
        active_trades = get_active_trades()
        trade_history = get_trade_history()
        account_info = get_latest_account_info()

        # Defensive check: ensure account_info is a dict and has required keys with numeric defaults
        if not isinstance(account_info, dict):
            account_info = {}
        for key in ['balance', 'available', 'equity', 'used_margin']:
            value = account_info.get(key)
            if value is None:
                account_info[key] = 0.0
            else:
                # Try to convert to float if possible
                try:
                    account_info[key] = float(value)
                except (ValueError, TypeError):
                    account_info[key] = 0.0

        trading_stats = calculate_trading_stats(trade_history)

        return render_template('trading.html',
            active_trades=active_trades,
            trade_history=trade_history,
            trading_stats=trading_stats,
            account=account_info)
    except Exception as e:
        logger.error(f"Error in trading route: {str(e)}")
        return render_template('trading.html', error=str(e))

@app.route('/performance')
def performance():
    """Performance dashboard route"""
    try:
        # Get trade history
        trade_history = get_trade_history(limit=1000)  # Get more trades for better analysis
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(trade_history)

        # Defensive: ensure keys expected by template exist
        if 'total_profit' not in performance_metrics:
            performance_metrics['total_profit'] = 0
        if 'total_profit_amount' not in performance_metrics:
            performance_metrics['total_profit_amount'] = performance_metrics.get('total_profit', 0)
        
        return render_template('performance.html', performance=performance_metrics)
    except Exception as e:
        logger.error(f"Error in performance route: {str(e)}")
        return render_template('performance.html', error=str(e))

@app.route('/alerts')
def alerts():
    """Alerts dashboard route"""
    try:
        # Get alerts data
        alerts = get_alerts()
        alert_history = get_alert_history()
        alert_stats = calculate_alert_stats()
        
        # Process alerts for display
        for alert in alerts:
            alert['status_color'] = {
                'active': 'success',
                'pending': 'warning',
                'triggered': 'info',
                'failed': 'danger'
            }.get(alert.get('status', ''), 'secondary')
            
            alert['created_at'] = alert.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')
        
        # Process alert history for display
        for history in alert_history:
            history['result_color'] = 'success' if history.get('status') == 'triggered' else 'danger'
            history['triggered_at'] = history.get('triggered_at', datetime.now()).strftime('%Y-%m-%d %H:%M')
        
        return render_template('alerts.html',
                             alerts=alerts,
                             alert_history=alert_history,
                             alert_stats=alert_stats)
    except Exception as e:
        logger.error(f"Error in alerts route: {str(e)}")
        return render_template('alerts.html', error=str(e))

@app.route('/strategy')
def strategy():
    """Strategy dashboard route"""
    try:
        # Get strategies data
        strategies = get_strategies()
        strategy_stats = calculate_strategy_stats()
        performance_data = calculate_performance_data()
        risk_data = calculate_risk_data()
        
        # Process strategies for display
        for strategy in strategies:
            strategy['status_color'] = {
                'active': 'success',
                'testing': 'warning',
                'stopped': 'danger'
            }.get(strategy.get('status', ''), 'secondary')
            
            strategy['updated_at'] = strategy.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')
            
            # Format entry/exit conditions for display
            strategy['entry_conditions'] = strategy.get('entry_conditions', [])
            strategy['exit_conditions'] = strategy.get('exit_conditions', [])
            
            # Ensure risk parameters exist
            if 'risk' not in strategy:
                strategy['risk'] = {
                    'stop_loss': 0,
                    'take_profit': 0,
                    'max_position': 0
                }
        
        # Defensive: ensure performance_data is dict and keys are lists, not methods
        if not isinstance(performance_data, dict):
            performance_data = {}
        for key in ['dates', 'values', 'benchmark']:
            if key not in performance_data or callable(performance_data[key]):
                performance_data[key] = []
        
        return render_template('strategy.html',
                             strategies=strategies,
                             strategy_stats=strategy_stats,
                             performance_data=performance_data,
                             risk_data=risk_data)
    except Exception as e:
        logger.error(f"Error in strategy route: {str(e)}")
        return render_template('strategy.html', error=str(e))

@app.route('/market')
def market():
    """Market dashboard route"""
    try:
        # Get market data
        market_data = get_market_data()
        market_stats = calculate_market_stats()
        
        # Get trading pairs
        trading_pairs = list(set(data.get('symbol') for data in market_data))
        
        # Calculate technical indicators for BTC/USDT
        technical_indicators = get_technical_indicators('BTC/USDT')
        
        # Process market data for display
        for data in market_data:
            data['trend_color'] = 'success' if float(data.get('change', 0)) >= 0 else 'danger'
            data['trend'] = 'Bullish' if float(data.get('change', 0)) >= 0 else 'Bearish'
        
        # Prepare chart data
        volume_data = {
            'timestamps': [data['timestamp'].strftime('%Y-%m-%d %H:%M') for data in market_data],
            'values': [float(data.get('volume', 0)) for data in market_data]
        }
        
        trend_data = {
            'timestamps': [data['timestamp'].strftime('%Y-%m-%d %H:%M') for data in market_data],
            'values': [float(data.get('change', 0)) for data in market_data]
        }
        
        return render_template('market.html',
                             market_data=market_data,
                             market_stats=market_stats,
                             trading_pairs=trading_pairs,
                             technical_indicators=technical_indicators,
                             volume_data=volume_data,
                             trend_data=trend_data)
    except Exception as e:
        logger.error(f"Error in market route: {str(e)}")
        return render_template('market.html', error=str(e))

@app.route('/api/trade', methods=['POST'])
def create_trade():
    """API endpoint to create a new trade"""
    try:
        trade_data = request.json
        trade_id = insert_trade(trade_data)
        return jsonify({"status": "success", "trade_id": str(trade_id)})
    except Exception as e:
        logger.error(f"Error creating trade: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        db = get_database()
        db.command('ping')
        return jsonify({"status": "healthy", "database": "connected"})
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """API endpoint to create a new alert"""
    try:
        alert_data = request.json
        alert_data['timestamp'] = datetime.utcnow()
        alert_data['status'] = 'active'
        
        db = get_database()
        result = db.alerts.insert_one(alert_data)
        
        return jsonify({
            "status": "success",
            "alert_id": str(result.inserted_id)
        })
    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """API endpoint to delete an alert"""
    try:
        db = get_database()
        result = db.alerts.delete_one({'_id': ObjectId(alert_id)})
        
        if result.deleted_count > 0:
            return jsonify({"status": "success"})
        else:
            return jsonify({
                "status": "error",
                "message": "Alert not found"
            }), 404
    except Exception as e:
        logger.error(f"Error deleting alert: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """API endpoint to create a new strategy"""
    try:
        strategy_data = request.json
        strategy_data['timestamp'] = datetime.utcnow()
        strategy_data['status'] = 'testing'  # New strategies start in testing mode
        strategy_data['performance'] = 0
        
        db = get_database()
        result = db.strategies.insert_one(strategy_data)
        
        return jsonify({
            "status": "success",
            "strategy_id": str(result.inserted_id)
        })
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/strategies/<strategy_id>', methods=['DELETE'])
def delete_strategy(strategy_id):
    """API endpoint to delete a strategy"""
    try:
        db = get_database()
        result = db.strategies.delete_one({'_id': ObjectId(strategy_id)})
        
        if result.deleted_count > 0:
            return jsonify({"status": "success"})
        else:
            return jsonify({
                "status": "error",
                "message": "Strategy not found"
            }), 404
    except Exception as e:
        logger.error(f"Error deleting strategy: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/market/price')
def get_price_data():
    """API endpoint to get price data for charts"""
    try:
        pair = request.args.get('pair', 'BTC/USDT')
        timeframe = request.args.get('timeframe', '1h')
        
        db = get_database()
        market_data = list(db.market_data.find({
            'symbol': pair,
            'timeframe': timeframe
        }).sort('timestamp', -1).limit(100))
        
        return jsonify({
            'timestamps': [data['timestamp'].strftime('%Y-%m-%d %H:%M') for data in market_data],
            'open': [float(data.get('open', 0)) for data in market_data],
            'high': [float(data.get('high', 0)) for data in market_data],
            'low': [float(data.get('low', 0)) for data in market_data],
            'close': [float(data.get('close', 0)) for data in market_data],
            'volume': [float(data.get('volume', 0)) for data in market_data],
            'volume_colors': ['#4723D9' if float(data.get('close', 0)) >= float(data.get('open', 0)) else '#dc3545' 
                            for data in market_data]
        })
    except Exception as e:
        logger.error(f"Error getting price data: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Initialize database collections
    try:
        init_collections()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 