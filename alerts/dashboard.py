"""
Main dashboard for the trading bot with real-time updates.
"""
import logging
import os
import json
import asyncio
from datetime import datetime, timedelta
import argparse
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from config.database import (
    get_database, get_market_data, get_trade_history,
    get_active_trades, get_performance_metrics,
    update_market_data, close_connections
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global database connection
db = None
db_lock = asyncio.Lock()
update_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    global db, update_task
    
    # Startup
    try:
        # Initialize database connection
        db = await get_db()
        
        # Start background tasks
        update_task = asyncio.create_task(update_market_data_task())
        logger.info("Started background tasks")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        # Shutdown
        if update_task:
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass
        
        if db:
            close_connections()
            db = None
        logger.info("Cleaned up resources")

app = FastAPI(title="Trading Bot Dashboard", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up templates directory
templates_dir = Path("templates")
templates = Jinja2Templates(directory=str(templates_dir))

# Add custom tojson filter to Jinja2 environment
def tojson_filter(obj):
    return json.dumps(obj, default=str)

templates.env.filters["tojson"] = tojson_filter

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._ping_tasks: dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            async with self._lock:
                self.active_connections.append(websocket)
                # Start ping task for this connection
                self._ping_tasks[websocket] = asyncio.create_task(self._ping_client(websocket))
                logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {str(e)}")
            try:
                await websocket.close(code=1006)  # Abnormal closure
            except Exception:
                pass

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                # Cancel and remove ping task
                if websocket in self._ping_tasks:
                    try:
                        self._ping_tasks[websocket].cancel()
                        await self._ping_tasks[websocket]
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error cancelling ping task: {str(e)}")
                    del self._ping_tasks[websocket]
                logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def _ping_client(self, websocket: WebSocket):
        """Send periodic pings to keep connection alive"""
        try:
            while True:
                await asyncio.sleep(30)  # Ping every 30 seconds
                try:
                    await websocket.send_json({"type": "ping"})
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error sending ping: {str(e)}")
                    break
        except asyncio.CancelledError:
            pass
        finally:
            await self.disconnect(websocket)

    async def broadcast(self, message: dict):
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except WebSocketDisconnect:
                    disconnected.append(connection)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {str(e)}")
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for connection in disconnected:
                await self.disconnect(connection)

manager = ConnectionManager()

async def get_db():
    """Get database connection with retry logic"""
    global db
    if db is not None:
        return db
        
    async with db_lock:
        if db is not None:  # Double check after acquiring lock
            return db
            
        max_retries = 3
        retry_delay = 1  # seconds
        
        for i in range(max_retries):
            try:
                # Try local MongoDB first
                db = get_database(remote=False)
                # Test connection
                db.command('ping')
                logger.info("Successfully connected to local MongoDB")
                return db
            except Exception as local_e:
                logger.warning(f"Local MongoDB connection failed: {str(local_e)}")
                try:
                    # Try remote MongoDB as fallback
                    db = get_database(remote=True)
                    db.command('ping')
                    logger.info("Successfully connected to remote MongoDB")
                    return db
                except Exception as remote_e:
                    logger.error(f"Remote MongoDB connection failed: {str(remote_e)}")
                    if i < max_retries - 1:  # Don't sleep on last attempt
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
        
        raise HTTPException(status_code=500, detail="Could not connect to database")

async def update_market_data_task():
    """Background task to update market data and broadcast to clients"""
    while True:
        try:
            if not manager.active_connections:
                await asyncio.sleep(1)
                continue

            # Get database connection
            db = await get_db()
            if not db:
                logger.error("No database connection available")
                await asyncio.sleep(5)
                continue
            
            # Get latest data
            trades = list(db.trades.find().sort("timestamp", -1).limit(50))
            active_trades = list(db.active_trades.find())
            alerts = list(db.alerts.find().sort("timestamp", -1))
            
            # Calculate performance stats
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('realized_pnl', 0) > 0)
            total_profit = sum(t.get('realized_pnl', 0) for t in trades)
            
            # Convert ObjectId to string for JSON serialization
            for trade in trades + active_trades:
                trade['_id'] = str(trade['_id'])
                if 'timestamp' in trade:
                    trade['timestamp'] = trade['timestamp'].isoformat()
            
            for alert in alerts:
                alert['_id'] = str(alert['_id'])
                if 'timestamp' in alert:
                    alert['timestamp'] = alert['timestamp'].isoformat()
            
            # Group alerts by status
            active_alerts = [a for a in alerts if a.get('status') == 'ACTIVE']
            triggered_alerts = [a for a in alerts if a.get('status') == 'TRIGGERED']
            completed_alerts = [a for a in alerts if a.get('status') == 'COMPLETED']
            
            # Prepare update message
            update = {
                'type': 'update',
                'data': {
                    'performance_stats': {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                        'total_profit': total_profit,
                        'active_positions': len(active_trades)
                    },
                    'recent_trades': trades[:10],
                    'active_trades': active_trades,
                    'alerts': {
                        'active': active_alerts,
                        'triggered': triggered_alerts,
                        'completed': completed_alerts,
                        'stats': {
                            'total_alerts': len(alerts),
                            'active_alerts': len(active_alerts),
                            'triggered_alerts': len(triggered_alerts),
                            'completed_alerts': len(completed_alerts)
                        }
                    }
                }
            }
            
            # Broadcast update to all connected clients
            await manager.broadcast(update)
            
        except Exception as e:
            logger.error(f"Error in market data update task: {str(e)}")
        
        # Wait before next update
        await asyncio.sleep(5)  # Update every 5 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the application starts"""
    try:
        # Initialize database connection
        await get_db()
        
        # Start background tasks
        asyncio.create_task(update_market_data_task())
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the application shuts down"""
    global db
    if db is not None:
        close_connections()
        db = None

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Render the main dashboard page"""
    try:
        # Get trading data from MongoDB
        db = await get_db()
        logger.info("Getting trading data from database")
        
        # Get active trades and ensure they're serializable
        active_trades_cursor = db.trades.find({"status": "active"})
        active_trades = []
        for trade in active_trades_cursor:
            trade_data = {
                "id": str(trade.get("_id", "")),
                "symbol": str(trade.get("symbol", "")),
                "side": str(trade.get("side", "")),
                "entry_price": float(trade.get("entry_price", 0.0)),
                "quantity": float(trade.get("quantity", 0.0)),
                "realized_pnl": float(trade.get("realized_pnl", 0.0)),
                "status": str(trade.get("status", "unknown"))
            }
            # Handle timestamp
            try:
                timestamp = trade.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.utcnow()
                trade_data["timestamp"] = timestamp.isoformat()
            except Exception:
                trade_data["timestamp"] = datetime.utcnow().isoformat()
            active_trades.append(trade_data)

        # Get trade history and ensure it's serializable
        trade_history_cursor = db.trades.find().sort("timestamp", -1).limit(50)
        trade_history = []
        current_date = datetime.utcnow().date()
        daily_pnl = 0.0
        
        for trade in trade_history_cursor:
            trade_data = {
                "id": str(trade.get("_id", "")),
                "symbol": str(trade.get("symbol", "")),
                "side": str(trade.get("side", "")),
                "entry_price": float(trade.get("entry_price", 0.0)),
                "exit_price": float(trade.get("exit_price", 0.0)),
                "quantity": float(trade.get("quantity", 0.0)),
                "realized_pnl": float(trade.get("realized_pnl", 0.0)),
                "status": str(trade.get("status", "unknown"))
            }
            
            # Handle timestamp and calculate daily PNL
            try:
                timestamp = trade.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.utcnow()
                
                trade_data["timestamp"] = timestamp.isoformat()
                
                # Add to daily PNL if trade is from today
                if timestamp.date() == current_date:
                    daily_pnl += float(trade.get("realized_pnl", 0.0))
                    
            except Exception:
                trade_data["timestamp"] = datetime.utcnow().isoformat()
            
            trade_history.append(trade_data)
        
        # Calculate performance stats with safe type conversion
        total_trades = len(trade_history)
        winning_trades = sum(1 for t in trade_history if float(t.get("realized_pnl", 0)) > 0)
        losing_trades = sum(1 for t in trade_history if float(t.get("realized_pnl", 0)) < 0)
        total_profit = sum(float(t.get("realized_pnl", 0)) for t in trade_history)
        
        performance_stats = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
            "total_profit": float(total_profit),
            "average_profit": float(total_profit / total_trades) if total_trades > 0 else 0.0,
            "profit_factor": 0.0,  # Calculate if needed
            "daily_profit": daily_pnl
        }
        
        # Get recent trades (already sorted in trade_history)
        recent_trades = trade_history[:10]
        
        # Verify data is JSON serializable (excluding request object)
        data_to_verify = {
            "performance_stats": performance_stats,
            "active_trades": active_trades,
            "recent_trades": recent_trades
        }
        
        try:
            json.dumps(data_to_verify)
            logger.info("Template data is JSON serializable")
        except TypeError as e:
            logger.error(f"Template data is not JSON serializable: {e}")
            raise HTTPException(status_code=500, detail="Data serialization error")
        
        # Add request object only when returning template response
        return templates.TemplateResponse("index.html", {
            "request": request,  # FastAPI Request object
            **data_to_verify   # Spread the verified data
        })
        
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Render the alerts page"""
    try:
        # Get alerts from MongoDB
        db = get_database()
        alerts = list(db.alerts.find().sort("timestamp", -1).limit(50))
        
        # Convert ObjectId to string for each alert
        for alert in alerts:
            alert['_id'] = str(alert['_id'])
        
        # Group alerts by status
        active_alerts = [a for a in alerts if a.get('status') == 'ACTIVE']
        triggered_alerts = [a for a in alerts if a.get('status') == 'TRIGGERED']
        completed_alerts = [a for a in alerts if a.get('status') == 'COMPLETED']
        
        return templates.TemplateResponse("alerts.html", {
            "request": request,
            "active_alerts": active_alerts,
            "triggered_alerts": triggered_alerts,
            "completed_alerts": completed_alerts,
            "alert_stats": {
                "total_alerts": len(alerts),
                "active_alerts": len(active_alerts),
                "triggered_alerts": len(triggered_alerts),
                "completed_alerts": len(completed_alerts)
            }
        })
    except Exception as e:
        logger.error(f"Error rendering alerts page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance", response_class=HTMLResponse)
async def performance_page(request: Request):
    """Render the performance page"""
    try:
        # Get trading data from MongoDB
        db = get_database()
        trades = list(db.trades.find().sort("timestamp", -1).limit(100))
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('realized_pnl', 0) < 0]
        
        total_profit = sum(t.get('realized_pnl', 0) for t in trades)
        winning_profit = sum(t.get('realized_pnl', 0) for t in winning_trades)
        losing_profit = abs(sum(t.get('realized_pnl', 0) for t in losing_trades))
        
        # Calculate daily performance
        daily_performance = {}
        for trade in trades:
            date = trade['timestamp'].date()
            if date not in daily_performance:
                daily_performance[date] = {
                    'profit': 0,
                    'trades': 0,
                    'winning_trades': 0
                }
            daily_performance[date]['profit'] += trade.get('realized_pnl', 0)
            daily_performance[date]['trades'] += 1
            if trade.get('realized_pnl', 0) > 0:
                daily_performance[date]['winning_trades'] += 1
        
        # Convert to list and sort by date
        daily_stats = [
            {
                'date': date,
                'profit': stats['profit'],
                'trades': stats['trades'],
                'win_rate': (stats['winning_trades'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            }
            for date, stats in daily_performance.items()
        ]
        daily_stats.sort(key=lambda x: x['date'], reverse=True)
        
        return templates.TemplateResponse("performance.html", {
            "request": request,
            "performance_stats": {
                "total_trades": total_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0,
                "total_profit": total_profit,
                "average_profit": total_profit / total_trades if total_trades > 0 else 0,
                "profit_factor": (winning_profit / losing_profit) if losing_profit > 0 else 0,
                "largest_win": max((t.get('realized_pnl', 0) for t in winning_trades), default=0),
                "largest_loss": min((t.get('realized_pnl', 0) for t in losing_trades), default=0),
                "average_win": winning_profit / len(winning_trades) if winning_trades else 0,
                "average_loss": losing_profit / len(losing_trades) if losing_trades else 0
            },
            "daily_performance": daily_stats,
            "recent_trades": trades[:20]  # Show last 20 trades
        })
    except Exception as e:
        logger.error(f"Error rendering performance page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/custom", response_class=HTMLResponse)
async def custom_dashboard(request: Request):
    """Render the custom data dashboard page"""
    try:
        # Get available symbols and timeframes
        db = get_database()
        market_data = list(db.market_data.find().sort("timestamp", -1).limit(100))
        
        # Get unique symbols and timeframes
        symbols = sorted(list(set(data['symbol'] for data in market_data)))
        timeframes = sorted(list(set(data['timeframe'] for data in market_data)))
        
        # Get latest market data for default symbol and timeframe
        default_symbol = symbols[0] if symbols else "BTCUSDT"
        default_timeframe = timeframes[0] if timeframes else "1m"
        
        latest_data = list(db.market_data.find({
            "symbol": default_symbol,
            "timeframe": default_timeframe
        }).sort("timestamp", -1).limit(100))
        
        # Format data for charts
        chart_data = {
            "timestamps": [d['timestamp'].isoformat() for d in latest_data],
            "prices": [d['close'] for d in latest_data],
            "volumes": [d['volume'] for d in latest_data]
        }
        
        return templates.TemplateResponse("custom_dashboard.html", {
            "request": request,
            "symbols": symbols,
            "timeframes": timeframes,
            "chart_data": chart_data,
            "current_symbol": default_symbol,
            "current_timeframe": default_timeframe
        })
    except Exception as e:
        logger.error(f"Error rendering custom dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading", response_class=HTMLResponse)
async def trading_page(request: Request):
    """Render the trading page"""
    try:
        db = await get_db()
        
        # Get trading data and convert to proper format
        trades_cursor = db.trades.find().sort("timestamp", -1).limit(100)
        trades = []
        for trade in trades_cursor:
            trades.append({
                'timestamp': trade.get('timestamp', datetime.now()),
                'symbol': str(trade.get('symbol', '')),
                'type': str(trade.get('type', '')),
                'entry_price': float(trade.get('entry_price', 0.0)),
                'exit_price': float(trade.get('exit_price', 0.0)),
                'size': float(trade.get('quantity', 0.0)),  # Using quantity as size
                'realized_pnl': float(trade.get('realized_pnl', 0.0)),
                'duration': str(trade.get('duration', '')),
                'strategy': str(trade.get('strategy', ''))
            })

        # Convert active trades
        active_trades_cursor = db.active_trades.find()
        active_trades = []
        for trade in active_trades_cursor:
            active_trades.append({
                'symbol': str(trade.get('symbol', '')),
                'type': str(trade.get('type', '')),
                'entry_price': float(trade.get('entry_price', 0.0)),
                'current_price': float(trade.get('current_price', 0.0)),
                'size': float(trade.get('quantity', 0.0)),  # Using quantity as size
                'unrealized_pnl': float(trade.get('unrealized_pnl', 0.0)),
                'duration': str(trade.get('duration', '')),
                'strategy': str(trade.get('strategy', ''))
            })
        
        # Calculate trading stats
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['realized_pnl'] > 0)
        total_profit = sum(t['realized_pnl'] for t in trades)
        
        trading_stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_profit': total_profit,
            'average_profit': total_profit / total_trades if total_trades > 0 else 0,
            'active_positions': len(active_trades)
        }
        
        return templates.TemplateResponse("trading.html", {
            "request": request,
            "trading_stats": trading_stats,
            "active_trades": active_trades,
            "recent_trades": trades[:20]
        })
    except Exception as e:
        logger.error(f"Error rendering trading page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market", response_class=HTMLResponse)
async def market_page(request: Request):
    """Render the market data page"""
    try:
        db = await get_db()
        
        # Get market data
        market_data = list(db.market_data.find().sort("timestamp", -1).limit(100))
        
        # Get unique symbols and timeframes
        symbols = sorted(list(set(data['symbol'] for data in market_data)))
        timeframes = sorted(list(set(data['timeframe'] for data in market_data)))
        
        # Get latest data for each symbol
        latest_prices = {}
        for symbol in symbols:
            latest = db.market_data.find_one({"symbol": symbol}, sort=[("timestamp", -1)])
            if latest:
                latest_prices[symbol] = {
                    'price': latest.get('close', 0),
                    'change': latest.get('price_change_24h', 0),
                    'volume': latest.get('volume', 0)
                }
        
        return templates.TemplateResponse("market.html", {
            "request": request,
            "symbols": symbols,
            "timeframes": timeframes,
            "latest_prices": latest_prices,
            "market_data": market_data[:20]  # Show last 20 data points
        })
    except Exception as e:
        logger.error(f"Error rendering market page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategy", response_class=HTMLResponse)
async def strategy_page(request: Request):
    """Render the strategy page"""
    try:
        # Get database connection
        db = await get_db()
        logger.info("Database connection established")
        
        # Get strategies data and ensure ObjectId is converted to string
        strategies = []
        cursor = db.strategies.find()
        strategy_list = list(cursor)  # Convert cursor to list first
        logger.info(f"Found {len(strategy_list)} strategies")
        
        for strategy in strategy_list:
            try:
                # Basic strategy data with safe type conversion
                base_data = {
                    'id': str(strategy.get('_id', '')),
                    'name': str(strategy.get('name', 'Unnamed Strategy')),
                    'type': str(strategy.get('type', 'Unknown')),
                    'status': str(strategy.get('status', 'inactive')),
                    'performance': 0.0  # Default value
                }
                
                # Safely convert performance to float
                try:
                    base_data['performance'] = float(strategy.get('performance', 0))
                except (TypeError, ValueError):
                    base_data['performance'] = 0.0
                
                # Safely convert lists
                base_data['pairs'] = [str(p) for p in strategy.get('pairs', [])] if isinstance(strategy.get('pairs'), list) else []
                base_data['timeframes'] = [str(t) for t in strategy.get('timeframes', [])] if isinstance(strategy.get('timeframes'), list) else []
                base_data['entry_conditions'] = [str(c) for c in strategy.get('entry_conditions', [])] if isinstance(strategy.get('entry_conditions'), list) else []
                base_data['exit_conditions'] = [str(c) for c in strategy.get('exit_conditions', [])] if isinstance(strategy.get('exit_conditions'), list) else []
                
                # Safely handle risk data
                risk_dict = strategy.get('risk', {})
                if not isinstance(risk_dict, dict):
                    risk_dict = {}
                    
                base_data['risk'] = {
                    'stop_loss': float(risk_dict.get('stop_loss', 0)),
                    'take_profit': float(risk_dict.get('take_profit', 0)),
                    'max_position': float(risk_dict.get('max_position', 0))
                }
                
                # Handle timestamp
                try:
                    timestamp = strategy.get('timestamp', datetime.now())
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    base_data['updated_at'] = timestamp.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    base_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                # Add status color
                base_data['status_color'] = {
                    'active': 'success',
                    'testing': 'warning',
                    'stopped': 'danger'
                }.get(base_data['status'].lower(), 'secondary')
                
                strategies.append(base_data)
                
            except Exception as e:
                logger.error(f"Error processing strategy: {e}")
                continue
        
        logger.info(f"Processed {len(strategies)} strategies successfully")
        
        # Simple statistics with safe type conversion
        strategy_stats = {
            'active': sum(1 for s in strategies if s['status'].lower() == 'active'),
            'profitable': sum(1 for s in strategies if s['performance'] > 0),
            'testing': sum(1 for s in strategies if s['status'].lower() == 'testing'),
            'stopped': sum(1 for s in strategies if s['status'].lower() == 'stopped')
        }
        
        # Generate dates for the last 30 days
        dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(30)]
        
        # Safe performance data
        performance_data = {
            'dates': dates,
            'values': [0.0] * 30,
            'benchmark': [0.0] * 30
        }
        
        # Safe risk data
        risk_data = {
            'strategies': [s['name'] for s in strategies],
            'sharpe_ratios': [0.0] * len(strategies)
        }
        
        # Prepare data for verification (excluding request object)
        data_to_verify = {
            "strategies": strategies,
            "strategy_stats": {
                'active': int(strategy_stats['active']),
                'profitable': int(strategy_stats['profitable']),
                'testing': int(strategy_stats['testing']),
                'stopped': int(strategy_stats['stopped'])
            },
            "performance_data": {
                'dates': [str(date) for date in performance_data['dates']],
                'values': [float(val) for val in performance_data['values']],
                'benchmark': [float(val) for val in performance_data['benchmark']]
            },
            "risk_data": {
                'strategies': [str(s) for s in risk_data['strategies']],
                'sharpe_ratios': [float(ratio) for ratio in risk_data['sharpe_ratios']]
            }
        }
        
        # Verify data is JSON serializable
        try:
            json.dumps(data_to_verify, default=str)
            logger.info("Template data is JSON serializable")
        except TypeError as e:
            logger.error(f"Template data is not JSON serializable: {e}")
            raise HTTPException(status_code=500, detail="Data serialization error")
        
        # Add request object only when returning template response
        return templates.TemplateResponse("strategy.html", {
            "request": request,  # FastAPI Request object
            **data_to_verify   # Spread the verified data
        })
        
    except Exception as e:
        logger.error(f"Error rendering strategy page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    try:
        await manager.connect(websocket)
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
                elif message.get('type') == 'subscribe':
                    # Handle subscription requests if needed
                    pass
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                break
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        await manager.disconnect(websocket)

@app.get("/api/market_data")
async def get_market_data(symbol: str, timeframe: str):
    """Get market data for a specific symbol and timeframe"""
    try:
        db = await get_db()
        
        # Get market data
        data = list(db.market_data.find(
            {"symbol": symbol, "timeframe": timeframe},
            {"_id": 0, "timestamp": 1, "close": 1}
        ).sort("timestamp", -1).limit(100))
        
        # Format data for chart
        return {
            "timestamps": [d["timestamp"].isoformat() for d in data],
            "prices": [d["close"] for d in data]
        }
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the dashboard application"""
    parser = argparse.ArgumentParser(description='Trading Bot Dashboard')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    args = parser.parse_args()
    
    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
