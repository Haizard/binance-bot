"""
Web dashboard for displaying interactive performance charts.
"""
import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import json
import asyncio
from typing import Dict, Any, List
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

app = FastAPI(title="Trading Performance Dashboard")

# Create templates directory if it doesn't exist
os.makedirs("visualization/templates", exist_ok=True)

# Mount static files directory
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup templates
templates = Jinja2Templates(directory="visualization/templates")

# Store active WebSocket connections
websocket_connections: List[WebSocket] = []

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Performance Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .chart-container {
            margin: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .metrics-panel {
            margin: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .metric-card {
            padding: 10px;
            margin: 5px;
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">Trading Performance Dashboard</span>
        </div>
    </nav>
    
    <div class="container-fluid">
        <!-- Performance Metrics -->
        <div class="row metrics-panel">
            <h4>Key Performance Metrics</h4>
            <div class="col" id="metrics-container">
                <!-- Metrics will be inserted here -->
            </div>
        </div>
        
        <!-- Charts -->
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <div id="equity-curve"></div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <div id="drawdown-chart"></div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <div id="win-rate-chart"></div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <div id="profit-distribution"></div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="chart-container">
                    <div id="trade-duration"></div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <div id="position-size"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket("ws://localhost:8000/ws");
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'charts') {
                // Update charts
                Object.entries(data.charts).forEach(([chartId, chartData]) => {
                    Plotly.newPlot(chartId, chartData.data, chartData.layout);
                });
            } else if (data.type === 'metrics') {
                // Update metrics
                updateMetrics(data.metrics);
            }
        };
        
        function updateMetrics(metrics) {
            const container = document.getElementById('metrics-container');
            let html = '<div class="row">';
            
            Object.entries(metrics).forEach(([key, value]) => {
                if (typeof value === 'number') {
                    value = value.toFixed(2);
                }
                html += `
                    <div class="col-md-3">
                        <div class="metric-card">
                            <h6>${key.replace(/_/g, ' ').toUpperCase()}</h6>
                            <h4>${value}</h4>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main dashboard page"""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time updates."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except:
        websocket_connections.remove(websocket)

async def broadcast_charts(charts: Dict[str, Any]):
    """Broadcast chart updates to all connected clients."""
    if not websocket_connections:
        return
        
    # Prepare chart data for broadcast
    chart_data = {
        'type': 'charts',
        'charts': {}
    }
    
    for chart_id, fig in charts.items():
        chart_data['charts'][chart_id] = {
            'data': fig.data,
            'layout': fig.layout
        }
    
    # Broadcast to all connections
    for connection in websocket_connections:
        try:
            await connection.send_json(chart_data)
        except:
            websocket_connections.remove(connection)

async def broadcast_metrics(metrics: Dict[str, Any]):
    """Broadcast metrics updates to all connected clients."""
    if not websocket_connections:
        return
        
    # Prepare metrics data
    metrics_data = {
        'type': 'metrics',
        'metrics': metrics
    }
    
    # Broadcast to all connections
    for connection in websocket_connections:
        try:
            await connection.send_json(metrics_data)
        except:
            websocket_connections.remove(connection)

def start_dashboard(host: str = "0.0.0.0", port: int = 8000):
    """Start the dashboard server."""
    uvicorn.run(app, host=host, port=port)

def load_trading_data(symbol: str, timeframe: str):
    """Load trading data for a specific symbol and timeframe"""
    try:
        data = {}
        
        # Load klines data
        klines_file = f'data/{symbol}_{timeframe}_klines.csv'
        logger.info(f"Loading klines data from {klines_file}")
        
        if os.path.exists(klines_file):
            data['klines'] = pd.read_csv(klines_file)
            if 'timestamp' in data['klines'].columns:
                data['klines']['timestamp'] = pd.to_datetime(data['klines']['timestamp'])
                # Sort by timestamp to ensure proper chart display
                data['klines'] = data['klines'].sort_values('timestamp')
                # Keep only the last 500 candles for better performance
                data['klines'] = data['klines'].tail(500)
                logger.info(f"Successfully loaded {len(data['klines'])} klines")
            else:
                logger.error("Timestamp column not found in klines data")
        else:
            logger.error(f"Klines file not found: {klines_file}")
        
        # Load trades data
        trades_file = f'data/{symbol}_trades.csv'
        if os.path.exists(trades_file):
            data['trades'] = pd.read_csv(trades_file)
            if 'time' in data['trades'].columns:
                data['trades']['time'] = pd.to_datetime(data['trades']['time'])
                # Sort by time
                data['trades'] = data['trades'].sort_values('time')
        
        return data
    except Exception as e:
        logger.error(f"Error loading trading data: {str(e)}")
        return None

def create_price_chart(klines_df):
    """Create price chart with candlesticks"""
    try:
        if klines_df is None or klines_df.empty:
            logger.error("No klines data available for price chart")
            return None
            
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(klines_df['timestamp']):
            klines_df['timestamp'] = pd.to_datetime(klines_df['timestamp'])
            
        # Ensure all required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        if not all(col in klines_df.columns for col in required_columns):
            logger.error(f"Missing required columns in klines data. Required: {required_columns}, Found: {klines_df.columns.tolist()}")
            return None
            
        # Sort data by timestamp to ensure proper display
        klines_df = klines_df.sort_values('timestamp')
        
        # Create figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=klines_df['timestamp'],
            open=klines_df['open'],
            high=klines_df['high'],
            low=klines_df['low'],
            close=klines_df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff3366'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Price Chart',
                font=dict(size=24, color='#ffffff')
            ),
            yaxis_title='Price',
            xaxis_title='Time',
            template='plotly_dark',
            plot_bgcolor='#2a2a2a',
            paper_bgcolor='#2a2a2a',
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            ),
            xaxis=dict(
                gridcolor='#404040',
                showgrid=True,
                rangeslider=dict(visible=False),
                type='date',
                title_font=dict(color='#ffffff'),
                tickfont=dict(color='#ffffff')
            ),
            yaxis=dict(
                gridcolor='#404040',
                showgrid=True,
                side='right',
                title_font=dict(color='#ffffff'),
                tickfont=dict(color='#ffffff'),
                tickformat=',.2f'
            )
        )
        
        logger.info("Successfully created price chart")
        return fig
    except Exception as e:
        logger.error(f"Error creating price chart: {str(e)}")
        return None

def create_volume_chart(klines_df):
    """Create volume chart"""
    try:
        if klines_df is None or klines_df.empty:
            logger.error("No klines data available for volume chart")
            return None
            
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(klines_df['timestamp']):
            klines_df['timestamp'] = pd.to_datetime(klines_df['timestamp'])
            
        # Sort data by timestamp to ensure proper display
        klines_df = klines_df.sort_values('timestamp')
        
        # Calculate colors based on price movement
        colors = ['#00ff88' if close > open else '#ff3366' 
                for close, open in zip(klines_df['close'], klines_df['open'])]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=klines_df['timestamp'],
            y=klines_df['volume'],
            name='Volume',
            marker_color=colors
        ))
        
        fig.update_layout(
            title=dict(
                text='Trading Volume',
                font=dict(size=24, color='#ffffff')
            ),
            yaxis_title='Volume',
            xaxis_title='Time',
            template='plotly_dark',
            plot_bgcolor='#2a2a2a',
            paper_bgcolor='#2a2a2a',
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            ),
            xaxis=dict(
                gridcolor='#404040',
                showgrid=True,
                type='date',
                title_font=dict(color='#ffffff'),
                tickfont=dict(color='#ffffff')
            ),
            yaxis=dict(
                gridcolor='#404040',
                showgrid=True,
                title_font=dict(color='#ffffff'),
                tickfont=dict(color='#ffffff'),
                tickformat=',.2f'
            )
        )
        
        logger.info("Successfully created volume chart")
        return fig
    except Exception as e:
        logger.error(f"Error creating volume chart: {str(e)}")
        return None

def create_pnl_chart(trades_df):
    """Create P&L chart"""
    if trades_df is not None and not trades_df.empty and 'realized_pnl' in trades_df.columns:
        cumulative_pnl = trades_df['realized_pnl'].cumsum()
        
        fig = go.Figure()
        
        # Add area fill under the line
        fig.add_trace(go.Scatter(
            x=trades_df['time'],
            y=cumulative_pnl,
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)',
            line=dict(color='#00ff88', width=2),
            name='Cumulative P&L'
        ))
        
        fig.update_layout(
            title=dict(
                text='Cumulative Profit/Loss',
                font=dict(size=24, color='#ffffff')
            ),
            yaxis_title='P&L',
            xaxis_title='Time',
            template='plotly_dark',
            plot_bgcolor='#2a2a2a',
            paper_bgcolor='#2a2a2a',
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            ),
            xaxis=dict(
                gridcolor='#404040',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='#404040',
                showgrid=True,
                zeroline=True,
                zerolinecolor='#808080'
            )
        )
        
        return fig
    return None

@app.get("/data/{symbol}/{timeframe}")
async def get_chart_data(symbol: str, timeframe: str):
    """Get chart data for a specific symbol and timeframe"""
    try:
        logger.info(f"Loading data for {symbol} {timeframe}")
        data = load_trading_data(symbol, timeframe)
        
        if not data:
            logger.error(f"No data available for {symbol} {timeframe}")
            return JSONResponse(
                content={"error": "No data available. Please ensure the real-time fetcher is running."},
                status_code=404
            )
        
        response = {}
        
        # Create price chart if klines data is available
        if 'klines' in data and not data['klines'].empty:
            klines_df = data['klines']
            price_fig = create_price_chart(klines_df)
            if price_fig:
                response['price_chart'] = price_fig.to_json()
            
            volume_fig = create_volume_chart(klines_df)
            if volume_fig:
                response['volume_chart'] = volume_fig.to_json()
        
        # Create P&L chart if trades data is available
        if 'trades' in data and not data['trades'].empty:
            pnl_fig = create_pnl_chart(data['trades'])
            if pnl_fig:
                response['pnl_chart'] = pnl_fig.to_json()
        
        if not response:
            logger.error("Failed to create any charts")
            return JSONResponse(
                content={"error": "Failed to create charts from available data"},
                status_code=500
            )
        
        logger.info("Successfully created charts")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        return JSONResponse(
            content={"error": f"Error loading data: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 