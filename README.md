# Trading Bot Dashboard

A comprehensive web-based dashboard system for monitoring and managing automated trading strategies.

## Features

### 1. Performance Dashboard
- Real-time performance metrics (total return, win rate, profit factor, Sharpe ratio)
- Equity curve and drawdown analysis
- Trade statistics and monthly performance breakdown
- Time-based analysis

### 2. Alerts Dashboard
- Alert management with CRUD operations
- Real-time alert monitoring
- Alert history tracking
- Customizable alert conditions

### 3. Strategy Dashboard
- Strategy management and performance tracking
- Risk analysis with Sharpe ratio calculations
- Strategy parameters configuration
- Performance comparison with benchmarks

### 4. Market Dashboard
- Real-time market overview
- Interactive price charts with multiple timeframes
- Volume and trend analysis
- Technical indicators (RSI, MACD, Moving Averages)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Haizard/working-algo-bot.git
cd working-algo-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure MongoDB connection:
Update the MongoDB connection URL in `config/database.py`

4. Start the dashboard:
```bash
python start_dashboards.py
```

## Configuration

- MongoDB connection settings in `config/database.py`
- Port configuration in `config/ports.yaml`

## Dependencies

- Flask
- MongoDB
- Plotly.js
- Bootstrap
- Boxicons

## License

MIT License

## Author

Haizard 