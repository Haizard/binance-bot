import asyncio
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import linalg as la
from binance import AsyncClient
from datetime import datetime, timedelta
import logging

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logger = logging.getLogger(__name__)

def pca_linear_model(x: pd.DataFrame, y: pd.Series, n_components: int, thresh: float= 0.01):
    # Center data at 0
    means = x.mean()
    x -= means
    x = x.dropna()

    # Find covariance and compute eigen vectors
    cov = np.cov(x, rowvar=False)
    evals , evecs = la.eigh(cov)
    # Sort eigenvectors by size of eigenvalue
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    # Create data set for model
    model_data = pd.DataFrame()
    for j in range(n_components):
         model_data['PC' + str(j)] = pd.Series( np.dot(x, evecs[j]) , index=x.index)
    
    cols = list(model_data.columns)
    model_data['target'] = y
    model_coefs = la.lstsq(model_data[cols], y)[0]
    model_data['pred'] = np.dot( model_data[cols], model_coefs)

    l_thresh = model_data['pred'].quantile(0.99)
    s_thresh = model_data['pred'].quantile(0.01)

    return model_coefs, evecs, means, l_thresh

async def train_and_save_rsi_pca_model():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)

    symbol = 'BTCUSDT' # Symbol to train on (can be configured)
    interval = AsyncClient.KLINE_INTERVAL_1HOUR # Interval for training data
    # Define the date range for training data
    # Using a recent historical period as an example
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2) # 2 years of data

    # Model parameters - should ideally match what's used in the agent
    rsi_periods = list(range(2, 25))
    n_components = 3
    lookahead = 6 # Should match the lookahead used in strategy development
    train_size_days = 365 * 1 # 1 year training window within the data fetch range

    logger.info(f"Fetching {interval} klines for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    # Fetch historical klines
    klines = await client.get_historical_klines(
        symbol,
        interval,
        start_date.strftime('%d %b, %Y %H:%M:%S'),
        end_date.strftime('%d %b, %Y %H:%M:%S')
    )
    await client.close_connection()
    logger.info(f"Fetched {len(klines)} klines.")

    if not klines:
        logger.error("No klines data fetched. Cannot train model.")
        return

    # Convert klines to DataFrame
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', \
                                       'close_time', 'quote_asset_volume', 'number_of_trades', \
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time')

    # Calculate RSI for multiple periods
    rsis_df = pd.DataFrame()
    for p in rsi_periods:
        rsis_df[p] = ta.rsi(df['close'], length=p)

    # Define the target variable (future price change)
    target = np.log(df['close']).diff(lookahead).shift(-lookahead)

    # Combine RSI and target, drop NaNs
    model_data = rsis_df.copy()
    model_data['target'] = target
    model_data = model_data.dropna()

    if model_data.empty:
        logger.error("Not enough data after calculating indicators and target to train model.")
        return

    # Select training window
    train_end_date = start_date + timedelta(days=train_size_days)
    train_data = model_data[model_data.index < train_end_date].copy()

    if train_data.empty:
        logger.error("No data in the training window. Cannot train model.")
        return

    x_train = train_data.drop('target', axis=1)
    y_train = train_data['target']

    logger.info("Training PCA-RSI model...")
    # Train the PCA-RSI model
    model_coefs, evecs, means, l_thresh = pca_linear_model(x_train, y_train, n_components)
    logger.info("Model training complete.")

    # Define paths to save parameters
    model_dir = 'models/rsi_pca'
    evecs_path = os.path.join(model_dir, 'pca_evecs.npy')
    means_path = os.path.join(model_dir, 'pca_means.npy')
    coefs_path = os.path.join(model_dir, 'model_coefs.npy')
    l_thresh_path = os.path.join(model_dir, 'long_threshold.txt')

    logger.info(f"Saving model parameters to {model_dir}...")
    # Save the parameters
    np.save(evecs_path, evecs)
    np.save(means_path, means)
    np.save(coefs_path, model_coefs)
    with open(l_thresh_path, 'w') as f:
        f.write(str(l_thresh))
    logger.info("Model parameters saved.")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(train_and_save_rsi_pca_model()) 