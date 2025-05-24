from mt5_connector import MT5Connector
from mp_support_resist import support_resistance_levels, sr_penetration_signal
from pip_pattern_miner import PIPPatternMiner, find_pips  # Added find_pips import
from wf_pip_miner import WFPIPMiner
import time
import numpy as np
import pandas as pd

class RealtimeTrader:
    def __init__(self, symbol: str, timeframe: str, strategy: str, volume: float):
        self.mt5 = MT5Connector()
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy
        self.volume = volume
        self.position_open = False
        
        # Initialize strategy-specific components
        if strategy == "support_resistance":
            self.lookback = 1440  # Changed from 365 to 1440 minutes (24 hours)
            self.first_w = 1.0
            self.atr_mult = 3.0
        elif strategy == "pip_pattern":
            self.pip_miner = PIPPatternMiner(
                n_pips=5,
                lookback=60,  # Changed from 24 to 60 minutes
                hold_period=15  # Changed from 6 to 15 minutes
            )
        elif strategy == "wf_pip":
            self.wf_miner = WFPIPMiner(
                n_pips=5,
                lookback=60,  # Changed from 24 to 60 minutes
                hold_period=15,  # Changed from 6 to 15 minutes
                train_size=1440 * 7,  # Changed to 7 days worth of minutes
                step_size=1440  # Changed to 1 day worth of minutes
            )
    
    def initialize(self, login: int, password: str, server: str):
        return self.mt5.initialize(login, password, server)
    
    def process_data(self, data: pd.DataFrame) -> float:
        try:
            if self.strategy == "support_resistance":
                levels = support_resistance_levels(
                    data, self.lookback, 
                    first_w=self.first_w, 
                    atr_mult=self.atr_mult
                )
                signal = sr_penetration_signal(data, levels)
                print(f"Support/Resistance signal strength: {signal[-1]:.4f}")
                # Return normalized signal between -1 and 1
                return np.clip(signal[-1] / 100, -1, 1)
                
            elif self.strategy == "pip_pattern":
                try:
                    self.pip_miner.train(np.log(data['close'].values))
                    pips_x, pips_y = find_pips(
                        np.log(data['close'].values[-24:]), 
                        5, 3
                    )
                    raw_signal = self.pip_miner.predict(pips_y)
                    print(f"PIP Pattern raw signal: {raw_signal:.4f}")
                    # Normalize and add threshold
                    return np.sign(raw_signal) if abs(raw_signal) > 15 else 0
                except RuntimeWarning as w:
                    print(f"Warning in PIP pattern calculation: {str(w)}")
                    return 0.0
                
            elif self.strategy == "wf_pip":
                arr = np.log(data['close'].values)
                raw_signal = self.wf_miner.update_signal(arr, len(arr)-1)
                print(f"WF PIP raw signal: {raw_signal:.4f}")
                # Normalize and add threshold
                return np.sign(raw_signal) if abs(raw_signal) > 20 else 0
                
        except Exception as e:
            print(f"Error processing data for {self.strategy}: {str(e)}")
            return 0.0
    
    def run(self, update_interval: int = 60):
        while True:
            try:
                print(f"\nChecking {self.symbol} for trading opportunities...")
                data = self.mt5.get_historical_data(
                    self.symbol, 
                    self.timeframe, 
                    1000
                )
                
                if data is None or len(data) == 0:
                    print(f"No data available for {self.symbol}, waiting {update_interval} seconds...")
                    time.sleep(update_interval)
                    continue
                
                # Get trading signal
                signal = self.process_data(data)
                print(f"Final normalized signal for {self.symbol}: {signal:.4f}")
                
                # Check if signal is strong enough to trade
                if abs(signal) < 0.5:  # Require at least 0.5 strength to trade
                    print(f"Signal not strong enough for {self.symbol}")
                    time.sleep(update_interval)
                    continue
                    
                current_price = data['close'].iloc[-1]
                print(f"Current price: {current_price:.5f}")
                
                if signal > 0 and not self.position_open:
                    print(f"Strong BUY signal ({signal:.4f}) detected for {self.symbol}")
                    stop_loss = current_price * 0.98
                    take_profit = current_price * 1.04
                    
                    success = self.mt5.place_order(
                        self.symbol, "BUY", self.volume,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    if success:
                        print(f"Successfully opened BUY position for {self.symbol}")
                        self.position_open = True
                    else:
                        print(f"Failed to open BUY position for {self.symbol}")
                
                elif signal < 0 and not self.position_open:
                    print(f"Strong SELL signal ({signal:.4f}) detected for {self.symbol}")
                    stop_loss = current_price * 1.02
                    take_profit = current_price * 0.96
                    
                    success = self.mt5.place_order(
                        self.symbol, "SELL", self.volume,
                        price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    if success:
                        print(f"Successfully opened SELL position for {self.symbol}")
                        self.position_open = True
                    else:
                        print(f"Failed to open SELL position for {self.symbol}")
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"Error in {self.symbol} trader: {str(e)}")
                print(f"Waiting {update_interval} seconds before retry...")
                time.sleep(update_interval)
    
    def shutdown(self):
        self.mt5.shutdown()






