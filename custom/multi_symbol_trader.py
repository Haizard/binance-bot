from typing import List, Dict
from realtime_trader import RealtimeTrader
from config import Config, TradingConfig
import threading
import time

class MultiSymbolTrader:
    def __init__(self, config: Config):
        self.config = config
        self.traders: Dict[str, RealtimeTrader] = {}
        self.running = False
        self.threads: List[threading.Thread] = []
    
    def initialize(self) -> bool:
        print("\nInitializing traders...")
        if not self.config.trading_configs:
            print("Error: No trading configurations available")
            return False
            
        for trading_config in self.config.trading_configs:
            print(f"\nInitializing trader for {trading_config.symbol}")
            print(f"Strategy: {trading_config.strategy}")
            print(f"Timeframe: {trading_config.timeframe}")
            print(f"Volume: {trading_config.volume}")
            
            trader = RealtimeTrader(
                symbol=trading_config.symbol,
                timeframe=trading_config.timeframe,
                strategy=trading_config.strategy,
                volume=trading_config.volume
            )
            
            try:
                if not trader.initialize(
                    login=self.config.mt5_login,
                    password=self.config.mt5_password,
                    server=self.config.mt5_server
                ):
                    print(f"Failed to initialize trader for {trading_config.symbol}")
                    self.shutdown()
                    return False
                    
                print(f"Successfully initialized trader for {trading_config.symbol}")
                self.traders[trading_config.symbol] = trader
                
            except Exception as e:
                print(f"Error initializing trader for {trading_config.symbol}: {str(e)}")
                self.shutdown()
                return False
        
        print(f"\nSuccessfully initialized {len(self.traders)} traders")
        return True
    
    def _run_trader(self, symbol: str, update_interval: int):
        trader = self.traders[symbol]
        while self.running:
            try:
                trader.run(update_interval=update_interval)
            except Exception as e:
                print(f"Error in {symbol} trader: {e}")
                time.sleep(update_interval)
    
    def run(self, update_interval: int = 60):
        self.running = True
        
        # Start a separate thread for each symbol
        for symbol in self.traders:
            thread = threading.Thread(
                target=self._run_trader,
                args=(symbol, update_interval),
                name=f"Trader-{symbol}"
            )
            thread.start()
            self.threads.append(thread)
        
        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()
    
    def shutdown(self):
        self.running = False
        for trader in self.traders.values():
            trader.shutdown()
        
        # Wait for all threads to complete
        for thread in self.threads:
            thread.join()
