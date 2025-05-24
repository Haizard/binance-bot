from dataclasses import dataclass
from typing import List
from dotenv import dotenv_values

@dataclass
class TradingConfig:
    symbol: str
    timeframe: str
    strategy: str
    volume: float

class Config:
    def __init__(self):
        # Load values directly without affecting environment
        config = dotenv_values(".env")
        
        # Debug prints
        print("Debug: Loading configuration...")
        print(f"MT5_LOGIN value from .env: '{config.get('MT5_LOGIN')}'")
        print(f"MT5_PASSWORD value from .env: '{config.get('MT5_PASSWORD')}'")
        print(f"MT5_SERVER value from .env: '{config.get('MT5_SERVER')}'")
        print(f"TRADING_PAIRS value from .env: '{config.get('TRADING_PAIRS')}'")
        
        self.mt5_login = config.get('MT5_LOGIN')
        self.mt5_password = config.get('MT5_PASSWORD')
        self.mt5_server = config.get('MT5_SERVER')
        
        # Parse trading configurations from environment
        self.trading_configs = self._parse_trading_pairs(config.get('TRADING_PAIRS', ''))
        
        # Debug print trading configs
        print("\nParsed Trading Configurations:")
        for cfg in self.trading_configs:
            print(f"Symbol: {cfg.symbol}, Timeframe: {cfg.timeframe}, Strategy: {cfg.strategy}, Volume: {cfg.volume}")
    
    def _parse_trading_pairs(self, trading_pairs_str: str) -> List[TradingConfig]:
        if not trading_pairs_str:
            print("Warning: No trading pairs configured")
            return []
            
        trading_pairs = trading_pairs_str.split(',')
        configs = []
        
        for pair in trading_pairs:
            if not pair:
                continue
            try:
                symbol, timeframe, strategy, volume = pair.strip().split(':')
                config = TradingConfig(
                    symbol=symbol.strip(),
                    timeframe=timeframe.strip(),
                    strategy=strategy.strip(),
                    volume=float(volume.strip())
                )
                configs.append(config)
            except ValueError as e:
                print(f"Warning: Invalid trading pair format: {pair}")
                print(f"Error details: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing trading pair {pair}: {str(e)}")
                continue
        
        if not configs:
            print("Warning: No valid trading configurations found")
            
        return configs
