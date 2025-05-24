from multi_symbol_trader import MultiSymbolTrader
from config import Config
import traceback

def main():
    try:
        # Load configuration
        config = Config()
        
        # Initialize multi-symbol trader
        trader = MultiSymbolTrader(config)
        
        if not trader.initialize():
            print("Failed to initialize traders")
            return
        
        try:
            # Start trading - check every 5 seconds instead of 60
            trader.run(update_interval=5)  
        except KeyboardInterrupt:
            print("Stopping traders...")
        finally:
            trader.shutdown()
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()


