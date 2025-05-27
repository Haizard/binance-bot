import os
import yaml
from binance.client import Client

def check_binance_balance():
    """Checks the Binance account balance using API keys from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        api_keys = config.get('api', {})
        api_key = api_keys.get('api_key')
        api_secret = api_keys.get('api_secret')

        if not api_key or not api_secret:
            print("Error: Binance API keys (api_key and api_secret) not found in config.yaml under the 'api' section.")
            return

        client = Client(api_key, api_secret)

        print("Successfully connected to Binance API. Fetching balances...")
        
        # Get account information
        account_info = client.get_account()
        
        print("\n--- Account Balances ---")
        for balance in account_info['balances']:
            # Only print non-zero balances
            if float(balance['free']) > 0 or float(balance['locked']) > 0:
                print(f"{balance['asset']}: Free={balance['free']}, Locked={balance['locked']}")
        print("------------------------")

    except Exception as e:
        print(f"An error occurred while checking Binance balance: {e}")

if __name__ == "__main__":
    check_binance_balance() 