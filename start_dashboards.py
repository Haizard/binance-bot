import subprocess
import yaml
import time
import os
from rich.console import Console
from rich.table import Table

console = Console()

def load_port_config():
    with open('config/ports.yaml', 'r') as f:
        return yaml.safe_load(f)

def start_dashboard(name, module, port):
    console.print(f"Starting {name} on port {port}...")
    return subprocess.Popen(['python', '-m', module, '--port', str(port)], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)

def main():
    # Load port configuration
    ports = load_port_config()
    
    # Create status table
    table = Table(title="Dashboard Status")
    table.add_column("Dashboard")
    table.add_column("URL")
    table.add_column("Status")
    
    # Start each dashboard component
    processes = []
    
    # Main Trading Dashboard
    proc = start_dashboard("Main Trading Dashboard", 
                         "alerts.dashboard", 
                         ports['main_dashboard'])
    processes.append(proc)
    table.add_row("Main Trading", f"http://localhost:{ports['main_dashboard']}", "✅ Running")
    
    # Performance Dashboard
    proc = start_dashboard("Performance Dashboard", 
                         "visualization.performance_dashboard", 
                         ports['performance_dashboard'])
    processes.append(proc)
    table.add_row("Performance", f"http://localhost:{ports['performance_dashboard']}", "✅ Running")
    
    # Alert Dashboard
    proc = start_dashboard("Alert Dashboard", 
                         "alerts.monitor", 
                         ports['alert_dashboard'])
    processes.append(proc)
    table.add_row("Alerts", f"http://localhost:{ports['alert_dashboard']}", "✅ Running")
    
    # Strategy Dashboard
    proc = start_dashboard("Strategy Dashboard", 
                         "analysis.strategy_dashboard", 
                         ports['strategy_dashboard'])
    processes.append(proc)
    table.add_row("Strategy Analysis", f"http://localhost:{ports['strategy_dashboard']}", "✅ Running")
    
    # Market Data Dashboard
    proc = start_dashboard("Market Data Dashboard", 
                         "data.market_dashboard", 
                         ports['market_dashboard'])
    processes.append(proc)
    table.add_row("Market Data", f"http://localhost:{ports['market_dashboard']}", "✅ Running")
    
    # Display dashboard information
    console.print("\n=== Trading Bot Dashboards ===\n")
    console.print(table)
    console.print("\nAll dashboards are now running. Press Ctrl+C to stop all dashboards.\n")
    
    try:
        # Keep the script running and monitor processes
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\nStopping all dashboards...")
        for proc in processes:
            proc.terminate()
        console.print("All dashboards stopped.")

if __name__ == "__main__":
    main() 