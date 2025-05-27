from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
from status_manager import StatusManager

console = Console()
manager = StatusManager()

STALE_THRESHOLD = 10  # seconds

ICONS = {
    'Data': '⏳',
    'Analysis': '🔍',
    'Trade': '💹',
    'Performance': '📈',
    'OK': '🟢',
    'STALE': '🟡',
    'ERROR': '🔴',
}

def get_health(last_update):
    try:
        t = time.mktime(time.strptime(last_update, '%Y-%m-%d %H:%M:%S'))
        if time.time() - t > STALE_THRESHOLD:
            return 'STALE'
        return 'OK'
    except Exception:
        return 'ERROR'

def build_table():
    statuses = manager.get_all_statuses()
    table = Table(title="Bot Live Status", expand=True)
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Last Update", style="green")
    table.add_column("Health", style="bold")
    for agent, stat in statuses.items():
        icon = ICONS.get(agent, '❔')
        status = stat.get('message', 'Idle')
        last_update = stat.get('timestamp', '-')
        health = get_health(last_update)
        health_icon = ICONS[health]
        table.add_row(f"{icon} {agent}", status, last_update, health_icon + ' ' + health)
    if not statuses:
        table.add_row("-", "No agent status yet", "-", ICONS['STALE'] + ' Waiting')
    return table

if __name__ == "__main__":
    with Live(build_table(), console=console, refresh_per_second=2) as live:
        while True:
            live.update(build_table())
            time.sleep(1) 