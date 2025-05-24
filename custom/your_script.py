from pip._vendor.rich.console import Console
from pip._vendor.rich.table import Table
from pip._vendor.rich.panel import Panel

console = Console()

# Create a fancy table
def display_data(data):
    table = Table(
        show_header=True,
        header_style="bold blue",
        border_style="blue",
        title="Data Overview",
        show_lines=True
    )
    
    # Add columns
    for column in data[0].keys():
        table.add_column(column, justify="center")
    
    # Add rows
    for row in data:
        table.add_row(*[str(v) for v in row.values()])
    
    # Display table in a panel
    console.print(Panel(table, border_style="cyan"))

# Example usage
sample_data = [
    {"ID": 1, "Name": "John", "Score": 95},
    {"ID": 2, "Name": "Alice", "Score": 88},
    {"ID": 3, "Name": "Bob", "Score": 92}
]

display_data(sample_data)
