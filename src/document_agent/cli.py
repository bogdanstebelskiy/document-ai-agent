import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def hello(name: str = typer.Option("World", help="Name to greet")) -> None:
    console.print(f"[bold green]Hello, {name}![/bold green]")
