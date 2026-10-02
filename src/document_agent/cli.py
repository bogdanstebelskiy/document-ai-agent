import typer
from rich.console import Console

from document_agent.models import get_llm, get_embeddings


app = typer.Typer()
console = Console()
error_console = Console(stderr=True)

@app.command()
def hello(name: str = typer.Option("World", help="Name to greet")) -> None:
    console.print(f"[bold green]Hello, {name}![/bold green]")

@app.command()
def stats() -> None:
    llm = get_llm()

    try:
        response = llm.invoke("ping")
        console.print(f"[bold blue]{llm.model}: {response.content}[/bold blue]")
    except Exception as e:
        error_console.print(f"[bold red]Error occurred while invoking LLM: {e}[/bold red]")

        raise typer.Exit(1)

    emb = get_embeddings()

    try:
        vector = emb.embed_query("ping")
        console.print(f"[bold blue]{emb.model}: {len(vector)}[/bold blue]")
    except Exception as e:
        error_console.print(f"[bold red]Error occurred while invoking Embeddings: {e}[/bold red]")

        raise typer.Exit(1)