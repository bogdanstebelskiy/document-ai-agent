import typer
from rich.console import Console
from rich.table import Table

from document_agent.ingest.pipeline import IngestPipeline
from document_agent.models import get_embeddings, get_llm

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
    except Exception as e:  # noqa: BLE001
        error_console.print(
            f"[bold red]Error occurred while invoking LLM: {e}[/bold red]"
        )

        raise typer.Exit(1)

    emb = get_embeddings()

    try:
        vector = emb.embed_query("ping")
        console.print(f"[bold blue]{emb.model}: {len(vector)}[/bold blue]")
    except Exception as e:  # noqa: BLE001
        error_console.print(
            f"[bold red]Error occurred while invoking Embeddings: {e}[/bold red]"
        )

        raise typer.Exit(1)


@app.command()
def ingest(dir_uri: str = typer.Option(..., help="Directory uri")) -> None:
    table = Table(title="Ingestion Report")

    table.add_column("added", style="green", no_wrap=True, justify="center")
    table.add_column("updated", style="blue", no_wrap=True, justify="center")
    table.add_column("skipped", style="white", no_wrap=True, justify="center")
    table.add_column("removed", style="red", no_wrap=True, justify="center")
    table.add_column("unsupported", style="yellow", no_wrap=True, justify="center")

    report = IngestPipeline().ingest(dir_uri)

    table.add_row(
        str(report.added),
        str(report.updated),
        str(report.skipped),
        str(report.removed),
        str(report.unsupported)
    )

    console.print(table)
