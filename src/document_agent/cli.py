import shutil
import uuid

import typer
from rich.console import Console
from rich.table import Table

from document_agent.config import settings
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
def ingest(
    dir_uri: str = typer.Option(..., help="Directory to ingest"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Clear the index before ingesting"),
) -> None:
    if rebuild:
        chroma_dir = settings.data_dir / "chroma"
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        manifest_db = settings.data_dir / "document_agent.sqlite"
        if manifest_db.exists():
            manifest_db.unlink()
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


@app.command()
def retrieval_eval(k: int = typer.Option(6, help="Top-k results to retrieve per question")) -> None:
    from document_agent.eval.run import hit_at_k, mrr, run_eval

    results = run_eval(k=k)

    summary = Table(title="Retrieval Eval", show_header=False, box=None)
    summary.add_column(style="bold")
    summary.add_column(justify="right")
    summary.add_row("questions", str(len(results)))
    summary.add_row(f"hit@{k}", f"{hit_at_k(results):.2%}")
    summary.add_row("MRR", f"{mrr(results):.4f}")
    console.print(summary)

    misses = [r for r in results if r.rank is None]
    if misses:
        miss_table = Table(title=f"Misses ({len(misses)})", show_lines=True)
        miss_table.add_column("question")
        miss_table.add_column("expected", style="yellow", no_wrap=True)
        for r in misses:
            miss_table.add_row(r.question, r.expected_source)
        console.print(miss_table)


def _make_checkpointer():
    import sqlite3
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver
    from document_agent.domain import Chunk

    serde = JsonPlusSerializer(allowed_msgpack_modules=[Chunk])
    return SqliteSaver(sqlite3.connect(settings.data_dir / "checkpoints.sqlite", check_same_thread=False), serde=serde)


@app.command()
def ask(
    question: str = typer.Argument(help="Question to ask your notes"),
    thread: str = typer.Option(None, "--thread", help="Thread ID for conversation memory"),
) -> None:
    from langchain_core.messages import AIMessageChunk
    from document_agent.agent.graph import build_graph

    thread_id = thread or str(uuid.uuid4())
    saver = _make_checkpointer()
    graph = build_graph(checkpointer=saver)
    citations = []

    console.print()
    for mode, chunk in graph.stream(
        {"question": question, "search_query": question, "messages": []},
        stream_mode=["messages", "values"],
        config={"configurable": {"thread_id": thread_id}},
    ):
        if mode == "messages":
            msg_chunk, metadata = chunk
            if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content and metadata.get("langgraph_node") in ("generate", "respond_direct"):
                print(msg_chunk.content, end="", flush=True)
        elif mode == "values" and "citations" in chunk:
            citations = chunk["citations"]

    print("\n")
    if citations:
        console.print("[dim]Sources:[/dim]")
        for source in citations:
            console.print(f"  - {source}")


@app.command()
def threads() -> None:
    from datetime import datetime, timezone

    saver = _make_checkpointer()
    seen = set()
    table = Table(title="Threads")
    table.add_column("thread", style="bold")
    table.add_column("messages", justify="right")
    table.add_column("last active", style="dim")

    for checkpoint_tuple in saver.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if thread_id in seen:
            continue
        seen.add(thread_id)

        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        ts = checkpoint_tuple.checkpoint.get("ts", "")
        if ts:
            dt = datetime.fromisoformat(ts).astimezone(timezone.utc)
            last_active = dt.strftime("%Y-%m-%d %H:%M UTC")
        else:
            last_active = "?"

        table.add_row(thread_id, str(len(messages)), last_active)

    if not seen:
        console.print("[dim]No threads found.[/dim]")
    else:
        console.print(table)
