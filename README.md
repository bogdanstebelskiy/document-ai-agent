# document-agent

CLI tool for ingesting local documents and asking questions about them. Uses RAG with a LangGraph agent that retrieves relevant chunks, rewrites queries when results are weak, and cites sources in answers.

Supports `.txt` and `.md` files. Conversations persist across runs via thread IDs.

## Stack

- **LLM/embeddings**: Ollama (default models: `qwen2.5:7b-instruct`, `nomic-embed-text`)
- **Vector store**: ChromaDB
- **Agent framework**: LangGraph
- **CLI**: Typer + Rich

## Usage

```bash
document-agent ingest --dir-uri ./notes
document-agent ask "What is this project about?"
document-agent ask "Tell me more" --thread <thread-id>
document-agent threads
document-agent stats
```

## Environment variables

All prefixed with `DA_`, configured via `.env` or shell.

| Variable | Default | Description |
|---|---|---|
| `DA_LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama chat model |
| `DA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `DA_DATA_DIR` | `.data` | Directory for ChromaDB, manifests, checkpoints |
| `DA_CHUNK_SIZE` | `500` | Max characters per chunk |
