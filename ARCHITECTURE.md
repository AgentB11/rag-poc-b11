# Architecture

This document describes how `rag-poc` is actually built: the pipeline stages, data flow, and the key design decisions behind them. For general RAG concepts (not specific to this repo), see [rag_overview.md](rag_overview.md).

## Pipeline overview

The system splits into two phases: an **offline indexing phase** that builds the searchable vector store from source documents, and an **online query phase** that answers questions against it.

```
OFFLINE (build the index) -- src/pipeline.py

+-------------------------+
| data/raw/*              |
| .txt . .pdf . .png/.jpg |
+-------------------------+
     |
     | ingest.py
     v
+---------------------+
| ingest.py           |
| pymupdf/pypdf (pdf) |
| easyocr (image)     |
| plain read (.txt)   |
+---------------------+
     |
     v
+----------------------+
| data/processed/*.txt |
+----------------------+
     |
     | chunk.py
     v
+--------------------------------+
| chunk.py                       |
| RecursiveCharacterTextSplitter |
| chunk_size=500, overlap=50     |
+--------------------------------+
     |
     v
+----------------------------+
| data/processed/chunks.json |
+----------------------------+
     |
     | embed.py
     v
+------------------------------+
| embed.py                     |
| sentence-transformers        |
| "all-MiniLM-L6-v2" (384-dim) |
+------------------------------+
     |
     v
+--------------------------------+
| data/processed/embeddings.json |
+--------------------------------+
     |
     | store.py
     v
+---------------------------+
| store.py                  |
| chromadb.PersistentClient |
| collection.upsert()       |
+---------------------------+
     |
     v
+------------------------------------+
| data/vector_store/                 |
| ChromaDB collection "banking_docs" |
+------------------------------------+


ONLINE (answer a question) -- main.py

+------------+
| user query |
+------------+
     |
     | retrieve.py
     v
+------------------------------------+
| retrieve.py                        |
| embed_query() + collection.query() |
| returns top_k chunks               |
| (source, chunk_id, text, distance) |
+------------------------------------+
     |
     | generate.py
     v
+-------------------------------------+
| generate.py                         |
| builds grounded prompt from         |
| retrieved chunks + query,           |
| sends to local Ollama (llama3.2:3b) |
| via ollama.chat()                   |
+-------------------------------------+
     |
     v
+--------+
| answer |
+--------+
```

## File responsibilities

| File | Role | Reads | Writes |
|---|---|---|---|
| `src/ingest.py` | Extract raw text from each source file type | `data/raw/*` | `data/processed/*.txt` |
| `src/chunk.py` | Split text into overlapping chunks | `data/processed/*.txt` | `data/processed/chunks.json` |
| `src/embed.py` | Turn chunks (and later, queries) into vectors | `data/processed/chunks.json` | `data/processed/embeddings.json` |
| `src/store.py` | Persist chunks + vectors into a searchable store | `data/processed/embeddings.json` | `data/vector_store/` (ChromaDB) |
| `src/retrieve.py` | Embed a query, semantic-search the store | `data/vector_store/` | — (returns results in memory) |
| `src/generate.py` | Turn retrieved chunks + query into a grounded answer | — (takes retrieval results as input) | — (returns answer in memory) |
| `src/pipeline.py` | Orchestrates the offline build: ingest → chunk → embed → store | `data/raw/*` | everything above |
| `main.py` | Orchestrates the online query: retrieve → generate | `data/vector_store/` | — |

## Key design decisions

**Staged persistence.** Every offline stage reads the *previous* stage's file from `data/processed/` rather than recomputing it in memory (e.g. `chunk.py` reads the `.txt` files `ingest.py` wrote, not the raw PDF/image again). This makes each stage's output independently inspectable and avoids redundant work — OCR and embedding are both expensive to redo.

**Offline/online split.** `pipeline.py` and `main.py` are separate entry points on purpose: building the index (slow, run occasionally) is a different operation from answering a question (fast, run per-query). `pipeline.py` has no query-handling logic; `main.py` has no ingestion/indexing logic.

**Local generation via Ollama, not the Anthropic API.** `generate.py` calls a local Ollama server (`llama3.2:3b`) instead of a hosted API, so this prototype runs fully offline/free with no API key. The Ollama server must be started manually (`ollama serve`) — see the project's saved environment notes for setup details.

**Small, CPU-friendly models throughout.** `all-MiniLM-L6-v2` (embedding) and `llama3.2:3b` (generation) were chosen because this machine has no GPU — both run acceptably fast on CPU alone.

## Running it

```bash
source venv/bin/activate

# rebuild the index from data/raw/
python -m src.pipeline

# ask a question against the built index
python main.py "What is a fixed deposit?"
```
