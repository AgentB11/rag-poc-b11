import json
from pathlib import Path

import chromadb

COLLECTION_NAME = "banking_docs"


def get_collection(persist_dir: Path):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(embedded_chunks: list[dict], persist_dir: Path) -> None:
    collection = get_collection(persist_dir)
    collection.upsert(
        ids=[f"{c['source']}::{c['chunk_id']}" for c in embedded_chunks],
        embeddings=[c["embedding"] for c in embedded_chunks],
        documents=[c["text"] for c in embedded_chunks],
        metadatas=[{"source": c["source"], "chunk_id": c["chunk_id"]} for c in embedded_chunks],
    )


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    processed_dir = project_dir / "data" / "processed"
    persist_dir = project_dir / "data" / "vector_store"

    embedded_chunks = json.loads((processed_dir / "embeddings.json").read_text(encoding="utf-8"))
    store_chunks(embedded_chunks, persist_dir)

    collection = get_collection(persist_dir)
    print(f"collection '{COLLECTION_NAME}' now has {collection.count()} chunks -> {persist_dir}")
