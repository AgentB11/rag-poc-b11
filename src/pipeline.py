import json
from pathlib import Path

from src.chunk import chunk_documents
from src.embed import embed_chunks
from src.ingest import ingest_directory, save_documents
from src.store import get_collection, store_chunks

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PERSIST_DIR = PROJECT_DIR / "data" / "vector_store"


def build_index() -> None:
    documents = ingest_directory(RAW_DIR)
    save_documents(documents, PROCESSED_DIR)
    print(f"ingested {len(documents)} documents -> {PROCESSED_DIR}")

    chunks = chunk_documents(documents)
    (PROCESSED_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"chunked into {len(chunks)} chunks -> {PROCESSED_DIR / 'chunks.json'}")

    embedded_chunks = embed_chunks(chunks)
    (PROCESSED_DIR / "embeddings.json").write_text(json.dumps(embedded_chunks, indent=2), encoding="utf-8")
    print(f"embedded {len(embedded_chunks)} chunks -> {PROCESSED_DIR / 'embeddings.json'}")

    store_chunks(embedded_chunks, PERSIST_DIR)
    collection = get_collection(PERSIST_DIR)
    print(f"stored in collection '{collection.name}' ({collection.count()} chunks) -> {PERSIST_DIR}")


if __name__ == "__main__":
    build_index()
