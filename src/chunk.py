import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_processed_documents(processed_dir: Path) -> list[dict]:
    return [
        {"source": path.name, "text": path.read_text(encoding="utf-8")}
        for path in sorted(processed_dir.glob("*.txt"))
    ]


def chunk_documents(documents: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})
    return chunks


if __name__ == "__main__":
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    documents = load_processed_documents(processed_dir)
    chunks = chunk_documents(documents)

    out_path = processed_dir / "chunks.json"
    out_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1

    print(f"{len(chunks)} chunks total -> {out_path}")
    for source, count in counts.items():
        print(f"  {source}: {count} chunks")

    print("\n--- sample chunk ---")
    print(chunks[0])
