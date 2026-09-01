import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    model = _get_model()
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False)
    return [{**chunk, "embedding": vector.tolist()} for chunk, vector in zip(chunks, vectors)]


def embed_query(text: str) -> list[float]:
    model = _get_model()
    return model.encode([text], show_progress_bar=False)[0].tolist()


if __name__ == "__main__":
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    chunks = json.loads((processed_dir / "chunks.json").read_text(encoding="utf-8"))
    embedded_chunks = embed_chunks(chunks)

    out_path = processed_dir / "embeddings.json"
    out_path.write_text(json.dumps(embedded_chunks, indent=2), encoding="utf-8")

    dims = len(embedded_chunks[0]["embedding"])
    print(f"{len(embedded_chunks)} chunks embedded ({dims}-dim vectors) -> {out_path}")
    print(f"sample vector (first 8 values): {embedded_chunks[0]['embedding'][:8]}")
