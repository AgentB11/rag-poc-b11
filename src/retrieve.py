import sys
from pathlib import Path

from src.embed import embed_query
from src.store import get_collection

TOP_K = 3


def semantic_search(query: str, persist_dir: Path, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection(persist_dir)
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for text, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"text": text, "source": metadata["source"], "chunk_id": metadata["chunk_id"], "distance": distance})
    return hits


if __name__ == "__main__":
    persist_dir = Path(__file__).resolve().parent.parent / "data" / "vector_store"
    query = " ".join(sys.argv[1:]) or "What is a fixed deposit?"

    print(f"query: {query!r}\n")
    for rank, hit in enumerate(semantic_search(query, persist_dir), start=1):
        print(f"#{rank} [{hit['source']} chunk {hit['chunk_id']}] distance={hit['distance']:.4f}")
        print(hit["text"][:200].replace("\n", " "))
        print()
