import sys
from pathlib import Path

import ollama

from src.retrieve import semantic_search

MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the context does not contain enough information to answer, say so clearly instead of guessing."
)


def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{hit['source']} chunk {hit['chunk_id']}]\n{hit['text']}" for hit in hits)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def generate_answer(query: str, hits: list[dict]) -> str:
    prompt = build_prompt(query, hits)
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    persist_dir = Path(__file__).resolve().parent.parent / "data" / "vector_store"
    query = " ".join(sys.argv[1:]) or "What is a fixed deposit?"

    hits = semantic_search(query, persist_dir)

    print(f"query: {query!r}\n")
    print("--- retrieved context ---")
    for hit in hits:
        print(f"[{hit['source']} chunk {hit['chunk_id']}] distance={hit['distance']:.4f}")

    answer = generate_answer(query, hits)
    print("\n--- answer ---")
    print(answer)
