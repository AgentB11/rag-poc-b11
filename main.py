import sys
from pathlib import Path

from src.generate import generate_answer
from src.retrieve import semantic_search

PERSIST_DIR = Path(__file__).resolve().parent / "data" / "vector_store"


def ask(query: str) -> str:
    hits = semantic_search(query, PERSIST_DIR)
    return generate_answer(query, hits)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    if not query:
        print('usage: python main.py "your question"')
        sys.exit(1)

    print(ask(query))
