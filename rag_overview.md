# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a generative language model. Instead of relying solely on knowledge baked into the model's weights during training, a RAG system first searches an external knowledge base for information relevant to the user's question, then passes that retrieved information to the language model as additional context when generating an answer.

## Pipeline stages

A typical RAG pipeline has five stages.

1. **Ingestion** — raw documents such as PDFs, images, or plain text files are loaded and their text content is extracted.
2. **Chunking** — the extracted text is split into smaller, semantically coherent pieces, since embedding models and language models both work better on bounded units of text rather than entire documents.
3. **Embedding** — each chunk is converted into a dense numerical vector using an embedding model, capturing its semantic meaning in a way that allows similar pieces of text to end up close together in vector space.
4. **Storage and retrieval** — the vectors are stored in a vector database, which supports fast similarity search, so that given a new query, the system can find the chunks whose embeddings are most similar to the query's embedding.
5. **Generation** — the retrieved chunks are inserted into a prompt alongside the original question, and a large language model uses that grounded context to produce an answer.

## Why RAG matters

The main benefit of RAG is that it lets a language model answer questions about information it was never trained on, such as private company documents, recent events, or a specific PDF the user just uploaded, without needing to retrain or fine-tune the model itself.
