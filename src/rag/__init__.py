"""
RAG Module — semantic retrieval-augmented generation for financial extraction.

Provides embedding, ChromaDB persistence, and retrieval services that inject
historical extraction context into the LLM's prompt for better accuracy.

Architecture:

    ChromaDB (per doc_type collections)
       │
       ├── save_to_vector_store()   ← called from ExtractionRepository after upsert
       └── retrieve_context()       ← called from Extractor before LLM invocation

Components:
    - embedder.py    → SentenceTransformer embedding function factory
    - chroma_client.py → ChromaDB persistent client management
    - repository.py  → write validated extractions to vector store
    - retriever.py   → query similar extractions by doc_type + semantic similarity
"""
