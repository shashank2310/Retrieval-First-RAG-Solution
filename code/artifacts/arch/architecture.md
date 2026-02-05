# RAG Hackathon Solution Architecture

## Overview
This document describes the architecture and design of the Retrieval-Augmented Generation (RAG) solution for the hackathon. The system is modular, configurable, and designed for reproducibility, cost optimization, and extensibility.

---

## Architecture Diagram

```
[Raw Articles/Docs]
      |
      v
[Preprocessing & Chunking]
      |
      v
[Embedding Generation (batched, cached)]
      |
      v
[Vector Store/Index (FAISS/Chroma)]
      |
      v
[Retrieval Engine (vector + BM25 hybrid)]
      |
      v
[Answer Generation (LLM, grounded in retrieved chunks)]
      |
      v
[results.json]
```

---

## Module Breakdown

- **Preprocessing & Chunking** (`preprocessing.py`)
  - Semantic chunking by sentence/paragraph with overlap
  - Configurable chunk size and strategy
  - Metadata preservation (article ID, title, URL)

- **Embedding Generation** (`embedding.py`)
  - Batched calls to OpenAI Embedding API
  - Local cache to minimize cost
  - Model selection via config

- **Vector Store & Indexing** (`vector_store.py`)
  - FAISS for local prototyping
  - Chroma for metadata-rich, scalable storage
  - Stores chunk embeddings, metadata, and text

- **Retrieval Engine** (`retrieval.py`)
  - Vector similarity (cosine)
  - BM25 lexical search
  - Hybrid retrieval via reciprocal rank fusion
  - Confidence thresholding

- **Answer Generation** (`answer_generation.py`)
  - Constructs prompt for LLM
  - Calls OpenAI GPT model with retrieved context
  - Cites sources by article ID

- **Pipeline Orchestration** (`main.py`)
  - Loads articles and questions from data files
  - Runs full pipeline and writes results

---

## Key Design Decisions

- **Chunking:** Semantic chunking with overlap, configurable size
- **Embedding Model:** OpenAI `text-embedding-3-small` (cost-effective, general-purpose)
- **Vector Store:** FAISS (local), Chroma (scalable, metadata)
- **Retrieval:** Hybrid (vector + BM25), reciprocal rank fusion
- **Batching & Caching:** Batched API calls, local cache for embeddings
- **Configurability:** All hyperparameters in `config.yaml`
- **Error Handling:** Rate limiting, retries, deterministic random seed

---

## Cost Optimization & Resource Management

- Batched embedding generation
- Embedding cache to avoid redundant API calls
- Rate limiting and retries for API calls
- Estimated cost: $0.0001 per 1K tokens for OpenAI embeddings
- For 100K chunks: ~$10

---

## Trade-offs & Limitations

- **Accuracy vs Speed vs Cost:** Small model for cost, hybrid retrieval for accuracy, local index for speed
- **Limitations:** Simple chunking may miss context, no cross-encoder re-ranking, limited to English, no-answer detection by threshold only

---

## Future Improvements

- Add semantic chunking with NLP
- Cross-encoder re-ranking
- Multilingual support
- More robust no-answer detection
- Production-grade vector DB integration

---

## Reproducibility & Testing

- All configs externalized
- API keys via `.env`
- Pytest suite for all modules and pipeline
- Deterministic random seed

---

## Output

- Answers to all questions in `results.json`, citing supporting article IDs

---

## References
- See `design-overview.json` for summary of design decisions
- See `setup-instructions.md` for environment setup and reproducibility

