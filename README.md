# 🚀 Retrieval-First RAG Solution

## 📌 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Setup & How to Run](#setup--how-to-run)
- [Configuration](#configuration)
- [Cost Optimization](#cost-optimization)
- [Chunking Strategy](#chunking-strategy)
- [Logging & Explainability](#logging--explainability)
- [Testing](#testing)
- [Artifacts](#artifacts)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
This project is a modular Retrieval-Augmented Generation (RAG) pipeline for the Retrieval-First RAG Hackathon. It helps large banks retrieve precise, compliant answers from massive knowledge repositories using semantic search and LLMs, with a focus on engineering best practices, cost optimization, and explainability.

## ✨ Features
- Modular codebase: preprocessing, embedding, vector store, retrieval, answer generation
- Semantic chunking with configurable size/overlap
- Batched embedding API calls (configurable batch size)
- Embedding caching to avoid redundant API calls
- Hybrid retrieval (vector + BM25) with reciprocal rank fusion
- No-answer detection: returns "I don't know" if no relevant chunk found
- Logging to both console and `artifacts/pipeline.log` for full traceability
- API call and cost tracking for embeddings and LLMs
- Configurable via YAML and `.env`
- Comprehensive test suite for all modules

## 🏗️ Architecture
- **Preprocessing:** Chunk documents with semantic overlap
- **Embedding:** Batched/cached OpenAI embeddings
- **Indexing:** FAISS vector store for fast similarity search
- **Retrieval:** Hybrid vector/BM25 search, reciprocal rank fusion
- **Answer Generation:** LLM generates answers, always citing sources
- **Output:** Results in `artifacts/results.json`

## ⚡ Setup & How to Run
1. Clone the repository
   ```sh
   git clone https://github.com/eft-hackathon/hackathon2-ai-explorers.git
   cd hackathon2-ai-explorers
   ```
2. Install dependencies
   ```sh
   pip install -r code/src/requirements.txt
   ```
3. Add your OpenAI API key to `code/src/.env`
   ```
   OPENAI_API_KEY=your-key-here
   ```
4. Configure chunking, batch size, and retrieval in `code/src/config/config.yaml`
5. Run the pipeline
   ```sh
   python code/src/main.py
   ```
6. View results in `artifacts/results.json`
7. View logs in `artifacts/pipeline.log`
8. Run tests (optional)
   ```sh
   pytest code/tests/
   ```

## ⚙️ Configuration
- All hyperparameters (chunk size, overlap, batch size, retrieval strategy) are in `code/src/config/config.yaml`
- API keys/secrets in `code/src/.env`

## 💸 Cost Optimization
- Embedding API calls are batched (default batch size: 32, can be increased for speed/cost)
- Embedding caching avoids redundant API calls
- API call counts and estimated costs are logged for both embeddings and LLMs
- Top-K filtering and confidence thresholds minimize unnecessary LLM calls

## 🧩 Chunking Strategy
- Semantic chunking by sentence/paragraph with overlap (configurable)
- Typical chunk size: 200–600 tokens; overlap: 10–20%
- Metadata (title, section) preserved for each chunk
- Tune chunk size/overlap in `config.yaml` for best retrieval quality

## 📋 Logging & Explainability
- All logs are written to both console and `artifacts/pipeline.log`
- Retrieval traces include similarity scores and reasoning
- No-answer events are logged
- API usage and cost are tracked and logged

## 🧪 Testing
- Comprehensive test suite in `code/tests/`
- Unit tests for chunking, embedding, retrieval, answer generation, vector store
- Integration tests for full pipeline and edge cases

## 📦 Artifacts
- `artifacts/results.json`: Submission mapping questions to ranked article IDs
- `artifacts/pipeline.log`: Full pipeline logs
- `artifacts/setup-instructions.md`: Environment and reproducible run steps
- `artifacts/design-overview.json`: Design summary and rationale
- `artifacts/answer-generation-prompt.md`: Prompt used for answer generation

## 🏗️ Tech Stack
- Python 3.10+
- OpenAI API (embeddings, LLM)
- FAISS / ChromaDB (vector store)
- PyYAML, python-dotenv, tqdm, pytest, rank_bm25, scikit-learn, numpy

## 👥 Team
- **Shashank Maheshwari** - [GitHub](https://github.com/shashank2310) | [LinkedIn](https://www.linkedin.com/in/maheshwari-shashank/)

---
For architecture details, see [artifacts/arch/architecture.md](code/artifacts/arch/architecture.md)
For setup instructions, see [artifacts/setup-instructions.md](code/artifacts/setup-instructions.md)
For design overview, see [artifacts/design-overview.json](code/artifacts/design-overview.json)
