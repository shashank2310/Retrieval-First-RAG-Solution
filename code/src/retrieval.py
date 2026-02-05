"""
Retrieval module for RAG pipeline. Supports vector, BM25, and hybrid retrieval with reciprocal rank fusion.
"""
import yaml
import os
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
import logging

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

RETRIEVAL_STRATEGY = config['retrieval']['strategy']
TOP_K = config['retrieval']['top_k']
FUSION = config['retrieval']['fusion']
CONFIDENCE_THRESHOLD = config['confidence']['threshold']

logging.basicConfig(level=logging.INFO)


def reciprocal_rank_fusion(vector_results, bm25_results, top_k):
    """
    Combines vector and BM25 results using reciprocal rank fusion.
    Returns list of chunk dicts (not just texts).
    """
    scores = {}
    for rank, (chunk, score) in enumerate(vector_results):
        scores[chunk['chunk_text']] = scores.get(chunk['chunk_text'], 0) + 1/(60+rank)
    for rank, (chunk, score) in enumerate(bm25_results):
        scores[chunk['chunk_text']] = scores.get(chunk['chunk_text'], 0) + 1/(60+rank)
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # Return chunk dicts, not just texts
    chunk_dicts = []
    all_chunks = {c['chunk_text']: c for c, _ in vector_results + bm25_results}
    for c, s in sorted_chunks[:top_k]:
        if c in all_chunks:
            chunk_dicts.append(all_chunks[c])
    return chunk_dicts


def bm25_search(query: str, chunks: List[Dict], top_k: int) -> List[Tuple[Dict, float]]:
    tokenized_corpus = [c['chunk_text'].split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]


def retrieve(query: str, query_embedding: List[float], vector_store, chunks: List[Dict]) -> List[Dict]:
    """
    Main retrieval entry point. Supports vector, BM25, and hybrid.
    Logs retrieval scores and reasoning for explainability.
    """
    vector_results = vector_store.search(query_embedding, top_k=TOP_K)
    bm25_results = bm25_search(query, chunks, TOP_K)
    logging.info(f"Retrieval for query: {query}")
    if RETRIEVAL_STRATEGY == 'vector':
        top_chunks = [c for c, s in vector_results if s >= CONFIDENCE_THRESHOLD]
        logging.info(f"Vector scores: {[s for c, s in vector_results]}")
    elif RETRIEVAL_STRATEGY == 'bm25':
        top_chunks = [c for c, s in bm25_results if s >= CONFIDENCE_THRESHOLD]
        logging.info(f"BM25 scores: {[s for c, s in bm25_results]}")
    else:
        # Hybrid
        top_chunks = reciprocal_rank_fusion(vector_results, bm25_results, TOP_K)
        logging.info(f"Hybrid (RRF) used. Vector scores: {[s for c, s in vector_results]}, BM25 scores: {[s for c, s in bm25_results]}")
    if not top_chunks:
        logging.info("No chunk met confidence threshold. Returning no-answer.")
    return top_chunks
