"""
Unit tests for retrieval module.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
from retrieval import retrieve

class DummyVectorStore:
    def search(self, query_embedding, top_k=5):
        return [({'chunk_text': 'a', 'article_id': 'id1'}, 0.9), ({'chunk_text': 'b', 'article_id': 'id2'}, 0.8)]

def test_vector_retrieval():
    import retrieval as ret
    ret.RETRIEVAL_STRATEGY = 'vector'
    chunks = [{'chunk_text': 'a', 'article_id': 'id1'}, {'chunk_text': 'b', 'article_id': 'id2'}]
    results = retrieve('query', [0.1]*5, DummyVectorStore(), chunks)
    assert isinstance(results, list)
    assert all('chunk_text' in c for c in results)
    ret.RETRIEVAL_STRATEGY = 'hybrid'  # restore

def test_bm25_retrieval():
    import retrieval as ret
    ret.RETRIEVAL_STRATEGY = 'bm25'
    chunks = [{'chunk_text': 'a', 'article_id': 'id1'}, {'chunk_text': 'b', 'article_id': 'id2'}]
    results = retrieve('query', [0.1]*5, DummyVectorStore(), chunks)
    assert isinstance(results, list)
    ret.RETRIEVAL_STRATEGY = 'hybrid'  # restore

def test_hybrid_retrieval():
    import retrieval as ret
    ret.RETRIEVAL_STRATEGY = 'hybrid'
    chunks = [{'chunk_text': 'a', 'article_id': 'id1'}, {'chunk_text': 'b', 'article_id': 'id2'}]
    results = retrieve('query', [0.1]*5, DummyVectorStore(), chunks)
    assert isinstance(results, list)
