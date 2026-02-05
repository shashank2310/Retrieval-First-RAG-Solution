"""
Unit tests for vector store.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
from vector_store import VectorStore

def test_faiss_add_and_search():
    dim = 5
    store = VectorStore(dim)
    chunks = [{'chunk_text': 'a', 'article_id': 'id1'}, {'chunk_text': 'b', 'article_id': 'id2'}]
    embeds = [[0.1]*dim, [0.2]*dim]
    store.add(embeds, chunks)
    results = store.search([0.1]*dim, top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert 'chunk_text' in results[0][0]

@pytest.mark.skipif('chromadb' not in sys.modules and not hasattr(__import__('builtins'), 'chromadb'), reason='chromadb not installed')
def test_chroma_add_and_search(monkeypatch):
    try:
        import chromadb
    except ImportError:
        pytest.skip('chromadb not installed')
    import vector_store as vs
    vs.VECTOR_TYPE = 'chroma'
    store = VectorStore(5)
    chunks = [{'chunk_text': 'a', 'article_id': 'id1'}]
    embeds = [[0.1]*5]
    store.add(embeds, chunks)
    results = store.search([0.1]*5, top_k=1)
    assert isinstance(results, list)
    vs.VECTOR_TYPE = 'faiss'  # restore
