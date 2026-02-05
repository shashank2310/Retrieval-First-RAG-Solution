"""
Unit tests for embedding module.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
from embedding import embed_chunks

def test_embed_chunks_cache(monkeypatch):
    # Monkeypatch OpenAI embedding call to avoid API usage
    monkeypatch.setattr('embedding.get_openai_embeddings', lambda texts, model: [[0.1]*5 for _ in texts])
    chunks = [{'chunk_text': 'test chunk'}]
    embeds = embed_chunks(chunks)
    assert isinstance(embeds, list)
    assert len(embeds) == 1
    assert isinstance(embeds[0], list)
    assert len(embeds[0]) == 5

def test_embed_chunks_empty():
    embeds = embed_chunks([])
    assert embeds == []
