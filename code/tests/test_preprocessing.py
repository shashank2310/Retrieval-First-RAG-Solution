import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
from preprocessing import chunk_document

def test_semantic_chunking_basic():
    text = "Sentence one. Sentence two! Sentence three?"
    chunks = chunk_document(text)
    assert isinstance(chunks, list)
    assert all('chunk_text' in c for c in chunks)
    assert len(chunks) >= 1

def test_fixed_chunking():
    import preprocessing as prep
    prep.STRATEGY = 'fixed'
    text = "word " * 1000
    chunks = chunk_document(text)
    assert len(chunks) > 1
    prep.STRATEGY = 'semantic'  # restore

def test_empty_input():
    chunks = chunk_document("")
    assert isinstance(chunks, list)
    assert len(chunks) == 0 or all(c['chunk_text'] == '' for c in chunks)
