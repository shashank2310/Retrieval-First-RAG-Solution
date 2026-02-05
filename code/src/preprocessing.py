"""
Preprocessing and chunking module for RAG pipeline.
"""
import re
from typing import List, Dict
import yaml
import os

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

CHUNK_SIZE = config['chunk']['size']
CHUNK_OVERLAP = config['chunk']['overlap']
STRATEGY = config['chunk']['strategy']


def semantic_chunk(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    Splits text into semantic chunks (by paragraph/sentence) with overlap.
    Returns list of dicts: {chunk_text, start_idx, end_idx}
    """
    # Simple sentence split (can be improved with NLP)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    token_count = 0
    idx = 0
    for sent in sentences:
        current.append(sent)
        token_count += len(sent.split())
        if token_count >= chunk_size:
            chunk_text = ' '.join(current)
            start = max(0, idx - overlap)
            end = idx + len(current)
            chunks.append({
                'chunk_text': chunk_text,
                'start_idx': start,
                'end_idx': end
            })
            # Overlap fix: avoid division by zero
            if overlap > 0 and len(sent.split()) > 0:
                keep_n = max(1, overlap // len(sent.split()))
                current = current[-keep_n:]
            else:
                current = []
            token_count = sum(len(s.split()) for s in current)
        idx += 1
    # Add last chunk
    if current:
        chunk_text = ' '.join(current)
        start = max(0, idx - overlap)
        end = idx + len(current)
        chunks.append({
            'chunk_text': chunk_text,
            'start_idx': start,
            'end_idx': end
        })
    return chunks


def chunk_document(doc: str) -> List[Dict]:
    """
    Main chunking entry point. Supports semantic/fixed chunking.
    """
    if STRATEGY == 'semantic':
        return semantic_chunk(doc, CHUNK_SIZE, CHUNK_OVERLAP)
    else:
        # Fixed-size chunking
        words = doc.split()
        chunks = []
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i+CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'chunk_text': chunk_text,
                'start_idx': i,
                'end_idx': i+len(chunk_words)
            })
        return chunks
