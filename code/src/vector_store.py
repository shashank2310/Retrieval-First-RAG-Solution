"""
Vector store and indexing module for RAG pipeline.
Supports FAISS and Chroma, with metadata storage.
"""
import os
import yaml
import numpy as np
from typing import List, Dict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

VECTOR_TYPE = config['vector_store']['type']
INDEX_TYPE = config['vector_store']['index']

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks = []
        self.metadata = []
        if VECTOR_TYPE == 'faiss':
            import faiss
            self.index = faiss.IndexFlatL2(dim) if INDEX_TYPE == 'flat' else faiss.IndexHNSWFlat(dim, 32)
        elif VECTOR_TYPE == 'chroma':
            import chromadb
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.create_collection('rag_chunks')
        else:
            raise ValueError('Unknown vector store type')

    def add(self, embeddings: List[List[float]], chunks: List[Dict]):
        if VECTOR_TYPE == 'faiss':
            import faiss
            arr = np.array(embeddings).astype('float32')
            self.index.add(arr)
            self.chunks.extend(chunks)
        elif VECTOR_TYPE == 'chroma':
            ids = [str(i) for i in range(len(embeddings))]
            self.collection.add(embeddings=embeddings, documents=[c['chunk_text'] for c in chunks], ids=ids, metadatas=chunks)
            self.chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5):
        if VECTOR_TYPE == 'faiss':
            import faiss
            arr = np.array([query_embedding]).astype('float32')
            D, I = self.index.search(arr, top_k)
            results = [(self.chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
            return results
        elif VECTOR_TYPE == 'chroma':
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return list(zip(results['metadatas'][0], results['distances'][0]))
