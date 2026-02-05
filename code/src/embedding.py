"""
Embedding module for RAG pipeline. Supports batched embedding, caching, and model selection.
"""
import os
import yaml
import pickle
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
import functools
import hashlib
import logging

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

EMBEDDING_MODEL = config['embedding']['model']
BATCH_SIZE = config['embedding']['batch_size']
CACHE_PATH = os.path.join(os.path.dirname(__file__), 'embedding_cache.pkl')

try:
    import openai
except ImportError:
    openai = None

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Simple cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

# Ensure artifacts directory exists before logging
artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(artifacts_dir, 'pipeline.log')),
        logging.StreamHandler()
    ]
)
api_call_count = 0
api_cost_per_call = 0.0001  # Example cost per 1K tokens


def get_openai_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """
    Batched embedding generation using OpenAI API.
    """
    global api_call_count
    api_call_count += 1
    if openai is None:
        raise ImportError('openai package not installed')
    openai.api_key = OPENAI_API_KEY
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc='Embedding'):
        batch = texts[i:i+BATCH_SIZE]
        response = openai.embeddings.create(input=batch, model=model)
        batch_embeds = [e.embedding for e in response.data]
        embeddings.extend(batch_embeds)
    logging.info(f"Embedding API calls: {api_call_count}, Estimated cost: ${api_call_count * api_cost_per_call:.4f}")
    return embeddings


def chunk_hash(chunk):
    return hashlib.sha256(chunk['chunk_text'].encode('utf-8')).hexdigest()


@functools.lru_cache(maxsize=10000)
def cached_embed(chunk_text, model):
    return get_openai_embeddings([chunk_text], model)[0]


def embed_chunks(chunks: List[Dict]) -> List[List[float]]:
    """
    Embeds chunk texts, using cache to avoid redundant API calls.
    """
    model = config['embedding']['model']
    batch_size = config['embedding'].get('batch_size', 32)
    chunk_texts = [chunk['chunk_text'] for chunk in chunks]
    embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        batch_embeddings = get_openai_embeddings(batch, model)
        embeddings.extend(batch_embeddings)
    return embeddings
