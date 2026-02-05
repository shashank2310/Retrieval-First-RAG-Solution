import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
from preprocessing import chunk_document
from embedding import embed_chunks
from vector_store import VectorStore
from retrieval import retrieve
from answer_generation import generate_answer

class DummyEmbeddings:
    def __init__(self, dim):
        self.dim = dim
    def embed(self, texts):
        return [[0.1]*self.dim for _ in texts]

@pytest.mark.skip(reason="Requires OpenAI API unless monkeypatched")
def test_full_pipeline(monkeypatch):
    # Monkeypatch embedding and answer generation
    monkeypatch.setattr('embedding.get_openai_embeddings', lambda texts, model: [[0.1]*5 for _ in texts])
    monkeypatch.setattr('answer_generation.openai', type('Dummy', (), {
        'chat': type('DummyChat', (), {
            'completions': type('DummyComp', (), {
                'create': lambda **kwargs: type('DummyResp', (), {
                    'choices': [type('DummyChoice', (), {
                        'message': type('DummyMsg', (), {'content': 'Test answer'})
                    })]
                })()
            })
        })
    }))
    doc = {'id': 'doc1', 'text': 'Policy text.'}
    chunks = chunk_document(doc['text'])
    for c in chunks:
        c['article_id'] = doc['id']
    embeds = embed_chunks(chunks)
    store = VectorStore(len(embeds[0]))
    store.add(embeds, chunks)
    q = 'What is the policy?'
    q_embed = [0.1]*len(embeds[0])
    top_chunks = retrieve(q, q_embed, store, chunks)
    answer = generate_answer(q, top_chunks)
    assert isinstance(answer, str)
    assert 'Test answer' in answer
