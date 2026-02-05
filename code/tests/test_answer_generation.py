import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
from answer_generation import generate_answer

def test_generate_answer(monkeypatch):
    # Monkeypatch OpenAI call to avoid API usage
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
    chunks = [{'chunk_text': 'context', 'article_id': 'id1'}]
    answer = generate_answer('What is the policy?', chunks)
    assert isinstance(answer, str)
    assert 'Test answer' in answer
