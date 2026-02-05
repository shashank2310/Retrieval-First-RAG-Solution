"""
Answer generation module for RAG pipeline. Constructs prompt and calls LLM.
"""
import os
from dotenv import load_dotenv
import openai
import logging

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

ANSWER_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'answer-generation-prompt.md')

if os.path.exists(ANSWER_PROMPT_PATH):
    with open(ANSWER_PROMPT_PATH, 'r') as f:
        BASE_PROMPT = f.read()
else:
    BASE_PROMPT = "You are a helpful assistant. Answer the question using only the provided context. Cite sources by article ID."

logging.basicConfig(level=logging.INFO)
llm_call_count = 0
llm_cost_per_call = 0.002  # Example cost per call


def generate_answer(question: str, retrieved_chunks: list) -> str:
    global llm_call_count
    context = '\n'.join([c['chunk_text'] for c in retrieved_chunks])
    if not retrieved_chunks:
        logging.info("No relevant chunks found. Returning 'I don't know'.")
        return "I don't know"
    prompt = f"{BASE_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    llm_call_count += 1
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    logging.info(f"LLM API calls: {llm_call_count}, Estimated cost: ${llm_call_count * llm_cost_per_call:.4f}")
    return response.choices[0].message.content.strip()
