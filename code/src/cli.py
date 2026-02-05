"""
CLI for user-driven RAG question answering.
Loads and indexes articles on startup, then answers user questions interactively.
"""
import os
import yaml
import json
import argparse
from preprocessing import chunk_document
from embedding import embed_chunks, get_openai_embeddings
from vector_store import VectorStore
from retrieval import retrieve
from answer_generation import generate_answer
import csv

class BankRAGAssistant:
    def __init__(self):
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(CONFIG_PATH, 'r') as f:
            self.config = yaml.safe_load(f)
        random_seed = self.config['seed']
        import random
        random.seed(random_seed)

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        articles_path = os.path.join(data_dir, 'input_article_details.jsonl')
        articles = []
        with open(articles_path, 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line)
                articles.append(article)

        all_chunks = []
        for article in articles:
            chunks = chunk_document(article['contents'])
            for c in chunks:
                c['article_id'] = article['id']
                c['title'] = article.get('title', '')
                c['url'] = article.get('url', '')
            all_chunks.extend(chunks)

        self.all_chunks = all_chunks
        self.embeddings = embed_chunks(all_chunks)
        vector_dim = len(self.embeddings[0])
        self.vector_store = VectorStore(vector_dim)
        self.vector_store.add(self.embeddings, all_chunks)

    def answer_user_question(self, question):
        q_embed = get_openai_embeddings([question], self.config['embedding']['model'])[0]
        top_chunks = retrieve(question, q_embed, self.vector_store, self.all_chunks)
        # Use top-K chunks for answer and article IDs
        answer = generate_answer(question, top_chunks)
        seen_ids = set()
        article_ids = []
        for c in top_chunks:
            aid = c['article_id']
            if aid not in seen_ids:
                article_ids.append(aid)
                seen_ids.add(aid)
        return {"question": question, "answer": answer, "article_ids": article_ids}

    def interactive_mode(self):
        print("Welcome to the Bank RAG Assistant.")
        print("Paste your question(s) below. Enter a blank line to finish.")
        user_questions = []
        while True:
            line = input()
            if line.strip() == "":
                break
            user_questions.append(line)
        results = []
        for q in user_questions:
            result = self.answer_user_question(q)
            print(f"\nQuestion: {q}\nAnswer: {result['answer']}\nCited Article IDs: {result['article_ids']}")
            results.append(result)
        # Save results to file (always code/artifacts/results.json)
        out_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'results.json')
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")

    def batch_mode(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        questions_path = os.path.join(data_dir, 'questions.csv')
        questions = []
        with open(questions_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row['question'])
        results = []
        for q in questions:
            result = self.answer_user_question(q)
            results.append(result)
        out_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'results.json')
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Batch results saved to {out_path}")
        print(f"Processed {len(questions)} questions.")


def main():
    parser = argparse.ArgumentParser(description="Bank RAG Assistant CLI")
    parser.add_argument('--batch', action='store_true', help='Run in batch mode using questions.csv')
    args = parser.parse_args()
    assistant = BankRAGAssistant()
    if args.batch:
        assistant.batch_mode()
    else:
        assistant.interactive_mode()

if __name__ == "__main__":
    main()
