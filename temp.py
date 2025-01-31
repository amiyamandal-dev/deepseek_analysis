import os
import time
import json
import faiss
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from functools import lru_cache
from typing import Optional, List, Dict, Any
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import ollama


import datasets as hf_datasets


import openai as openai_python_client


load_dotenv()


class FaissVectorStore:
    """
    Minimal Faiss-based vector store using inner product (for normalized embeddings).
    Stores documents, metadata, and their embeddings.
    """
    def __init__(self, embedding_dim: int):
        # IndexFlatIP uses inner product for similarity (good for normalized embeddings).
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents = []
        self.metadata = []

    def add_texts(self, texts: List[str], embeddings: np.ndarray, meta: List[Dict[str, Any]]):
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.documents.extend(texts)
        self.metadata.extend(meta)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Ensure shape is (batch_size, dim)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, top_k)
        # distances shape = (1, top_k), indices shape = (1, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            # Retrieve doc & metadata
            doc_text = self.documents[idx]
            doc_meta = self.metadata[idx]
            score = float(distances[0][rank])
            results.append({
                "text": doc_text,
                "metadata": doc_meta,
                "score": score
            })
        return results

class BGELarge:

    @lru_cache(maxsize=None)
    def encode(self, text: str) -> np.ndarray:
        """
        Return a vector embedding of shape (embedding_dim,).
        Replace with your real BGE call if needed.
        e.g., self.ollama_client.run(model="bge-m3", input=data)[0]
        """
        embedding = ollama.embed(model='bge-m3',input=text)
        embedding = embedding['embeddings'][0]
        return np.array(embedding, dtype=np.float32)

class GPT:
    def __init__(self):
        self.client = openai_python_client.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.system_prompt = '''When answering questions, follow these steps:
1. Analyze the context to determine its relevance to the question.
2. If the context is relevant to the question, use the information provided in the context to answer the question.
3. If the context does not mention the question, rely on your own knowledge to answer the question.
4. Provide a clear and accurate response to the question, incorporating relevant information from the context or your own knowledge.

Please ensure that your responses are well-structured, concise, and address the specific question effectively. Your answers should demonstrate a thorough understanding of the question and provide valuable insights or information.'''

    def generate(self, model: str, prompt: str) -> Optional[str]:
        try:
            r = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error generating response with GPT: {e}") from e


class DeepSeek:
    def __init__(self):
        self.client = openai_python_client.OpenAI(api_key=os.environ['API_KEY'], base_url="https://api.deepseek.com")
        self.system_prompt = '''Step 1: Analyze context for answering questions.\n"
        "Step 2: Decide context is relevant with question or not relevant with question.\n "
        "Step 3: If any topic about question mentioned in context, use that information for question.\n "
        "Step 4: If context has not mention on question, ignore that context I give you and use your self knowledge.\n "
        "Step 5: Answer the question.\n'''

    def generate(self, model: str, prompt: str) -> Optional[str]:
        try:
            # Possibly: openai_python_client.api_base = "https://api.deepseek.com"
            r = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error generating response with DeepSeek: {e}") from e


class Judge:
    def __init__(self, api_key: str):
        self.client = openai_python_client.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.system_prompt = "As a judge, your role is to compare the LLM’s response to the correct answer. If they align, return “Correct.” If they do not align, return “Incorrect.” Use no other words. Your task is to provide a clear assessment of alignment between the LLM’s response and the correct answer. Stick to these guidelines and focus on accuracy."

    def evaluate(self, question: str, correct: str, response: str) -> int:
        """
        Returns 1 if judged 'Correct', else 0.
        """
        prompt = f"**Question**: {question}\n\n**Correct answer**: {correct}\n\n**LLM's response**: {response}"
        try:
            r = self.client.chat.completions.create(
                model="gpt-4o",  # or whichever judge model you prefer
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0
            )
            c = r.choices[0].message.content.strip()
            return 1 if "Correct".lower() in c.lower() else 0
        except Exception as e:
            raise Exception(f"Error generating judge response: {e}") from e


def exponential_backoff(func, *args, max_retries=3, **kwargs):
    """
    Generic exponential backoff helper. Tries func(*args, **kwargs).
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise e

class RAG:
    def __init__(self, model, vector_store: FaissVectorStore, embedder: BGELarge, top_n=2):
        self.model = model
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_n = top_n

    def forward(self, prompt_template: str, question: str, simple_answer: str, model_id: str):
        """
        RAG approach:
        1) Embed question
        2) Query vector store
        3) Combine top docs as context
        4) Generate final answer with LLM
        Returns (list_of_context_docs, final_answer_string).
        """
        # 1) Encode question
        question_emb = self.embedder.encode(question)

        # 2) Retrieve from Faiss
        retrieved_docs = self.vector_store.search(question_emb, top_k=self.top_n)
        if not retrieved_docs:
            # If nothing found, fallback to simple answer
            return [], simple_answer

        # Combine the top docs into a single context string
        context_texts = [doc["text"] for doc in retrieved_docs]
        combined_context = "\n\n".join(context_texts)

        # 3) Format final prompt
        #    Example prompt template: "\n'''{0}'''\n\n**Question**: {1}\n\n"
        #    We'll pass context => {0}, question => {1}
        final_prompt = prompt_template.format(combined_context, question)

        # 4) Generate with retry
        try:
            ans = exponential_backoff(self.model.generate, model_id, final_prompt, max_retries=3)
            return retrieved_docs, ans
        except Exception:
            return "I don't know", "Error"


def load_dataset(slice_percentage: str = "1") -> List[dict]:
    """
    Load a portion of HotPotQA (fullwiki) from Hugging Face.
    slice_percentage can be e.g. '1' (1%), '5' (5%), etc.
    Returns a list of dicts with keys: question, answer, context, title, ...
    If your dataset has different keys, adapt as needed.
    """
    ds = hf_datasets.load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{slice_percentage}%]")
    # Convert to a list of dicts for easy iteration
    return list(ds)


def build_vector_store(
    docs: List[str], titles: List[str], embedder: BGELarge
) -> FaissVectorStore:
    """
    Build and populate a FaissVectorStore given docs & titles.
    We'll use BGE to encode each doc, then store them.
    """
    embedding_dim = embedder.encode("test").shape[0]
    vs = FaissVectorStore(embedding_dim)

    # Encode docs
    embeddings = []
    metadata = []
    for i, text in enumerate(docs):
        emb = embedder.encode(text)
        embeddings.append(emb)
        metadata.append({"title": titles[i], "id": i})

    embeddings = np.array(embeddings, dtype=np.float32)
    vs.add_texts(docs, embeddings, metadata)
    return vs


def evaluate_row(
    row: dict,
    evaluator,  # GPT or DeepSeek
    model_id: str,
    judge: Judge,
    rag_pipeline: RAG
):
    """
    Evaluate a single row using:
    - Simple approach (no context)
    - RAG approach (with retrieved context)
    Return a dict of results.
    """
    prompt_template = "\n'''{0}'''\n\n**Question**: {1}\n\n"
    question = row["question"]
    correct_answer = row["answer"]


    simple_prompt = prompt_template.format("", question)
    try:
        result_simple = evaluator.generate(model_id, simple_prompt)
    except Exception:
        result_simple = "Error in simple generation"

    # 2) RAG approach
    context, result_rag = rag_pipeline.forward(prompt_template, question, result_simple, model_id)

    # 3) Evaluate correctness with the Judge
    acc_simple = judge.evaluate(question, correct_answer, result_simple)
    acc_rag = judge.evaluate(question, correct_answer, result_rag)

    # Return dictionary with relevant fields
    return {
        "question": question,
        "truth": correct_answer,
        "prediction_simple": acc_simple,
        "prediction_rag": acc_rag,
        "prediction_response_simple": result_simple,
        "prediction_response_rag": result_rag,
        "context": context,  # list of retrieved docs or []
    }


def load_models() -> dict:

    gpt_model = GPT()
    deepseek_model = DeepSeek()

    return {
        # You can use the same GPT object for multiple model_id calls if desired
        "gpt-4o": gpt_model,
        "deepseek-chat": deepseek_model
    }

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline with Faiss, BGE, GPT/DeepSeek.")
    parser.add_argument('--output_dir', type=str, default='results', help="Where to save CSV results.")
    parser.add_argument('--dataset_slice', type=str, default='1', help="Percentage of dataset slice to load, e.g., '1' for 1%.")
    args = parser.parse_args()

    # 1) Load the dataset
    ds = load_dataset(args.dataset_slice)
    documents = []
    titles = []
    for i, row in enumerate(ds):
        if "context" in row and row["context"]:
            combined_text = "\n".join([f"{c[0]}: {c[1]}" for c in row["context"]])
        else:
            combined_text = f"Placeholder text for doc {i}"
        documents.append(combined_text)
        if "title" in row and row["title"]:
            titles.append(row["title"])
        else:
            titles.append(f"doc-{i}")

    bge_embedder = BGELarge() 
    vector_store = build_vector_store(documents, titles, bge_embedder)

    # 3) Load your GPT/DeepSeek models
    models = load_models()

    # 4) Initialize a judge
    judge = Judge(api_key=os.environ.get("OPENAI_API_KEY", ""))

    # 5) Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 6) Evaluate each model
    for model_name, evaluator in models.items():
        print(f"\nEvaluating model: {model_name}\n")

        # Create the RAG pipeline for this model
        rag_pipeline = RAG(model=evaluator, vector_store=vector_store, embedder=bge_embedder, top_n=5)

        # We'll store results in a list of dict
        results = []

        for row in tqdm(ds, desc=f"{model_name}"):
            try:
                result = evaluate_row(row, evaluator, model_name, judge, rag_pipeline)
                results.append(result)
            except Exception as e:
                # If an error occurs for a row, you can log or handle it
                print(f"Error evaluating row: {e}")

        # 7) Convert to DataFrame and save
        df = pd.DataFrame(results)
        safe_name = model_name.replace('/', '_')
        out_path = os.path.join(args.output_dir, f"{safe_name}_results.xlsx")
        try:
            df.to_excel(out_path, index=False)
            print(f"Saved results to {out_path}")
        except Exception as e:
            print(f"Failed to save to Excel: {e}")
            # Fallback to saving as CSV
            out_path_csv = os.path.join(args.output_dir, f"{safe_name}_results.csv")
            df.to_csv(out_path_csv, index=False)
            print(f"Saved results to {out_path_csv} as CSV instead")


if __name__ == "__main__":
    main()
