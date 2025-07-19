import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import requests

# === Параметры ===
PERSIST_DIR = Path("chroma_data").absolute()
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "deepseek-r1"


model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path=str(PERSIST_DIR))
collection = client.get_collection(name="ru_bq_collection")


def strip_think_sections(raw: str) -> str:
    parts = raw.split("</think>")
    return "".join(parts[1:]).strip() if len(parts) > 1 else raw.strip()


def ask_ollama_http(prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    instruction = "Дай максимально лаконичный ответ на данный вопрос.\n\n"
    payload_prompt = instruction + prompt

    url = f"{OLLAMA_HOST}/v1/completions"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": payload_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    if "choices" in data and data["choices"]:
        return strip_think_sections(data["choices"][0]["text"])
    return str(data)


def query_chromadb(query_text: str, top_k=10) -> dict:
    # Вектор запроса
    query_vector = model.encode([query_text]).astype(np.float32)

    # Поиск по ChromaDB
    results = collection.query(query_embeddings=query_vector, n_results=top_k)

    # Формируем промпт
    context_chunks = results['documents'][0]
    prompt_lines = ["Вы получили следующие фрагменты текста, релевантные запросу:"]
    for i, chunk in enumerate(context_chunks, start=1):
        prompt_lines.append(f"{i}) {chunk}")
    prompt_lines.append("")
    prompt_lines.append(f"На основе этих фрагментов ответьте на вопрос: «{query_text}»")
    prompt = "\n".join(prompt_lines)

    # Ответ от LLM
    answer = ask_ollama_http(prompt)

    return {
        "query": query_text,
        "answer": answer,
        "chunks": context_chunks,
        "metadatas": results['metadatas'][0],
        "distances": results['distances'][0],
        "prompt": prompt
    }
