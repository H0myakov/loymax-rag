import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
import requests

# Настройка логирования
logging.basicConfig(filename='chromadb.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PERSIST_DIR = Path("chroma_data").absolute()


def query_chromadb(query_text, collection_name="ru_bq_collection", top_k=10):
    logging.info(f"Инициализация PersistentClient для запроса: {query_text}")
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_collection(name=collection_name)

    logging.info("Загрузка SentenceTransformer и векторизация запроса")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query_text]).astype(np.float32)

    logging.info(f"Выполнение поиска по вектору. Top_k={top_k}")
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )

    logging.info(f"Найдено {len(results['ids'][0])} результатов")

    # Сбор контекста для промпта
    context_chunks = results['documents'][0]
    prompt_lines = ["Вы получили следующие фрагменты текста, релевантные запросу:"]
    for i, chunk in enumerate(context_chunks, start=1):
        prompt_lines.append(f"{i}) {chunk}")
    prompt_lines.append("")
    prompt_lines.append(f"На основе этих фрагментов ответьте на вопрос: «{query_text}»")
    prompt = "\n".join(prompt_lines)

    # Генерация через Ollama
    answer = ask_ollama_http(prompt)

    return results, answer, prompt


OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "deepseek-r1"


def strip_think_sections(raw: str) -> str:
    parts = raw.split("</think>")
    if len(parts) > 1:
        final_part = "".join(parts[1:]).strip()
        return final_part
    else:
        return raw.strip()


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
        raw_text = data["choices"][0]["text"]
        return strip_think_sections(raw_text)
    else:
        return str(data)


if __name__ == "__main__":
    print("Введите вопрос для обработки (для выхода введите 'exit'):")

    while True:
        query_text = input("> ").strip()

        if query_text.lower() == 'exit':
            print("Выход из программы.")
            break

        if not query_text:
            print("Пожалуйста, введите непустой вопрос.")
            continue

        results, answer, prompt = query_chromadb(query_text, top_k=10)

        print(f"Запрос: {query_text}")
        print("\n=== Найденные чанки ===")
        for i, (text, meta, distance) in enumerate(
                zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
            print(f"{i + 1}. UID: {meta['uid']}, расстояние: {distance:.4f}")
            print(f"Текст: {text}\n")

        print("\n=== Сформированный prompt ===\n")
        print(prompt)

        print("\n=== Ответ модели Ollama ===\n")
        print(answer)

        with open("README.md", "a", encoding='utf-8') as f:
            f.write("\n## Результаты работы с ChromaDB + Ollama\n")
            f.write(f"- Запрос: {query_text}\n")
            f.write(f"- Ответ: {answer}\n")
            f.write(f"- Кол-во чанков: {len(results['documents'][0])}\n")
            f.write(f"- Пример расстояния: {min(results['distances'][0]):.4f}\n")
