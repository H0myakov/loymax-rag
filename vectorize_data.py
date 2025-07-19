import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Настройка логирования
logging.basicConfig(filename='vectorize.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
def load_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Векторизация текстов
def vectorize_texts(data, model):
    texts = [item['text'] for item in data if item['text'] and item['text'].strip()]
    logging.info(f"Найдено текстов для векторизации: {len(texts)}")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts

# Сохранение результатов
def save_vectors(embeddings, texts, output_path):
    data_with_vectors = [
        {'uid': data[i]['uid'], 'ru_wiki_pageid': data[i]['ru_wiki_pageid'], 'text': text, 'vector': embedding.tolist()}
        for i, (embedding, text) in enumerate(zip(embeddings, texts))
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_with_vectors, f, ensure_ascii=False, indent=2)
    logging.info(f"Векторы сохранены в {output_path}")

if __name__ == "__main__":
    input_path = Path("data/processed_RuBQ_2.0_paragraphs.json")
    output_path = Path("data/vectored_RuBQ_2.0_paragraphs.json")

    # Загрузка модели
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda') # Модель с размером вектора 384
    logging.info("Модель загружена: all-MiniLM-L6-v2")

    # Загрузка и обработка данных
    data = load_data(input_path)
    embeddings, texts = vectorize_texts(data, model)

    # Сохранение результатов
    save_vectors(embeddings, texts, output_path)