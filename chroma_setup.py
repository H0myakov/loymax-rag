import json
import numpy as np
import chromadb
from pathlib import Path
import logging
import os

# Настройка логирования
logging.basicConfig(
    filename='chromadb.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Загрузка данных из JSON-файла
def load_data(input_path):
    logging.info(f"Загрузка данных из {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Настройка ChromaDB и загрузка данных
def setup_chromadb(data, collection_name="ru_bq_collection", batch_size=5000):
    # Директория для хранения данных ChromaDB
    persist_dir = Path("chroma_data").absolute()
    os.makedirs(persist_dir, exist_ok=True)

    # Создание клиента ChromaDB (новый API)
    logging.info(f"Создание PersistentClient в директории {persist_dir}")
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Попытка получить или пересоздать коллекцию
    try:
        collection = client.get_collection(name=collection_name)
        logging.warning(f"Коллекция {collection_name} уже существует, удаляем и создаём заново...")
        collection.delete()
        collection = client.create_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
        logging.info(f"Создана новая коллекция {collection_name}")

    # Подготовка данных
    ids = [str(item['uid']) for item in data]
    texts = [item['text'] for item in data]
    vectors = np.array([item['vector'] for item in data], dtype=np.float32)
    metadatas = [{"uid": str(item['uid']), "ru_wiki_pageid": str(item['ru_wiki_pageid'])} for item in data]

    # Загрузка данных по пакетам
    for i in range(0, len(data), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        logging.info(
            f"Добавление пакета {i // batch_size + 1} из {len(data) // batch_size + 1} (с {i} по {min(i + batch_size, len(data))})"
        )
        collection.add(
            ids=batch_ids,
            embeddings=batch_vectors,
            documents=batch_texts,
            metadatas=batch_metadatas
        )

    logging.info(f"Успешно добавлено {len(data)} записей в коллекцию.")
    logging.info(f"Содержимое директории {persist_dir}: {os.listdir(persist_dir)}")


if __name__ == "__main__":
    input_path = Path("data/vectored_RuBQ_2.0_paragraphs.json")

    # Проверка наличия файла
    if not input_path.exists():
        logging.error(f"Файл {input_path} не найден.")
        print(f"Файл не найден: {input_path}")
    else:
        data = load_data(input_path)
        setup_chromadb(data)
        print("Данные успешно загружены и сохранены в ChromaDB.")
