import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Загрузка данных
def load_data(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Подготовка векторов с сохранением uid
def prepare_vectors(data):
    uids = [item['uid'] for item in data]
    texts = [item['text'] for item in data]
    vectors = np.array([item['vector'] for item in data])
    return uids, texts, vectors

# Поиск похожих текстов
def find_similar_texts(query_text, uids, texts, vectors, top_k=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query_text])[0].reshape(1, -1)

    # Вычисляем косинусное сходство
    similarities = cosine_similarity(query_vector, vectors)[0]

    # Сортируем по убыванию сходства и берем топ-k
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]
    top_similarities = similarities[top_indices]
    top_uids = [uids[i] for i in top_indices]
    top_texts = [texts[i] for i in top_indices]
    top_vectors = [vectors[i].tolist() for i in top_indices]

    return list(zip(top_uids, top_texts, top_vectors, top_similarities))[:top_k]

if __name__ == "__main__":
    input_path = Path("data/vectored_RuBQ_2.0_paragraphs.json")

    # Загрузка данных
    data = load_data(input_path)
    uids, texts, vectors = prepare_vectors(data)

    # Пример запроса
    query_text = "футбольный клуб в россии"
    similar_results = find_similar_texts(query_text, uids, texts, vectors, top_k=10)

    # Вывод результатов
    print(f"Запрос: {query_text}")
    print("Топ-10 похожих текстов с векторами:")
    for uid, text, vector, similarity in similar_results:
        print(f"UID: {uid}")
        print(f"Текст: {text}")
        print(f"Сходство: {similarity:.4f}")
        print("---")