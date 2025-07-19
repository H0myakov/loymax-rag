import json
import requests
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(filename='parse_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_and_parse_json(url, output_path):
    try:
        # Скачивание файла
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Датасет успешно загружен. Тип данных: {type(data)}")

        # Проверка структуры
        if not isinstance(data, list):
            logging.error("Данные не являются списком. Проверь структуру JSON.")
            raise ValueError("Ожидался список словарей.")

        logging.info(f"Количество записей: {len(data)}")
        logging.info(f"Пример первой записи: {data[0]}")

        # Сохранение спарсенных данных
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Данные сохранены в {output_path}")

    except Exception as e:
        logging.error(f"Ошибка при парсинге или загрузке: {e}")
        raise


if __name__ == "__main__":
    dataset_url = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json"
    output_path = Path("data/parsed_RuBQ_2.0_paragraphs.json")
    download_and_parse_json(dataset_url, output_path)