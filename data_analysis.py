import pandas as pd
import requests
import json
import logging

# Настройка логирования
logging.basicConfig(filename='data_analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def download_dataset(url, output_path):
    """Скачивает датасет и сохраняет его локально."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(response.json(), f, ensure_ascii=False)
        logging.info(f"Датасет успешно скачан и сохранен в {output_path}")
    except Exception as e:
        logging.error(f"Ошибка при скачивании датасета: {e}")
        raise


def analyze_dataset(file_path):
    """Анализирует датасет и возвращает статистику."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Проверка наличия ключей и базовый анализ
        stats = {
            'total_documents': len(df),
            'average_text_length': df['text'].str.len().mean() if 'text' in df.columns else 0,
            'empty_documents': len(
                df[df['text'].isna() | (df['text'].str.strip() == '')]) if 'text' in df.columns else 0,
            'duplicate_uids': df['uid'].duplicated().sum() if 'uid' in df.columns else 0,
            'duplicate_texts': df['text'].duplicated().sum() if 'text' in df.columns else 0
        }

        logging.info("Результаты анализа данных:")
        for key, value in stats.items():
            logging.info(f"{key}: {value}")

        return stats
    except Exception as e:
        logging.error(f"Ошибка при анализе датасета: {e}")
        raise


if __name__ == "__main__":
    dataset_url = "https://raw.githubusercontent.com/vladislavneon/RuBQ/refs/heads/master/RuBQ_2.0/RuBQ_2.0_paragraphs.json"
    output_path = "data/RuBQ_2.0_paragraphs.json"
    download_dataset(dataset_url, output_path)
    stats = analyze_dataset(output_path)

    # Сохранение результатов в README
    with open("README.md", "w", encoding='utf-8') as f:
        f.write("# Анализ данных RuBQ 2.0\n\n")
        f.write("## Результаты анализа\n")
        for key, value in stats.items():
            f.write(f"- {key}: {value}\n")