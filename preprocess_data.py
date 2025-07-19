import pandas as pd
import json
import logging
import re
import yaml
from pathlib import Path

# Настройка логирования
logging.basicConfig(filename='preprocess.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Загрузка конфигурации
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Предобработка текста
def preprocess_text(text, min_length=10):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = text.lower()
        if len(text) < min_length:
            return None
        return text
    except Exception as e:
        logging.error(f"Ошибка при предобработке текста: {e}")
        return None


# Проверка качества данных
def quality_check(df, config):
    logging.info(f"Столбцы датафрейма: {df.columns.tolist()}")
    if 'text' not in df.columns:
        logging.error(f"Столбцы в датафрейме: {df.columns.tolist()}")
        raise KeyError("Столбец 'text' не найден. Проверь структуру данных.")

    logging.info(f"Тип данных в столбце 'text': {df['text'].dtype}")
    initial_count = len(df)
    filtered_df = df.copy()

    try:
        logging.info(f"Количество записей до обработки: {len(filtered_df)}")
        empty_count = 0

        # Проверка и удаление дубликатов по uid
        duplicate_uids = filtered_df[filtered_df['uid'].duplicated(keep=False)]
        if not duplicate_uids.empty:
            logging.warning(
                f"Найдено {len(duplicate_uids)} записей с дублирующимися uid: {duplicate_uids['uid'].tolist()}")
            filtered_df = filtered_df.drop_duplicates(subset=['uid'], keep='first')

        # Проверка и удаление дубликатов по тексту
        duplicate_texts = filtered_df[filtered_df['text'].duplicated(keep=False)]
        if not duplicate_texts.empty:
            logging.warning(f"Найдено {len(duplicate_texts)} записей с дублирующимися текстами")
            filtered_df = filtered_df.drop_duplicates(subset=['text'], keep='first')

        # Применение предобработки текста
        filtered_df['text'] = filtered_df['text'].apply(preprocess_text, min_length=config.get('min_length', 10))
        filtered_df = filtered_df.dropna(subset=['text'])

        final_count = len(filtered_df)
        logging.info(f"Изначальное количество документов: {initial_count}")
        logging.info(f"Осталось документов после фильтрации: {final_count}")
        logging.info(f"Удалено документов: {initial_count - final_count}")

        return filtered_df
    except Exception as e:
        logging.error(f"Ошибка при проверке качества: {e}")
        raise


if __name__ == "__main__":
    data_path = Path("data/parsed_RuBQ_2.0_paragraphs.json")
    config_path = Path("configs/config.yaml")

    if not config_path.exists():
        config = {
            'data_path': str(data_path),
            'min_length': 10,
            'output_path': 'data/processed_RuBQ_2.0_paragraphs.json'
        }
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"Тип данных после загрузки: {type(data)}, первые 2 записи: {data[:2]}")
    df = pd.DataFrame(data)

    config = load_config(config_path)
    processed_df = quality_check(df, config)

    processed_df.to_json(config['output_path'], orient='records', force_ascii=False)
    logging.info(f"Обработанные данные сохранены в {config['output_path']}")

    with open("README.md", "a", encoding='utf-8') as f:
        f.write("\n## Результаты предобработки\n")
        f.write(f"- Изначальное количество документов: {len(df)}\n")
        f.write(f"- Осталось документов после фильтрации: {len(processed_df)}\n")
        f.write(f"- Удалено документов: {len(df) - len(processed_df)}\n")