
FROM python:3.12-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем все файлы проекта в контейнер
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт, на котором будет работать FastAPI
EXPOSE 8000

# Команда для запуска FastAPI
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]
