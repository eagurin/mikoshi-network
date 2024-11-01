# Этап 1: Используем минимальный базовый образ Python для сборки
FROM python:3.11-slim AS builder

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем необходимые системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements.txt для установки зависимостей (используем кеширование Docker)
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip
# RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Копируем остальной исходный код приложения
COPY . .

# Этап 2: Создаем финальный образ с минимальным размером
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем установленное приложение из этапа сборки
COPY --from=builder /app /app

# Создаем пользователя для работы приложения
RUN useradd -ms /bin/bash appuser
USER appuser

# Открываем необходимый порт
EXPOSE 9099

# RUN apt-get install python3.11-venv
# RUN python3 -m venv venv
# RUN . venv/bin/activate
# RUN pip install --upgrade pip
# RUN pip install  --upgrade  -r requirements.txt
RUN pip install --upgrade 'r2r[core,ingestion-bundle,hatchet]'
# RUN r2r serve --config-name=full_local_llm --full --docker --project-name=mikoshi-network

# Команда запуска приложения
CMD ["python3", "-m", "r2r", "serve", "--config-name=local_llm", "--project-name=r2r_default"]
