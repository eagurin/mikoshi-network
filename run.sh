#!/bin/bash

# Проверка, что скрипт выполняется с правами суперпользователя
if [[ $EUID -ne 0 ]]; then
   echo "Пожалуйста, запустите скрипт с правами суперпользователя (sudo)." 
   exit 1
fi

echo "Установка необходимых системных пакетов..."
apt update
apt install -y python3.11 python3-pip python3.11-venv docker-compose

echo "Создание и активация виртуального окружения..."
python3.11 -m venv venv
source venv/bin/activate

echo "Обновление pip..."
pip install --upgrade pip

echo "Установка зависимостей из requirements.txt..."
pip install -r requirements.txt
pip install --upgrade 'r2r[core,ingestion-bundle,hatchet]'

echo "Обновление и перезапуск R2R..."
r2r docker-down
r2r update

echo "Остановка и удаление Docker-контейнеров и ресурсов..."
docker-compose down --remove-orphans
docker system prune --all --force --volumes

echo "Запуск Docker Compose с указанным проектом и профилем..."
docker-compose --project-name r2r-network --profile postgres up -d --build

echo "Запуск скриптов внутри контейнеров..."
docker-compose exec open-webui bash -c "./start.sh" || true
docker-compose exec pipelines bash -c "./start.sh" || true

echo "Установка Python и зависимостей внутри контейнера pipelines..."
commands="
apt update && \
apt install -y python3.11 python3.11-venv && \
python3.11 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt && \
pip install --upgrade 'r2r[core,ingestion-bundle,hatchet]'
"
docker-compose exec pipelines bash -c "$commands" || true

echo "Работа с Ollama внутри контейнера..."
ollama_commands="
ollama pull llama3.1 && \
ollama pull llama3.2 && \
ollama pull mxbai-embed-large
"
docker-compose exec ollama bash -c "$ollama_commands" || true

echo "Установка прав на выполнение для start.sh..."
docker-compose exec pipelines bash -c "chmod +x start.sh" || true

echo "Просмотр логов Docker Compose..."
docker-compose logs -f
