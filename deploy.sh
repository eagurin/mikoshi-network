#!/bin/bash

# Загрузка compose.full.yaml
curl -o docker-compose.full.yml https://raw.githubusercontent.com/SciPhi-AI/R2R/main/py/compose.full.yaml

# Запуск docker-compose с нужными параметрами
docker compose -f docker-compose.full.yml -f docker-compose.mikoshi.yml --profile postgres --project-name mikoshi up -d
