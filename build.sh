#!/bin/bash

# Exit on error
set -e

# Указываем имя окружения
CONDA_ENV_NAME="r2r_env"

# Создание conda-окружения и активация его
echo "Создаю conda окружение с именем $CONDA_ENV_NAME c Python 3.11..."
conda create -n "$CONDA_ENV_NAME" python=3.11 -y

echo "Активирую окружение $CONDA_ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Обновляем pip
echo "Обновляю pip..."
pip install --upgrade pip

# Устанавливаем r2r с необходимыми зависимостями
echo "Устанавливаю r2r и необходимые зависимости..."
pip install --upgrade 'r2r[core,ingestion-bundle,hatchet]'

# Обновляем r2r
echo "Обновляю r2r..."
r2r update

# Запускаем r2r с нужными опциями
echo "Запускаю r2r serve..."
r2r serve --docker --config-name=full --full --build

echo "Готово!"
