FROM ghcr.io/open-webui/pipelines:main

# Обновите репозитории и установите необходимые пакеты для установки Python 3.12
RUN apt-get update && apt-get install -y wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev libbz2-dev

# Скачайте и установите Python 3.12
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz \
    && tar -xf Python-3.12.0.tgz \
    && cd Python-3.12.0 \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make altinstall

# Удалите исходники для очистки
RUN rm -rf Python-3.12.0 Python-3.12.0.tgz

# Установите pip для Python 3.12
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.12

# Установите зависимости для вашего приложения с помощью pip3.12
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt


# Экспортируйте необходимый порт
EXPOSE 9099

# CMD ["sh", "-c", "./start.sh"]

# Команда запуска приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9099"]