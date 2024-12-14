# Этап сборки пакета
FROM python:3.11-slim AS builder

# Установка необходимых пакетов для сборки C++ модуля
RUN apt-get update && \
    apt-get install -y build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /packages

# Копируем логику сборки (pyproject.toml, setup.py, MANIFEST.in, C++ файлы)
COPY ./linalg-packages /packages

# Установка необходимых пакетов для сборки
RUN pip install --upgrade pip && pip install build pybind11 numpy

# Сборка пакета
RUN python3 -m build


# Основной этап
FROM pytorchlightning/pytorch_lightning:latest-py3.11-torch2.1-cuda12.1.0

WORKDIR /app

# Копируем зависимости (requirements.txt) и устанавливаем их
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем собранный пакет из образа builder и устанавливаем его
COPY --from=builder /packages/dist/*.whl /app/
RUN pip install /app/*.whl

# Копируем остальные файлы проекта (train.py, configs, dataset.py, lightning_module.py, data_module.py и т.д.)
COPY . /app

# По умолчанию запускаем train.py
CMD ["python3", "train.py"]
