# MLOps ДЗ-2: Модульное обучение с PyTorch Lightning, Hydra и DVC

## Цель ДЗ-2

- Продолжить работу с пакетом, созданным в ДЗ-1.
- Реализовать процесс обучения модели с использованием PyTorch Lightning.
- Организовать конфигурирование через Hydra.
- Подготовить инфраструктуру для управления данными и результатами экспериментов с помощью DVC.

## Задачи

1. **Использование созданного C++ расширения из ДЗ-1**  
   В ДЗ-1 был создан C++ модуль для вычисления энтропии. В этом задании требуется использовать этот модуль при подготовке данных в `torch.utils.data.Dataset` (в методе `__getitem__`).

2. **PyTorch Lightning**  
   - Реализовать `pl.LightningDataModule` и `pl.LightningModule`.
   - Обернуть процесс обучения в Lightning, чтобы удобно запускать обучение, валидацию и тестирование.

3. **Hydra для конфигурирования**  
   - Вся конфигурация должна быть определена в YAML-файлах.  
   - Не должно быть "магических чисел" или констант в коде — все параметры (размер датасета, размер батча, параметры модели, количество эпох и т.д.) должны браться из конфигов.
   - Точка входа: `train.py` с декоратором `@hydra.main`.

4. **DVC для версионирования данных и результатов**  
   - Подготовить проект к использованию DVC для хранения данных, моделей и результатов обучения.
   - Добавить примеры интеграции (например, `dvc add` для данных или модели, `dvc.yaml` для стадий обучения). Фактическая реализация полной CI/CD с DVC может выполняться отдельно, но структура и команды должны быть готовы.

5. **Docker Compose**  
   - Использовать Docker для сборки и запуска окружения.
   - В `Dockerfile` обеспечить сборку C++ пакета из `linalg-packages`.
   - В `docker-compose.yml` запустить контейнер, который будет выполнять обучение.
   - Использовать базовый образ PyTorch Lightning, чтобы не устанавливать Lightning вручную.

## Шаги реализации

1. **C++ биндинги и пакет**  
   Как в ДЗ-1, у нас есть пакет `linalg-packages`, содержащий C++ код и Pybind11 биндинги. В этом задании мы используем уже готовый пакет.

2. **Dataset и DataModule**  
   - Создать класс `ProbDistributionDataset`, который в `__getitem__` генерирует случайное распределение вероятностей, вычисляет для него энтропию через наш C++ модуль и возвращает `(probs_tensor, entropy_tensor)`.
   - Создать `ProbDistDataModule`, наследующий `pl.LightningDataModule`, который инкапсулирует логику разделения датасета на train/val/test.

3. **LightningModule**  
   - Создать `EntropyPredictor` как `pl.LightningModule` для предсказания энтропии или другой задачи, связанной с нашими данными.
   - Определить `forward`, `training_step`, `validation_step`, `test_step` и `configure_optimizers`.

4. **Hydra**  
   - Разбить конфигурацию на несколько YAML-файлов в папке `configs`:  
     - `config.yaml` (главный),
     - `data.yaml` (параметры датасета),
     - `model.yaml` (параметры модели),
     - `training.yaml` (параметры тренера и обучения).
   - В `train.py` использовать декоратор `@hydra.main`, чтобы загружать конфигурации и инициализировать модель, датамодуль, тренер.

5. **DVC**  
   - Инициализировать DVC в проекте (`dvc init`).
   - Добавить необходимые файлы (например, данные, модельные веса после обучения) под DVC (`dvc add`).
   - Подготовить `dvc.yaml` для процессов обучения (опционально).
   - Это позволит в будущем отслеживать версии данных и результатов.

6. **Docker и Docker Compose**  
   - В `Dockerfile` использовать базовый образ от PyTorch Lightning, чтобы не устанавливать Lightning.
   - Собрать C++ пакет внутри образа и установить его.
   - Установить необходимые Python-пакеты.
   - Скопировать коды `train.py`, `dataset.py`, `data_module.py`, `lightning_module.py`, `configs/`.
   - `docker-compose.yml` для запуска контейнера с обучением, а также при необходимости запуска MinIO или других сервисов.

## Запуск проекта

1. Собрать и запустить через Docker Compose:
   ```bash
   docker compose up --build