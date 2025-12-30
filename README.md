# Fashion-MNIST Classifier

Автор: Шатов Александр Витальевич

---

## Описание проекта

Полное описание: [Google Docs](https://docs.google.com/document/d/10TbP6OVHRLdVVGO-_ssQUhbQ4XXfMs81lsKKf4NLJwQ/edit?usp=sharing)

### Постановка задачи

Разработать систему для автоматической классификации изображений предметов одежды из набора данных Fashion-MNIST с помощью сверточной нейронной сети (CNN).

### Формат данных

**Входные данные:**

- Изображение 28×28 пикселей, форматы: PNG, JPG, TIFF
- Telegram-бот → FastAPI сервис получает numpy array `(1, 28, 28)`

**Выходные данные:**

- Предсказанный класс (0-9)
- Текстовое описание класса
- Вероятность предсказания

**Классы одежды:** 0. T-shirt/top

1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

### Датасет Fashion-MNIST

- **Train:** 60,000 примеров
- **Test:** 10,000 примеров
- **Размер:** 28×28 grayscale
- **Классы:** 10

**Источники:**

- [GitHub](https://github.com/zalandoresearch/fashion-mnist)
- [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

### Валидация

- **Train:** 48,000 (80%)
- **Validation:** 12,000 (20%, random_state=42)
- **Test:** 10,000 (отдельный набор)

### Метрики

- **Основная:** Accuracy
- **Дополнительные:** F1-score (macro/micro), Confusion Matrix

### Модели

**Baseline:**

- MLP (1 скрытый слой, 784 → 256 → 10)
- Ожидаемая accuracy: ~75-80%

**Основная модель:**

- CNN (2-3 conv layers + MaxPooling + FC)
- Dropout регуляризация
- Достигнутая accuracy: 91.46%

### Deployment (планируется)

- Telegram-бот (python-telegram-bot)
- Backend: FastAPI
- Инфраструктура: Docker Compose

---

## Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### 1. Clone Repository

```

git clone https://github.com/Mileeena/HSE-ML-Ops2

```

### 2. Create Virtual Environment

```

python -m venv .venv
source .venv/bin/activate # Linux/macOS

# .venv\Scripts\activate # Windows

```

### 3. Install Dependencies

**С UV (рекомендуется):**

```

pip install uv
uv pip install -e ".[dev]"

```

**Или с pip:**

```

pip install -e ".[dev]"

```

### 4. Setup Pre-commit

```

pre-commit install
pre-commit run --all-files

```

### 5. Start Infrastructure

```

docker-compose up -d
docker-compose ps # проверить статус

```

**Доступ:**

- MLflow UI: http://localhost:8080
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

### 6. Configure DVC

```

dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin

```

### 7. Download Data

```

# Скачать напрямую

python commands.py download

# Или из DVC remote

python commands.py dvc-pull

```

Данные сохраняются в `./data/FashionMNIST/`

---

## Train

### Базовое обучение

```

python commands.py train

```

Или:

```

python -m fashion_mnist_classifier.training.train

```

**Процесс:**

1. Загрузка конфигурации из `configs/config.yaml`
2. Автоматическое получение данных через DVC
3. Обучение CNN модели (10 эпох)
4. Логирование в MLflow (http://localhost:8080)
5. Сохранение чекпоинтов в `checkpoints/`

### Изменение параметров

```

# Learning rate

python -m fashion_mnist_classifier.training.train model.lr=0.0001

# Количество эпох

python -m fashion_mnist_classifier.training.train training.max_epochs=20

# Batch size

python -m fashion_mnist_classifier.training.train data.batch_size=128

# Несколько параметров

python -m fashion_mnist_classifier.training.train \
 model.lr=0.0005 \
 training.max_epochs=20 \
 data.batch_size=128

```

### Baseline модель

```

python -m fashion_mnist_classifier.training.train model=baseline

```

### Доступные конфигурации

**Data:** `configs/data/fashion_mnist.yaml`
**Models:**

- `configs/model/cnn.yaml` (default)
- `configs/model/baseline.yaml`

**Training:** `configs/training/default.yaml`
**Logging:** `configs/logging/mlflow.yaml`

---

## Результаты

**CNN (10 epochs):**

- Test Accuracy: 91.46%
- Test F1-macro: 91.49%
- Validation Loss: 0.215

**Baseline MLP:**

- Test Accuracy: ~78-82%

---

## Структура проекта

```

fashion-mnist-classifier/
├── fashion_mnist_classifier/ # Основной пакет
│ ├── data/ # Dataset, download
│ ├── models/ # CNN, baseline
│ ├── training/ # Training pipeline
│ └── utils/ # Helpers
├── configs/ # Hydra конфигурации
├── commands.py # CLI interface
├── docker-compose.yml # Infrastructure
├── pyproject.toml # Dependencies
└── README.md

```

---

## Технологии

- PyTorch 2.0+ / PyTorch Lightning 2.1+
- Hydra 1.3+ (конфигурация)
- MLflow 2.10+ (experiment tracking)
- DVC 3.0+ (версионирование данных)
- MinIO (S3-compatible storage)
- PostgreSQL 16 (MLflow backend)
- Docker Compose

---

```

```
