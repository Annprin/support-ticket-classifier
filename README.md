# Telco Churn Predictor

ML-проект для прогнозирования оттока клиентов телеком-компании.
Модель реализована в виде нейронной сети (MLP) на PyTorch и оформлена по стандартам Hugging Face (`PreTrainedModel`).

Проект:

* полностью воспроизводим;
* использует **DVC** для версионирования данных и моделей;
* использует **MLflow** для трекинга экспериментов;
* упакован в **Docker-образ** для офлайн-инференса;
* развёртывается как **онлайн-сервис через TorchServe**.

---

## 1. Бизнес-цель проекта

Разработка ML-сервиса, который по характеристикам клиента предсказывает вероятность его ухода (churn).

### Бизнес-задачи

* **Идентификация группы риска** — выявление клиентов с высокой вероятностью оттока
* **Таргетированное удержание** — персонализированные предложения только для клиентов группы риска
* **Оптимизация затрат** — снижение расходов на удержание за счёт отказа от массовых скидок

---

## 2. Набор данных

* **Источник:** Telco Customer Churn Dataset (Kaggle)
* **Размер:** ~7 000 клиентов
* **Признаки:** категориальные и числовые (`gender`, `SeniorCitizen`, `tenure`, `Contract`, `MonthlyCharges` и др.)
* **Целевая переменная:** `Churn` (Yes / No)

Датасет требует предобработки:

* One-Hot Encoding категориальных признаков
* Масштабирование числовых признаков
* Обработка пропусков

Вся предобработка зашита в DVC-пайплайн.

---

## 3. Структура проекта и хранение артефактов

### Git (код и конфигурации)

* `src/` — код проекта
* `config.yaml` — конфигурация экспериментов
* `dvc.yaml`, `dvc.lock` — описание и фиксация DVC-пайплайна
* `reports/metrics.json` — метрики валидации
* `mlruns/` — директория MLflow (**не отслеживается Git**)
* `torchserve/` — артефакты и конфиги TorchServe (handler, config, model.pt и т.д.)
* `model-store/` — хранилище `.mar` архивов (обычно не коммитится, зависит от подхода)

### DVC (данные и модели)

* **Raw data:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (отслеживается через `.dvc`)
* **Processed data:** `data/processed/` (output стадии `prepare`)
* **Модель:** `models/churn_mlp_model/`

  * `config.json`
  * `model.safetensors` (или веса модели)
  * `preprocessor.joblib`

Все большие файлы и артефакты хранятся **вне Git**, но версионируются через **DVC**.

---

## 4. DVC-пайплайн

```text
prepare   →   train   →   evaluate
```

* **prepare** — загрузка и предобработка данных
* **train** — обучение модели и сохранение артефактов
* **evaluate** — валидация модели и сохранение метрик

Пайплайн описан в `dvc.yaml`, все версии зафиксированы в `dvc.lock`.

---

## 5. MLflow: трекинг экспериментов

Каждый запуск обучения (`python -m src.train --config config.yaml`) создаёт отдельный MLflow run с:

**Параметрами:** `epochs`, `learning_rate`, `batch_size`, `random_seed`, `hidden_layers`, `dropout` и т.д.
**Метриками:** accuracy, F1 (Churn), precision (Churn), train loss
**Артефактами:** `models/churn_mlp_model/`, `reports/metrics.json`, `config.yaml`, `dvc.lock`

### Запуск MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Открыть:

```
http://127.0.0.1:5000
```

---

## 6. Docker: офлайн-инференс

### Подготовка модели

Перед сборкой Docker-образа подтяни модель и данные:

```bash
dvc pull
```

### Сборка

```bash
docker build -t ml-app:v1 .
```

### Запуск

Контейнер выполняет офлайн-инференс через `python -m src.predict`.

**Вход:** CSV с сырыми признаками (как в исходном датасете).
Если колонка `Churn` присутствует — игнорируется.

**Выход:** CSV с предсказанным классом и вероятностью.

```bash
docker run --rm \
  -v "$(pwd)/sample.csv:/app/input.csv" \
  -v "$(pwd)/outputs:/app/outputs" \
  ml-app:v1 \
  --config config.yaml \
  --input_path /app/input.csv \
  --output_path /app/outputs/preds.csv \
  --proba
```

---

## 7. TorchServe: онлайн-сервис

В этом проекте модель развёрнута как REST-сервис через **TorchServe**.

### 7.1 Подготовка артефактов для TorchServe

1. Экспорт модели в `state_dict` (например, `torchserve/model.pt`)
2. Реализация кастомного обработчика `torchserve/handler.py` (preprocess + inference + postprocess)
3. Создание `.mar` архива через `torch-model-archiver`:

```bash
torch-model-archiver \
  --model-name mymodel \
  --version 1.0 \
  --serialized-file torchserve/model.pt \
  --handler torchserve/handler.py \
  --extra-files "models/churn_mlp_model/config.json,models/churn_mlp_model/preprocessor.joblib" \
  --export-path model-store \
  --force
```

> Если команда `torch-model-archiver` не найдена — установи TorchServe tooling:

```bash
pip install torchserve torch-model-archiver torch-workflow-archiver
```

### 7.2 Сборка Docker-образа TorchServe

Dockerfile для сервиса: `Dockerfile.torchserve`

```bash
docker build -t mymodel-serve:v1 -f Dockerfile.torchserve .
```

### 7.3 Запуск контейнера

```bash
docker run -d --name churn-serve \
  -p 8080:8080 -p 8081:8081 \
  mymodel-serve:v1
```

Полезные эндпоинты:

* Management API (модели): `http://localhost:8081/models`
* Inference API (предикт): `http://localhost:8080/predictions/<model_name>`

Проверить, что модель видна:

```bash
curl http://localhost:8081/models
curl "http://localhost:8081/models/mymodel?verbose=true"
```

### 7.4 Формат входных данных (REST)

Файл `sample_input.json` — JSON, содержащий **одну запись** или список записей.

Пример (одна запись):

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.35,
  "TotalCharges": 860.2
}
```

Запрос на инференс:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  --data-binary @sample_input.json \
  http://localhost:8080/predictions/mymodel
```

Ожидаемый ответ: JSON с предсказанием класса/вероятности (зависит от реализации `handler.py`).

### 7.5 Конфигурация TorchServe

Файл: `torchserve/config.properties`
Типичные параметры:

* `inference_address=http://0.0.0.0:8080`
* `management_address=http://0.0.0.0:8081`
* `metrics_address=http://0.0.0.0:8082`
* `default_response_timeout=120`

### 7.6 Troubleshooting (важно для Mac / ARM / sklearn)

Если воркеры TorchServe падают с ошибкой вида:

```
ImportError: ... libgomp ... cannot allocate memory in static TLS block
```

Решение (в Dockerfile.torchserve):

* установить системную библиотеку `libgomp1`
* ограничить число потоков OpenMP
* выставить `LD_PRELOAD` (важно для worker subprocess)

Пример:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

Логи сервиса:

```bash
docker logs churn-serve --tail 200
```

---

## 8. Как запустить проект с нуля (полная воспроизводимость)

```bash
git clone <REPO_URL>
cd <PROJECT_DIR>

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "dvc[s3]"

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=ru-central1

dvc pull
dvc repro
```

---

## 9. Воспроизводимость

Проект гарантирует:

* полную воспроизводимость данных и моделей;
* восстановление любой версии проекта:

```bash
git checkout <commit>
dvc pull
dvc repro
```

* отсутствие крупных файлов и моделей в Git;
* трекинг всех экспериментов через MLflow;
* воспроизводимый Docker-образ для офлайн-инференса;
* Docker-контейнер TorchServe, поднимающий REST API для онлайн-предикта.