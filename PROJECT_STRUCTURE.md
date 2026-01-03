# 🏗️ Структура проекта FuncXauusd

## 📁 Полная структура директорий

```
FuncXauusd/
│
├── .gitignore                          # Исключения Git (созд создан)
├── .gitkeep файлы                      # Для пустых директорий в Git
├── README.md                           # Главная документация (создан)
├── requirements.txt                    # Python зависимости (создан)
├── PROJECT_STRUCTURE.md                # Этот файл
│
├── config/                             # Конфигурации
│   ├── training_config.yaml           # Главный конфиг (создан)
│   └── secrets.yaml.example           # Пример конфига с секретами
│
├── data/                               # Данные (не в Git)
│   ├── raw/                           # Исходные CSV файлы
│   │   ├── .gitkeep
│   │   └── XAUUSD_H1.csv              # Загрузить из MT5
│   └── processed/                      # Обработанные данные
│       └── .gitkeep
│
├── models/                             # Модели (не в Git)
│   ├── trained/                       # Обученные .cbm файлы
│   │   └── .gitkeep
│   └── exported/                       # ONNX + MQL файлы
│       └── .gitkeep
│
├── src/                                # Исходный код Python
│   ├── __init__.py
│   │
│   ├── data/                          # Работа с данными
│   │   ├── __init__.py
│   │   └── loader.py                  # ✅ СОЗДАТЬ
│   │
│   ├── features/                      # Feature engineering
│   │   ├── __init__.py
│   │   └── engineering.py             # ✅ СОЗДАТЬ
│   │
│   ├── labeling/                      # Разметка данных
│   │   ├── __init__.py
│   │   └── strategies.py              # ✅ АДАПТИРОВАТЬ из labeling_lib.py
│   │
│   ├── models/                        # Обучение моделей
│   │   ├── __init__.py
│   │   ├── trainer.py                 # ✅ СОЗДАТЬ (основная логика)
│   │   └── validator.py               # ✅ СОЗДАТЬ
│   │
│   ├── export/                        # Экспорт моделей
│   │   ├── __init__.py
│   │   └── onnx_exporter.py           # ✅ АДАПТИРОВАТЬ из export_lib.py
│   │
│   └── backtesting/                   # Тестирование
│       ├── __init__.py
│       └── tester.py                  # ✅ АДАПТИРОВАТЬ из tester_lib.py
│
├── scripts/                            # Исполняемые скрипты
│   └── train_cluster_model.py         # ✅ Главный скрипт (создан)
│
├── mql5/                               # MetaTrader 5 код
│   ├── Experts/                       # Советники
│   │   └── OneDirectionBot.mq5        # ✅ АДАПТИРОВАТЬ
│   └── Include/                        # Include файлы
│       └── ModelInclude.mqh.template  # ✅ Шаблон (создается при экспорте)
│
├── notebooks/                          # Jupyter notebooks (опционально)
│   ├── .gitkeep
│   └── exploratory_analysis.ipynb     # Для экспериментов
│
├── tests/                              # Юнит-тесты (TODO для будущих агентов)
│   ├── __init__.py
│   └── test_labeling.py
│
├── logs/                               # Логи (не в Git)
│   └── .gitkeep
│
├── results/                            # Результаты экспериментов (не в Git)
│   ├── plots/
│   │   └── .gitkeep
│   └── reports/
│       └── .gitkeep
│
└── docs/                               # Документация
    ├── implementation_plan.md          # ✅ СОЗДАТЬ (6 этапов из документа)
    ├── api_reference.md                # TODO
    └── architecture.md                 # TODO
```

## 🔧 Инструкции по созданию файлов

### ✅ УЖЕ СОЗДАНЫ (в артефактах):
1. `.gitignore`
2. `README.md`
3. `requirements.txt`
4. `config/training_config.yaml`
5. `scripts/train_cluster_model.py`

### 📝 НУЖНО СОЗДАТЬ:

#### 1. Базовые модули (`src/`)

**`src/__init__.py`** (пустой):
```python
"""FuncXauusd - Production ML Trading System"""
__version__ = "1.0.0"
```

**`src/data/__init__.py`** (пустой)

**`src/data/loader.py`**:
```python
# Функции:
# - load_price_data(config) -> pd.DataFrame
# - cache_prices(config) -> None
# - get_cached_prices() -> pd.DataFrame
```

**`src/features/__init__.py`** (пустой)

**`src/features/engineering.py`**:
```python
# Функции:
# - create_features(data, periods, meta_periods) -> pd.DataFrame
# Аналог get_features() из one_direction_clusters.py
```

**`src/labeling/__init__.py`**:
```python
from .strategies import get_labels_one_direction
```

**`src/labeling/strategies.py`**:
```python
# Взять из labeling_lib.py:
# - calculate_labels_one_direction (с @njit)
# - get_labels_one_direction
```

**`src/models/__init__.py`** (пустой)

**`src/models/trainer.py`**:
```python
# Класс ClusterModelTrainer:
# - __init__(config)
# - train_all_clusters() -> list
# - _train_single_cluster(cluster_id) -> dict
# Логика из fit_model() и train_with_config()
```

**`src/models/validator.py`**:
```python
# Функции:
# - validate_class_balance(labels) -> bool
# - validate_sample_size(data, min_samples) -> bool
```

**`src/export/__init__.py`**:
```python
from .onnx_exporter import export_to_onnx
```

**`src/export/onnx_exporter.py`**:
```python
# Функции из export_lib.py:
# - export_to_onnx(model_main, model_meta, config, r2_score)
# - _generate_mql_include(config, periods, meta_periods)
```

**`src/backtesting/__init__.py`**:
```python
from .tester import tester_one_direction, test_model_one_direction
```

**`src/backtesting/tester.py`**:
```python
# Из tester_lib.py:
# - process_data_one_direction (с @jit)
# - tester_one_direction
# - test_model_one_direction
```

#### 2. Документация

**`docs/implementation_plan.md`**:
```markdown
# План реализации (6 этапов)

## Этап 1: Переформатирование данных
[Текст из первого документа]

## Этап 2: Динамические чекпоинты
[Текст из первого документа]

...

## Этап 6: Система деградации
[Текст из первого документа]
```

#### 3. MQL5 код

**`mql5/Experts/OneDirectionBot.mq5`**:
- Взять из предоставленного файла
- ИСПРАВИТЬ размеры массивов (уже указано в комментариях)

**`mql5/Include/ModelInclude.mqh.template`**:
- Шаблон для генерации через Python
- Генерируется автоматически при экспорте

#### 4. .gitkeep файлы

Создать пустые файлы `.gitkeep` в:
- `data/raw/`
- `data/processed/`
- `models/trained/`
- `models/exported/`
- `logs/`
- `results/plots/`
- `results/reports/`
- `notebooks/`

### 5. Дополнительные файлы (опционально)

**`config/secrets.yaml.example`**:
```yaml
# Пример конфига с секретными данными
mt5:
  account: "YOUR_ACCOUNT"
  password: "YOUR_PASSWORD"
  server: "YOUR_BROKER"
```

**`.github/workflows/ci.yml`** (если нужен CI/CD):
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
```

## 🚀 Порядок действий для агентов

### Фаза 1: Базовая структура
1. ✅ Создать `.gitignore`, `README.md`, `requirements.txt`
2. ✅ Создать `config/training_config.yaml`
3. ✅ Создать `scripts/train_cluster_model.py`
4. Создать все `.gitkeep` файлы
5. Создать `docs/implementation_plan.md`

### Фаза 2: Модули данных и признаков
1. Создать `src/data/loader.py`
2. Создать `src/features/engineering.py`
3. Адаптировать `src/labeling/strategies.py` из `labeling_lib.py`

### Фаза 3: Модули обучения
1. Создать `src/models/trainer.py` (ключевой модуль!)
2. Создать `src/models/validator.py`
3. Адаптировать `src/backtesting/tester.py` из `tester_lib.py`

### Фаза 4: Экспорт и интеграция
1. Адаптировать `src/export/onnx_exporter.py` из `export_lib.py`
2. Исправить `mql5/Experts/OneDirectionBot.mq5`
3. Создать шаблон `mql5/Include/ModelInclude.mqh.template`

### Фаза 5: Тестирование
1. Загрузить `data/raw/XAUUSD_H1.csv`
2. Запустить `python scripts/train_cluster_model.py`
3. Проверить экспорт ONNX моделей
4. Протестировать советник на демо-счете

## 📌 Ключевые принципы

### Модульность
- Каждый модуль = одна ответственность
- Переиспользование кода
- Легкое тестирование

### Конфигурируемость
- Все параметры в `config/training_config.yaml`
- Никаких хардкодов
- Поддержка мультифрейма через конфиг

### Производительность
- Numba JIT для критических циклов
- Кэширование данных
- Параллелизация где возможно

### Воспроизводимость
- Фиксированный `random_seed = 42`
- Детерминированность CatBoost
- Логирование всех операций

## ⚠️ Критические моменты

### 1. Размеры массивов в MQL5
**ИСПРАВЛЕНИЕ** в `OneDirectionBot.mq5`:
```cpp
double f[10];  // было ArraySize(features), должно быть 10
double f_m[1]; // было ArraySize(features_m), должно быть 1
```

### 2. Разделение признаков
При создании признаков:
- **Основные**: `std` (стандартное отклонение)
- **Мета**: `skewness` (асимметрия)

### 3. Кластеризация
- Использовать `StandardScaler` перед KMeans
- Кластеризовать ТОЛЬКО по мета-признакам
- Обучать модели для каждого кластера отдельно

### 4. Экспорт ONNX
- Сохранять 2 модели: main + meta
- Генерировать .mqh файл с правильными функциями
- Проверять размерности входов/выходов

## 🎯 Готовность к GitHub

### Что включить в первый коммит:
- ✅ `.gitignore`
- ✅ `README.md`
- ✅ `requirements.txt`
- ✅ `PROJECT_STRUCTURE.md` (этот файл)
- ✅ `config/training_config.yaml`
- ✅ `scripts/train_cluster_model.py`
- `.gitkeep` файлы в пустых директориях
- `docs/implementation_plan.md`
- Базовые `__init__.py` файлы

### Что НЕ включать:
- ❌ `data/` (кроме .gitkeep)
- ❌ `models/` (кроме .gitkeep)
- ❌ `logs/`
- ❌ `results/`
- ❌ `*.pyc`, `__pycache__/`
- ❌ `.vscode/`, `.idea/`

## 📚 Дополнительные ресурсы

### Для изучения:
- [CatBoost Documentation](https://catboost.ai/docs/)
- [ONNX Documentation](https://onnx.ai/onnx/intro/)
- [MQL5 Documentation](https://www.mql5.com/en/docs)

### Полезные команды:

```bash
# Инициализация Git
git init
git add .
git commit -m "Initial commit: Project structure"

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск обучения
python scripts/train_cluster_model.py

# Проверка структуры
tree -L 3 -I '__pycache__|*.pyc|venv'
```

---

**Статус:** Фундамент готов к разработке 🚀  
**Следующий шаг:** Создание модулей в `src/`