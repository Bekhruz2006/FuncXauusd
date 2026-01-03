"""
FuncXauusd - Production ML Trading System

Кластерная торговая система для XAUUSD с автоматическим поиском 
оптимальных конфигураций и двухуровневой архитектурой моделей.

Architecture:
    - Main Model: торговые сигналы на основе std-признаков
    - Meta Model: фильтрация кластеров на основе skewness-признаков
    - KMeans Clustering: автоматическое разбиение рыночных режимов

Key Features:
    - Автопоиск гиперпараметров (40 итераций)
    - Walk-forward валидация с чекпоинтами
    - ONNX экспорт для MetaTrader 5
    - Integrated risk management (ATR-based SL/TP)
"""

__version__ = "1.0.0"
__author__ = "Trading Systems Engineering Team"
__license__ = "Proprietary"

# Package metadata
PROJECT_NAME = "FuncXauusd"
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "H1", "H4", "D1", "W1", "MN"]
RANDOM_SEED = 42  # Для воспроизводимости

# Критические пороги
MIN_CLASS_BALANCE = 0.2
MIN_SAMPLES_PER_CLUSTER = 100
TARGET_VAL_ACCURACY = 0.75
TARGET_R2_SCORE = 0.95

# Импорты подмодулей
from . import data
from . import features
from . import labeling
from . import models
from . import export
from . import backtesting

__all__ = [
    'data',
    'features',
    'labeling',
    'models',
    'export',
    'backtesting'
]