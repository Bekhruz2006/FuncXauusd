"""
Модуль работы с данными: загрузка, предобработка, синтетическая генерация.
"""

from .loader import MultiTimeframeDataLoader
from .preprocessor import DataPreprocessor
from .regime_detector import RegimeDetector
from .synthetic_generator import SyntheticDataGenerator

__all__ = [
    'MultiTimeframeDataLoader',
    'DataPreprocessor', 
    'RegimeDetector',
    'SyntheticDataGenerator'
]