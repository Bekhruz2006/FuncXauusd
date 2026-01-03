"""Модуль стратегий разметки данных"""

from .strategies import (
    get_labels_one_direction,
    validate_labels,
    print_label_distribution
)

__all__ = [
    'get_labels_one_direction',
    'validate_labels',
    'print_label_distribution'
]