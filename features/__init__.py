"""
Модуль вычисления признаков: атомарные функции, макро-признаки,
мультитаймфреймовая агрегация.
"""

from .atomic_functions import AtomicFunctionLibrary
from .macro_features import MacroFeatureComputer
from .multitimeframe import MultiTimeframeAggregator

__all__ = [
    'AtomicFunctionLibrary',
    'MacroFeatureComputer',
    'MultiTimeframeAggregator'
]