"""
Стратегии разметки данных для обучения

Реализованы методы:
    - get_labels_one_direction: Разметка для однонаправленной торговли
    
Принципы разметки:
    - Label = 1: Ожидается движение в заданном направлении на markup пунктов
    - Label = 0: Движение не достигнет markup в заданном окне (min_bars, max_bars)
    - Используется случайный горизонт прогноза для робастности
"""

import random
import numpy as np
import pandas as pd
from numba import njit
from typing import Literal


@njit
def calculate_labels_one_direction(
    close_data: np.ndarray,
    markup: float,
    min_bars: int,
    max_bars: int,
    direction: str
) -> np.ndarray:
    """
    Расчет меток для однонаправленной торговли (Numba-optimized)
    
    Логика:
        - Для каждого бара берется случайное окно [min_bars, max_bars]
        - Проверяется, достигнет ли цена markup в этом окне
        - Label = 1 если достигнет, 0 если нет
    
    Args:
        close_data: Массив цен закрытия
        markup: Порог движения цены (в пунктах инструмента)
        min_bars: Минимальный горизонт прогноза
        max_bars: Максимальный горизонт прогноза
        direction: 'buy' или 'sell'
    
    Returns:
        np.ndarray: Массив меток [0, 1]
    
    Example:
        >>> close = np.array([100, 101, 102, 99, 98])
        >>> labels = calculate_labels_one_direction(close, 2.0, 1, 3, 'buy')
        >>> # labels[0] = 1 если в следующих 1-3 барах цена >= 102
    """
    labels = np.empty(len(close_data) - max_bars, dtype=np.float64)
    
    for i in range(len(labels)):
        # Случайный горизонт прогноза
        rand = random.randint(min_bars, max_bars)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]
        
        if direction == "sell":
            # Для продажи: цена должна упасть на markup
            if (future_pr + markup) < curr_pr:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
                
        elif direction == "buy":
            # Для покупки: цена должна вырасти на markup
            if (future_pr - markup) > curr_pr:
                labels[i] = 1.0
            else:
                labels[i] = 0.0
        else:
            # Некорректное направление - все нули
            labels[i] = 0.0
    
    return labels


def get_labels_one_direction(
    dataset: pd.DataFrame,
    markup: float,
    min_bars: int = 1,
    max_bars: int = 15,
    direction: Literal['buy', 'sell'] = 'buy'
) -> pd.DataFrame:
    """
    Создание меток для однонаправленной торговли
    
    Args:
        dataset: DataFrame с колонкой 'close'
        markup: Порог движения для сигнала (например, 0.25 для XAUUSD)
        min_bars: Минимальное окно в будущее
        max_bars: Максимальное окно в будущее
        direction: 'buy' (рост) или 'sell' (падение)
    
    Returns:
        pd.DataFrame: Исходные данные + колонка 'labels'
    
    Raises:
        ValueError: Если direction не 'buy' или 'sell'
        ValueError: Если 'close' отсутствует в dataset
    
    Example:
        >>> df = pd.DataFrame({'close': [100, 102, 101, 99, 103]})
        >>> labeled = get_labels_one_direction(df, markup=1.5, direction='buy')
        >>> labeled['labels']
        0    1.0  # 102 > 100 + 1.5
        1    0.0  # 99 < 102 + 1.5
        2    1.0  # 103 > 101 + 1.5
    """
    # Валидация
    if direction not in ['buy', 'sell']:
        raise ValueError(f"direction должен быть 'buy' или 'sell', получено: {direction}")
    
    if 'close' not in dataset.columns:
        raise ValueError("В dataset отсутствует колонка 'close'")
    
    if len(dataset) < max_bars + 100:
        raise ValueError(
            f"Недостаточно данных: {len(dataset)} баров "
            f"(минимум {max_bars + 100})"
        )
    
    # Расчет меток
    close_data = dataset['close'].values
    labels = calculate_labels_one_direction(
        close_data,
        markup,
        min_bars,
        max_bars,
        direction
    )
    
    # Усечение датасета до длины меток
    result = dataset.iloc[:len(labels)].copy()
    result['labels'] = labels
    
    # Удаление NaN (если были)
    result = result.dropna()
    
    # Статистика
    total = len(result)
    positive = (result['labels'] == 1.0).sum()
    balance = positive / total if total > 0 else 0
    
    print(f"📊 Разметка ({direction}):")
    print(f"  • Всего: {total} примеров")
    print(f"  • Сигналов (1): {positive} ({balance:.1%})")
    print(f"  • Ожиданий (0): {total - positive} ({1-balance:.1%})")
    print(f"  • Баланс классов: {balance:.3f}")
    
    return result


# === ДОПОЛНИТЕЛЬНЫЕ СТРАТЕГИИ РАЗМЕТКИ ===

def get_labels_bidirectional(
    dataset: pd.DataFrame,
    markup: float,
    min_bars: int = 1,
    max_bars: int = 15
) -> pd.DataFrame:
    """
    Двунаправленная разметка (Buy/Sell/Hold)
    
    Args:
        dataset: DataFrame с колонкой 'close'
        markup: Порог движения
        min_bars: Минимальное окно
        max_bars: Максимальное окно
    
    Returns:
        pd.DataFrame: Данные с метками [0: Wait, 1: Buy, 2: Sell]
    
    Note:
        В текущей архитектуре не используется (только one-direction),
        но может быть полезно для будущих экспериментов
    """
    close_data = dataset['close'].values
    labels = []
    
    for i in range(len(close_data) - max_bars):
        rand = random.randint(min_bars, max_bars)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]
        
        if (future_pr - markup) > curr_pr:
            labels.append(1.0)  # Buy
        elif (future_pr + markup) < curr_pr:
            labels.append(2.0)  # Sell
        else:
            labels.append(0.0)  # Wait
    
    result = dataset.iloc[:len(labels)].copy()
    result['labels'] = labels
    
    return result.dropna()


def validate_labels(dataset: pd.DataFrame,
                   min_balance: float = 0.2) -> tuple:
    """
    Валидация качества разметки
    
    Args:
        dataset: Размеченные данные с колонкой 'labels'
        min_balance: Минимальный баланс классов
    
    Returns:
        (bool, str): (валидность, сообщение)
    
    Проверки:
        - Наличие обоих классов
        - Баланс классов >= min_balance
        - Отсутствие NaN в метках
    """
    if 'labels' not in dataset.columns:
        return False, "Отсутствует колонка 'labels'"
    
    labels = dataset['labels']
    
    # Проверка на NaN
    if labels.isna().any():
        return False, f"NaN в метках: {labels.isna().sum()} шт."
    
    # Подсчет классов
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        return False, f"Только один класс: {unique_labels}"
    
    # Баланс классов
    total = len(labels)
    positive = (labels == 1.0).sum()
    balance = positive / total
    
    if balance < min_balance or balance > (1 - min_balance):
        return False, f"Дисбаланс классов: {balance:.3f} (мин {min_balance})"
    
    return True, f"OK (баланс: {balance:.3f})"


def print_label_distribution(dataset: pd.DataFrame) -> None:
    """
    Вывод распределения меток
    
    Полезно для анализа качества разметки
    """
    if 'labels' not in dataset.columns:
        print("⚠️ Колонка 'labels' не найдена")
        return
    
    labels = dataset['labels']
    counts = labels.value_counts().sort_index()
    total = len(labels)
    
    print(f"\n📊 Распределение меток:")
    for label, count in counts.items():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {int(label)}: {count:6d} ({pct:5.1f}%) {bar}")


def analyze_label_sequences(dataset: pd.DataFrame) -> dict:
    """
    Анализ последовательностей меток
    
    Проверяет:
        - Средняя длина последовательностей одинаковых меток
        - Максимальная длина последовательности
        - Количество переключений
    
    Returns:
        dict: Статистика последовательностей
    """
    labels = dataset['labels'].values
    
    # Поиск переключений
    switches = np.diff(labels) != 0
    n_switches = switches.sum()
    
    # Длины последовательностей
    sequence_lengths = []
    current_length = 1
    
    for i in range(1, len(labels)):
        if labels[i] == labels[i-1]:
            current_length += 1
        else:
            sequence_lengths.append(current_length)
            current_length = 1
    sequence_lengths.append(current_length)
    
    return {
        'total_labels': len(labels),
        'n_switches': n_switches,
        'avg_sequence_length': np.mean(sequence_lengths),
        'max_sequence_length': np.max(sequence_lengths),
        'min_sequence_length': np.min(sequence_lengths)
    }