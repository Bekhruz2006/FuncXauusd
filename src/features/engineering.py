"""
Feature Engineering для торговой системы

Создание признаков:
    - Main features: Standard Deviation на различных периодах
    - Meta features: Skewness для кластеризации

Принципы:
    - Все признаки рассчитываются на основе цен закрытия
    - Периоды задаются в конфигурации
    - Признаки нормализованы и очищены от NaN
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.stats import skew


def create_features(data: pd.DataFrame,
                   periods: List[int],
                   meta_periods: List[int] = None) -> pd.DataFrame:
    """
    Создание полного набора признаков для обучения
    
    Args:
        data: DataFrame с колонкой 'close'
        periods: Периоды для основных признаков (std)
        meta_periods: Периоды для мета-признаков (skewness)
    
    Returns:
        pd.DataFrame: Данные с добавленными признаками
        
    Структура выходных данных:
        - Индекс: datetime
        - close: исходная цена
        - feat_0, feat_1, ...: std-признаки
        - meta_0, meta_1, ...: skewness-признаки (если заданы)
    """
    if 'close' not in data.columns:
        raise ValueError("Отсутствует колонка 'close'")
    
    result = data[['close']].copy()
    
    # === ОСНОВНЫЕ ПРИЗНАКИ (Standard Deviation) ===
    print(f"📊 Создание признаков: {len(periods)} std-периодов", end='')
    
    for idx, period in enumerate(periods):
        result[f'feat_{idx}'] = _calculate_rolling_std(
            result['close'], 
            period
        )
    
    # === МЕТА-ПРИЗНАКИ (Skewness) ===
    if meta_periods is not None and len(meta_periods) > 0:
        print(f" + {len(meta_periods)} skewness-периодов")
        
        for idx, period in enumerate(meta_periods):
            result[f'meta_{idx}'] = _calculate_rolling_skewness(
                result['close'],
                period
            )
    else:
        print()
    
    # Удаление NaN (появляются из-за rolling операций)
    initial_len = len(result)
    result = result.dropna()
    dropped = initial_len - len(result)
    
    if dropped > 0:
        print(f"  ⚠ Удалено {dropped} NaN строк (из rolling окон)")
    
    print(f"  ✓ Итого признаков: {len(result.columns) - 1}")
    
    return result


def _calculate_rolling_std(series: pd.Series, period: int) -> pd.Series:
    """
    Расчет скользящего стандартного отклонения
    
    Args:
        series: Временной ряд цен
        period: Период окна
    
    Returns:
        pd.Series: Rolling standard deviation
    """
    return series.rolling(window=period).std()


def _calculate_rolling_skewness(series: pd.Series, period: int) -> pd.Series:
    """
    Расчет скользящей асимметрии (skewness)
    
    Skewness показывает направление и величину асимметрии распределения:
        - skew > 0: хвост справа (цены росли)
        - skew < 0: хвост слева (цены падали)
        - skew ≈ 0: симметричное распределение
    
    Args:
        series: Временной ряд цен
        period: Период окна
    
    Returns:
        pd.Series: Rolling skewness
    """
    # Используем scipy.stats.skew через apply
    return series.rolling(window=period).apply(
        lambda x: skew(x, bias=False),
        raw=True
    )


def create_features_multiframe(
    primary_data: pd.DataFrame,
    secondary_data_dict: dict,
    periods: List[int]
) -> pd.DataFrame:
    """
    Создание признаков с нескольких таймфреймов (для будущей реализации)
    
    Args:
        primary_data: Основной таймфрейм
        secondary_data_dict: {timeframe: DataFrame} с высшими ТФ
        periods: Периоды для признаков
    
    Returns:
        pd.DataFrame: Данные с мультифреймовыми признаками
    """
    result = create_features(primary_data, periods)
    
    # TODO: Добавить признаки с высших таймфреймов
    # Например: дневной RSI, недельный High/Low и т.д.
    
    return result


def normalize_features(features: pd.DataFrame,
                      method: str = 'standard') -> pd.DataFrame:
    """
    Нормализация признаков (опционально)
    
    Args:
        features: DataFrame с признаками
        method: 'standard' (z-score) или 'minmax'
    
    Returns:
        pd.DataFrame: Нормализованные признаки
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Отделяем close от признаков
    close = features['close']
    feat_cols = [col for col in features.columns if col != 'close']
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")
    
    normalized = pd.DataFrame(
        scaler.fit_transform(features[feat_cols]),
        columns=feat_cols,
        index=features.index
    )
    
    normalized['close'] = close
    
    return normalized


# === ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (для экспериментов) ===

def add_momentum_features(data: pd.DataFrame,
                         periods: List[int]) -> pd.DataFrame:
    """
    Добавление моментум-признаков
    
    Args:
        data: Данные с колонкой 'close'
        periods: Периоды для расчета
    
    Returns:
        pd.DataFrame: Данные с добавленными признаками
    """
    result = data.copy()
    
    for period in periods:
        # Rate of Change
        result[f'roc_{period}'] = result['close'].pct_change(period)
        
        # Momentum
        result[f'mom_{period}'] = result['close'].diff(period)
    
    return result.dropna()


def add_volatility_features(data: pd.DataFrame,
                           periods: List[int]) -> pd.DataFrame:
    """
    Добавление волатильных признаков
    
    Args:
        data: Данные с колонкой 'close'
        periods: Периоды для расчета
    
    Returns:
        pd.DataFrame: Данные с добавленными признаками
    """
    result = data.copy()
    
    for period in periods:
        # Historical Volatility (std of returns)
        returns = result['close'].pct_change()
        result[f'hvol_{period}'] = returns.rolling(period).std()
        
        # Average True Range (упрощенная версия без high/low)
        result[f'atr_{period}'] = result['close'].diff().abs().rolling(period).mean()
    
    return result.dropna()


def add_mean_reversion_features(data: pd.DataFrame,
                                periods: List[int]) -> pd.DataFrame:
    """
    Добавление mean-reversion признаков
    
    Args:
        data: Данные с колонкой 'close'
        periods: Периоды для расчета
    
    Returns:
        pd.DataFrame: Данные с добавленными признаками
    """
    result = data.copy()
    
    for period in periods:
        # Z-score (отклонение от скользящей средней)
        ma = result['close'].rolling(period).mean()
        std = result['close'].rolling(period).std()
        result[f'zscore_{period}'] = (result['close'] - ma) / std
        
        # Bollinger Bands distance
        upper = ma + 2 * std
        lower = ma - 2 * std
        result[f'bb_dist_{period}'] = (result['close'] - ma) / (upper - lower)
    
    return result.dropna()


# === УТИЛИТЫ ===

def get_feature_columns(df: pd.DataFrame,
                       prefix: str = 'feat_') -> List[str]:
    """
    Получение списка колонок с признаками
    
    Args:
        df: DataFrame
        prefix: Префикс признаков ('feat_', 'meta_')
    
    Returns:
        list: Список имен колонок
    """
    return [col for col in df.columns if col.startswith(prefix)]


def validate_features(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Валидация созданных признаков
    
    Проверки:
        - Отсутствие inf значений
        - Отсутствие NaN
        - Отсутствие константных признаков
    
    Returns:
        (bool, str): (валидность, сообщение об ошибке)
    """
    # Проверка на inf
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        return False, "Обнаружены inf значения"
    
    # Проверка на NaN
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        return False, f"Обнаружены NaN в колонках: {nan_cols}"
    
    # Проверка на константные признаки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [
        col for col in numeric_cols 
        if df[col].nunique() == 1
    ]
    
    if constant_cols:
        return False, f"Константные признаки: {constant_cols}"
    
    return True, "OK"


def print_feature_stats(df: pd.DataFrame) -> None:
    """
    Вывод статистики по признакам
    
    Полезно для отладки и проверки качества признаков
    """
    feat_cols = get_feature_columns(df, 'feat_')
    meta_cols = get_feature_columns(df, 'meta_')
    
    print(f"\n📈 Статистика признаков:")
    print(f"  • Основных (std): {len(feat_cols)}")
    print(f"  • Мета (skewness): {len(meta_cols)}")
    print(f"  • Всего строк: {len(df)}")
    
    if len(feat_cols) > 0:
        print(f"\n  Диапазоны std-признаков:")
        for col in feat_cols[:5]:  # Первые 5
            print(f"    {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")
    
    if len(meta_cols) > 0:
        print(f"\n  Диапазоны skewness-признаков:")
        for col in meta_cols:
            print(f"    {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")