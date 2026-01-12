"""
Advanced Feature Engineering для FuncXauusd (Production Grade).

Генерация признаков для глубокого обучения на данных 2004-2025.
Принципы:
    1. Стационарность: Использование Log Returns и относительных отклонений.
    2. Мульти-доменность: Волатильность, Импульс, Тренд, Время.
    3. Оптимизация памяти: Downcasting до float32 для обработки 20 лет истории.

Определяет логику создания признаков и утилиты для работы с колонками.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import skew, kurtosis

# =============================================================================
# ГЛАВНЫЙ КОНТРОЛЛЕР
# =============================================================================

def create_features(data: pd.DataFrame,
                   periods: List[int],
                   meta_periods: List[int] = None) -> pd.DataFrame:
    """
    Создание полного набора профессиональных признаков.
    
    Args:
        data: DataFrame с OHLCV данными
        periods: Список периодов для скользящих окон (например, [14, 50, 200])
        meta_periods: Периоды для статистических мета-признаков
    
    Returns:
        pd.DataFrame: Очищенный датафрейм с признаками
    """
    # Проверка обязательных колонок
    req_cols = ['open', 'high', 'low', 'close']
    if not all(c in data.columns for c in req_cols):
        raise ValueError(f"Missing required columns: {req_cols}")
    
    # Работаем с копией, чтобы не ломать исходный кэш
    df = data.copy()
    
    # Конвертация в float32 для экономии памяти (важно для 20 лет истории)
    for col in req_cols:
        df[col] = df[col].astype('float32')
    if 'volume' in df.columns:
        df['volume'] = df['volume'].astype('float32')

    print(f"📊 Feature Engineering Started (Rows: {len(df)})")
    
    # 1. Базовые лог-доходности (Log Returns) - Основа стационарности
    # Вместо абсолютных цен используем изменения
    df['feat_log_ret'] = np.log(df['close'] / df['close'].shift(1)).astype('float32')

    # 2. Волатильность (Volatility Features)
    print(f"  • Calculating Volatility & ATR...")
    df = _add_volatility_features(df, periods)
    
    # 3. Импульс и Осцилляторы (Momentum & Oscillators)
    print(f"  • Calculating RSI, MACD, ROC...")
    df = _add_momentum_features(df, periods)
    
    # 4. Трендовые метрики (Trend & Mean Reversion)
    print(f"  • Calculating Distance to MA & Bollinger...")
    df = _add_trend_features(df, periods)
    
    # 5. Временные признаки (Cyclical Time Encoding)
    print(f"  • Encoding Cyclical Time (Hour/Day)...")
    df = _add_time_features(df)
    
    # 6. Лаги (Lagged Features)
    # Добавляем историю возвратов, чтобы модель видела паттерны
    print(f"  • Generating Lags...")
    lags = [1, 2, 3, 5, 8]
    for lag in lags:
        df[f'feat_lag_ret_{lag}'] = df['feat_log_ret'].shift(lag)

    # 7. Мета-признаки (Higher Order Statistics) - для кластеризации
    if meta_periods:
        print(f"  • Calculating Meta Features (Skew/Kurt)...")
        for p in meta_periods:
            # Skewness (Асимметрия)
            df[f'meta_skew_{p}'] = df['feat_log_ret'].rolling(p).apply(
                lambda x: skew(x, bias=False), raw=True
            ).astype('float32')
            
            # Kurtosis (Эксцесс) - "Толстые хвосты"
            df[f'meta_kurt_{p}'] = df['feat_log_ret'].rolling(p).apply(
                lambda x: kurtosis(x, bias=False), raw=True
            ).astype('float32')

    # Очистка NaN (возникают в начале датасета из-за самых длинных периодов)
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    
    print(f"  ✓ Done. Features: {len(get_feature_columns(df))}. Dropped NaN: {dropped}")
    
    return df

# =============================================================================
# ГРУППЫ ПРИЗНАКОВ
# =============================================================================

def _add_volatility_features(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Расчет продвинутых метрик волатильности"""
    
    # TR (True Range) классический
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift(1)).abs()
    l_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    
    for p in periods:
        # ATR нормализованный к цене (чтобы был сопоставим в 2004 и 2024)
        atr = tr.rolling(p).mean()
        df[f'feat_atr_norm_{p}'] = (atr / df['close']).astype('float32')
        
        # Rolling Std (Vol) на лог-доходностях (не на цене!)
        df[f'feat_volatility_{p}'] = df['feat_log_ret'].rolling(p).std().astype('float32')
        
    return df

def _add_momentum_features(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Расчет RSI, MACD, ROC"""
    
    for p in periods:
        # ROC (Rate of Change)
        df[f'feat_roc_{p}'] = df['close'].pct_change(p).astype('float32')
        
        # RSI (Relative Strength Index) - Векторизированный
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        rs = gain / loss
        # Нормализуем RSI в диапазон [0, 1] для нейросети (вместо 0-100)
        df[f'feat_rsi_{p}'] = (100 - (100 / (1 + rs))) / 100.0
        df[f'feat_rsi_{p}'] = df[f'feat_rsi_{p}'].astype('float32')

    # MACD (стандартные 12, 26, 9) - добавляем как один мощный признак
    # Используем EMA
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    # Нормализуем MACD гистограмму к цене
    df['feat_macd_hist_norm'] = ((macd - signal) / df['close']).astype('float32')
    
    return df

def _add_trend_features(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Расчет отклонений от средних и каналов"""
    
    for p in periods:
        # SMA Distance: (Price - SMA) / SMA
        sma = df['close'].rolling(p).mean()
        df[f'feat_dist_sma_{p}'] = ((df['close'] - sma) / sma).astype('float32')
        
        # Bollinger Bands Position
        # Показывает, где цена находится относительно полос (0 = low, 1 = high, >1 breakout)
        std = df['close'].rolling(p).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        # Защита от деления на ноль
        bb_range = (upper - lower).replace(0, 1e-6)
        df[f'feat_bb_pos_{p}'] = ((df['close'] - lower) / bb_range).astype('float32')
        
    return df

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Циклическое кодирование времени.
    Важно для H1: модель поймет разницу между Азиатской и Американской сессией.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Пытаемся восстановить индекс если он потерян
        if 'time' in df.columns:
            timestamps = pd.to_datetime(df['time'])
        else:
            return df # Не можем посчитать время
    else:
        timestamps = df.index.to_series()

    # Час дня (0-23) -> Sin/Cos
    hour = timestamps.dt.hour
    df['feat_hour_sin'] = np.sin(2 * np.pi * hour / 24).astype('float32')
    df['feat_hour_cos'] = np.cos(2 * np.pi * hour / 24).astype('float32')
    
    # День недели (0-6) -> Sin/Cos
    dayofweek = timestamps.dt.dayofweek
    df['feat_day_sin'] = np.sin(2 * np.pi * dayofweek / 7).astype('float32')
    df['feat_day_cos'] = np.cos(2 * np.pi * dayofweek / 7).astype('float32')
    
    # День года (сезонность)
    dayofyear = timestamps.dt.dayofyear
    df['feat_year_sin'] = np.sin(2 * np.pi * dayofyear / 365.25).astype('float32')
    df['feat_year_cos'] = np.cos(2 * np.pi * dayofyear / 365.25).astype('float32')
    
    return df

# =============================================================================
# УТИЛИТЫ
# =============================================================================

def create_features_multiframe(
    primary_data: pd.DataFrame,
    secondary_data_dict: Dict[str, pd.DataFrame],
    periods: List[int]
) -> pd.DataFrame:
    """
    Создание признаков с нескольких таймфреймов.
    Обеспечивает защиту от заглядывания в будущее (Look-ahead bias).
    """
    # Сначала считаем признаки для основного ТФ
    result = create_features(primary_data, periods)
    
    # Для каждого высшего ТФ
    for tf_name, tf_data in secondary_data_dict.items():
        # Считаем признаки на высшем ТФ
        tf_feats = create_features(tf_data, periods)
        
        # Оставляем только 'feat_' колонки
        cols_to_merge = [c for c in tf_feats.columns if c.startswith('feat_')]
        tf_feats = tf_feats[cols_to_merge]
        
        # Переименовываем
        tf_feats.columns = [f"{c}_{tf_name}" for c in tf_feats.columns]
        
        # Merge с ffill (Forward Fill)
        # ВАЖНО: reindex(method='ffill') берет последнее доступное значение
        # Это корректно симулирует real-time: в 14:15 мы знаем Close H1 свечи за 14:00
        aligned = tf_feats.reindex(result.index, method='ffill')
        
        # Добавляем к результату
        result = pd.concat([result, aligned], axis=1)
        
    return result.dropna()

def get_feature_columns(df: pd.DataFrame, prefix: str = 'feat_') -> List[str]:
    """Получение списка колонок по префиксу"""
    return [col for col in df.columns if col.startswith(prefix)]

def validate_features(df: pd.DataFrame) -> Tuple[bool, str]:
    """Строгая проверка качества данных"""
    if df.empty:
        return False, "Empty dataframe"
    
    # Проверка на NaN
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        return False, f"NaN found in: {nan_cols[:3]}..."
        
    # Проверка на бесконечности
    numeric_df = df.select_dtypes(include=[np.number])
    if np.isinf(numeric_df).any().any():
        inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
        return False, f"Inf found in: {inf_cols[:3]}..."
        
    return True, "OK"

def print_feature_stats(df: pd.DataFrame) -> None:
    """Вывод статистики для контроля распределений"""
    feat_cols = get_feature_columns(df, 'feat_')
    meta_cols = get_feature_columns(df, 'meta_')
    
    print(f"\n📈 Feature Statistics:")
    print(f"  • Total Features: {len(feat_cols)}")
    print(f"  • Meta Features:  {len(meta_cols)}")
    print(f"  • Memory Usage:   {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Проверка стационарности (грубая)
    if len(feat_cols) > 0:
        ex_col = feat_cols[0]
        print(f"  • Example ({ex_col}):")
        print(f"    Mean: {df[ex_col].mean():.5f}")
        print(f"    Std:  {df[ex_col].std():.5f}")
        print(f"    Min/Max: {df[ex_col].min():.5f} / {df[ex_col].max():.5f}")