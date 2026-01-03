"""
Загрузка и кэширование исторических ценовых данных

Поддержка:
    - CSV формат MT5 (разделитель ';' или пробел)
    - Автоматическое определение формата
    - Кэширование в Parquet для быстрой перезагрузки
    - Валидация данных
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Union

# Глобальный кэш для избежания повторной загрузки
_PRICE_CACHE: Optional[pd.DataFrame] = None


def load_price_data(config: dict, force_reload: bool = False) -> pd.DataFrame:
    """
    Загрузка исторических OHLCV данных
    
    Args:
        config: Конфигурация с путями и параметрами
        force_reload: Принудительная перезагрузка из CSV
    
    Returns:
        pd.DataFrame: Данные с индексом datetime и колонкой 'close'
    
    Raises:
        FileNotFoundError: Если CSV файл не найден
        ValueError: Если данные некорректны
    """
    global _PRICE_CACHE
    
    # Проверка кэша
    if _PRICE_CACHE is not None and not force_reload:
        print(f"✓ Данные загружены из памяти ({len(_PRICE_CACHE)} баров)")
        return _PRICE_CACHE.copy()
    
    symbol = config['symbol']['name']
    raw_path = Path(config['data']['paths']['raw'])
    csv_file = raw_path / f"{symbol}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV файл не найден: {csv_file}\n"
            f"Экспортируйте данные из MetaTrader 5"
        )
    
    print(f"📂 Загрузка данных из {csv_file.name}...")
    
    # Попытка определить формат
    df = _load_csv_auto_detect(csv_file)
    
    # Валидация
    _validate_price_data(df)
    
    # Кэширование
    _PRICE_CACHE = df
    
    print(f"✓ Загружено {len(df)} баров ({df.index[0]} - {df.index[-1]})")
    
    return df.copy()


def _load_csv_auto_detect(filepath: Path) -> pd.DataFrame:
    """
    Автоматическое определение формата CSV и загрузка
    
    Поддерживаемые форматы:
        1. MT5 экспорт с разделителем ';'
        2. MT5 экспорт с пробелами
        3. Стандартный CSV с запятыми
    """
    # Формат 1: разделитель ';'
    try:
        df = pd.read_csv(filepath, sep=';', parse_dates=['Date'])
        if 'Date' in df.columns and 'Close' in df.columns:
            return _normalize_mt5_format(df)
    except:
        pass
    
    # Формат 2: разделитель пробел (как в оригинальном коде)
    try:
        df = pd.read_csv(filepath, sep=r'\s+')
        if '<DATE>' in df.columns and '<CLOSE>' in df.columns:
            return _normalize_mt5_space_format(df)
    except:
        pass
    
    # Формат 3: стандартный CSV
    try:
        df = pd.read_csv(filepath)
        if 'time' in df.columns and 'close' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df[['close']].dropna()
    except:
        pass
    
    raise ValueError(
        f"Не удалось определить формат файла {filepath.name}\n"
        f"Поддерживаемые форматы:\n"
        f"  1. MT5 экспорт с ';' (Date;Open;High;Low;Close;Volume)\n"
        f"  2. MT5 экспорт с пробелами (<DATE> <TIME> <OPEN> ...)\n"
        f"  3. Стандартный CSV (time,open,high,low,close,volume)"
    )


def _normalize_mt5_format(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализация MT5 формата с разделителем ';'"""
    result = pd.DataFrame()
    result['time'] = pd.to_datetime(df['Date'])
    result['close'] = df['Close'].astype(float)
    result.set_index('time', inplace=True)
    return result.dropna()


def _normalize_mt5_space_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализация MT5 формата с пробелами
    Формат: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL>
    """
    result = pd.DataFrame()
    result['time'] = df['<DATE>'] + ' ' + df['<TIME>']
    result['time'] = pd.to_datetime(result['time'], format='mixed')
    result['close'] = df['<CLOSE>'].astype(float)
    result.set_index('time', inplace=True)
    return result.dropna()


def _validate_price_data(df: pd.DataFrame) -> None:
    """
    Валидация загруженных данных
    
    Проверки:
        - Наличие индекса datetime
        - Наличие колонки 'close'
        - Отсутствие NaN
        - Отсутствие нулевых/отрицательных цен
        - Минимальное количество данных
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс должен быть DatetimeIndex")
    
    if 'close' not in df.columns:
        raise ValueError("Отсутствует колонка 'close'")
    
    if df['close'].isna().any():
        raise ValueError(
            f"Обнаружены NaN значения: {df['close'].isna().sum()} шт."
        )
    
    if (df['close'] <= 0).any():
        raise ValueError("Обнаружены нулевые или отрицательные цены")
    
    if len(df) < 1000:
        raise ValueError(
            f"Недостаточно данных: {len(df)} баров (минимум 1000)"
        )


def cache_prices(config: dict) -> None:
    """
    Предварительная загрузка и кэширование данных в памяти
    
    Использование:
        >>> cache_prices(config)
        >>> # Теперь load_price_data() будет мгновенной
    """
    load_price_data(config, force_reload=True)


def get_cached_prices() -> Optional[pd.DataFrame]:
    """
    Получение закэшированных данных без перезагрузки
    
    Returns:
        pd.DataFrame или None если кэш пуст
    """
    return _PRICE_CACHE.copy() if _PRICE_CACHE is not None else None


def clear_cache() -> None:
    """Очистка кэша данных"""
    global _PRICE_CACHE
    _PRICE_CACHE = None


def load_multiframe_data(config: dict) -> Dict[str, pd.DataFrame]:
    """
    Загрузка данных с нескольких таймфреймов (для будущей реализации)
    
    Args:
        config: Конфигурация с включенным multiframe
    
    Returns:
        dict: {timeframe: DataFrame}
    """
    if not config['data']['multiframe']['enabled']:
        raise ValueError("Multiframe отключен в конфигурации")
    
    timeframes = config['data']['multiframe']['timeframes']
    data = {}
    
    for tf in timeframes:
        # TODO: реализовать загрузку разных таймфреймов
        print(f"⚠️ Загрузка {tf}: не реализовано")
    
    return data


# Дополнительные утилиты

def resample_to_timeframe(df: pd.DataFrame, 
                         target_tf: str) -> pd.DataFrame:
    """
    Ресемплинг данных на другой таймфрейм
    
    Args:
        df: Исходные данные
        target_tf: Целевой таймфрейм ('5m', 'H1', 'D1' и т.д.)
    
    Returns:
        pd.DataFrame: Ресемплированные данные
    """
    # Маппинг таймфреймов на pandas freq
    tf_map = {
        '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
        'H1': '1H', 'H4': '4H', 'D1': '1D', 'W1': '1W', 'MN': '1M'
    }
    
    if target_tf not in tf_map:
        raise ValueError(f"Неподдерживаемый таймфрейм: {target_tf}")
    
    freq = tf_map[target_tf]
    
    # Если есть OHLC - агрегируем правильно
    if 'open' in df.columns:
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df.columns else 'mean'
        })
    else:
        # Только close
        resampled = df.resample(freq).last()
    
    return resampled.dropna()


def align_timeframes(primary_df: pd.DataFrame,
                    secondary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Выравнивание двух таймфреймов по индексу первичного
    
    Использование: добавление высших ТФ как контекст
    """
    # Форвард-филл для заполнения пропусков
    aligned = secondary_df.reindex(primary_df.index, method='ffill')
    return aligned