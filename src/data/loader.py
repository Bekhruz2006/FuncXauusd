"""
Загрузка и кэширование исторических ценовых данных.
Сохраняет OHLCV для расчета ATR и Target.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

# Глобальный кэш
_PRICE_CACHE: Optional[pd.DataFrame] = None

def load_price_data(config: dict, force_reload: bool = False) -> pd.DataFrame:
    global _PRICE_CACHE
    
    if _PRICE_CACHE is not None and not force_reload:
        print(f"✓ Данные из кэша ({len(_PRICE_CACHE)} баров)")
        return _PRICE_CACHE.copy()
    
    symbol = config['symbol']['name']
    raw_path = Path(config['data']['paths']['raw'])
    csv_file = raw_path / f"{symbol}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_file}")
    
    print(f"📂 Загрузка: {csv_file.name}")
    df = _load_csv_auto_detect(csv_file)
    _validate_price_data(df)
    
    _PRICE_CACHE = df
    print(f"✓ Загружено {len(df)} баров. Columns: {list(df.columns)}")
    return df.copy()

def get_cached_prices() -> Optional[pd.DataFrame]:
    """Получение данных из кэша без перезагрузки"""
    global _PRICE_CACHE
    return _PRICE_CACHE.copy() if _PRICE_CACHE is not None else None

def clear_cache() -> None:
    """Очистка кэша"""
    global _PRICE_CACHE
    _PRICE_CACHE = None

def cache_prices(config: dict) -> None:
    """Принудительная загрузка в кэш"""
    load_price_data(config, force_reload=True)

def _load_csv_auto_detect(filepath: Path) -> pd.DataFrame:
    # 1. MT5 Export (separator ';')
    try:
        df = pd.read_csv(filepath, sep=';', parse_dates=['Date'])
        if 'Date' in df.columns and 'Close' in df.columns:
            return _normalize_mt5_semicolon(df)
    except:
        pass
    
    # 2. MT5 Export (separator space/tab)
    try:
        df = pd.read_csv(filepath, sep=r'\s+')
        if '<DATE>' in df.columns:
            return _normalize_mt5_space(df)
    except:
        pass
        
    # 3. Standard CSV
    try:
        df = pd.read_csv(filepath)
        if 'time' in df.columns and 'close' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df
    except:
        pass
        
    raise ValueError(f"Неизвестный формат файла: {filepath}")

def _normalize_mt5_semicolon(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализация формата Date;Open;High;Low;Close;Volume"""
    res = pd.DataFrame()
    res['time'] = pd.to_datetime(df['Date'])
    
    # ВАЖНО: Сохраняем все колонки и приводим к нижнему регистру
    res['open'] = df['Open'].astype(float)
    res['high'] = df['High'].astype(float)
    res['low'] = df['Low'].astype(float)
    res['close'] = df['Close'].astype(float)
    
    if 'Volume' in df.columns:
        res['volume'] = df['Volume'].astype(float)
    elif 'TickVol' in df.columns:
        res['volume'] = df['TickVol'].astype(float)
        
    res.set_index('time', inplace=True)
    return res.dropna()

def _normalize_mt5_space(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализация формата <DATE> <TIME> ..."""
    res = pd.DataFrame()
    res['time'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    res['open'] = df['<OPEN>'].astype(float)
    res['high'] = df['<HIGH>'].astype(float)
    res['low'] = df['<LOW>'].astype(float)
    res['close'] = df['<CLOSE>'].astype(float)
    res['volume'] = df['<TICKVOL>'].astype(float)
    res.set_index('time', inplace=True)
    return res.dropna()

def _validate_price_data(df: pd.DataFrame) -> None:
    req = ['open', 'high', 'low', 'close']
    if not all(c in df.columns for c in req):
        raise ValueError(f"Отсутствуют обязательные колонки. Найдены: {list(df.columns)}")
    if df.isnull().values.any():
        raise ValueError("Найдены NaN значения в ценах")