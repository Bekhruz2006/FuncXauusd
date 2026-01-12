import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from src.data.loader import _load_csv_auto_detect, _validate_price_data

class MultiframeLoader:
    def __init__(self, data_path: str, symbol: str, primary_tf: str, context_tfs: List[str]):
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.primary_tf = primary_tf
        self.context_tfs = context_tfs
        
    def load_timeframe(self, tf: str) -> pd.DataFrame:
        """Загрузка конкретного таймфрейма с перебором имен"""
        # Варианты имен: XAUUSD_M15.csv, XAUUSD_15M.csv, XAUUSD_15m.csv
        candidates = [
            f"{self.symbol}_{tf}.csv",
            f"{self.symbol}_{tf.upper()}.csv",
            
            # Обработка инверсии (15M vs M15)
            f"{self.symbol}_{self._invert_tf_name(tf)}.csv"
        ]
        
        for fname in candidates:
            path = self.data_path / fname
            if path.exists():
                print(f"  Load {tf}: {fname} found!")
                try:
                    df = _load_csv_auto_detect(path)
                    _validate_price_data(df)
                    # Удаляем дубликаты индекса
                    df = df[~df.index.duplicated(keep='first')]
                    return df
                except Exception as e:
                    print(f"  Error loading {fname}: {e}")
        
        print(f"⚠️ Файл для {tf} не найден. Искал: {candidates}")
        raise FileNotFoundError
        
    def _invert_tf_name(self, tf: str) -> str:
        """Преобразует M15 -> 15M и наоборот"""
        tf = tf.upper()
        if tf.startswith('M') and tf[1:].isdigit():
            return f"{tf[1:]}M"
        if tf.endswith('M') and tf[:-1].isdigit():
            return f"M{tf[:-1]}"
        if tf.startswith('H') and tf[1:].isdigit():
            return f"{tf[1:]}H"
        return tf

    def load_all(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for tf in self.context_tfs:
            try:
                data[tf] = self.load_timeframe(tf)
            except FileNotFoundError:
                pass # Просто пропускаем
        return data

def add_multiframe_to_existing(
    primary_data: pd.DataFrame,
    data_path: str,
    symbol: str,
    primary_tf: str,
    context_tfs: List[str]
) -> pd.DataFrame:
    """
    Добавление контекстных признаков
    """
    loader = MultiframeLoader(data_path, symbol, primary_tf, context_tfs)
    context_data = loader.load_all()
    
    result = primary_data.copy()
    
    for tf, df in context_data.items():
        # Ресемплинг/Выравнивание (Forward Fill)
        # Важно: берем только данные из прошлого (shift не нужен при ffill, но следим за data leakage)
        # reindex с ffill берет ближайшее значение СЗАДИ (из прошлого)
        aligned = df.reindex(result.index, method='ffill')
        
        # Если данных мало в начале, ffill даст NaN -> fillna(0) или dropna() в конце
        
        # 1. Тренд (EMA diff)
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        trend = (ema_fast - ema_slow) / ema_slow
        result[f'trend_{tf}'] = trend.reindex(result.index, method='ffill')
        
        # 2. Положение в диапазоне (Stoch style)
        rng = (df['high'] - df['low']).replace(0, 1) # Защита от деления на 0
        pos = (df['close'] - df['low']) / rng
        result[f'pos_{tf}'] = pos.reindex(result.index, method='ffill')
        
        # 3. Волатильность (ATR ratio)
        tr = np.maximum(df['high'] - df['low'], abs(df['close'] - df['open']))
        atr = tr.rolling(14).mean()
        # Нормализуем ATR к цене
        atr_norm = atr / df['close']
        result[f'vol_{tf}'] = atr_norm.reindex(result.index, method='ffill')
        
    # Удаляем строки, где мультифрейм еще не подгрузился (начало истории)
    return result.dropna()