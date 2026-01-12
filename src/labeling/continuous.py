import numpy as np
import pandas as pd
from numba import njit
from typing import Literal

@njit
def calculate_regression_target(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    direction: int,  # 1 = Buy, -1 = Sell
    max_bars: int,
    decay: float
) -> np.ndarray:
    n = len(close)
    targets = np.zeros(n, dtype=np.float64)
    
    for i in range(n - max_bars):
        entry_price = close[i]
        volatility = atr[i]
        
        if volatility <= 0 or np.isnan(volatility):
            targets[i] = -1.0 # Default bad
            continue
            
        best_normalized_profit = -2.0 # Start with stop loss level
        
        for j in range(1, max_bars + 1):
            future_idx = i + j
            
            if direction == 1: # Buy
                profit = high[future_idx] - entry_price
            else: # Sell
                profit = entry_price - low[future_idx]
            
            ratio = profit / volatility
            weighted_ratio = ratio * (decay ** j)
            
            if weighted_ratio > best_normalized_profit:
                best_normalized_profit = weighted_ratio
        
        # Clip values to avoid extreme outliers impacting RMSE
        targets[i] = max(min(best_normalized_profit, 5.0), -2.0)
        
    return targets

def get_continuous_labels(
    dataset: pd.DataFrame,
    max_bars: int = 24,
    direction: Literal['buy', 'sell'] = 'buy',
    decay_factor: float = 0.96
) -> pd.DataFrame:
    data = dataset.copy()
    
    # 1. Проверки
    req = ['close', 'high', 'low']
    if not all(c in data.columns for c in req):
        raise ValueError(f"Нет колонок {req}")
        
    # 2. ATR
    if 'atr' not in data.columns:
        from src.risk.atr_manager import calculate_atr
        data['atr'] = calculate_atr(data)
    
    # Заполняем пропуски в ATR
    data['atr'] = data['atr'].fillna(method='bfill').fillna(method='ffill')
    
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    atr = data['atr'].values
    
    # Защита от нулей в ATR
    atr[atr == 0] = 0.0001
    
    dir_int = 1 if direction == 'buy' else -1
    
    targets = calculate_regression_target(
        close, high, low, atr, dir_int, max_bars, decay_factor
    )
    
    data['labels'] = targets
    
    # Удаляем последние строки, где таргет не рассчитан
    data = data.iloc[:-max_bars]
    
    # Финальная очистка
    data = data.dropna()
    
    # Проверка на константность (для отладки)
    unique_vals = data['labels'].nunique()
    if unique_vals < 2:
        print(f"⚠️ ВНИМАНИЕ: Целевая переменная содержит всего {unique_vals} уникальных значений!")
    
    print(f"✅ Разметка завершена. Rows: {len(data)}, Target mean: {data['labels'].mean():.4f}")
    return data