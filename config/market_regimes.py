"""
Определение рыночных режимов и их характеристик.
Используется для классификации и специализации агентов.
"""

from enum import Enum
from typing import Dict, List, Callable
from dataclasses import dataclass
import numpy as np


class RegimeType(Enum):
    """Типы рыночных режимов"""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    VOLATILE = "volatile"
    RANGING = "ranging"
    TRANSITION = "transition"


@dataclass
class RegimeCharacteristics:
    """Характеристики режима для специализации агентов"""
    name: str
    volatility_range: tuple  # (min, max) ATR percentile
    trend_strength_range: tuple  # (min, max) ADX или custom metric
    mean_reversion_idx: tuple  # Hurst exponent или аналог
    preferred_entry_logic: List[str]  # Рекомендуемые атомы
    risk_multiplier: float  # Модификатор размера позиции
    max_hold_bars: int  # Максимальная длительность позиции
    

class RegimeDefinitions:
    """Центральный реестр определений режимов"""
    
    REGIMES: Dict[RegimeType, RegimeCharacteristics] = {
        RegimeType.TREND_UP: RegimeCharacteristics(
            name="Устойчивый восходящий тренд",
            volatility_range=(0.2, 0.6),
            trend_strength_range=(0.5, 1.0),
            mean_reversion_idx=(0.6, 1.0),  # Более трендовый
            preferred_entry_logic=["momentum_breakout", "pullback_entry", "ema_crossover"],
            risk_multiplier=1.2,
            max_hold_bars=48  # 2 дня на H1
        ),
        
        RegimeType.TREND_DOWN: RegimeCharacteristics(
            name="Устойчивый нисходящий тренд",
            volatility_range=(0.2, 0.6),
            trend_strength_range=(-1.0, -0.5),
            mean_reversion_idx=(0.6, 1.0),
            preferred_entry_logic=["counter_trend_long", "oversold_bounce", "divergence_entry"],
            risk_multiplier=0.8,  # Консервативнее против тренда
            max_hold_bars=24
        ),
        
        RegimeType.VOLATILE: RegimeCharacteristics(
            name="Высокая волатильность",
            volatility_range=(0.7, 1.0),
            trend_strength_range=(-0.3, 0.3),
            mean_reversion_idx=(0.3, 0.6),
            preferred_entry_logic=["volatility_compression", "range_breakout", "gap_fade"],
            risk_multiplier=0.6,  # Сильно снижаем риск
            max_hold_bars=12
        ),
        
        RegimeType.RANGING: RegimeCharacteristics(
            name="Боковое движение",
            volatility_range=(0.0, 0.4),
            trend_strength_range=(-0.2, 0.2),
            mean_reversion_idx=(0.0, 0.4),  # Сильная ревертация
            preferred_entry_logic=["mean_reversion", "support_resistance", "bollinger_bounce"],
            risk_multiplier=1.0,
            max_hold_bars=36
        ),
        
        RegimeType.TRANSITION: RegimeCharacteristics(
            name="Переходный период",
            volatility_range=(0.3, 0.7),
            trend_strength_range=(-0.5, 0.5),
            mean_reversion_idx=(0.2, 0.8),
            preferred_entry_logic=["pattern_recognition", "multi_tf_confirmation"],
            risk_multiplier=0.5,  # Очень консервативно
            max_hold_bars=8
        )
    }
    
    @staticmethod
    def get_regime(regime_type: RegimeType) -> RegimeCharacteristics:
        """Получение характеристик режима по типу"""
        try:
            return RegimeDefinitions.REGIMES[regime_type]
        except KeyError:
            raise ValueError(f"Неизвестный тип режима: {regime_type}")
    
    @staticmethod
    def classify_regime(volatility_percentile: float, 
                       trend_strength: float, 
                       mean_reversion: float) -> RegimeType:
        """
        Классификация текущего состояния рынка в режим.
        
        Args:
            volatility_percentile: Процентиль волатильности (0-1)
            trend_strength: Сила тренда (-1 до 1)
            mean_reversion: Индекс возврата к среднему (0-1)
        
        Returns:
            Определенный режим
        """
        try:
            # Простая эвристика для начальной классификации
            if volatility_percentile > 0.7:
                return RegimeType.VOLATILE
            
            if abs(trend_strength) < 0.2 and mean_reversion < 0.4:
                return RegimeType.RANGING
            
            if trend_strength > 0.5 and mean_reversion > 0.6:
                return RegimeType.TREND_UP
            
            if trend_strength < -0.5 and mean_reversion > 0.6:
                return RegimeType.TREND_DOWN
            
            return RegimeType.TRANSITION
            
        except Exception as e:
            raise RuntimeError(f"Ошибка классификации режима: {e}")
    
    @staticmethod
    def get_pathological_scenarios() -> List[Dict]:
        """
        Генерация патологических сценариев для Stage 0.
        
        Returns:
            Список словарей с параметрами сценариев
        """
        return [
            {
                "name": "sharp_gap",
                "gap_size": 50,  # пунктов
                "frequency": 0.05,  # 5% баров
                "direction": "random"
            },
            {
                "name": "prolonged_flat",
                "duration": 168,  # 1 неделя
                "max_range": 0.002,  # 0.2% от цены
                "volume_multiplier": 0.3
            },
            {
                "name": "false_breakout",
                "breakout_size": 30,
                "reversal_size": 50,
                "frequency": 0.1
            },
            {
                "name": "whipsaw",
                "amplitude": 40,
                "frequency": 0.15,
                "duration": 12  # 12 часов
            },
            {
                "name": "liquidity_crisis",
                "spread_multiplier": 5.0,
                "volume_multiplier": 0.1,
                "duration": 24
            }
        ]


# Функции-детекторы для вычисления метрик режима
def compute_volatility_percentile(returns: np.ndarray, window: int = 100) -> float:
    """Процентиль волатильности относительно скользящего окна"""
    try:
        rolling_std = np.array([returns[max(0, i-window):i].std() 
                                for i in range(window, len(returns))])
        current_vol = returns[-window:].std()
        percentile = np.searchsorted(np.sort(rolling_std), current_vol) / len(rolling_std)
        return np.clip(percentile, 0.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Ошибка вычисления volatility_percentile: {e}")


def compute_trend_strength(prices: np.ndarray, window: int = 50) -> float:
    """Сила тренда через линейную регрессию (-1 до 1)"""
    try:
        x = np.arange(window)
        y = prices[-window:]
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = np.tanh(slope / (y.mean() * 0.01))  # Нормализация
        return np.clip(normalized_slope, -1.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Ошибка вычисления trend_strength: {e}")


def compute_mean_reversion_index(prices: np.ndarray, window: int = 100) -> float:
    """Индекс возврата к среднему через упрощенный Hurst exponent (0-1)"""
    try:
        lags = range(2, min(20, window // 5))
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]
        # Преобразуем Hurst (0.5=random, >0.5=trend, <0.5=mean-revert) в [0,1]
        mri = 1.0 - abs(hurst - 0.5) * 2.0
        return np.clip(mri, 0.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Ошибка вычисления mean_reversion_index: {e}")