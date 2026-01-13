"""
Мультитаймфреймовый агрегатор - иерархическая система принятия решений
с координацией между стратегическим, тактическим и исполнительным уровнями.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from config.hyperparameters import HYPERPARAMS
from config.market_regimes import RegimeType
from utils.logger import LOGGER


class TimeframeLevel(Enum):
    """Уровни таймфреймовой иерархии"""
    STRATEGIC = "strategic"      # H4, D1
    TACTICAL = "tactical"        # H1, 30M
    EXECUTION = "execution"      # 15M, 5M, M1


class TimeframeSignal:
    """Сигнал с одного таймфрейма"""
    
    def __init__(self,
                 timeframe: str,
                 level: TimeframeLevel,
                 bias: float,
                 confidence: float,
                 regime: Optional[str] = None,
                 volatility_regime: Optional[str] = None):
        
        self.timeframe = timeframe
        self.level = level
        self.bias = bias  # -1 (bearish) до 1 (bullish)
        self.confidence = confidence  # 0 до 1
        self.regime = regime
        self.volatility_regime = volatility_regime
        self.timestamp = None
    
    def __repr__(self) -> str:
        return (f"TimeframeSignal({self.timeframe}, bias={self.bias:.2f}, "
                f"conf={self.confidence:.2f}, regime={self.regime})")


class MultiTimeframeAggregator:
    """Агрегатор мультитаймфреймовых сигналов с иерархической логикой"""
    
    def __init__(self,
                 timeframe_hierarchy: Optional[Dict[str, str]] = None,
                 min_confirmation_timeframes: int = 2):
        
        if timeframe_hierarchy is None:
            self.timeframe_hierarchy = {
                'H4': TimeframeLevel.STRATEGIC,
                'H1': TimeframeLevel.TACTICAL,
                '30M': TimeframeLevel.TACTICAL,
                '15M': TimeframeLevel.EXECUTION,
                'M5': TimeframeLevel.EXECUTION,
                'M1': TimeframeLevel.EXECUTION
            }
        else:
            self.timeframe_hierarchy = timeframe_hierarchy
        
        self.min_confirmation_timeframes = min_confirmation_timeframes
        self.signal_cache: Dict[str, TimeframeSignal] = {}
        
        LOGGER.info(f"Инициализация мультитаймфреймового агрегатора: "
                   f"{len(self.timeframe_hierarchy)} таймфреймов")
    
    def compute_hierarchical_signals(self, 
                                     data_dict: Dict[str, pd.DataFrame],
                                     current_bar: Optional[int] = None) -> Dict[str, TimeframeSignal]:
        """
        Вычисление иерархических сигналов со всех таймфреймов.
        
        Args:
            data_dict: {timeframe: DataFrame} с предобработанными данными
            current_bar: Индекс текущего бара (None = последний)
        
        Returns:
            Словарь {timeframe: TimeframeSignal}
        """
        try:
            signals = {}
            
            # 1. Вычисляем сигналы для каждого таймфрейма
            for tf, df in data_dict.items():
                if tf not in self.timeframe_hierarchy:
                    continue
                
                try:
                    signal = self._compute_single_timeframe_signal(df, tf, current_bar)
                    signals[tf] = signal
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка вычисления сигнала {tf}: {e}")
                    continue
            
            # 2. Иерархическая координация сигналов
            coordinated_signals = self._coordinate_signals_hierarchically(signals)
            
            self.signal_cache = coordinated_signals
            
            LOGGER.debug(f"Вычислено сигналов: {len(coordinated_signals)}")
            return coordinated_signals
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления иерархических сигналов: {e}", exc_info=True)
            return {}
    
    def _compute_single_timeframe_signal(self,
                                         df: pd.DataFrame,
                                         timeframe: str,
                                         current_bar: Optional[int] = None) -> TimeframeSignal:
        """Вычисление сигнала для одного таймфрейма"""
        try:
            if current_bar is None:
                current_bar = len(df) - 1
            
            if current_bar < 50:  # Минимум данных для анализа
                return TimeframeSignal(
                    timeframe=timeframe,
                    level=self.timeframe_hierarchy[timeframe],
                    bias=0.0,
                    confidence=0.0
                )
            
            # Анализируем последние данные
            lookback = min(100, current_bar)
            recent_data = df.iloc[current_bar-lookback:current_bar+1]
            
            # Вычисляем bias через несколько методов
            trend_bias = self._compute_trend_bias(recent_data)
            momentum_bias = self._compute_momentum_bias(recent_data)
            volume_bias = self._compute_volume_bias(recent_data)
            
            # Взвешенная комбинация
            weights = {'trend': 0.5, 'momentum': 0.3, 'volume': 0.2}
            combined_bias = (
                weights['trend'] * trend_bias +
                weights['momentum'] * momentum_bias +
                weights['volume'] * volume_bias
            )
            
            # Confidence на основе согласованности
            biases = [trend_bias, momentum_bias, volume_bias]
            confidence = 1.0 - np.std(biases)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Определение режима
            regime = self._determine_regime(recent_data)
            volatility_regime = self._determine_volatility_regime(recent_data)
            
            signal = TimeframeSignal(
                timeframe=timeframe,
                level=self.timeframe_hierarchy[timeframe],
                bias=combined_bias,
                confidence=confidence,
                regime=regime,
                volatility_regime=volatility_regime
            )
            
            signal.timestamp = df.index[current_bar]
            
            return signal
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления сигнала для {timeframe}: {e}")
            return TimeframeSignal(
                timeframe=timeframe,
                level=self.timeframe_hierarchy[timeframe],
                bias=0.0,
                confidence=0.0
            )
    
    def _compute_trend_bias(self, df: pd.DataFrame) -> float:
        """Вычисление трендового bias"""
        try:
            close = df['Close']
            
            # EMA crossover
            ema_fast = close.ewm(span=20, adjust=False).mean()
            ema_slow = close.ewm(span=50, adjust=False).mean()
            
            ema_diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
            ema_bias = np.tanh(ema_diff * 100)
            
            # Линейная регрессия
            x = np.arange(len(close))
            slope = np.polyfit(x, close.values, 1)[0]
            slope_bias = np.tanh(slope / (close.mean() * 0.01))
            
            # Комбинирование
            trend_bias = (ema_bias + slope_bias) / 2
            
            return np.clip(trend_bias, -1.0, 1.0)
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления trend_bias: {e}")
            return 0.0
    
    def _compute_momentum_bias(self, df: pd.DataFrame) -> float:
        """Вычисление импульсного bias"""
        try:
            close = df['Close']
            
            # ROC
            roc = (close.iloc[-1] - close.iloc[-14]) / close.iloc[-14]
            roc_bias = np.tanh(roc * 10)
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain.iloc[-1] / (avg_loss.iloc[-1] + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            rsi_bias = (rsi - 50) / 50
            
            # Комбинирование
            momentum_bias = (roc_bias + rsi_bias) / 2
            
            return np.clip(momentum_bias, -1.0, 1.0)
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления momentum_bias: {e}")
            return 0.0
    
    def _compute_volume_bias(self, df: pd.DataFrame) -> float:
        """Вычисление объемного bias"""
        try:
            close = df['Close']
            volume = df['Volume']
            
            # Направление цены
            price_direction = np.sign(close.diff())
            
            # Взвешивание объемом
            volume_weighted = (price_direction * volume).iloc[-20:].sum()
            total_volume = volume.iloc[-20:].sum()
            
            volume_bias = volume_weighted / (total_volume + 1e-8)
            
            return np.clip(volume_bias, -1.0, 1.0)
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления volume_bias: {e}")
            return 0.0
    
    def _determine_regime(self, df: pd.DataFrame) -> str:
        """Определение рыночного режима"""
        try:
            if 'regime_type' in df.columns:
                return df['regime_type'].iloc[-1]
            
            # Простая эвристика
            close = df['Close']
            returns = close.pct_change().dropna()
            
            volatility = returns.std()
            trend = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
            
            if abs(trend) > 0.05 and trend > 0:
                return RegimeType.TREND_UP.value
            elif abs(trend) > 0.05 and trend < 0:
                return RegimeType.TREND_DOWN.value
            elif volatility > returns.rolling(50).std().mean() * 1.5:
                return RegimeType.VOLATILE.value
            else:
                return RegimeType.RANGING.value
                
        except Exception as e:
            LOGGER.error(f"Ошибка определения режима: {e}")
            return RegimeType.TRANSITION.value
    
    def _determine_volatility_regime(self, df: pd.DataFrame) -> str:
        """Определение режима волатильности"""
        try:
            returns = df['Close'].pct_change().dropna()
            current_vol = returns.iloc[-20:].std()
            historical_vol = returns.std()
            
            if current_vol < historical_vol * 0.5:
                return "low"
            elif current_vol > historical_vol * 1.5:
                return "high"
            else:
                return "medium"
                
        except Exception:
            return "medium"
    
    def _coordinate_signals_hierarchically(self, 
                                          signals: Dict[str, TimeframeSignal]) -> Dict[str, TimeframeSignal]:
        """
        Иерархическая координация сигналов.
        Старшие таймфреймы могут модифицировать младшие.
        """
        try:
            coordinated = signals.copy()
            
            # Получаем стратегический bias (старший ТФ)
            strategic_signals = [s for s in signals.values() 
                               if s.level == TimeframeLevel.STRATEGIC]
            
            if not strategic_signals:
                return coordinated
            
            # Усредняем стратегический bias
            strategic_bias = np.mean([s.bias for s in strategic_signals])
            strategic_confidence = np.mean([s.confidence for s in strategic_signals])
            
            # Модифицируем тактические и исполнительные сигналы
            for tf, signal in coordinated.items():
                if signal.level == TimeframeLevel.TACTICAL:
                    # Тактический уровень может ослаблять, но не противоречить стратегии
                    if np.sign(signal.bias) != np.sign(strategic_bias) and strategic_confidence > 0.6:
                        # Снижаем confidence противоречащих сигналов
                        signal.confidence *= 0.5
                        LOGGER.debug(f"{tf}: снижена confidence из-за противоречия стратегии")
                
                elif signal.level == TimeframeLevel.EXECUTION:
                    # Исполнительный уровень строго следует тактике/стратегии
                    tactical_signals = [s for s in coordinated.values() 
                                      if s.level == TimeframeLevel.TACTICAL]
                    
                    if tactical_signals:
                        tactical_bias = np.mean([s.bias for s in tactical_signals])
                        
                        # Если противоречит, сильно снижаем confidence
                        if np.sign(signal.bias) != np.sign(tactical_bias):
                            signal.confidence *= 0.3
                            LOGGER.debug(f"{tf}: снижена confidence из-за противоречия тактике")
            
            return coordinated
            
        except Exception as e:
            LOGGER.error(f"Ошибка координации сигналов: {e}")
            return signals
    
    def get_aggregated_signal(self, 
                             signals: Optional[Dict[str, TimeframeSignal]] = None,
                             require_confirmation: bool = True) -> Tuple[float, float, Dict]:
        """
        Получение агрегированного сигнала со всех таймфреймов.
        
        Args:
            signals: Словарь сигналов (использует кэш если None)
            require_confirmation: Требовать подтверждения с нескольких ТФ
        
        Returns:
            (aggregated_bias, aggregated_confidence, metadata)
        """
        try:
            if signals is None:
                signals = self.signal_cache
            
            if not signals:
                return 0.0, 0.0, {}
            
            # Взвешивание по уровням иерархии
            level_weights = {
                TimeframeLevel.STRATEGIC: 0.5,
                TimeframeLevel.TACTICAL: 0.3,
                TimeframeLevel.EXECUTION: 0.2
            }
            
            weighted_bias = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            signal_directions = []
            
            for signal in signals.values():
                weight = level_weights.get(signal.level, 0.2)
                weight *= signal.confidence  # Взвешиваем по уверенности
                
                weighted_bias += signal.bias * weight
                weighted_confidence += signal.confidence * weight
                total_weight += weight
                
                signal_directions.append(np.sign(signal.bias))
            
            if total_weight > 0:
                aggregated_bias = weighted_bias / total_weight
                aggregated_confidence = weighted_confidence / total_weight
            else:
                aggregated_bias = 0.0
                aggregated_confidence = 0.0
            
            # Проверка confirmation
            if require_confirmation:
                confirmation_count = sum(1 for d in signal_directions if d == np.sign(aggregated_bias))
                
                if confirmation_count < self.min_confirmation_timeframes:
                    aggregated_confidence *= 0.5
                    LOGGER.debug(f"Недостаточно подтверждений: {confirmation_count}/{self.min_confirmation_timeframes}")
            
            metadata = {
                'num_signals': len(signals),
                'confirmation_count': sum(1 for d in signal_directions if d == np.sign(aggregated_bias)),
                'signal_agreement': np.mean([1 if d == np.sign(aggregated_bias) else 0 for d in signal_directions]),
                'dominant_regime': self._get_dominant_regime(signals)
            }
            
            return aggregated_bias, aggregated_confidence, metadata
            
        except Exception as e:
            LOGGER.error(f"Ошибка агрегации сигнала: {e}", exc_info=True)
            return 0.0, 0.0, {}
    
    def _get_dominant_regime(self, signals: Dict[str, TimeframeSignal]) -> str:
        """Определение доминирующего режима"""
        try:
            regimes = [s.regime for s in signals.values() if s.regime]
            
            if not regimes:
                return "unknown"
            
            # Режим с наибольшим весом
            from collections import Counter
            regime_counts = Counter(regimes)
            dominant = regime_counts.most_common(1)[0][0]
            
            return dominant
            
        except Exception:
            return "unknown"
    
    def should_enter_trade(self,
                          aggregated_bias: float,
                          aggregated_confidence: float,
                          metadata: Dict,
                          min_bias_threshold: float = 0.3,
                          min_confidence_threshold: float = 0.5) -> Tuple[bool, str]:
        """
        Решение о входе в сделку на основе агрегированного сигнала.
        
        Returns:
            (should_enter, reason)
        """
        try:
            # Проверка порогов
            if abs(aggregated_bias) < min_bias_threshold:
                return False, f"Слабый bias: {aggregated_bias:.3f} < {min_bias_threshold}"
            
            if aggregated_confidence < min_confidence_threshold:
                return False, f"Низкая confidence: {aggregated_confidence:.3f} < {min_confidence_threshold}"
            
            # Проверка согласованности
            agreement = metadata.get('signal_agreement', 0)
            if agreement < 0.6:
                return False, f"Низкая согласованность сигналов: {agreement:.2f}"
            
            # Проверка режима
            regime = metadata.get('dominant_regime', 'unknown')
            if regime == RegimeType.TRANSITION.value:
                return False, "Переходный режим - высокая неопределенность"
            
            # Направление входа
            direction = "LONG" if aggregated_bias > 0 else "SHORT"
            
            # Для long-bias стратегии отфильтровываем SHORT
            if direction == "SHORT":
                return False, "SHORT сигнал отклонен (long-bias стратегия)"
            
            reason = (f"{direction} вход: bias={aggregated_bias:.3f}, "
                     f"conf={aggregated_confidence:.3f}, regime={regime}")
            
            return True, reason
            
        except Exception as e:
            LOGGER.error(f"Ошибка принятия решения о входе: {e}")
            return False, f"Ошибка: {e}"
    
    def compute_dynamic_position_size(self,
                                     aggregated_confidence: float,
                                     volatility_regime: str,
                                     base_risk_percent: float = 0.01) -> float:
        """
        Динамический расчет размера позиции.
        
        Args:
            aggregated_confidence: Уверенность агрегированного сигнала
            volatility_regime: Режим волатильности (low/medium/high)
            base_risk_percent: Базовый процент риска
        
        Returns:
            Скорректированный процент риска
        """
        try:
            # Масштабируем по confidence
            risk = base_risk_percent * aggregated_confidence
            
            # Корректируем по волатильности
            volatility_multipliers = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.6
            }
            
            multiplier = volatility_multipliers.get(volatility_regime, 1.0)
            risk *= multiplier
            
            # Ограничиваем диапазон
            risk = np.clip(risk, 0.005, 0.025)
            
            return risk
            
        except Exception as e:
            LOGGER.error(f"Ошибка расчета размера позиции: {e}")
            return base_risk_percent


def create_timeframe_features(data_dict: Dict[str, pd.DataFrame],
                              target_timeframe: str = 'H1') -> pd.DataFrame:
    """
    Создание мультитаймфреймовых признаков для целевого таймфрейма.
    
    Args:
        data_dict: {timeframe: DataFrame}
        target_timeframe: Целевой таймфрейм для признаков
    
    Returns:
        DataFrame с MTF признаками
    """
    try:
        if target_timeframe not in data_dict:
            raise ValueError(f"Целевой ТФ {target_timeframe} не найден")
        
        target_df = data_dict[target_timeframe]
        mtf_features = pd.DataFrame(index=target_df.index)
        
        # Добавляем признаки со старших таймфреймов
        higher_tfs = ['H4', 'D1']
        
        for htf in higher_tfs:
            if htf not in data_dict:
                continue
            
            htf_df = data_dict[htf]
            
            # Ресамплинг на target timeframe
            htf_resampled = htf_df.reindex(target_df.index, method='ffill')
            
            # Добавляем ключевые признаки
            mtf_features[f'{htf}_close'] = htf_resampled['Close']
            mtf_features[f'{htf}_trend'] = (
                htf_resampled['Close'] - htf_resampled['Close'].shift(1)
            ) / htf_resampled['Close'].shift(1)
            
            if 'regime_type' in htf_resampled.columns:
                mtf_features[f'{htf}_regime'] = htf_resampled['regime_type']
        
        LOGGER.info(f"Создано {len(mtf_features.columns)} MTF признаков для {target_timeframe}")
        return mtf_features
        
    except Exception as e:
        LOGGER.error(f"Ошибка создания MTF признаков: {e}")
        return pd.DataFrame(index=target_df.index)