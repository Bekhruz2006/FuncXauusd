"""
Библиотека атомарных функций - базовые строительные блоки для агентов.
Абстрактные индикаторы независимые от конкретной технической реализации.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union
from scipy import signal as scipy_signal
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

from utils.logger import LOGGER


class AtomicFunction:
    """Базовый класс для атомарной функции"""
    
    def __init__(self, name: str, description: str, parameters: Dict):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.execution_count = 0
        self.error_count = 0
    
    def compute(self, df: pd.DataFrame, **kwargs) -> Union[pd.Series, float, bool]:
        """Вычисление функции. Переопределяется в подклассах"""
        raise NotImplementedError
    
    def validate_inputs(self, df: pd.DataFrame) -> bool:
        """Валидация входных данных"""
        try:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [col for col in required_columns if col not in df.columns]
            
            if missing:
                LOGGER.warning(f"{self.name}: отсутствуют колонки {missing}")
                return False
            
            if len(df) < 2:
                LOGGER.warning(f"{self.name}: недостаточно данных")
                return False
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Ошибка валидации входов {self.name}: {e}")
            return False


class TrendAtom(AtomicFunction):
    """Атом определения трендовости"""
    
    def __init__(self, window: int = 50, method: str = 'ema'):
        super().__init__(
            name=f"trend_{method}_{window}",
            description=f"Определение тренда методом {method} с окном {window}",
            parameters={'window': window, 'method': method}
        )
        self.window = window
        self.method = method
    
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Вычисление силы и направления тренда.
        
        Returns:
            Series со значениями от -1 (сильный downtrend) до 1 (сильный uptrend)
        """
        try:
            self.execution_count += 1
            
            if not self.validate_inputs(df):
                return pd.Series(0, index=df.index)
            
            if len(df) < self.window:
                return pd.Series(0, index=df.index)
            
            close = df['Close']
            
            if self.method == 'ema':
                trend_signal = self._ema_trend(close)
            elif self.method == 'linreg':
                trend_signal = self._linreg_trend(close)
            elif self.method == 'adx':
                trend_signal = self._adx_trend(df)
            else:
                trend_signal = self._ema_trend(close)
            
            return trend_signal
            
        except Exception as e:
            self.error_count += 1
            LOGGER.error(f"Ошибка вычисления {self.name}: {e}")
            return pd.Series(0, index=df.index)
    
    def _ema_trend(self, close: pd.Series) -> pd.Series:
        """Тренд через EMA crossover"""
        try:
            ema_fast = close.ewm(span=self.window // 2, adjust=False).mean()
            ema_slow = close.ewm(span=self.window, adjust=False).mean()
            
            diff = (ema_fast - ema_slow) / ema_slow
            normalized = np.tanh(diff * 100)
            
            return pd.Series(normalized, index=close.index)
            
        except Exception as e:
            LOGGER.error(f"Ошибка EMA тренда: {e}")
            return pd.Series(0, index=close.index)
    
    def _linreg_trend(self, close: pd.Series) -> pd.Series:
        """Тренд через скользящую линейную регрессию"""
        try:
            trend_values = []
            
            for i in range(len(close)):
                if i < self.window:
                    trend_values.append(0)
                    continue
                
                window_data = close.iloc[i-self.window:i].values
                x = np.arange(self.window).reshape(-1, 1)
                
                try:
                    model = LinearRegression()
                    model.fit(x, window_data)
                    slope = model.coef_[0]
                    
                    normalized_slope = np.tanh(slope / (window_data.mean() * 0.01))
                    trend_values.append(normalized_slope)
                    
                except Exception:
                    trend_values.append(0)
            
            return pd.Series(trend_values, index=close.index)
            
        except Exception as e:
            LOGGER.error(f"Ошибка LinReg тренда: {e}")
            return pd.Series(0, index=close.index)
    
    def _adx_trend(self, df: pd.DataFrame) -> pd.Series:
        """Тренд через ADX индикатор"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=self.window).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=self.window).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=self.window).mean() / atr)
            
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
            adx = dx.rolling(window=self.window).mean()
            
            trend_direction = np.where(plus_di > minus_di, 1, -1)
            trend_strength = adx / 100.0
            
            trend_signal = trend_direction * trend_strength
            
            return pd.Series(trend_signal, index=df.index).fillna(0)
            
        except Exception as e:
            LOGGER.error(f"Ошибка ADX тренда: {e}")
            return pd.Series(0, index=df.index)


class MomentumAtom(AtomicFunction):
    """Атом измерения импульса"""
    
    def __init__(self, period: int = 14, method: str = 'roc'):
        super().__init__(
            name=f"momentum_{method}_{period}",
            description=f"Измерение импульса методом {method} за {period} периодов",
            parameters={'period': period, 'method': method}
        )
        self.period = period
        self.method = method
    
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Вычисление импульса движения.
        
        Returns:
            Series с нормализованными значениями импульса
        """
        try:
            self.execution_count += 1
            
            if not self.validate_inputs(df):
                return pd.Series(0, index=df.index)
            
            close = df['Close']
            
            if self.method == 'roc':
                momentum = self._rate_of_change(close)
            elif self.method == 'rsi':
                momentum = self._rsi_momentum(close)
            elif self.method == 'stochastic':
                momentum = self._stochastic_momentum(df)
            else:
                momentum = self._rate_of_change(close)
            
            return momentum
            
        except Exception as e:
            self.error_count += 1
            LOGGER.error(f"Ошибка вычисления {self.name}: {e}")
            return pd.Series(0, index=df.index)
    
    def _rate_of_change(self, close: pd.Series) -> pd.Series:
        """Rate of Change индикатор"""
        try:
            roc = (close - close.shift(self.period)) / close.shift(self.period) * 100
            normalized = np.tanh(roc / 10)
            return normalized.fillna(0)
            
        except Exception as e:
            LOGGER.error(f"Ошибка ROC: {e}")
            return pd.Series(0, index=close.index)
    
    def _rsi_momentum(self, close: pd.Series) -> pd.Series:
        """RSI как мера импульса"""
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.period).mean()
            avg_loss = loss.rolling(window=self.period).mean()
            
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            normalized = (rsi - 50) / 50
            
            return normalized.fillna(0)
            
        except Exception as e:
            LOGGER.error(f"Ошибка RSI: {e}")
            return pd.Series(0, index=close.index)
    
    def _stochastic_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Stochastic Oscillator как мера импульса"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            lowest_low = low.rolling(window=self.period).min()
            highest_high = high.rolling(window=self.period).max()
            
            k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
            
            normalized = (k - 50) / 50
            
            return normalized.fillna(0)
            
        except Exception as e:
            LOGGER.error(f"Ошибка Stochastic: {e}")
            return pd.Series(0, index=df.index)


class VolatilityCompressionAtom(AtomicFunction):
    """Атом обнаружения сжатия волатильности"""
    
    def __init__(self, window: int = 20, threshold_percentile: float = 0.2):
        super().__init__(
            name=f"vol_compression_{window}",
            description=f"Обнаружение сжатия волатильности за {window} периодов",
            parameters={'window': window, 'threshold': threshold_percentile}
        )
        self.window = window
        self.threshold_percentile = threshold_percentile
    
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Обнаружение периодов сжатия волатильности.
        
        Returns:
            Series с значениями от 0 (нет сжатия) до 1 (сильное сжатие)
        """
        try:
            self.execution_count += 1
            
            if not self.validate_inputs(df):
                return pd.Series(0, index=df.index)
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            # True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR
            atr = tr.rolling(window=self.window).mean()
            
            # Bollinger Bands width
            sma = close.rolling(window=self.window).mean()
            std = close.rolling(window=self.window).std()
            bb_width = (std * 2) / sma
            
            # Нормализация к историческим значениям
            atr_rolling = atr.rolling(window=self.window * 5).apply(
                lambda x: np.percentile(x, self.threshold_percentile * 100)
            )
            bb_rolling = bb_width.rolling(window=self.window * 5).apply(
                lambda x: np.percentile(x, self.threshold_percentile * 100)
            )
            
            # Сжатие когда текущие значения ниже порога
            atr_compression = np.clip(1 - (atr / (atr_rolling + 1e-8)), 0, 1)
            bb_compression = np.clip(1 - (bb_width / (bb_rolling + 1e-8)), 0, 1)
            
            # Комбинирование
            compression = (atr_compression + bb_compression) / 2
            
            return compression.fillna(0)
            
        except Exception as e:
            self.error_count += 1
            LOGGER.error(f"Ошибка вычисления {self.name}: {e}")
            return pd.Series(0, index=df.index)


class DivergenceAtom(AtomicFunction):
    """Атом обнаружения дивергенций"""
    
    def __init__(self, lookback: int = 14, price_window: int = 5):
        super().__init__(
            name=f"divergence_{lookback}",
            description=f"Обнаружение дивергенций за {lookback} периодов",
            parameters={'lookback': lookback, 'price_window': price_window}
        )
        self.lookback = lookback
        self.price_window = price_window
    
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Обнаружение бычьих/медвежьих дивергенций.
        
        Returns:
            Series: 1 (бычья дивергенция), -1 (медвежья), 0 (нет)
        """
        try:
            self.execution_count += 1
            
            if not self.validate_inputs(df):
                return pd.Series(0, index=df.index)
            
            close = df['Close']
            
            # Вычисляем RSI для сравнения
            rsi = self._compute_rsi(close, period=self.lookback)
            
            # Поиск локальных экстремумов
            price_highs = self._find_peaks(close, self.price_window)
            price_lows = self._find_troughs(close, self.price_window)
            rsi_highs = self._find_peaks(rsi, self.price_window)
            rsi_lows = self._find_troughs(rsi, self.price_window)
            
            divergence = pd.Series(0, index=df.index)
            
            # Бычья дивергенция: цена делает lower low, RSI делает higher low
            for i in range(self.lookback, len(df)):
                recent_price_lows = price_lows[max(0, i-self.lookback):i]
                recent_rsi_lows = rsi_lows[max(0, i-self.lookback):i]
                
                if len(recent_price_lows) >= 2 and len(recent_rsi_lows) >= 2:
                    if (close.iloc[recent_price_lows[-1]] < close.iloc[recent_price_lows[-2]] and
                        rsi.iloc[recent_rsi_lows[-1]] > rsi.iloc[recent_rsi_lows[-2]]):
                        divergence.iloc[i] = 1
                
                # Медвежья дивергенция: цена делает higher high, RSI делает lower high
                recent_price_highs = price_highs[max(0, i-self.lookback):i]
                recent_rsi_highs = rsi_highs[max(0, i-self.lookback):i]
                
                if len(recent_price_highs) >= 2 and len(recent_rsi_highs) >= 2:
                    if (close.iloc[recent_price_highs[-1]] > close.iloc[recent_price_highs[-2]] and
                        rsi.iloc[recent_rsi_highs[-1]] < rsi.iloc[recent_rsi_highs[-2]]):
                        divergence.iloc[i] = -1
            
            return divergence
            
        except Exception as e:
            self.error_count += 1
            LOGGER.error(f"Ошибка вычисления {self.name}: {e}")
            return pd.Series(0, index=df.index)
    
    def _compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Вспомогательная функция для RSI"""
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)
            
        except Exception:
            return pd.Series(50, index=close.index)
    
    def _find_peaks(self, series: pd.Series, window: int) -> List[int]:
        """Поиск локальных максимумов"""
        try:
            peaks = []
            for i in range(window, len(series) - window):
                if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                    peaks.append(i)
            return peaks
        except Exception:
            return []
    
    def _find_troughs(self, series: pd.Series, window: int) -> List[int]:
        """Поиск локальных минимумов"""
        try:
            troughs = []
            for i in range(window, len(series) - window):
                if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                    troughs.append(i)
            return troughs
        except Exception:
            return []


class SupportResistanceAtom(AtomicFunction):
    """Атом определения уровней поддержки/сопротивления"""
    
    def __init__(self, lookback: int = 50, num_levels: int = 3, tolerance: float = 0.02):
        super().__init__(
            name=f"support_resistance_{lookback}",
            description=f"Определение S/R уровней за {lookback} периодов",
            parameters={'lookback': lookback, 'num_levels': num_levels, 'tolerance': tolerance}
        )
        self.lookback = lookback
        self.num_levels = num_levels
        self.tolerance = tolerance
    
    def compute(self, df: pd.DataFrame, **kwargs) -> Dict[str, List[float]]:
        """
        Вычисление уровней поддержки и сопротивления.
        
        Returns:
            Dict с ключами 'support' и 'resistance', значения - списки уровней
        """
        try:
            self.execution_count += 1
            
            if not self.validate_inputs(df):
                return {'support': [], 'resistance': []}
            
            if len(df) < self.lookback:
                return {'support': [], 'resistance': []}
            
            recent_data = df.iloc[-self.lookback:]
            
            # Кластеризация цен High и Low
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            
            resistance_levels = self._find_price_clusters(highs, self.num_levels)
            support_levels = self._find_price_clusters(lows, self.num_levels)
            
            return {
                'support': sorted(support_levels),
                'resistance': sorted(resistance_levels)
            }
            
        except Exception as e:
            self.error_count += 1
            LOGGER.error(f"Ошибка вычисления {self.name}: {e}")
            return {'support': [], 'resistance': []}
    
    def _find_price_clusters(self, prices: np.ndarray, num_clusters: int) -> List[float]:
        """Поиск кластеров цен"""
        try:
            from sklearn.cluster import KMeans
            
            if len(prices) < num_clusters:
                return list(prices)
            
            prices_reshaped = prices.reshape(-1, 1)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(prices_reshaped)
            
            centers = kmeans.cluster_centers_.flatten()
            
            return list(centers)
            
        except Exception as e:
            LOGGER.error(f"Ошибка кластеризации цен: {e}")
            return []
    
    def compute_distance_to_levels(self, df: pd.DataFrame) -> pd.Series:
        """Вычисление расстояния текущей цены до ближайших уровней"""
        try:
            levels = self.compute(df)
            all_levels = levels['support'] + levels['resistance']
            
            if not all_levels:
                return pd.Series(0, index=df.index)
            
            close = df['Close']
            distances = []
            
            for price in close:
                min_distance = min([abs(price - level) / price for level in all_levels])
                distances.append(min_distance)
            
            return pd.Series(distances, index=df.index)
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления расстояний: {e}")
            return pd.Series(0, index=df.index)


class VolumePatternAtom(AtomicFunction):
    """Атом анализа паттернов объема"""
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"volume_pattern_{window}",
            description=f"Анализ паттернов объема за {window} периодов",
            parameters={'window': window}
        )
        self.window = window
    
    def compute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Вычисление признаков объема.
        
        Returns:
            DataFrame с колонками volume_spike, volume_trend, volume_profile
        """
        try:
            self.execution_count += 1
            
            if not self.validate_inputs(df):
                return pd.DataFrame(index=df.index)
            
            volume = df['Volume']
            
            features = pd.DataFrame(index=df.index)
            
            # Volume spike (превышение среднего)
            vol_ma = volume.rolling(window=self.window).mean()
            vol_std = volume.rolling(window=self.window).std()
            features['volume_spike'] = (volume - vol_ma) / (vol_std + 1e-8)
            
            # Volume trend
            volume_change = volume.pct_change(periods=self.window)
            features['volume_trend'] = np.tanh(volume_change)
            
            # Volume profile (относительная позиция в диапазоне)
            vol_min = volume.rolling(window=self.window).min()
            vol_max = volume.rolling(window=self.window).max()
            features['volume_profile'] = (volume - vol_min) / (vol_max - vol_min + 1e-8)
            
            # On-Balance Volume normalized
            obv = self._compute_obv(df)
            obv_ma = obv.rolling(window=self.window).mean()
            obv_std = obv.rolling(window=self.window).std()
            features['obv_normalized'] = (obv - obv_ma) / (obv_std + 1e-8)
            
            return features.fillna(0)
            
        except Exception as e:
            self.error_count += 1
            LOGGER.error(f"Ошибка вычисления {self.name}: {e}")
            return pd.DataFrame(index=df.index)
    
    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        try:
            close = df['Close']
            volume = df['Volume']
            
            obv = [0]
            
            for i in range(1, len(df)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            
            return pd.Series(obv, index=df.index)
            
        except Exception:
            return pd.Series(0, index=df.index)


class AtomicFunctionLibrary:
    """Центральная библиотека всех атомарных функций"""
    
    def __init__(self):
        self.functions: Dict[str, AtomicFunction] = {}
        self.categories: Dict[str, List[str]] = {
            'trend': [],
            'momentum': [],
            'volatility': [],
            'pattern': [],
            'volume': []
        }
        
        self._initialize_library()
        
        LOGGER.info(f"Инициализирована библиотека атомарных функций: {len(self.functions)} функций")
    
    def _initialize_library(self) -> None:
        """Инициализация всех доступных функций"""
        try:
            # Trend функции
            for window in [20, 50, 100]:
                for method in ['ema', 'linreg', 'adx']:
                    func = TrendAtom(window=window, method=method)
                    self.register_function(func, category='trend')
            
            # Momentum функции
            for period in [7, 14, 21]:
                for method in ['roc', 'rsi', 'stochastic']:
                    func = MomentumAtom(period=period, method=method)
                    self.register_function(func, category='momentum')
            
            # Volatility функции
            for window in [10, 20, 40]:
                func = VolatilityCompressionAtom(window=window)
                self.register_function(func, category='volatility')
            
            # Pattern функции
            for lookback in [14, 21, 28]:
                func = DivergenceAtom(lookback=lookback)
                self.register_function(func, category='pattern')
            
            for lookback in [30, 50, 100]:
                func = SupportResistanceAtom(lookback=lookback)
                self.register_function(func, category='pattern')
            
            # Volume функции
            for window in [10, 20, 30]:
                func = VolumePatternAtom(window=window)
                self.register_function(func, category='volume')
            
        except Exception as e:
            LOGGER.error(f"Ошибка инициализации библиотеки: {e}", exc_info=True)
    
    def register_function(self, func: AtomicFunction, category: str) -> None:
        """Регистрация новой функции в библиотеке"""
        try:
            self.functions[func.name] = func
            
            if category in self.categories:
                self.categories[category].append(func.name)
            else:
                self.categories[category] = [func.name]
                
        except Exception as e:
            LOGGER.error(f"Ошибка регистрации функции {func.name}: {e}")
    
    def get_function(self, name: str) -> Optional[AtomicFunction]:
        """Получение функции по имени"""
        return self.functions.get(name)
    
    def get_functions_by_category(self, category: str) -> List[AtomicFunction]:
        """Получение всех функций категории"""
        try:
            func_names = self.categories.get(category, [])
            return [self.functions[name] for name in func_names if name in self.functions]
        except Exception as e:
            LOGGER.error(f"Ошибка получения функций категории {category}: {e}")
            return []
    
    def compute_all(self, df: pd.DataFrame, categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Вычисление всех функций библиотеки"""
        try:
            results = pd.DataFrame(index=df.index)
            
            target_categories = categories or list(self.categories.keys())
            
            for category in target_categories:
                functions = self.get_functions_by_category(category)
                
                for func in functions:
                    try:
                        result = func.compute(df)
                        
                        if isinstance(result, pd.Series):
                            results[func.name] = result
                        elif isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f"{func.name}_{col}"] = result[col]
                        
                    except Exception as e:
                        LOGGER.error(f"Ошибка вычисления {func.name}: {e}")
                        continue
            
            LOGGER.info(f"Вычислено {len(results.columns)} атомарных признаков")
            return results
            
        except Exception as e:
            LOGGER.error(f"Ошибка массового вычисления: {e}", exc_info=True)
            return pd.DataFrame(index=df.index)
    
    def get_statistics(self) -> pd.DataFrame:
        """Статистика использования функций"""
        try:
            stats = []
            
            for name, func in self.functions.items():
                stats.append({
                    'name': name,
                    'executions': func.execution_count,
                    'errors': func.error_count,
                    'error_rate': func.error_count / (func.execution_count + 1e-8)
                })
            
            return pd.DataFrame(stats)
            
        except Exception as e:
            LOGGER.error(f"Ошибка получения статистики: {e}")
            return pd.DataFrame()