"""
Детектор рыночных режимов с использованием PELT алгоритма для обнаружения
точек смены распределения и классификации сегментов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from scipy.stats import ks_2samp, anderson_ksamp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import ruptures as rpt

from config.hyperparameters import HYPERPARAMS
from config.market_regimes import (
    RegimeType, RegimeDefinitions,
    compute_volatility_percentile, compute_trend_strength, compute_mean_reversion_index
)
from utils.logger import LOGGER


class RegimeDetector:
    """Детектор и классификатор рыночных режимов"""
    
    def __init__(self,
                 num_regimes: Optional[int] = None,
                 pelt_penalty: Optional[float] = None,
                 min_segment_length: Optional[int] = None,
                 feature_windows: Optional[Dict[str, int]] = None):
        
        self.num_regimes = num_regimes or HYPERPARAMS.regime.num_regimes
        self.pelt_penalty = pelt_penalty or HYPERPARAMS.regime.pelt_penalty
        self.min_segment_length = min_segment_length or HYPERPARAMS.regime.min_segment_length
        
        self.feature_windows = feature_windows or {
            'volatility': 50,
            'trend': 50,
            'mean_reversion': 100
        }
        
        self.detected_changepoints: Dict[str, List[int]] = {}
        self.regime_segments: Dict[str, List[Dict]] = {}
        self.regime_features: Dict[str, pd.DataFrame] = {}
        self.regime_classifier: Optional[KMeans] = None
        
        LOGGER.info(f"Инициализация детектора режимов: {self.num_regimes} режимов, "
                   f"penalty={self.pelt_penalty}")
    
    def detect_regimes(self, 
                       data_dict: Dict[str, pd.DataFrame],
                       base_timeframe: str = 'H1') -> Dict[str, pd.DataFrame]:
        """
        Полный цикл детекции режимов для всех таймфреймов.
        
        Args:
            data_dict: {timeframe: DataFrame}
            base_timeframe: Базовый ТФ для основной детекции
        
        Returns:
            Данные с добавленными метками режимов
        """
        try:
            LOGGER.info(f"Начало детекции режимов на базовом ТФ: {base_timeframe}")
            
            if base_timeframe not in data_dict:
                raise ValueError(f"Базовый ТФ {base_timeframe} не найден")
            
            # 1. Детекция точек смены на базовом ТФ
            base_df = data_dict[base_timeframe].copy()
            changepoints = self._detect_changepoints_pelt(base_df, base_timeframe)
            self.detected_changepoints[base_timeframe] = changepoints
            
            # 2. Сегментация данных по changepoints
            segments = self._create_segments(base_df, changepoints, base_timeframe)
            self.regime_segments[base_timeframe] = segments
            
            # 3. Вычисление признаков для каждого сегмента
            segment_features = self._compute_segment_features(base_df, segments, base_timeframe)
            
            # 4. Кластеризация сегментов в режимы
            regime_labels = self._cluster_segments(segment_features, base_timeframe)
            
            # 5. Присвоение меток режимов всем барам
            base_df = self._assign_regime_labels(base_df, segments, regime_labels, base_timeframe)
            
            # 6. Пропагация меток на другие таймфреймы
            labeled_data = {base_timeframe: base_df}
            
            for tf, df in data_dict.items():
                if tf != base_timeframe:
                    try:
                        df_labeled = self._propagate_regime_labels(
                            df, base_df, base_timeframe, tf
                        )
                        labeled_data[tf] = df_labeled
                    except Exception as e:
                        LOGGER.error(f"Ошибка пропагации меток на {tf}: {e}")
                        labeled_data[tf] = df
            
            LOGGER.info(f"Детекция режимов завершена: {len(changepoints)} точек смены, "
                       f"{len(segments)} сегментов")
            
            return labeled_data
            
        except Exception as e:
            LOGGER.error(f"Критическая ошибка детекции режимов: {e}", exc_info=True)
            raise
    
    def _detect_changepoints_pelt(self, df: pd.DataFrame, tf: str) -> List[int]:
        """
        Детекция точек смены распределения с помощью PELT алгоритма.
        
        Args:
            df: DataFrame с данными
            tf: Название таймфрейма
        
        Returns:
            Список индексов точек смены
        """
        try:
            # Подготовка сигнала для детекции
            signal_data = self._prepare_signal_for_detection(df)
            
            if signal_data is None or len(signal_data) < self.min_segment_length * 2:
                LOGGER.warning(f"{tf}: недостаточно данных для детекции")
                return []
            
            # PELT алгоритм с различными моделями
            models_to_try = ['l2', 'rbf', 'normal', 'ar']
            best_changepoints = []
            best_score = float('inf')
            
            for model in models_to_try:
                try:
                    algo = rpt.Pelt(model=model, min_size=self.min_segment_length, jump=1)
                    algo.fit(signal_data)
                    changepoints = algo.predict(pen=self.pelt_penalty)
                    
                    # Удаляем последнюю точку (конец данных)
                    if changepoints and changepoints[-1] == len(signal_data):
                        changepoints = changepoints[:-1]
                    
                    # Оценка качества детекции
                    score = self._evaluate_changepoint_quality(signal_data, changepoints)
                    
                    if score < best_score:
                        best_score = score
                        best_changepoints = changepoints
                        
                except Exception as e:
                    LOGGER.debug(f"Модель {model} не сработала: {e}")
                    continue
            
            # Фильтрация слишком близких точек
            best_changepoints = self._filter_close_changepoints(
                best_changepoints, min_distance=self.min_segment_length
            )
            
            LOGGER.info(f"{tf}: обнаружено {len(best_changepoints)} точек смены (модель: best_score={best_score:.4f})")
            
            return best_changepoints
            
        except Exception as e:
            LOGGER.error(f"Ошибка PELT детекции {tf}: {e}", exc_info=True)
            return []
    
    def _prepare_signal_for_detection(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Подготовка многомерного сигнала для детекции изменений"""
        try:
            features = []
            
            # Цена закрытия (нормализованная)
            close_norm = (df['Close'] - df['Close'].mean()) / df['Close'].std()
            features.append(close_norm.values)
            
            # Доходности
            if 'returns' in df.columns:
                returns_norm = (df['returns'] - df['returns'].mean()) / (df['returns'].std() + 1e-8)
                features.append(returns_norm.fillna(0).values)
            
            # Волатильность
            if 'returns_volatility' in df.columns:
                vol_norm = (df['returns_volatility'] - df['returns_volatility'].mean()) / (df['returns_volatility'].std() + 1e-8)
                features.append(vol_norm.fillna(0).values)
            
            # Объем
            volume_norm = (df['Volume'] - df['Volume'].mean()) / (df['Volume'].std() + 1e-8)
            features.append(volume_norm.values)
            
            # Комбинирование в многомерный сигнал
            signal_matrix = np.column_stack(features)
            
            return signal_matrix
            
        except Exception as e:
            LOGGER.error(f"Ошибка подготовки сигнала: {e}")
            return None
    
    def _evaluate_changepoint_quality(self, signal: np.ndarray, changepoints: List[int]) -> float:
        """Оценка качества детектированных точек смены"""
        try:
            if not changepoints or len(changepoints) == 0:
                return float('inf')
            
            # Метрика: среднее различие между соседними сегментами
            total_score = 0.0
            prev_cp = 0
            
            for cp in changepoints:
                if cp <= prev_cp or cp >= len(signal):
                    continue
                
                segment1 = signal[prev_cp:cp]
                segment2 = signal[cp:min(cp + (cp - prev_cp), len(signal))]
                
                if len(segment1) > 5 and len(segment2) > 5:
                    # Статистическое различие (Kolmogorov-Smirnov)
                    try:
                        ks_stat, _ = ks_2samp(segment1.flatten(), segment2.flatten())
                        total_score += ks_stat
                    except Exception:
                        continue
                
                prev_cp = cp
            
            avg_score = total_score / len(changepoints) if changepoints else float('inf')
            
            # Инвертируем: чем больше различие, тем лучше
            return 1.0 / (avg_score + 1e-8)
            
        except Exception as e:
            LOGGER.error(f"Ошибка оценки changepoints: {e}")
            return float('inf')
    
    def _filter_close_changepoints(self, changepoints: List[int], min_distance: int) -> List[int]:
        """Фильтрация слишком близких точек смены"""
        try:
            if not changepoints:
                return []
            
            filtered = [changepoints[0]]
            
            for cp in changepoints[1:]:
                if cp - filtered[-1] >= min_distance:
                    filtered.append(cp)
            
            return filtered
            
        except Exception as e:
            LOGGER.error(f"Ошибка фильтрации changepoints: {e}")
            return changepoints
    
    def _create_segments(self, df: pd.DataFrame, changepoints: List[int], tf: str) -> List[Dict]:
        """Создание сегментов на основе точек смены"""
        try:
            segments = []
            
            if not changepoints:
                # Один сегмент на все данные
                segments.append({
                    'start_idx': 0,
                    'end_idx': len(df) - 1,
                    'start_date': df.index[0],
                    'end_date': df.index[-1],
                    'length': len(df)
                })
                return segments
            
            # Добавляем начало и конец
            all_points = [0] + changepoints + [len(df)]
            
            for i in range(len(all_points) - 1):
                start_idx = all_points[i]
                end_idx = all_points[i + 1] - 1
                
                if end_idx <= start_idx:
                    continue
                
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_date': df.index[start_idx],
                    'end_date': df.index[end_idx],
                    'length': end_idx - start_idx + 1
                })
            
            LOGGER.debug(f"{tf}: создано {len(segments)} сегментов")
            return segments
            
        except Exception as e:
            LOGGER.error(f"Ошибка создания сегментов {tf}: {e}")
            return []
    
    def _compute_segment_features(self, 
                                   df: pd.DataFrame, 
                                   segments: List[Dict], 
                                   tf: str) -> pd.DataFrame:
        """Вычисление признаков для каждого сегмента"""
        try:
            feature_list = []
            
            for seg_idx, seg in enumerate(segments):
                try:
                    seg_data = df.iloc[seg['start_idx']:seg['end_idx']+1]
                    
                    if len(seg_data) < 10:
                        continue
                    
                    features = {
                        'segment_id': seg_idx,
                        'length': seg['length']
                    }
                    
                    # Волатильность
                    returns = seg_data['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        features['volatility'] = returns.std()
                        features['volatility_percentile'] = compute_volatility_percentile(
                            returns.values, window=min(len(returns), self.feature_windows['volatility'])
                        )
                    else:
                        features['volatility'] = 0
                        features['volatility_percentile'] = 0.5
                    
                    # Сила тренда
                    features['trend_strength'] = compute_trend_strength(
                        seg_data['Close'].values,
                        window=min(len(seg_data), self.feature_windows['trend'])
                    )
                    
                    # Индекс возврата к среднему
                    features['mean_reversion_idx'] = compute_mean_reversion_index(
                        seg_data['Close'].values,
                        window=min(len(seg_data), self.feature_windows['mean_reversion'])
                    )
                    
                    # Дополнительные признаки
                    features['avg_volume'] = seg_data['Volume'].mean()
                    features['price_range'] = (seg_data['High'].max() - seg_data['Low'].min()) / seg_data['Close'].mean()
                    features['avg_body_ratio'] = seg_data['body_ratio'].mean() if 'body_ratio' in seg_data.columns else 0
                    
                    # Направление тренда
                    price_change = (seg_data['Close'].iloc[-1] - seg_data['Close'].iloc[0]) / seg_data['Close'].iloc[0]
                    features['price_change_pct'] = price_change
                    
                    # Число up/down свечей
                    if 'candle_direction' in seg_data.columns:
                        features['up_candles_ratio'] = (seg_data['candle_direction'] == 1).sum() / len(seg_data)
                    else:
                        features['up_candles_ratio'] = 0.5
                    
                    feature_list.append(features)
                    
                except Exception as e:
                    LOGGER.warning(f"Ошибка вычисления признаков сегмента {seg_idx}: {e}")
                    continue
            
            features_df = pd.DataFrame(feature_list)
            self.regime_features[tf] = features_df
            
            LOGGER.debug(f"{tf}: вычислено {len(features_df)} наборов признаков сегментов")
            return features_df
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления признаков сегментов {tf}: {e}")
            return pd.DataFrame()
    
    def _cluster_segments(self, features_df: pd.DataFrame, tf: str) -> np.ndarray:
        """Кластеризация сегментов в режимы"""
        try:
            if features_df.empty or len(features_df) < self.num_regimes:
                LOGGER.warning(f"{tf}: недостаточно сегментов для кластеризации")
                return np.zeros(len(features_df), dtype=int)
            
            # Выбор признаков для кластеризации
            cluster_features = [
                'volatility_percentile', 'trend_strength', 'mean_reversion_idx',
                'price_range', 'avg_body_ratio', 'up_candles_ratio'
            ]
            
            available_features = [f for f in cluster_features if f in features_df.columns]
            X = features_df[available_features].values
            
            # Нормализация
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means кластеризация
            best_labels = None
            best_inertia = float('inf')
            
            for n_init in [10, 20, 50]:
                try:
                    kmeans = KMeans(
                        n_clusters=self.num_regimes,
                        init='k-means++',
                        n_init=n_init,
                        max_iter=300,
                        random_state=42
                    )
                    labels = kmeans.fit_predict(X_scaled)
                    
                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        best_labels = labels
                        self.regime_classifier = kmeans
                        
                except Exception as e:
                    LOGGER.debug(f"Кластеризация с n_init={n_init} не удалась: {e}")
                    continue
            
            if best_labels is None:
                LOGGER.error(f"{tf}: кластеризация не удалась")
                return np.zeros(len(features_df), dtype=int)
            
            # Логирование распределения
            unique, counts = np.unique(best_labels, return_counts=True)
            distribution = dict(zip(unique, counts))
            LOGGER.info(f"{tf}: распределение сегментов по кластерам: {distribution}")
            
            return best_labels
            
        except Exception as e:
            LOGGER.error(f"Ошибка кластеризации {tf}: {e}", exc_info=True)
            return np.zeros(len(features_df), dtype=int)
    
    def _assign_regime_labels(self, 
                              df: pd.DataFrame, 
                              segments: List[Dict],
                              regime_labels: np.ndarray,
                              tf: str) -> pd.DataFrame:
        """Присвоение меток режимов всем барам"""
        try:
            df['regime'] = -1
            df['regime_confidence'] = 0.0
            
            for seg_idx, seg in enumerate(segments):
                if seg_idx >= len(regime_labels):
                    continue
                
                regime_label = regime_labels[seg_idx]
                start_idx = seg['start_idx']
                end_idx = seg['end_idx']
                
                df.iloc[start_idx:end_idx+1, df.columns.get_loc('regime')] = regime_label
                
                # Confidence снижается к границам сегмента
                seg_length = end_idx - start_idx + 1
                confidence_curve = self._compute_confidence_curve(seg_length)
                df.iloc[start_idx:end_idx+1, df.columns.get_loc('regime_confidence')] = confidence_curve
            
            # Интерпретация кластеров в понятные режимы
            df = self._interpret_regime_clusters(df, tf)
            
            LOGGER.debug(f"{tf}: метки режимов присвоены")
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка присвоения меток {tf}: {e}")
            return df
    
    def _compute_confidence_curve(self, segment_length: int) -> np.ndarray:
        """Вычисление кривой уверенности для сегмента"""
        try:
            # Параболическая кривая: максимум в центре
            x = np.linspace(-1, 1, segment_length)
            confidence = 1.0 - x**2
            return confidence
            
        except Exception:
            return np.ones(segment_length)
    
    def _interpret_regime_clusters(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Интерпретация числовых кластеров в осмысленные режимы"""
        try:
            if 'regime' not in df.columns:
                return df
            
            # Вычисление характеристик каждого кластера
            cluster_characteristics = {}
            
            for cluster_id in df['regime'].unique():
                if cluster_id < 0:
                    continue
                
                cluster_data = df[df['regime'] == cluster_id]
                
                if len(cluster_data) < 10:
                    continue
                
                returns = cluster_data['Close'].pct_change().dropna()
                
                characteristics = {
                    'volatility_percentile': compute_volatility_percentile(returns.values, window=50),
                    'trend_strength': compute_trend_strength(cluster_data['Close'].values, window=50),
                    'mean_reversion_idx': compute_mean_reversion_index(cluster_data['Close'].values, window=100)
                }
                
                # Классификация в RegimeType
                regime_type = RegimeDefinitions.classify_regime(
                    characteristics['volatility_percentile'],
                    characteristics['trend_strength'],
                    characteristics['mean_reversion_idx']
                )
                
                cluster_characteristics[cluster_id] = {
                    'regime_type': regime_type,
                    **characteristics
                }
            
            # Создание нового столбца с интерпретированными режимами
            df['regime_type'] = df['regime'].map(
                lambda x: cluster_characteristics.get(x, {}).get('regime_type', RegimeType.TRANSITION).value
                if x >= 0 else 'unknown'
            )
            
            LOGGER.debug(f"{tf}: кластеры интерпретированы: {cluster_characteristics}")
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка интерпретации кластеров {tf}: {e}")
            return df
    
    def _propagate_regime_labels(self, 
                                  target_df: pd.DataFrame,
                                  base_df: pd.DataFrame,
                                  base_tf: str,
                                  target_tf: str) -> pd.DataFrame:
        """Пропагация меток режимов с базового ТФ на целевой"""
        try:
            target_df['regime'] = -1
            target_df['regime_type'] = 'unknown'
            target_df['regime_confidence'] = 0.0
            
            # Метод: forward fill с базового ТФ
            for idx in target_df.index:
                try:
                    # Найти ближайший бар в базовом ТФ
                    closest_base_idx = base_df.index.asof(idx)
                    
                    if pd.isna(closest_base_idx):
                        continue
                    
                    base_row = base_df.loc[closest_base_idx]
                    
                    target_df.at[idx, 'regime'] = base_row['regime']
                    target_df.at[idx, 'regime_type'] = base_row['regime_type']
                    target_df.at[idx, 'regime_confidence'] = base_row['regime_confidence'] * 0.8  # Снижаем на 20%
                    
                except Exception:
                    continue
            
            LOGGER.debug(f"{target_tf}: метки спропагированы с {base_tf}")
            return target_df
            
        except Exception as e:
            LOGGER.error(f"Ошибка пропагации меток {base_tf}->{target_tf}: {e}")
            return target_df
    
    def get_regime_summary(self, tf: str) -> Dict:
        """Получение сводной статистики по режимам"""
        try:
            if tf not in self.regime_segments:
                return {}
            
            segments = self.regime_segments[tf]
            features = self.regime_features.get(tf, pd.DataFrame())
            
            summary = {
                'total_segments': len(segments),
                'avg_segment_length': np.mean([s['length'] for s in segments]),
                'regime_distribution': {},
                'avg_features_by_regime': {}
            }
            
            if not features.empty and 'regime' in features.columns:
                for regime in features['regime'].unique():
                    regime_data = features[features['regime'] == regime]
                    summary['regime_distribution'][int(regime)] = len(regime_data)
                    
                    summary['avg_features_by_regime'][int(regime)] = {
                        'volatility': float(regime_data['volatility_percentile'].mean()),
                        'trend': float(regime_data['trend_strength'].mean()),
                        'mean_reversion': float(regime_data['mean_reversion_idx'].mean())
                    }
            
            return summary
            
        except Exception as e:
            LOGGER.error(f"Ошибка создания сводки режимов {tf}: {e}")
            return {}


def validate_regime_consistency(df: pd.DataFrame, window: int = 10) -> float:
    """Валидация консистентности режимов (нет частых переключений)"""
    try:
        if 'regime' not in df.columns:
            return 0.0
        
        regime_changes = (df['regime'] != df['regime'].shift(1)).sum()
        change_frequency = regime_changes / len(df)
        
        # Метрика: чем меньше частота переключений, тем лучше
        consistency_score = 1.0 - min(change_frequency * 10, 1.0)
        
        return consistency_score
        
    except Exception as e:
        LOGGER.error(f"Ошибка валидации консистентности: {e}")
        return 0.0