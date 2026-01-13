"""
Предобработка данных: очистка, заполнение пропусков, Z-нормализация, 
фильтрация низколиквидных периодов.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from scipy import stats
from sklearn.preprocessing import RobustScaler

from config.hyperparameters import HYPERPARAMS
from utils.logger import LOGGER


class DataPreprocessor:
    """Предобработчик мультитаймфреймовых данных"""
    
    def __init__(self,
                 zscore_window: Optional[int] = None,
                 fill_method: str = 'ffill',
                 remove_outliers: bool = True,
                 outlier_threshold: float = 3.0,
                 low_liquidity_hours: Optional[List[int]] = None):
        
        self.zscore_window = zscore_window or HYPERPARAMS.data.zscore_window
        self.fill_method = fill_method
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.low_liquidity_hours = low_liquidity_hours or HYPERPARAMS.data.low_liquidity_hours
        
        self.scalers: Dict[str, Dict[str, RobustScaler]] = {}
        self.preprocessing_stats: Dict[str, Dict] = {}
        
        LOGGER.info(f"Инициализация препроцессора: zscore_window={self.zscore_window}")
    
    def preprocess_all(self, 
                       data_dict: Dict[str, pd.DataFrame],
                       mark_low_liquidity: bool = True,
                       compute_returns: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Полная предобработка всех таймфреймов.
        
        Args:
            data_dict: {timeframe: DataFrame}
            mark_low_liquidity: Отметить низколиквидные периоды
            compute_returns: Вычислить доходности
        
        Returns:
            Предобработанные данные
        """
        processed_data = {}
        
        for tf, df in data_dict.items():
            try:
                LOGGER.info(f"Предобработка {tf}...")
                
                df_processed = df.copy()
                
                # 1. Заполнение пропусков
                df_processed = self._fill_missing_values(df_processed, tf)
                
                # 2. Удаление выбросов
                if self.remove_outliers:
                    df_processed = self._remove_outliers(df_processed, tf)
                
                # 3. Отметка низколиквидных периодов
                if mark_low_liquidity:
                    df_processed = self._mark_low_liquidity_periods(df_processed, tf)
                
                # 4. Вычисление доходностей
                if compute_returns:
                    df_processed = self._compute_returns(df_processed, tf)
                
                # 5. Z-нормализация признаков
                df_processed = self._apply_zscore_normalization(df_processed, tf)
                
                # 6. Добавление технических мета-признаков
                df_processed = self._add_technical_metafeatures(df_processed, tf)
                
                processed_data[tf] = df_processed
                
                # Статистика
                stats = self._compute_preprocessing_stats(df, df_processed, tf)
                self.preprocessing_stats[tf] = stats
                
                LOGGER.info(f"{tf} предобработан: {len(df_processed)} баров, "
                          f"{len(df_processed.columns)} признаков")
                
            except Exception as e:
                LOGGER.error(f"Ошибка предобработки {tf}: {e}", exc_info=True)
                continue
        
        LOGGER.info(f"Предобработка завершена для {len(processed_data)} таймфреймов")
        return processed_data
    
    def _fill_missing_values(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Заполнение пропущенных значений"""
        try:
            missing_before = df.isnull().sum().sum()
            
            if missing_before == 0:
                return df
            
            LOGGER.debug(f"{tf}: заполнение {missing_before} пропусков методом {self.fill_method}")
            
            if self.fill_method == 'ffill':
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')  # Для первых строк
            elif self.fill_method == 'interpolate':
                df = df.interpolate(method='linear', limit_direction='both')
            elif self.fill_method == 'mean':
                df = df.fillna(df.mean())
            else:
                raise ValueError(f"Неизвестный метод заполнения: {self.fill_method}")
            
            missing_after = df.isnull().sum().sum()
            
            if missing_after > 0:
                LOGGER.warning(f"{tf}: осталось {missing_after} пропусков после заполнения")
                df = df.dropna()
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка заполнения пропусков {tf}: {e}")
            return df
    
    def _remove_outliers(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Удаление выбросов на основе Z-score"""
        try:
            price_columns = ['Open', 'High', 'Low', 'Close']
            original_length = len(df)
            
            for col in price_columns:
                if col in df.columns:
                    z_scores = np.abs(stats.zscore(df[col]))
                    df = df[z_scores < self.outlier_threshold]
            
            removed = original_length - len(df)
            
            if removed > 0:
                LOGGER.warning(f"{tf}: удалено {removed} выбросов "
                             f"({removed/original_length*100:.2f}%)")
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка удаления выбросов {tf}: {e}")
            return df
    
    def _mark_low_liquidity_periods(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Отметка периодов с низкой ликвидностью"""
        try:
            df['is_low_liquidity'] = 0
            
            # Отметка по часам
            df['hour'] = df.index.hour
            df.loc[df['hour'].isin(self.low_liquidity_hours), 'is_low_liquidity'] = 1
            
            # Отметка выходных (если есть данные)
            df['dayofweek'] = df.index.dayofweek
            df.loc[df['dayofweek'].isin([5, 6]), 'is_low_liquidity'] = 1
            
            # Отметка по объему (ниже 10-го перцентиля)
            volume_threshold = df['Volume'].quantile(0.10)
            df.loc[df['Volume'] < volume_threshold, 'is_low_liquidity'] = 1
            
            low_liq_count = (df['is_low_liquidity'] == 1).sum()
            LOGGER.debug(f"{tf}: {low_liq_count} баров низкой ликвидности "
                        f"({low_liq_count/len(df)*100:.2f}%)")
            
            df = df.drop(['hour', 'dayofweek'], axis=1)
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка отметки низкой ликвидности {tf}: {e}")
            df['is_low_liquidity'] = 0
            return df
    
    def _compute_returns(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Вычисление различных типов доходностей"""
        try:
            # Простые доходности
            df['returns'] = df['Close'].pct_change()
            
            # Логарифмические доходности
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Доходности с лагами
            for lag in [2, 3, 5]:
                df[f'returns_lag{lag}'] = df['Close'].pct_change(periods=lag)
            
            # Волатильность доходностей (скользящее окно)
            df['returns_volatility'] = df['returns'].rolling(window=20).std()
            
            # Заполнение первых NaN
            df = df.fillna(method='bfill')
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления доходностей {tf}: {e}")
            return df
    
    def _apply_zscore_normalization(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Применение скользящей Z-нормализации к признакам"""
        try:
            features_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for feature in features_to_normalize:
                if feature in df.columns:
                    try:
                        rolling_mean = df[feature].rolling(window=self.zscore_window, min_periods=1).mean()
                        rolling_std = df[feature].rolling(window=self.zscore_window, min_periods=1).std()
                        
                        # Избегаем деления на ноль
                        rolling_std = rolling_std.replace(0, 1e-8)
                        
                        zscore_feature_name = f'{feature}_zscore'
                        df[zscore_feature_name] = (df[feature] - rolling_mean) / rolling_std
                        
                        # Ограничиваем экстремальные значения
                        df[zscore_feature_name] = df[zscore_feature_name].clip(-5, 5)
                        
                    except Exception as e:
                        LOGGER.warning(f"Ошибка Z-нормализации {feature} в {tf}: {e}")
                        continue
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка Z-нормализации {tf}: {e}")
            return df
    
    def _add_technical_metafeatures(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Добавление технических мета-признаков"""
        try:
            # Истинный диапазон (True Range)
            df['true_range'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            
            # Средний истинный диапазон (ATR)
            for period in [14, 21]:
                df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()
            
            # Тело и тени свечей
            df['candle_body'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            
            # Нормализованные тени
            df['body_ratio'] = df['candle_body'] / (df['High'] - df['Low'] + 1e-8)
            
            # Направление свечи
            df['candle_direction'] = np.where(df['Close'] > df['Open'], 1, -1)
            
            # Размах H-L относительно цены
            df['hl_pct'] = (df['High'] - df['Low']) / df['Close']
            
            # Позиция закрытия в диапазоне H-L
            df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
            
            df = df.fillna(method='bfill')
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка добавления мета-признаков {tf}: {e}")
            return df
    
    def _compute_preprocessing_stats(self, 
                                     df_original: pd.DataFrame,
                                     df_processed: pd.DataFrame,
                                     tf: str) -> Dict:
        """Вычисление статистики предобработки"""
        try:
            stats = {
                'original_rows': len(df_original),
                'processed_rows': len(df_processed),
                'removed_rows': len(df_original) - len(df_processed),
                'original_columns': len(df_original.columns),
                'processed_columns': len(df_processed.columns),
                'added_features': len(df_processed.columns) - len(df_original.columns),
                'missing_values': int(df_processed.isnull().sum().sum()),
                'low_liquidity_bars': int((df_processed['is_low_liquidity'] == 1).sum())
            }
            
            return stats
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления статистики {tf}: {e}")
            return {}
    
    def get_feature_names(self, timeframe: str) -> List[str]:
        """Получение списка всех признаков для таймфрейма"""
        try:
            if timeframe in self.preprocessing_stats:
                # Возвращаем список из preprocessed данных
                return []  # Реализовать при необходимости
            return []
        except Exception as e:
            LOGGER.error(f"Ошибка получения списка признаков: {e}")
            return []
    
    def save_preprocessing_report(self, output_path: Optional[str] = None) -> None:
        """Сохранение отчета по предобработке"""
        try:
            if not self.preprocessing_stats:
                LOGGER.warning("Статистика предобработки отсутствует")
                return
            
            report_df = pd.DataFrame(self.preprocessing_stats).T
            
            if output_path:
                report_df.to_csv(output_path, index=True)
                LOGGER.info(f"Отчет предобработки сохранен: {output_path}")
            else:
                LOGGER.info(f"\n{report_df}")
                
        except Exception as e:
            LOGGER.error(f"Ошибка сохранения отчета: {e}")


# Вспомогательные функции
def robust_scale_feature(series: pd.Series, quantile_range: Tuple[float, float] = (0.25, 0.75)) -> pd.Series:
    """Робастное масштабирование признака с использованием квантилей"""
    try:
        scaler = RobustScaler(quantile_range=quantile_range)
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=series.index)
    except Exception as e:
        LOGGER.error(f"Ошибка робастного масштабирования: {e}")
        return series