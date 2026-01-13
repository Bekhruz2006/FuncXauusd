"""
Загрузчик мультитаймфреймовых данных с валидацией и синхронизацией.
Обрабатывает CSV файлы с различными таймфреймами и приводит к единому формату.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from config.hyperparameters import HYPERPARAMS
from config.paths import PATHS
from utils.logger import LOGGER


class MultiTimeframeDataLoader:
    """Загрузчик и синхронизатор мультитаймфреймовых данных"""
    
    def __init__(self, 
                 data_dir: Optional[Path] = None,
                 timeframes: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 validate_on_load: bool = True):
        
        self.data_dir = data_dir or PATHS.RAW_DATA_DIR
        self.timeframes = timeframes or HYPERPARAMS.data.timeframes
        self.start_date = start_date or HYPERPARAMS.data.start_date
        self.end_date = end_date or HYPERPARAMS.data.end_date
        self.validate_on_load = validate_on_load
        
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.sync_index: Optional[pd.DatetimeIndex] = None
        self.validation_stats: Dict[str, Dict] = {}
        
        LOGGER.info(f"Инициализация загрузчика данных для ТФ: {self.timeframes}")
    
    def load_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузка всех таймфреймов с валидацией.
        
        Returns:
            Словарь {timeframe: DataFrame} с загруженными данными
        """
        loaded_data = {}
        failed_timeframes = []
        
        for tf in self.timeframes:
            try:
                LOGGER.info(f"Загрузка данных для {tf}...")
                df = self._load_single_timeframe(tf)
                
                if df is not None and not df.empty:
                    loaded_data[tf] = df
                    self.data_cache[tf] = df
                    LOGGER.info(f"{tf} загружен: {len(df)} баров")
                else:
                    failed_timeframes.append(tf)
                    LOGGER.warning(f"Не удалось загрузить {tf}: пустой датасет")
                    
            except Exception as e:
                failed_timeframes.append(tf)
                LOGGER.error(f"Ошибка загрузки {tf}: {e}", exc_info=True)
        
        if failed_timeframes:
            LOGGER.warning(f"Не загружены таймфреймы: {failed_timeframes}")
        
        if not loaded_data:
            raise ValueError("Не удалось загрузить ни одного таймфрейма")
        
        LOGGER.info(f"Загружено {len(loaded_data)} из {len(self.timeframes)} таймфреймов")
        return loaded_data
    
    def _load_single_timeframe(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Загрузка одного таймфрейма из CSV.
        
        Args:
            timeframe: Название таймфрейма (M1, M5, H1, etc.)
        
        Returns:
            DataFrame с данными или None при ошибке
        """
        try:
            filepath = PATHS.get_raw_data_path(timeframe)
            
            if not filepath.exists():
                LOGGER.error(f"Файл не найден: {filepath}")
                return None
            
            # Чтение CSV с различными разделителями
            separators = [';', ',', '\t']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(
                        filepath,
                        sep=sep,
                        parse_dates=['Date'],
                        index_col='Date',
                        encoding='utf-8'
                    )
                    if len(df.columns) >= 5:  # OHLCV
                        break
                except Exception:
                    continue
            
            if df is None or df.empty:
                LOGGER.error(f"Не удалось распарсить {filepath}")
                return None
            
            # Стандартизация названий колонок
            df = self._standardize_columns(df)
            
            # Фильтрация по датам
            df = self._filter_by_date_range(df)
            
            # Валидация данных
            if self.validate_on_load:
                validation_passed, stats = self._validate_data(df, timeframe)
                self.validation_stats[timeframe] = stats
                
                if not validation_passed:
                    LOGGER.warning(f"Валидация {timeframe} выявила проблемы: {stats}")
            
            # Сортировка по индексу
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Критическая ошибка загрузки {timeframe}: {e}", exc_info=True)
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приведение названий колонок к единому стандарту"""
        try:
            column_mapping = {
                'open': 'Open', 'OPEN': 'Open',
                'high': 'High', 'HIGH': 'High',
                'low': 'Low', 'LOW': 'Low',
                'close': 'Close', 'CLOSE': 'Close',
                'volume': 'Volume', 'VOLUME': 'Volume', 'vol': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Отсутствуют колонки: {missing_columns}")
            
            # Оставляем только необходимые колонки
            df = df[required_columns]
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка стандартизации колонок: {e}")
            raise
    
    def _filter_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация данных по временному диапазону"""
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            original_length = len(df)
            df = df[(df.index >= start) & (df.index <= end)]
            filtered_length = len(df)
            
            if filtered_length < original_length:
                LOGGER.debug(f"Отфильтровано {original_length - filtered_length} баров")
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Ошибка фильтрации по датам: {e}")
            return df
    
    def _validate_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, Dict]:
        """
        Комплексная валидация данных.
        
        Returns:
            (validation_passed, statistics_dict)
        """
        stats = {
            'total_bars': len(df),
            'missing_values': 0,
            'negative_values': 0,
            'zero_volume_bars': 0,
            'ohlc_inconsistencies': 0,
            'duplicate_timestamps': 0,
            'outliers': 0,
            'gaps': 0
        }
        
        validation_passed = True
        
        try:
            # Проверка на пропуски
            missing = df.isnull().sum().sum()
            stats['missing_values'] = int(missing)
            if missing > 0:
                validation_passed = False
                LOGGER.warning(f"{timeframe}: {missing} пропущенных значений")
            
            # Проверка на отрицательные значения в ценах
            price_columns = ['Open', 'High', 'Low', 'Close']
            negative = (df[price_columns] < 0).sum().sum()
            stats['negative_values'] = int(negative)
            if negative > 0:
                validation_passed = False
                LOGGER.warning(f"{timeframe}: {negative} отрицательных значений цен")
            
            # Проверка нулевого объема
            zero_volume = (df['Volume'] == 0).sum()
            stats['zero_volume_bars'] = int(zero_volume)
            if zero_volume > len(df) * 0.1:  # Более 10%
                LOGGER.warning(f"{timeframe}: {zero_volume} баров с нулевым объемом")
            
            # Проверка консистентности OHLC
            ohlc_check = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close'])
            )
            inconsistencies = ohlc_check.sum()
            stats['ohlc_inconsistencies'] = int(inconsistencies)
            if inconsistencies > 0:
                validation_passed = False
                LOGGER.error(f"{timeframe}: {inconsistencies} нарушений OHLC логики")
            
            # Проверка дубликатов временных меток
            duplicates = df.index.duplicated().sum()
            stats['duplicate_timestamps'] = int(duplicates)
            if duplicates > 0:
                validation_passed = False
                LOGGER.warning(f"{timeframe}: {duplicates} дублирующихся временных меток")
            
            # Проверка выбросов (цены выше 3 сигм)
            returns = df['Close'].pct_change()
            outliers = (np.abs(returns) > returns.std() * 3).sum()
            stats['outliers'] = int(outliers)
            if outliers > len(df) * 0.05:  # Более 5%
                LOGGER.warning(f"{timeframe}: {outliers} потенциальных выбросов")
            
            # Проверка гэпов во времени
            time_diffs = df.index.to_series().diff()
            expected_diff = self._get_expected_timedelta(timeframe)
            gaps = (time_diffs > expected_diff * 2).sum()
            stats['gaps'] = int(gaps)
            if gaps > 0:
                LOGGER.info(f"{timeframe}: {gaps} временных гэпов обнаружено")
            
        except Exception as e:
            LOGGER.error(f"Ошибка валидации {timeframe}: {e}", exc_info=True)
            validation_passed = False
        
        return validation_passed, stats
    
    def _get_expected_timedelta(self, timeframe: str) -> timedelta:
        """Получение ожидаемой разницы между барами для таймфрейма"""
        try:
            tf_map = {
                'M1': timedelta(minutes=1),
                'M5': timedelta(minutes=5),
                '15M': timedelta(minutes=15),
                '30M': timedelta(minutes=30),
                'H1': timedelta(hours=1),
                'H4': timedelta(hours=4),
                'D1': timedelta(days=1)
            }
            return tf_map.get(timeframe, timedelta(hours=1))
        except Exception:
            return timedelta(hours=1)
    
    def synchronize_timeframes(self, 
                               base_timeframe: str = 'H1',
                               method: str = 'outer') -> Dict[str, pd.DataFrame]:
        """
        Синхронизация всех таймфреймов к единой временной сетке.
        
        Args:
            base_timeframe: Базовый таймфрейм для синхронизации
            method: Метод объединения ('inner', 'outer', 'left')
        
        Returns:
            Словарь синхронизированных DataFrame
        """
        try:
            if not self.data_cache:
                LOGGER.warning("Кэш данных пуст, загружаем...")
                self.load_all_timeframes()
            
            if base_timeframe not in self.data_cache:
                raise ValueError(f"Базовый ТФ {base_timeframe} не найден в кэше")
            
            base_index = self.data_cache[base_timeframe].index
            synchronized_data = {}
            
            for tf, df in self.data_cache.items():
                try:
                    if tf == base_timeframe:
                        synchronized_data[tf] = df.copy()
                    else:
                        # Ресамплинг к базовому таймфрейму
                        resampled = self._resample_to_base(df, base_index, method)
                        synchronized_data[tf] = resampled
                    
                    LOGGER.debug(f"Синхронизирован {tf}: {len(synchronized_data[tf])} баров")
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка синхронизации {tf}: {e}")
                    continue
            
            self.sync_index = base_index
            LOGGER.info(f"Синхронизация завершена для {len(synchronized_data)} таймфреймов")
            
            return synchronized_data
            
        except Exception as e:
            LOGGER.error(f"Критическая ошибка синхронизации: {e}", exc_info=True)
            raise
    
    def _resample_to_base(self, 
                          df: pd.DataFrame, 
                          base_index: pd.DatetimeIndex,
                          method: str) -> pd.DataFrame:
        """Ресамплинг данных к базовому индексу"""
        try:
            # Создаем пустой DataFrame с базовым индексом
            resampled = pd.DataFrame(index=base_index)
            
            # Метод заполнения: forward fill для младших ТФ
            if method == 'ffill' or len(df.index) > len(base_index):
                merged = resampled.join(df, how='left')
                merged = merged.fillna(method='ffill')
            else:
                # Для старших ТФ используем backfill
                merged = resampled.join(df, how='left')
                merged = merged.fillna(method='bfill')
            
            # Удаляем строки, где все еще есть NaN
            merged = merged.dropna()
            
            return merged
            
        except Exception as e:
            LOGGER.error(f"Ошибка ресамплинга: {e}")
            raise
    
    def get_data_splits(self, 
                        train_ratio: Optional[float] = None,
                        val_ratio: Optional[float] = None,
                        test_ratio: Optional[float] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Разделение данных на train/validation/test для каждого таймфрейма.
        
        Returns:
            {timeframe: {'train': df, 'val': df, 'test': df}}
        """
        try:
            train_ratio = train_ratio or HYPERPARAMS.data.train_ratio
            val_ratio = val_ratio or HYPERPARAMS.data.validation_ratio
            test_ratio = test_ratio or HYPERPARAMS.data.test_ratio
            
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError(f"Сумма пропорций != 1.0: {train_ratio + val_ratio + test_ratio}")
            
            splits = {}
            
            for tf, df in self.data_cache.items():
                try:
                    total_len = len(df)
                    train_end = int(total_len * train_ratio)
                    val_end = int(total_len * (train_ratio + val_ratio))
                    
                    splits[tf] = {
                        'train': df.iloc[:train_end].copy(),
                        'val': df.iloc[train_end:val_end].copy(),
                        'test': df.iloc[val_end:].copy()
                    }
                    
                    LOGGER.debug(f"{tf} split: train={len(splits[tf]['train'])}, "
                               f"val={len(splits[tf]['val'])}, test={len(splits[tf]['test'])}")
                    
                except Exception as e:
                    LOGGER.error(f"Ошибка разделения {tf}: {e}")
                    continue
            
            LOGGER.info(f"Данные разделены для {len(splits)} таймфреймов")
            return splits
            
        except Exception as e:
            LOGGER.error(f"Ошибка разделения данных: {e}", exc_info=True)
            raise
    
    def get_validation_report(self) -> pd.DataFrame:
        """Получение отчета по валидации всех таймфреймов"""
        try:
            if not self.validation_stats:
                LOGGER.warning("Статистика валидации отсутствует")
                return pd.DataFrame()
            
            report_df = pd.DataFrame(self.validation_stats).T
            return report_df
            
        except Exception as e:
            LOGGER.error(f"Ошибка создания отчета: {e}")
            return pd.DataFrame()


# Вспомогательные функции модуля
def detect_timeframe_from_data(df: pd.DataFrame) -> str:
    """Автоопределение таймфрейма по частоте данных"""
    try:
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= timedelta(minutes=1):
            return "M1"
        elif median_diff <= timedelta(minutes=5):
            return "M5"
        elif median_diff <= timedelta(minutes=15):
            return "15M"
        elif median_diff <= timedelta(minutes=30):
            return "30M"
        elif median_diff <= timedelta(hours=1):
            return "H1"
        elif median_diff <= timedelta(hours=4):
            return "H4"
        else:
            return "D1"
            
    except Exception:
        return "UNKNOWN"