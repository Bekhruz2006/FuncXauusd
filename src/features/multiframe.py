"""
Мультифреймовые признаки для торговой системы

Реализация Приоритета 2:
    - Загрузка данных с разных таймфреймов (1m - 1Month)
    - Создание признаков с высших ТФ как контекст
    - Синхронизация и выравнивание таймфреймов
    - Интеграция в основную систему признаков

Идея:
    Модель обучается на основном таймфрейме (например, H1),
    но использует контекстную информацию с высших таймфреймов:
    - Дневной тренд (D1)
    - Недельные High/Low (W1)
    - Месячная волатильность (MN)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


# Маппинг таймфреймов на pandas freq
TIMEFRAME_MAP = {
    '1m': '1T',
    '5m': '5T',
    '15m': '15T',
    '30m': '30T',
    'H1': '1H',
    '1h': '1H',
    'H4': '4H',
    '4h': '4H',
    'D1': '1D',
    '1d': '1D',
    'W1': '1W',
    '1w': '1W',
    'MN': '1M',
    '1Month': '1M'
}

# Иерархия таймфреймов (от меньшего к большему)
TIMEFRAME_HIERARCHY = [
    '1m', '5m', '15m', '30m', 'H1', 'H4', 'D1', 'W1', 'MN'
]


class MultiframeLoader:
    """
    Загрузчик данных с нескольких таймфреймов
    
    Attributes:
        data_path: Путь к директории с CSV файлами
        symbol: Название инструмента (например, 'XAUUSD')
        primary_tf: Основной таймфрейм для обучения
        context_tfs: Список высших ТФ для контекста
    """
    
    def __init__(self,
                 data_path: str,
                 symbol: str,
                 primary_tf: str = 'H1',
                 context_tfs: Optional[List[str]] = None):
        """
        Args:
            data_path: Путь к директории с данными
            symbol: Инструмент (XAUUSD, EURUSD и т.д.)
            primary_tf: Основной таймфрейм
            context_tfs: Высшие ТФ (по умолчанию D1, W1, MN)
        """
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.primary_tf = primary_tf
        
        # Контекстные ТФ по умолчанию
        if context_tfs is None:
            self.context_tfs = self._get_higher_timeframes(primary_tf)
        else:
            self.context_tfs = context_tfs
        
        print(f"📊 MultiframeLoader инициализирован:")
        print(f"  • Symbol: {symbol}")
        print(f"  • Primary TF: {primary_tf}")
        print(f"  • Context TFs: {', '.join(self.context_tfs)}")
    
    def _get_higher_timeframes(self, base_tf: str) -> List[str]:
        """
        Получение высших таймфреймов относительно базового
        
        Args:
            base_tf: Базовый таймфрейм
        
        Returns:
            list: Список высших ТФ
        """
        try:
            base_idx = TIMEFRAME_HIERARCHY.index(base_tf)
        except ValueError:
            # Если не найден, возвращаем D1, W1, MN
            return ['D1', 'W1', 'MN']
        
        # Берем все выше базового
        higher = TIMEFRAME_HIERARCHY[base_idx + 1:]
        
        # Ограничиваем разумным количеством (2-3 ТФ)
        return higher[:3] if higher else []
    
    def load_timeframe(self, timeframe: str) -> pd.DataFrame:
        """
        Загрузка данных одного таймфрейма
        
        Args:
            timeframe: Таймфрейм для загрузки
        
        Returns:
            pd.DataFrame: Данные с индексом datetime
        
        Raises:
            FileNotFoundError: Если файл не найден
        """
        # Поиск файла (разные варианты именования)
        possible_names = [
            f"{self.symbol}_{timeframe}.csv",
            f"{self.symbol.lower()}_{timeframe}.csv",
            f"{self.symbol}_{timeframe.lower()}.csv",
            f"{self.symbol.upper()}_{timeframe.upper()}.csv",
        ]
        
        filepath = None
        for name in possible_names:
            candidate = self.data_path / name
            if candidate.exists():
                filepath = candidate
                break
        
        if filepath is None:
            raise FileNotFoundError(
                f"Не найден файл для {timeframe}. "
                f"Искал: {possible_names}"
            )
        
        # Загрузка (используем универсальный парсер)
        df = self._parse_csv(filepath)
        
        print(f"  ✓ {timeframe}: {len(df)} баров "
              f"({df.index[0]} - {df.index[-1]})")
        
        return df
    
    def _parse_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Универсальный парсер MT5 CSV
        
        Поддерживает:
            - Разделитель ';'
            - Разделитель пробел
            - Формат Date;Open;High;Low;Close;Volume
        """
        # Попытка 1: разделитель ';'
        try:
            df = pd.read_csv(filepath, sep=';', parse_dates=['Date'])
            if 'Date' in df.columns and 'Close' in df.columns:
                result = pd.DataFrame()
                result['time'] = pd.to_datetime(df['Date'])
                result['open'] = df['Open'].astype(float)
                result['high'] = df['High'].astype(float)
                result['low'] = df['Low'].astype(float)
                result['close'] = df['Close'].astype(float)
                result['volume'] = df['Volume'].astype(float)
                result.set_index('time', inplace=True)
                return result.dropna()
        except:
            pass
        
        # Попытка 2: разделитель пробел
        try:
            df = pd.read_csv(filepath, sep=r'\s+')
            if '<DATE>' in df.columns:
                result = pd.DataFrame()
                result['time'] = df['<DATE>'] + ' ' + df['<TIME>']
                result['time'] = pd.to_datetime(result['time'], format='mixed')
                result['open'] = df['<OPEN>'].astype(float)
                result['high'] = df['<HIGH>'].astype(float)
                result['low'] = df['<LOW>'].astype(float)
                result['close'] = df['<CLOSE>'].astype(float)
                result['volume'] = df['<TICKVOL>'].astype(float)
                result.set_index('time', inplace=True)
                return result.dropna()
        except:
            pass
        
        raise ValueError(f"Не удалось распарсить {filepath.name}")
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Загрузка всех таймфреймов
        
        Returns:
            dict: {timeframe: DataFrame}
        """
        data = {}
        
        print(f"\n📥 Загрузка таймфреймов:")
        
        # Основной ТФ
        data[self.primary_tf] = self.load_timeframe(self.primary_tf)
        
        # Контекстные ТФ
        for tf in self.context_tfs:
            try:
                data[tf] = self.load_timeframe(tf)
            except FileNotFoundError as e:
                print(f"  ⚠️ Пропущен {tf}: {e}")
                continue
        
        return data
    
    def resample_to_primary(self,
                           higher_tf_data: pd.DataFrame,
                           primary_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ресемплинг высшего ТФ на основной
        
        Args:
            higher_tf_data: Данные высшего ТФ
            primary_data: Данные основного ТФ
        
        Returns:
            pd.DataFrame: Выровненные данные
        """
        # Forward-fill для заполнения пропусков
        aligned = higher_tf_data.reindex(
            primary_data.index,
            method='ffill'
        )
        
        return aligned


class MultiframeFeatureBuilder:
    """
    Построитель мультифреймовых признаков
    
    Создает признаки на основе нескольких таймфреймов:
        - Позиция цены относительно дневных High/Low
        - Недельный тренд
        - Месячная волатильность
        - Межфреймовая дивергенция
    """
    
    def __init__(self,
                 loader: MultiframeLoader,
                 feature_config: Optional[Dict] = None):
        """
        Args:
            loader: Загрузчик данных
            feature_config: Конфигурация признаков
        """
        self.loader = loader
        self.config = feature_config or self._default_config()
        self.data: Dict[str, pd.DataFrame] = {}
    
    def _default_config(self) -> Dict:
        """Конфигурация признаков по умолчанию"""
        return {
            'use_price_position': True,    # Позиция в High/Low range
            'use_trend_direction': True,   # Направление тренда
            'use_volatility_ratio': True,  # Отношение волатильностей
            'use_divergence': True,        # Межфреймовая дивергенция
            'use_ma_distance': True        # Расстояние до скользящих средних
        }
    
    def load_data(self) -> None:
        """Загрузка всех необходимых таймфреймов"""
        self.data = self.loader.load_all()
    
    def build_features(self) -> pd.DataFrame:
        """
        Построение полного набора мультифреймовых признаков
        
        Returns:
            pd.DataFrame: Основной ТФ с добавленными признаками
        """
        if not self.data:
            self.load_data()
        
        # Основной датафрейм
        primary_data = self.data[self.loader.primary_tf].copy()
        result = primary_data[['close']].copy()
        
        print(f"\n🔧 Построение мультифреймовых признаков:")
        
        # Для каждого высшего ТФ
        for tf in self.loader.context_tfs:
            if tf not in self.data:
                continue
            
            higher_data = self.data[tf]
            
            # Выравнивание на основной ТФ
            aligned = self.loader.resample_to_primary(
                higher_data,
                primary_data
            )
            
            # Создание признаков
            features = self._create_context_features(
                primary_data,
                aligned,
                tf
            )
            
            # Добавление к результату
            for col in features.columns:
                result[col] = features[col]
            
            print(f"  ✓ {tf}: {len(features.columns)} признаков")
        
        # Удаление NaN
        initial_len = len(result)
        result = result.dropna()
        dropped = initial_len - len(result)
        
        if dropped > 0:
            print(f"  ℹ️ Удалено {dropped} NaN строк")
        
        print(f"\n  📊 Итого: {len(result.columns) - 1} мультифреймовых признаков")
        
        return result
    
    def _create_context_features(self,
                                primary: pd.DataFrame,
                                context: pd.DataFrame,
                                tf_name: str) -> pd.DataFrame:
        """
        Создание признаков для одного высшего ТФ
        
        Args:
            primary: Основной ТФ
            context: Выровненный высший ТФ
            tf_name: Название ТФ (для имен колонок)
        
        Returns:
            pd.DataFrame: Признаки
        """
        features = pd.DataFrame(index=primary.index)
        
        # 1. Позиция цены в High/Low range
        if self.config['use_price_position'] and 'high' in context.columns:
            high = context['high']
            low = context['low']
            close = primary['close']
            
            range_size = high - low
            price_position = (close - low) / range_size.replace(0, 1)
            
            features[f'price_pos_{tf_name}'] = price_position
        
        # 2. Направление тренда (через EMA)
        if self.config['use_trend_direction']:
            close = context['close']
            
            # Быстрая и медленная EMA
            ema_fast = close.ewm(span=10, adjust=False).mean()
            ema_slow = close.ewm(span=30, adjust=False).mean()
            
            # Выравнивание на основной ТФ
            ema_fast_aligned = ema_fast.reindex(primary.index, method='ffill')
            ema_slow_aligned = ema_slow.reindex(primary.index, method='ffill')
            
            trend = (ema_fast_aligned - ema_slow_aligned) / ema_slow_aligned
            features[f'trend_{tf_name}'] = trend
        
        # 3. Отношение волатильностей
        if self.config['use_volatility_ratio']:
            # Волатильность высшего ТФ
            context_vol = context['close'].pct_change().rolling(14).std()
            context_vol_aligned = context_vol.reindex(
                primary.index,
                method='ffill'
            )
            
            # Волатильность основного ТФ
            primary_vol = primary['close'].pct_change().rolling(14).std()
            
            # Отношение
            vol_ratio = primary_vol / context_vol_aligned.replace(0, 1)
            features[f'vol_ratio_{tf_name}'] = vol_ratio
        
        # 4. Дивергенция (разница направлений)
        if self.config['use_divergence']:
            # ROC (Rate of Change) на обоих ТФ
            primary_roc = primary['close'].pct_change(5)
            context_roc = context['close'].pct_change(5)
            context_roc_aligned = context_roc.reindex(
                primary.index,
                method='ffill'
            )
            
            # Дивергенция = знак(primary_roc) != знак(context_roc)
            divergence = np.sign(primary_roc) * np.sign(context_roc_aligned)
            features[f'divergence_{tf_name}'] = divergence
        
        # 5. Расстояние до MA высшего ТФ
        if self.config['use_ma_distance']:
            ma = context['close'].rolling(20).mean()
            ma_aligned = ma.reindex(primary.index, method='ffill')
            
            distance = (primary['close'] - ma_aligned) / ma_aligned
            features[f'ma_dist_{tf_name}'] = distance
        
        return features


# === ИНТЕГРАЦИЯ С ОСНОВНОЙ СИСТЕМОЙ ===

def create_multiframe_features(
    data_path: str,
    symbol: str,
    primary_tf: str = 'H1',
    context_tfs: Optional[List[str]] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Удобная функция для создания мультифреймовых признаков
    
    Args:
        data_path: Путь к данным
        symbol: Инструмент
        primary_tf: Основной таймфрейм
        context_tfs: Высшие ТФ
        config: Конфигурация признаков
    
    Returns:
        pd.DataFrame: Данные с мультифреймовыми признаками
    
    Example:
        >>> df = create_multiframe_features(
        >>>     data_path='./data/raw',
        >>>     symbol='XAUUSD',
        >>>     primary_tf='H1',
        >>>     context_tfs=['D1', 'W1']
        >>> )
        >>> # Теперь df содержит признаки с D1 и W1
    """
    # Загрузчик
    loader = MultiframeLoader(
        data_path,
        symbol,
        primary_tf,
        context_tfs
    )
    
    # Построитель признаков
    builder = MultiframeFeatureBuilder(loader, config)
    
    # Создание признаков
    features = builder.build_features()
    
    return features


def add_multiframe_to_existing(
    primary_data: pd.DataFrame,
    data_path: str,
    symbol: str,
    primary_tf: str,
    context_tfs: List[str]
) -> pd.DataFrame:
    """
    Добавление мультифреймовых признаков к существующим данным
    
    Args:
        primary_data: Основные данные с признаками
        data_path: Путь к данным высших ТФ
        symbol: Инструмент
        primary_tf: Основной ТФ
        context_tfs: Высшие ТФ
    
    Returns:
        pd.DataFrame: Данные с добавленными мультифреймовыми признаками
    """
    # Создание мультифреймовых признаков
    multiframe = create_multiframe_features(
        data_path,
        symbol,
        primary_tf,
        context_tfs
    )
    
    # Объединение с основными данными
    result = primary_data.copy()
    
    # Добавление только новых колонок
    for col in multiframe.columns:
        if col not in result.columns:
            result[col] = multiframe[col]
    
    # Синхронизация по индексу
    result = result.loc[result.index.isin(multiframe.index)]
    
    return result.dropna()


# === ВАЛИДАЦИЯ ===

def validate_multiframe_data(data: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
    """
    Валидация мультифреймовых данных
    
    Проверки:
        - Наличие основного ТФ
        - Достаточность данных
        - Корректность временных диапазонов
    
    Args:
        data: Словарь с данными разных ТФ
    
    Returns:
        (valid, errors): Валидность и список ошибок
    """
    errors = []
    
    if not data:
        errors.append("Нет данных")
        return False, errors
    
    # Проверка размеров
    for tf, df in data.items():
        if len(df) < 1000:
            errors.append(f"{tf}: недостаточно данных ({len(df)} < 1000)")
    
    # Проверка временных диапазонов
    date_ranges = {
        tf: (df.index.min(), df.index.max())
        for tf, df in data.items()
    }
    
    # Все ТФ должны иметь пересекающиеся диапазоны
    all_starts = [start for start, _ in date_ranges.values()]
    all_ends = [end for _, end in date_ranges.values()]
    
    latest_start = max(all_starts)
    earliest_end = min(all_ends)
    
    if latest_start >= earliest_end:
        errors.append(
            f"Временные диапазоны не пересекаются: "
            f"{latest_start} >= {earliest_end}"
        )
    
    return len(errors) == 0, errors