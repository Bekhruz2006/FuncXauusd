"""
Вычисление макро-признаков: корреляции с индексами, календарные эффекты,
глобальные рыночные условия.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import pearsonr, spearmanr
import requests
from io import StringIO

from utils.logger import LOGGER


class MacroFeatureComputer:
    """Вычислитель макро-признаков для рынка"""
    
    def __init__(self,
                 correlation_window: int = 30,
                 cache_external_data: bool = True):
        
        self.correlation_window = correlation_window
        self.cache_external_data = cache_external_data
        
        self.external_data_cache: Dict[str, pd.DataFrame] = {}
        self.computed_features: Dict[str, pd.DataFrame] = {}
        
        LOGGER.info(f"Инициализация вычислителя макро-признаков: window={correlation_window}")
    
    def compute_all_macro_features(self, 
                                    df: pd.DataFrame,
                                    include_calendar: bool = True,
                                    include_correlations: bool = True,
                                    include_global_conditions: bool = True) -> pd.DataFrame:
        """
        Вычисление всех макро-признаков.
        
        Args:
            df: DataFrame с OHLCV данными
            include_calendar: Календарные эффекты
            include_correlations: Корреляции с индексами
            include_global_conditions: Глобальные условия
        
        Returns:
            DataFrame с макро-признаками
        """
        try:
            macro_features = pd.DataFrame(index=df.index)
            
            if include_calendar:
                calendar_features = self.compute_calendar_features(df)
                macro_features = macro_features.join(calendar_features)
            
            if include_correlations:
                correlation_features = self.compute_correlation_features(df)
                macro_features = macro_features.join(correlation_features)
            
            if include_global_conditions:
                global_features = self.compute_global_market_conditions(df)
                macro_features = macro_features.join(global_features)
            
            # Интеракции между признаками
            interaction_features = self.compute_feature_interactions(macro_features)
            macro_features = macro_features.join(interaction_features)
            
            LOGGER.info(f"Вычислено {len(macro_features.columns)} макро-признаков")
            return macro_features
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления макро-признаков: {e}", exc_info=True)
            return pd.DataFrame(index=df.index)
    
    def compute_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Календарные эффекты и сезонность.
        
        Returns:
            DataFrame с календарными признаками
        """
        try:
            features = pd.DataFrame(index=df.index)
            
            # Временные признаки
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['day_of_month'] = df.index.day
            features['week_of_year'] = df.index.isocalendar().week
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
            
            # Циклические преобразования для непрерывности
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Торговые сессии
            features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
            features['is_european_session'] = ((features['hour'] >= 7) & (features['hour'] < 16)).astype(int)
            features['is_american_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
            features['is_overlap_session'] = ((features['hour'] >= 13) & (features['hour'] < 16)).astype(int)
            
            # Выходные и праздники
            features['is_weekend'] = (features['day_of_week'].isin([5, 6])).astype(int)
            features['is_month_start'] = (features['day_of_month'] <= 5).astype(int)
            features['is_month_end'] = (features['day_of_month'] >= 25).astype(int)
            
            # День недели эффекты (one-hot encoding)
            for dow in range(7):
                features[f'dow_{dow}'] = (features['day_of_week'] == dow).astype(int)
            
            # Месяц эффекты
            for month in range(1, 13):
                features[f'month_{month}'] = (features['month'] == month).astype(int)
            
            # Относительная позиция в месяце
            days_in_month = df.index.to_series().dt.days_in_month
            features['month_progress'] = features['day_of_month'] / days_in_month
            
            LOGGER.debug(f"Вычислено {len(features.columns)} календарных признаков")
            return features
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления календарных признаков: {e}")
            return pd.DataFrame(index=df.index)
    
    def compute_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Корреляции с внешними индексами (доллар, облигации).
        
        Returns:
            DataFrame с корреляционными признаками
        """
        try:
            features = pd.DataFrame(index=df.index)
            
            # Симуляция внешних индексов (в реальности загрузка через API)
            external_indices = self._simulate_external_indices(df)
            
            for index_name, index_data in external_indices.items():
                try:
                    # Синхронизация по времени
                    aligned_index = index_data.reindex(df.index, method='ffill')
                    
                    if aligned_index.isna().all():
                        continue
                    
                    # Корреляция доходностей
                    gold_returns = df['Close'].pct_change()
                    index_returns = aligned_index.pct_change()
                    
                    rolling_corr = gold_returns.rolling(window=self.correlation_window).corr(index_returns)
                    features[f'corr_{index_name}'] = rolling_corr
                    
                    # Лаговые корреляции
                    for lag in [1, 3, 5]:
                        lagged_corr = gold_returns.rolling(window=self.correlation_window).corr(
                            index_returns.shift(lag)
                        )
                        features[f'corr_{index_name}_lag{lag}'] = lagged_corr
                    
                    # Относительная сила
                    features[f'relative_strength_{index_name}'] = (
                        df['Close'] / aligned_index
                    ).pct_change(periods=self.correlation_window)
                    
                except Exception as e:
                    LOGGER.warning(f"Ошибка вычисления корреляции с {index_name}: {e}")
                    continue
            
            features = features.fillna(0)
            
            LOGGER.debug(f"Вычислено {len(features.columns)} корреляционных признаков")
            return features
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления корреляций: {e}")
            return pd.DataFrame(index=df.index)
    
    def _simulate_external_indices(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Симуляция внешних индексов (DXY, TLT).
        В production заменить на реальную загрузку данных.
        """
        try:
            indices = {}
            
            # Симуляция индекса доллара (DXY) - обратная корреляция с золотом
            np.random.seed(42)
            gold_returns = df['Close'].pct_change()
            
            # DXY движется противоположно золоту с шумом
            dxy_returns = -0.7 * gold_returns + np.random.randn(len(df)) * 0.005
            dxy_price = 100 * (1 + dxy_returns).cumprod()
            indices['dxy'] = pd.Series(dxy_price.values, index=df.index)
            
            # Симуляция облигаций (TLT) - положительная корреляция
            tlt_returns = 0.5 * gold_returns + np.random.randn(len(df)) * 0.003
            tlt_price = 100 * (1 + tlt_returns).cumprod()
            indices['tlt'] = pd.Series(tlt_price.values, index=df.index)
            
            # Симуляция S&P500 - слабая корреляция
            sp500_returns = 0.2 * gold_returns + np.random.randn(len(df)) * 0.01
            sp500_price = 3000 * (1 + sp500_returns).cumprod()
            indices['sp500'] = pd.Series(sp500_price.values, index=df.index)
            
            return indices
            
        except Exception as e:
            LOGGER.error(f"Ошибка симуляции индексов: {e}")
            return {}
    
    def compute_global_market_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Глобальные рыночные условия и режимы.
        
        Returns:
            DataFrame с глобальными признаками
        """
        try:
            features = pd.DataFrame(index=df.index)
            
            # Рыночная волатильность (VIX proxy)
            returns = df['Close'].pct_change()
            features['market_volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Изменение волатильности
            features['volatility_change'] = features['market_volatility'].pct_change(periods=5)
            
            # Risk-on / Risk-off индикатор
            # Комбинация объема и волатильности
            volume_normalized = (df['Volume'] - df['Volume'].rolling(50).mean()) / df['Volume'].rolling(50).std()
            vol_normalized = (features['market_volatility'] - features['market_volatility'].rolling(50).mean()) / features['market_volatility'].rolling(50).std()
            
            features['risk_appetite'] = volume_normalized - vol_normalized
            
            # Тренд глобальной ликвидности (proxy через объем)
            features['liquidity_trend'] = df['Volume'].rolling(window=30).mean().pct_change(periods=30)
            
            # Momentum рынка
            for period in [5, 10, 20]:
                features[f'market_momentum_{period}'] = df['Close'].pct_change(periods=period)
            
            # Режим волатильности (low/medium/high)
            vol_quantiles = features['market_volatility'].rolling(window=100).quantile(0.33)
            vol_quantiles_high = features['market_volatility'].rolling(window=100).quantile(0.66)
            
            features['vol_regime'] = 1  # medium
            features.loc[features['market_volatility'] < vol_quantiles, 'vol_regime'] = 0  # low
            features.loc[features['market_volatility'] > vol_quantiles_high, 'vol_regime'] = 2  # high
            
            # Макро тренд (долгосрочный)
            for window in [50, 100, 200]:
                sma = df['Close'].rolling(window=window).mean()
                features[f'macro_trend_{window}'] = (df['Close'] - sma) / sma
            
            # Индикатор кризиса (экстремальные движения)
            extreme_moves = (returns.abs() > returns.rolling(100).std() * 3).astype(int)
            features['crisis_indicator'] = extreme_moves.rolling(window=10).sum()
            
            features = features.fillna(0)
            
            LOGGER.debug(f"Вычислено {len(features.columns)} глобальных признаков")
            return features
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления глобальных условий: {e}")
            return pd.DataFrame(index=df.index)
    
    def compute_feature_interactions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисление интеракций между признаками.
        
        Args:
            features_df: DataFrame с уже вычисленными признаками
        
        Returns:
            DataFrame с интеракционными признаками
        """
        try:
            interactions = pd.DataFrame(index=features_df.index)
            
            # Список ключевых признаков для интеракций
            key_features = []
            
            for col in features_df.columns:
                if any(x in col for x in ['corr_', 'market_volatility', 'risk_appetite', 'momentum']):
                    key_features.append(col)
            
            # Попарные произведения (ограничиваем количество)
            interaction_count = 0
            max_interactions = 20
            
            for i, feat1 in enumerate(key_features[:10]):
                for feat2 in key_features[i+1:10]:
                    if interaction_count >= max_interactions:
                        break
                    
                    try:
                        interactions[f'interact_{feat1}_{feat2}'] = (
                            features_df[feat1] * features_df[feat2]
                        )
                        interaction_count += 1
                    except Exception:
                        continue
            
            # Квадратичные термы для важных признаков
            for feat in ['market_volatility', 'risk_appetite']:
                if feat in features_df.columns:
                    interactions[f'{feat}_squared'] = features_df[feat] ** 2
            
            interactions = interactions.fillna(0)
            
            LOGGER.debug(f"Вычислено {len(interactions.columns)} интеракционных признаков")
            return interactions
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления интеракций: {e}")
            return pd.DataFrame(index=features_df.index)
    
    def compute_economic_calendar_features(self, 
                                          df: pd.DataFrame,
                                          events: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Признаки на основе экономического календаря.
        
        Args:
            df: DataFrame с данными
            events: Список событий [{date, importance, currency}, ...]
        
        Returns:
            DataFrame с признаками календаря
        """
        try:
            features = pd.DataFrame(index=df.index)
            
            if events is None:
                # Симуляция важных событий
                events = self._simulate_economic_events(df)
            
            # Расстояние до следующего события
            features['days_to_next_event'] = 999
            features['event_importance'] = 0
            
            for event in events:
                try:
                    event_date = pd.to_datetime(event['date'])
                    importance = event.get('importance', 1)
                    
                    # Обновляем для баров до события
                    mask = df.index <= event_date
                    days_diff = (event_date - df.index[mask]).days
                    
                    # Обновляем если это ближайшее событие
                    update_mask = days_diff < features.loc[mask, 'days_to_next_event']
                    features.loc[mask[update_mask], 'days_to_next_event'] = days_diff[update_mask]
                    features.loc[mask[update_mask], 'event_importance'] = importance
                    
                except Exception as e:
                    LOGGER.warning(f"Ошибка обработки события: {e}")
                    continue
            
            # Индикатор близости события
            features['event_proximity'] = np.exp(-features['days_to_next_event'] / 7)
            features['weighted_event_impact'] = features['event_proximity'] * features['event_importance']
            
            LOGGER.debug(f"Вычислено {len(features.columns)} признаков экономического календаря")
            return features
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления признаков календаря: {e}")
            return pd.DataFrame(index=df.index)
    
    def _simulate_economic_events(self, df: pd.DataFrame) -> List[Dict]:
        """Симуляция экономических событий"""
        try:
            events = []
            
            # Генерируем событие раз в месяц
            date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='MS')
            
            for date in date_range:
                # FOMC, NFP и другие важные события
                events.append({
                    'date': date + timedelta(days=np.random.randint(1, 28)),
                    'importance': np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]),
                    'currency': 'USD'
                })
            
            return events
            
        except Exception as e:
            LOGGER.error(f"Ошибка симуляции событий: {e}")
            return []
    
    def get_feature_importance_proxy(self, features_df: pd.DataFrame, target: pd.Series) -> pd.Series:
        """
        Приблизительная оценка важности признаков через корреляцию.
        
        Args:
            features_df: DataFrame с признаками
            target: Целевая переменная (например, будущая доходность)
        
        Returns:
            Series с важностью каждого признака
        """
        try:
            importance_scores = {}
            
            for col in features_df.columns:
                try:
                    # Пропускаем константные признаки
                    if features_df[col].std() < 1e-8:
                        importance_scores[col] = 0.0
                        continue
                    
                    # Корреляция Спирмена (устойчива к выбросам)
                    valid_mask = ~(features_df[col].isna() | target.isna())
                    
                    if valid_mask.sum() < 10:
                        importance_scores[col] = 0.0
                        continue
                    
                    corr, pvalue = spearmanr(
                        features_df.loc[valid_mask, col],
                        target[valid_mask]
                    )
                    
                    # Взвешиваем по значимости
                    importance_scores[col] = abs(corr) * (1 - pvalue)
                    
                except Exception as e:
                    LOGGER.debug(f"Ошибка оценки важности {col}: {e}")
                    importance_scores[col] = 0.0
            
            importance_series = pd.Series(importance_scores).sort_values(ascending=False)
            
            LOGGER.info(f"Топ-5 важных признаков: {importance_series.head().to_dict()}")
            return importance_series
            
        except Exception as e:
            LOGGER.error(f"Ошибка вычисления важности признаков: {e}")
            return pd.Series()


def fetch_external_data_api(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Загрузка внешних данных через API (заглушка для расширения).
    
    Args:
        symbol: Тикер инструмента (DXY, TLT, etc.)
        start_date: Начальная дата
        end_date: Конечная дата
    
    Returns:
        DataFrame с данными или None
    """
    try:
        # Заглушка для реальной реализации API
        LOGGER.warning(f"API загрузка не реализована для {symbol}. Используется симуляция.")
        return None
        
    except Exception as e:
        LOGGER.error(f"Ошибка загрузки данных для {symbol}: {e}")
        return None