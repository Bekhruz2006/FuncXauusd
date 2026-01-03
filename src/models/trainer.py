"""
Обучение кластерной торговой системы

Архитектура:
    1. Кластеризация данных по мета-признакам (skewness)
    2. Обучение отдельной модели для каждого кластера:
        - Main Model: торговые сигналы (std-признаки)
        - Meta Model: фильтр кластера (skewness-признаки)
    3. Валидация и отбор лучшей модели
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier

from src.data.loader import load_price_data
from src.features.engineering import create_features, get_feature_columns
from src.labeling.strategies import get_labels_one_direction
from src.models.validator import (
    validate_class_balance,
    validate_sample_size,
    validate_cluster_sizes
)
from src.backtesting.tester import test_model_one_direction


class ClusterModelTrainer:
    """
    Тренер кластерной системы
    
    Workflow:
        1. Загрузка данных
        2. Создание признаков (std + skewness)
        3. Разметка данных
        4. Кластеризация по meta-признакам
        5. Обучение модели для каждого кластера
        6. Валидация и отбор лучшей
    
    Attributes:
        config: Конфигурация обучения
        data: Исторические данные
        clusters: Метки кластеров
        models: Обученные модели по кластерам
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Конфигурация с параметрами обучения
        """
        self.config = config
        self.data = None
        self.clusters = None
        self.models = {}
        
        print(f"\n{'='*70}")
        print(f"  🎯 CLUSTER MODEL TRAINER")
        print(f"{'='*70}")
        print(f"  Symbol: {config['symbol']['name']}")
        print(f"  Direction: {config['trading']['direction'].upper()}")
        print(f"  N Clusters: {config.get('n_clusters', config['clustering']['n_clusters'])}")
        print(f"{'='*70}\n")
    
    def train_all_clusters(self) -> List[Dict]:
        """
        Обучение моделей для всех кластеров
        
        Returns:
            list: Список результатов для каждого кластера
            [
                {
                    'cluster': 0,
                    'model': CatBoostClassifier,
                    'meta_model': CatBoostClassifier,
                    'val_acc': 0.78,
                    'r2': 0.92,
                    'samples': 1200,
                    'balance': 0.45,
                    'dataset': DataFrame  # для тестирования
                },
                ...
            ]
        """
        # 1. Подготовка данных
        print("📊 Подготовка данных...")
        self._prepare_data()
        
        # 2. Кластеризация
        print(f"\n🔬 Кластеризация...")
        self._perform_clustering()
        
        # 3. Обучение для каждого кластера
        results = []
        n_clusters = len(np.unique(self.clusters))
        
        print(f"\n🎓 Обучение {n_clusters} моделей...")
        
        for cluster_id in range(n_clusters):
            print(f"\n  Кластер {cluster_id}:")
            
            try:
                result = self._train_single_cluster(cluster_id)
                
                if result is not None:
                    results.append(result)
                    print(f"    ✓ Val Acc: {result['val_acc']:.4f} | "
                          f"R²: {result['r2']:.4f} | "
                          f"Samples: {result['samples']}")
                else:
                    print(f"    ✗ Пропущен")
                    
            except Exception as e:
                print(f"    ⚠️ Ошибка: {e}")
                continue
        
        print(f"\n{'─'*70}")
        print(f"  ✅ Обучено моделей: {len(results)}/{n_clusters}")
        print(f"{'─'*70}\n")
        
        return results
    
    def _prepare_data(self) -> None:
        """Загрузка, создание признаков и разметка"""
        # Загрузка цен
        prices = load_price_data(self.config)
        
        # Создание признаков
        periods = self.config['periods']
        meta_periods = self.config['periods_meta']
        
        features = create_features(prices, periods, meta_periods)
        
        # Разметка
        labeled = get_labels_one_direction(
            features,
            markup=self.config['markup'],
            min_bars=self.config['trading']['labeling']['min_bars'],
            max_bars=self.config['trading']['labeling']['max_bars'],
            direction=self.config['trading']['direction']
        )
        
        self.data = labeled
        
        # Валидация
        valid, errors = self._validate_data()
        if not valid:
            raise ValueError(f"Данные не прошли валидацию: {errors}")
    
    def _validate_data(self) -> Tuple[bool, List[str]]:
        """Валидация подготовленных данных"""
        errors = []
        
        # Размер
        min_samples = self.config.get('min_samples', 1000)
        valid, msg = validate_sample_size(self.data, min_samples)
        if not valid:
            errors.append(msg)
        
        # Баланс классов
        min_balance = self.config['validation']['criteria']['min_class_balance']
        valid, balance, msg = validate_class_balance(
            self.data['labels'],
            min_balance
        )
        if not valid:
            errors.append(msg)
        
        return len(errors) == 0, errors
    
    def _perform_clustering(self) -> None:
        """
        Кластеризация данных по мета-признакам
        
        Алгоритм:
            1. Извлечение meta-признаков (skewness)
            2. Нормализация через StandardScaler
            3. KMeans кластеризация
            4. Валидация качества кластеров
        """
        # Извлечение meta-признаков
        meta_cols = get_feature_columns(self.data, 'meta_')
        
        if len(meta_cols) == 0:
            raise ValueError("Нет мета-признаков для кластеризации")
        
        meta_features = self.data[meta_cols].values
        
        # Нормализация
        scaler = StandardScaler()
        meta_scaled = scaler.fit_transform(meta_features)
        
        # KMeans
        n_clusters = self.config.get('n_clusters', 
                                    self.config['clustering']['n_clusters'])
        random_state = self.config['clustering']['random_state']
        n_init = self.config['clustering']['n_init']
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init
        )
        
        self.clusters = kmeans.fit_predict(meta_scaled)
        
        # Валидация кластеров
        min_cluster_size = self.config.get('min_samples', 100)
        valid, sizes, msg = validate_cluster_sizes(
            self.clusters,
            min_cluster_size
        )
        
        print(f"  Кластеров: {n_clusters}")
        print(f"  Размеры: {sizes}")
        
        if not valid:
            print(f"  ⚠️ Предупреждение: {msg}")
    
    def _train_single_cluster(self, cluster_id: int) -> Optional[Dict]:
        """
        Обучение модели для одного кластера
        
        Args:
            cluster_id: ID кластера
        
        Returns:
            dict: Результаты обучения или None при ошибке
        """
        # Отбор данных кластера
        cluster_mask = self.clusters == cluster_id
        cluster_data = self.data[cluster_mask].copy()
        
        # Проверка размера
        min_samples = self.config.get('min_samples', 100)
        if len(cluster_data) < min_samples:
            print(f"    Мало данных: {len(cluster_data)} < {min_samples}")
            return None
        
        # Проверка баланса
        min_balance = self.config['validation']['criteria']['min_class_balance']
        valid, balance, msg = validate_class_balance(
            cluster_data['labels'],
            min_balance
        )
        
        if not valid:
            print(f"    Дисбаланс: {balance:.3f} < {min_balance}")
            return None
        
        # Разделение на train/test
        train_data, test_data = self._split_data(cluster_data)
        
        # Обучение Main Model (торговые сигналы)
        main_model = self._train_main_model(train_data, test_data)
        
        # Обучение Meta Model (фильтр кластера)
        meta_model = self._train_meta_model(train_data, test_data)
        
        # Валидация на тестовых данных
        val_acc = main_model.score(
            test_data[get_feature_columns(test_data, 'feat_')],
            test_data['labels']
        )
        
        # Подготовка датасета для R² теста
        test_dataset = self._prepare_test_dataset(
            cluster_data,
            main_model,
            meta_model
        )
        
        # Расчет R² (качество торговой стратегии)
        r2 = test_model_one_direction(
            dataset=test_dataset,
            result=[main_model, meta_model],
            config=self.config,
            plt=False
        )
        
        return {
            'cluster': cluster_id,
            'model': main_model,
            'meta_model': meta_model,
            'val_acc': val_acc,
            'r2': r2,
            'samples': len(cluster_data),
            'balance': balance,
            'dataset': test_dataset
        }
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Разделение на train/test с сохранением временного порядка"""
        train_size = self.config['validation']['train_size']
        shuffle = self.config['validation']['shuffle']
        stratify = self.config['validation']['stratify']
        random_state = self.config['validation']['random_state']
        
        if shuffle and stratify:
            # Стратифицированное разделение
            train_data, test_data = train_test_split(
                data,
                train_size=train_size,
                shuffle=True,
                stratify=data['labels'],
                random_state=random_state
            )
        else:
            # Временное разделение
            split_idx = int(len(data) * train_size)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
        
        return train_data, test_data
    
    def _train_main_model(self,
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame) -> CatBoostClassifier:
        """
        Обучение основной модели (торговые сигналы)
        
        Использует std-признаки для предсказания направления
        """
        # Признаки и метки
        feat_cols = get_feature_columns(train_data, 'feat_')
        X_train = train_data[feat_cols]
        y_train = train_data['labels'].astype('int16')
        X_test = test_data[feat_cols]
        y_test = test_data['labels'].astype('int16')
        
        # Параметры модели
        model_params = self.config['model']['main']['params'].copy()
        
        # Переопределение из конфига поиска
        if 'iterations' in self.config:
            model_params['iterations'] = self.config['iterations']
        if 'depth' in self.config:
            model_params['depth'] = self.config['depth']
        
        # Обучение
        model = CatBoostClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=model_params.get('early_stopping_rounds', 50),
            plot=False
        )
        
        return model
    
    def _train_meta_model(self,
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame) -> CatBoostClassifier:
        """
        Обучение мета-модели (фильтр кластера)
        
        Использует skewness-признаки для определения режима рынка
        """
        # Мета-признаки
        meta_cols = get_feature_columns(train_data, 'meta_')
        
        # Если мета-признаков нет, используем все признаки
        if len(meta_cols) == 0:
            meta_cols = get_feature_columns(train_data, 'feat_')
        
        X_train = train_data[meta_cols]
        y_train = train_data['labels'].astype('int16')
        X_test = test_data[meta_cols]
        y_test = test_data['labels'].astype('int16')
        
        # Параметры мета-модели (обычно проще чем main)
        meta_params = self.config['model']['meta']['params'].copy()
        
        # Обучение
        meta_model = CatBoostClassifier(**meta_params)
        meta_model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=meta_params.get('early_stopping_rounds', 30),
            plot=False
        )
        
        return meta_model
    
    def _prepare_test_dataset(self,
                             data: pd.DataFrame,
                             main_model: CatBoostClassifier,
                             meta_model: CatBoostClassifier) -> pd.DataFrame:
        """
        Подготовка датасета для R² теста
        
        Добавляет предсказания моделей как labels и meta_labels
        """
        dataset = data.copy()
        
        # Признаки
        feat_cols = get_feature_columns(dataset, 'feat_')
        meta_cols = get_feature_columns(dataset, 'meta_')
        
        if len(meta_cols) == 0:
            meta_cols = feat_cols
        
        # Предсказания
        dataset['labels'] = main_model.predict_proba(dataset[feat_cols])[:, 1]
        dataset['meta_labels'] = meta_model.predict_proba(dataset[meta_cols])[:, 1]
        
        # Бинаризация (порог 0.5)
        dataset['labels'] = dataset['labels'].apply(lambda x: 1.0 if x >= 0.5 else 0.0)
        dataset['meta_labels'] = dataset['meta_labels'].apply(lambda x: 1.0 if x >= 0.5 else 0.0)
        
        return dataset


# === ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ===

def select_best_model(results: List[Dict],
                     metric: str = 'val_acc') -> Dict:
    """
    Выбор лучшей модели из результатов
    
    Args:
        results: Список результатов обучения
        metric: Метрика для сравнения ('val_acc', 'r2')
    
    Returns:
        dict: Лучшая модель
    """
    if not results:
        raise ValueError("Нет результатов для выбора")
    
    return max(results, key=lambda x: x[metric])


def save_model(result: Dict, filepath: str) -> None:
    """
    Сохранение модели на диск
    
    Args:
        result: Результат обучения с моделями
        filepath: Путь для сохранения (.cbm)
    """
    result['model'].save_model(filepath)
    print(f"✓ Модель сохранена: {filepath}")


def load_model(filepath: str) -> CatBoostClassifier:
    """
    Загрузка модели с диска
    
    Args:
        filepath: Путь к файлу .cbm
    
    Returns:
        CatBoostClassifier: Загруженная модель
    """
    model = CatBoostClassifier()
    model.load_model(filepath)
    print(f"✓ Модель загружена: {filepath}")
    return model