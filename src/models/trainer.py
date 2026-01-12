import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from src.data.loader import load_price_data
from src.features.engineering import create_features, get_feature_columns
from src.labeling.continuous import get_continuous_labels
from src.features.multiframe import add_multiframe_to_existing

class ClusterModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.data = None
        self.clusters = None
        
    def train_all_clusters(self) -> List[Dict]:
        self._prepare_data()
        
        n_clusters = self.config['clustering'].get('n_clusters', 1)
        
        # Если данных мало, отключаем кластеризацию
        if len(self.data) < 500:
            print("⚠️ Мало данных, кластеризация отключена.")
            n_clusters = 1

        if n_clusters > 1:
            self._perform_clustering()
        else:
            self.clusters = np.zeros(len(self.data), dtype=int)
        
        results = []
        for cluster_id in range(n_clusters):
            print(f"\n⚡ Обучение кластера {cluster_id}...")
            res = self._train_single_cluster(cluster_id)
            if res:
                results.append(res)
        return results
    
    def _prepare_data(self) -> None:
        print("📥 Загрузка данных...")
        prices = load_price_data(self.config)
        
        # 1. Базовые признаки
        features = create_features(
            prices, 
            self.config['periods'], 
            self.config['periods_meta']
        )
        
        # 2. Мультифреймовые признаки
        if self.config['data']['multiframe']['enabled']:
            print("🌐 Добавление мультифреймовых признаков...")
            try:
                features = add_multiframe_to_existing(
                    primary_data=features,
                    data_path=self.config['data']['paths']['raw'],
                    symbol=self.config['symbol']['name'].split('_')[0],
                    primary_tf=self.config['symbol']['timeframe'],
                    context_tfs=self.config['data']['multiframe']['timeframes']
                )
                print(f"   Итого колонок: {len(features.columns)}")
            except Exception as e:
                print(f"⚠️ Ошибка мультифрейма: {e}")
                print("   Продолжаем только с базовыми признаками.")
        
        # Восстановление OHLC для расчета таргета
        aligned_prices = prices.loc[features.index]
        features['high'] = aligned_prices['high']
        features['low'] = aligned_prices['low']
        features['open'] = aligned_prices['open'] # Нужно для ATR иногда
        
        # 3. Разметка
        print("🏷️ Генерация целевой переменной...")
        self.data = get_continuous_labels(
            features,
            max_bars=self.config['trading']['labeling']['max_bars'],
            direction=self.config['trading']['direction'],
            decay_factor=self.config['trading']['labeling'].get('decay', 0.96)
        )
            
    def _perform_clustering(self) -> None:
        meta_cols = get_feature_columns(self.data, 'meta_')
        if not meta_cols:
            meta_cols = get_feature_columns(self.data, 'feat_')[:5]
            
        X = self.data[meta_cols].values
        # Защита от NaN при кластеризации
        if np.isnan(X).any():
            X = np.nan_to_num(X)
            
        X = StandardScaler().fit_transform(X)
        
        kmeans = KMeans(
            n_clusters=self.config['clustering']['n_clusters'],
            random_state=42,
            n_init=10
        )
        self.clusters = kmeans.fit_predict(X)
        
    def _train_single_cluster(self, cluster_id: int) -> Optional[Dict]:
        mask = self.clusters == cluster_id
        cluster_data = self.data[mask].copy()
        
        if len(cluster_data) < 100:
            print(f"   Пропуск кластера {cluster_id}: мало данных ({len(cluster_data)})")
            return None
        
        # Разделение Train/Test (без перемешивания для временных рядов)
        split_idx = int(len(cluster_data) * 0.8)
        train_df = cluster_data.iloc[:split_idx]
        test_df = cluster_data.iloc[split_idx:]
        
        # Отбор признаков (исключаем служебные)
        all_cols = cluster_data.columns
        exclude_cols = ['labels', 'target', 'open', 'high', 'low', 'close', 'volume', 'atr']
        feat_cols = [c for c in all_cols if c not in exclude_cols]
        
        # === 1. MAIN MODEL (Regression) ===
        params = self.config['model']['main']['params'].copy()
        if 'custom_loss' in params: del params['custom_loss']
        
        # Проверка на константный таргет в регрессии
        if train_df['labels'].nunique() <= 1:
            print("   ⚠️ Ошибка: Target содержит только одно значение. Пропуск.")
            return None

        model = CatBoostRegressor(**params)
        model.fit(
            train_df[feat_cols], train_df['labels'],
            eval_set=(test_df[feat_cols], test_df['labels']),
            early_stopping_rounds=50,
            verbose=False
        )
        
        r2 = model.score(test_df[feat_cols], test_df['labels'])
        print(f"   Cluster {cluster_id} Main R2: {r2:.4f}")
        
        # === 2. META MODEL (Classifier) ===
        # Исправление: Обучаем мета-модель различать хорошие и плохие входы
        # Если кластеризации нет, мета-модель учится на ошибках основной модели
        
        meta_params = self.config['model']['meta']['params'].copy()
        meta_model = CatBoostClassifier(**meta_params)
        
        # Создаем бинарный таргет для мета-модели:
        # 1 = если основная модель предсказала > 0.5 И реальность > 0.5 (True Positive)
        # 0 = все остальное
        # Но для простоты сейчас: 1 = реальная прибыль > 0.5, 0 = иначе
        meta_target = (train_df['labels'] > 0.5).astype(int)
        
        # ВАЖНО: Проверка, что есть оба класса (0 и 1)
        if meta_target.nunique() > 1:
            meta_model.fit(train_df[feat_cols], meta_target, verbose=False)
        else:
            # Если класс только один (например, все сделки прибыльные или все убыточные),
            # создаем фиктивную модель для совместимости с ONNX
            print("   ⚠️ Meta target const. Creating dummy meta model.")
            dummy_X = train_df[feat_cols].iloc[:2]
            dummy_y = [0, 1] # Искусственные классы
            meta_model.fit(dummy_X, dummy_y, verbose=False)
        
        return {
            'cluster': cluster_id,
            'model': model,
            'meta_model': meta_model,
            'val_acc': r2, # Используем R2 как метрику
            'r2': r2,
            'dataset': test_df
        }

def select_best_model(results: List[Dict], metric: str = 'val_acc') -> Dict:
    if not results:
        raise ValueError("Нет результатов обучения")
    return max(results, key=lambda x: x.get(metric, -float('inf')))

def save_model(result: Dict, filepath: str) -> None:
    if 'model' in result:
        result['model'].save_model(filepath)

def load_model(filepath: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(filepath)
    return model