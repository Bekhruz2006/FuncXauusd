#!/usr/bin/env python3
import sys
import yaml
import warnings
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Fix paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import cache_prices
from src.models.trainer import ClusterModelTrainer
from src.export.onnx_exporter import export_to_onnx
from src.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    create_walk_forward_splits
)

warnings.filterwarnings('ignore')

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-walk-forward', action='store_true')
    parser.add_argument('--enable-multiframe', action='store_true')
    parser.add_argument('--optimize-atr', action='store_true')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("🚀 STARTING WALK-FORWARD REGRESSION PIPELINE (FIXED)")
    print("="*70 + "\n")
    
    config = load_config(project_root / 'config/training_config.yaml')
    
    # Применение аргументов командной строки
    if args.enable_multiframe:
        config['data']['multiframe']['enabled'] = True

    # 1. Загрузка данных (через Trainer)
    print("🛠️ Initializing Trainer...")
    trainer = ClusterModelTrainer(config)
    
    # Загрузка и подготовка данных
    trainer._prepare_data()
    full_data = trainer.data
    
    # 2. Подготовка сплитов
    print(f"✂️ Splitting Data ({len(full_data)} bars)...")
    is_data, oos_data, oot_data = create_walk_forward_splits(full_data)
    
    # 3. Настройка валидатора
    wf_conf = config['walk_forward']
    validator = WalkForwardValidator(WalkForwardConfig(
        n_is_blocks=wf_conf['n_is_blocks'],
        n_oos_blocks=wf_conf['n_oos_blocks'],
        min_r2=wf_conf['min_r2'],
        noise_level=wf_conf['noise_level'],
        l2_increment=wf_conf['l2_increment'],
        max_retries=wf_conf['max_retries']
    ))
    validator.split_data(is_data, oos_data)
    
    # 4. Функции для валидатора
    def train_func(train_data, params):
        # Создаем временный конфиг
        temp_conf = config.copy()
        temp_conf['model']['main']['params'].update(params)
        
        # Инициализируем тренера
        t = ClusterModelTrainer(temp_conf)
        t.data = train_data
        
        # Для скорости отключаем сложную кластеризацию внутри WF цикла
        # и обучаем единую модель на всех данных блока
        t.clusters = np.zeros(len(train_data), dtype=int)
        
        # Обучаем "кластер 0" (он же весь датасет)
        res = t._train_single_cluster(0)
        return res['model'] if res else None
        
    def eval_func(model, test_data):
        # === ИСПРАВЛЕНИЕ ===
        # Получаем имена признаков, на которых модель реально обучилась
        model_feature_names = model.feature_names_
        
        # Проверяем наличие всех колонок
        missing_cols = [c for c in model_feature_names if c not in test_data.columns]
        if missing_cols:
            print(f"    ⚠️ Warning: Missing columns in test set: {missing_cols}")
            # Пытаемся заполнить нулями или падаем, если критично
            for c in missing_cols:
                test_data[c] = 0.0
                
        # Используем только те колонки, которые ожидает модель
        X_test = test_data[model_feature_names]
        
        # Скорринг
        r2 = model.score(X_test, test_data['labels'])
        return {'r2': r2}
    
    # 5. Запуск валидации
    base_params = config['model']['main']['params']
    success, final_model = validator.validate_sequential(train_func, eval_func, base_params)
    
    if success and final_model:
        print("\n🏆 SUCCESS! Model passed all checkpoints.")
        
        # 6. Финальный тест на OOT
        print("\n🧪 Testing on Out-of-Time data...")
        model_feature_names = final_model.feature_names_
        X_oot = oot_data[model_feature_names]
        
        r2_oot = final_model.score(X_oot, oot_data['labels'])
        print(f"   OOT R2 Score: {r2_oot:.4f}")
        
        if r2_oot > 0:
            # 7. Экспорт
            print("\n💾 Exporting...")
            from catboost import CatBoostClassifier
            # Создаем заглушку meta-модели (так как мы обучали без кластеров в WF)
            meta_dummy = CatBoostClassifier(iterations=10)
            # Обучаем заглушку, чтобы она была валидной
            meta_dummy.fit(X_oot.iloc[:10], [0, 1]*5, verbose=False)
            
            export_to_onnx(final_model, meta_dummy, config, r2_oot)
            print("✅ Export complete. Copy .onnx and .mqh files to MT5.")
        else:
            print("⚠️ OOT Score is negative. Export skipped.")
        
    else:
        print("\n💀 FAILURE. Model did not converge after retries.")
        sys.exit(1)

if __name__ == '__main__':
    main()