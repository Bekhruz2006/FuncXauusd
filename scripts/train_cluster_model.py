#!/usr/bin/env python3
"""
Основной скрипт обучения кластерной торговой модели
Реализует автопоиск оптимальной конфигурации с валидацией
"""

import sys
import time
import yaml
import warnings
from pathlib import Path
from datetime import datetime

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_price_data, cache_prices
from src.features.engineering import create_features
from src.labeling.strategies import get_labels_one_direction
from src.models.trainer import ClusterModelTrainer
from src.models.validator import validate_model
from src.export.onnx_exporter import export_to_onnx
from src.backtesting.tester import test_model_one_direction

warnings.filterwarnings('ignore')

# ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================
CACHED_PRICES = None
BEST_GLOBAL_MODEL = None
SEARCH_HISTORY = []


def load_config(config_path: str = "config/training_config.yaml") -> dict:
    """Загрузка конфигурации из YAML"""
    config_file = project_root / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_search_configs(base_config: dict) -> list:
    """
    Генерация конфигураций для поиска
    
    Returns:
        list: Список конфигураций для тестирования
    """
    from itertools import product
    
    search_space = base_config['search']['space']
    configs = []
    
    # Комбинаторный поиск
    for (markup, n_clusters, periods, meta_periods, 
         depth, iterations, min_samples) in product(
        search_space['markup'],
        search_space['n_clusters'],
        search_space['periods'],
        search_space['meta_periods'],
        search_space['model_depth'],
        search_space['model_iterations'],
        search_space['min_samples']
    ):
        config = {
            **base_config,
            'markup': markup,
            'n_clusters': n_clusters,
            'periods': periods,
            'periods_meta': meta_periods,
            'depth': depth,
            'iterations': iterations,
            'min_samples': min_samples
        }
        configs.append(config)
    
    return configs


def prioritize_configs(configs: list) -> list:
    """
    Приоритизация конфигураций по ожидаемому качеству
    Предпочитаем средние значения и более сложные модели
    """
    def priority_score(c):
        score = 0
        score += abs(c['markup'] - 0.25) * 10
        score += abs(c['n_clusters'] - 8) * 5
        score += abs(c['depth'] - 5) * 3
        score += abs(c['iterations'] - 700) / 100
        score -= c['min_samples'] / 1000  # Больше данных = лучше
        return score
    
    return sorted(configs, key=priority_score)


def train_single_config(config: dict, iteration: int) -> dict:
    """
    Обучение со всеми кластерами для данной конфигурации
    
    Args:
        config: Конфигурация обучения
        iteration: Номер текущей итерации
    
    Returns:
        dict: Результаты обучения или None
    """
    print(f"\n{'─'*70}")
    print(f"🔄 Попытка {iteration}/{config['search']['max_iterations']}")
    print(f"{'─'*70}")
    print(f"  Конфигурация:")
    print(f"    • Markup: {config['markup']}")
    print(f"    • Кластеров: {config['n_clusters']}")
    print(f"    • Признаков: {len(config['periods'])}")
    print(f"    • Мета-периодов: {config['periods_meta']}")
    print(f"    • Глубина: {config['depth']}")
    print(f"    • Итераций: {config['iterations']}")
    print(f"    • Мин. примеров: {config['min_samples']}")
    
    try:
        # Инициализация тренера
        trainer = ClusterModelTrainer(config)
        
        # Обучение всех кластеров
        results = trainer.train_all_clusters()
        
        if not results or len(results) == 0:
            print(f"  ❌ Ни одна модель не обучена")
            return None
        
        # Выбор лучшей модели (по Val Accuracy)
        best_model = max(results, key=lambda x: x['val_acc'])
        
        print(f"\n  🏆 Лучшая модель кластера {best_model['cluster']}:")
        print(f"    • Val Acc: {best_model['val_acc']:.4f} ⭐")
        print(f"    • R²: {best_model['r2']:.4f}")
        
        return {
            'best_model': best_model,
            'config': config,
            'all_models': results
        }
        
    except Exception as e:
        print(f"  ⚠️ Ошибка: {e}")
        return None


def print_final_results(best_model_result: dict, search_history: list, 
                       elapsed_time: float):
    """Вывод финальных результатов поиска"""
    print(f"\n{'='*70}")
    print(f"  📊 РЕЗУЛЬТАТЫ ПОИСКА")
    print(f"{'='*70}")
    print(f"\n⏱ Время поиска: {elapsed_time/60:.1f} минут")
    print(f"✅ Протестировано конфигураций: {len(search_history)}")
    
    if not best_model_result:
        print("\n❌ Не удалось найти подходящую модель!")
        return False
    
    best = best_model_result['best_model']
    best_config = best_model_result['config']
    
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ:")
    print(f"┌─────────────────────────────────────────┐")
    print(f"│  Кластер:        {best['cluster']:<5}                    │")
    print(f"│  Val Accuracy:   {best['val_acc']:.4f} ⭐⭐⭐            │")
    print(f"│  R²:             {best['r2']:.4f}                  │")
    print(f"│  Примеров:       {best['samples']:<7}                  │")
    print(f"│  Баланс:         {best['balance']:.2f}                    │")
    print(f"└─────────────────────────────────────────┘")
    
    print(f"\n⚙️ ОПТИМАЛЬНАЯ КОНФИГУРАЦИЯ:")
    print(f"  • Markup: {best_config['markup']}")
    print(f"  • Кластеров: {best_config['n_clusters']}")
    print(f"  • Признаков: {len(best_config['periods'])}")
    print(f"  • Мета-периодов: {best_config['periods_meta']}")
    print(f"  • Глубина: {best_config['depth']}")
    print(f"  • Итераций: {best_config['iterations']}")
    print(f"  • Мин. примеров: {best_config['min_samples']}")
    
    return True


def print_search_history(history: list, top_n: int = 10):
    """Вывод истории поиска"""
    if len(history) <= 1:
        return
    
    print(f"\n📊 История поиска (топ-{top_n}):")
    print(f"┌─────┬──────────┬──────────┬──────────┬──────────┐")
    print(f"│  №  │  Markup  │  Val Acc │  R²      │ Кластеров│")
    print(f"├─────┼──────────┼──────────┼──────────┼──────────┤")
    
    sorted_history = sorted(history, 
                           key=lambda x: x['best_model']['val_acc'], 
                           reverse=True)
    
    for i, h in enumerate(sorted_history[:top_n], 1):
        m = h['best_model']
        c = h['config']
        print(f"│ {i:<3} │  {c['markup']:.2f}    │  {m['val_acc']:.4f}  │  {m['r2']:.4f}  │    {c['n_clusters']:<2}    │")
    
    print(f"└─────┴──────────┴──────────┴──────────┴──────────┘\n")


def main():
    """Главная функция"""
    print("\n" + "="*70)
    print(" "*8 + "🎯 АВТОПОИСК ОПТИМАЛЬНОЙ КОНФИГУРАЦИИ (40 попыток) 🎯")
    print("="*70 + "\n")
    
    # Загрузка конфигурации
    config = load_config()
    
    print("📋 Параметры поиска:")
    print(f"  • Целевая Val Acc: ≥{config['search']['targets']['val_accuracy']:.2f} (75%+)")
    print(f"  • Целевой R²: ≥{config['search']['targets']['r2_score']:.2f}")
    print(f"  • Максимум попыток: {config['search']['max_iterations']}")
    print(f"  • Период данных: {config['data']['backward']} - {config['data']['full_forward']}")
    
    # Кэширование данных
    print("\n🔄 Загрузка и кэширование данных...")
    cache_prices(config)
    
    # Генерация конфигураций
    print(f"\n🔍 Генерация конфигураций...")
    all_configs = generate_search_configs(config)
    print(f"📊 Сгенерировано {len(all_configs)} конфигураций")
    
    # Приоритизация
    prioritized_configs = prioritize_configs(all_configs)
    configs_to_test = prioritized_configs[:config['search']['max_iterations']]
    print(f"⚡ Будет протестировано {len(configs_to_test)} лучших\n")
    
    # Основной цикл поиска
    start_time = time.time()
    target_acc = config['search']['targets']['val_accuracy']
    
    for idx, test_config in enumerate(configs_to_test, 1):
        result = train_single_config(test_config, idx)
        
        if result is None:
            continue
        
        SEARCH_HISTORY.append(result)
        best_model = result['best_model']
        
        # Проверка достижения цели
        if best_model['val_acc'] >= target_acc:
            print(f"\n{'='*70}")
            print(f"  🎉 ДОСТИГНУТА ЦЕЛЕВАЯ ТОЧНОСТЬ НА ПОПЫТКЕ {idx}!")
            print(f"  Val Accuracy: {best_model['val_acc']:.4f} ≥ {target_acc}")
            print(f"{'='*70}")
            BEST_GLOBAL_MODEL = result
            break
        
        # Обновление лучшей модели
        if (BEST_GLOBAL_MODEL is None or 
            best_model['val_acc'] > BEST_GLOBAL_MODEL['best_model']['val_acc']):
            BEST_GLOBAL_MODEL = result
            print(f"  ⭐ Новый лидер! Val Acc: {best_model['val_acc']:.4f}")
    
    elapsed = time.time() - start_time
    
    # Вывод результатов
    if not print_final_results(BEST_GLOBAL_MODEL, SEARCH_HISTORY, elapsed):
        return 1
    
    print_search_history(SEARCH_HISTORY)
    
    # Финальное тестирование
    print(f"\n📈 Финальное тестирование с визуализацией...\n")
    best = BEST_GLOBAL_MODEL['best_model']
    best_config = BEST_GLOBAL_MODEL['config']
    
    try:
        R2_final = test_model_one_direction(
            dataset=best['dataset'],
            result=[best['model'], best['meta_model']],
            config=best_config,
            plt=True
        )
        
        print(f"\n{'='*70}")
        print(f"  ✅ ФИНАЛЬНЫЙ R² НА ТЕСТЕ: {R2_final:.4f}")
        print(f"  🎯 VAL ACCURACY: {best['val_acc']:.4f}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ Ошибка тестирования: {e}")
        R2_final = best['r2']
    
    # Экспорт
    print(f"\n💾 Экспорт в MetaTrader 5...")
    try:
        export_to_onnx(
            model_main=best['model'],
            model_meta=best['meta_model'],
            config=best_config,
            r2_score=R2_final
        )
        
        print(f"\n{'='*70}")
        print(f"  ✅ АВТОПОИСК ЗАВЕРШЁН УСПЕШНО!")
        print(f"{'='*70}")
        print(f"\n📁 Файлы находятся в: {config['export']['paths']['onnx']}")
        print(f"\n💡 Модель кластера {best['cluster']} готова к торговле!")
        print(f"   Val Acc: {best['val_acc']:.4f} | R²: {R2_final:.4f}")
        
    except Exception as e:
        print(f"\n❌ Ошибка экспорта: {e}")
        return 1
    
    # Статус достижения цели
    status = 'ДА' if best['val_acc'] >= target_acc else f'НЕТ (лучший {best["val_acc"]:.4f})'
    print(f"\nЦель достигнута: {status}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())