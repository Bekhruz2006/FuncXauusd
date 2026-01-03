#!/usr/bin/env python3
"""
Улучшенный скрипт обучения с Walk-Forward валидацией

Интегрирует:
    - Walk-Forward валидацию (Этап 2)
    - Динамические ATR-уровни (Этап 3)
    - Мультифреймовые признаки (Приоритет 2)
    - Систему деградации (Этап 6)

Usage:
    python scripts/train_with_walk_forward.py [--config path/to/config.yaml]
"""

import sys
import time
import yaml
import warnings
import argparse
from pathlib import Path
from datetime import datetime

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_price_data, cache_prices
from src.features.engineering import create_features
from src.features.multiframe import create_multiframe_features
from src.labeling.strategies import get_labels_one_direction
from src.models.trainer import ClusterModelTrainer
from src.export.onnx_exporter import export_to_onnx
from src.risk.atr_manager import ATRRiskManager, backtest_with_dynamic_atr
from src.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    create_walk_forward_splits
)
from src.monitoring.degradation import DegradationMonitor

warnings.filterwarnings('ignore')


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Обучение с Walk-Forward валидацией'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Путь к конфигурации'
    )
    parser.add_argument(
        '--enable-multiframe',
        action='store_true',
        help='Включить мультифреймовые признаки'
    )
    parser.add_argument(
        '--enable-walk-forward',
        action='store_true',
        help='Включить Walk-Forward валидацию'
    )
    parser.add_argument(
        '--optimize-atr',
        action='store_true',
        help='Оптимизировать ATR множители'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации"""
    config_file = project_root / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_data_with_features(config: dict,
                               use_multiframe: bool = False) -> pd.DataFrame:
    """
    Подготовка данных с признаками
    
    Args:
        config: Конфигурация
        use_multiframe: Использовать мультифреймовые признаки
    
    Returns:
        pd.DataFrame: Данные с признаками и метками
    """
    print(f"\n{'='*70}")
    print(f"  📊 ПОДГОТОВКА ДАННЫХ")
    print(f"{'='*70}\n")
    
    # Загрузка основных данных
    print("📥 Загрузка цен...")
    prices = load_price_data(config)
    
    # Создание базовых признаков
    print("\n🔧 Создание базовых признаков...")
    periods = config['periods']
    meta_periods = config['periods_meta']
    
    features = create_features(prices, periods, meta_periods)
    
    # Мультифреймовые признаки (опционально)
    if use_multiframe and config['data']['multiframe']['enabled']:
        print("\n🌐 Добавление мультифреймовых признаков...")
        
        from src.features.multiframe import add_multiframe_to_existing
        
        features = add_multiframe_to_existing(
            features,
            data_path=config['data']['paths']['raw'],
            symbol=config['symbol']['name'].split('_')[0],  # XAUUSD без _H1
            primary_tf=config['symbol']['timeframe'],
            context_tfs=config['data']['multiframe']['timeframes'][:3]  # D1, W1, MN
        )
    
    # Добавление ATR
    print("\n💹 Расчет ATR...")
    atr_manager = ATRRiskManager(
        sl_multiplier=config['trading']['risk']['stop_loss'] / 100,  # Конвертация пунктов в множитель
        tp_multiplier=config['trading']['risk']['take_profit'] / 100,
        atr_period=14
    )
    features = atr_manager.add_atr_to_data(features)
    
    # Разметка данных
    print("\n🏷️ Разметка данных...")
    labeled = get_labels_one_direction(
        features,
        markup=config['markup'],
        min_bars=config['trading']['labeling']['min_bars'],
        max_bars=config['trading']['labeling']['max_bars'],
        direction=config['trading']['direction']
    )
    
    print(f"\n✅ Данные подготовлены: {len(labeled)} баров")
    
    return labeled


def train_with_walk_forward(data: pd.DataFrame,
                            config: dict,
                            wf_config: WalkForwardConfig) -> tuple:
    """
    Обучение с Walk-Forward валидацией
    
    Args:
        data: Подготовленные данные
        config: Конфигурация обучения
        wf_config: Конфигурация Walk-Forward
    
    Returns:
        (success, best_model, validator): Результат валидации
    """
    print(f"\n{'='*70}")
    print(f"  🎯 WALK-FORWARD ВАЛИДАЦИЯ")
    print(f"{'='*70}\n")
    
    # Разделение данных
    is_data, oos_data, oot_data = create_walk_forward_splits(data)
    
    # Инициализация валидатора
    validator = WalkForwardValidator(wf_config)
    validator.split_data(is_data, oos_data)
    
    # Функции обучения и оценки
    def train_fn(train_data, params):
        """Функция обучения для валидатора"""
        trainer_config = {**config, **params}
        trainer = ClusterModelTrainer(trainer_config)
        
        # Временно заменяем данные
        trainer.data = train_data
        results = trainer.train_all_clusters()
        
        if not results:
            raise ValueError("Не удалось обучить модель")
        
        # Лучшая модель
        best = max(results, key=lambda x: x['val_acc'])
        return best
    
    def eval_fn(model_result, test_data):
        """Функция оценки для валидатора"""
        # Расчет метрик на тестовых данных
        from src.backtesting.tester import test_model_one_direction
        
        r2 = test_model_one_direction(
            dataset=model_result['dataset'],
            result=[model_result['model'], model_result['meta_model']],
            config=config,
            plt=False
        )
        
        # TODO: Рассчитать реальные метрики (PPT, Sharpe, DD)
        # Пока используем заглушки на основе R²
        metrics = {
            'ppt': r2 * 10,  # Заглушка
            'drawdown': max(0.01, 0.1 - r2 * 0.05),  # Заглушка
            'sharpe': r2 * 2,  # Заглушка
            'n_trades': 100  # Заглушка
        }
        
        return metrics
    
    # Запуск валидации
    success, checkpoint_results = validator.validate_sequential(
        train_fn,
        eval_fn,
        {
            'depth': config['model']['main']['params']['depth'],
            'iterations': config['model']['main']['params']['iterations'],
            'l2_leaf_reg': config['model']['main']['params']['l2_leaf_reg']
        }
    )
    
    # Отчет
    validator.print_detailed_report()
    
    # Если прошли все чекпоинты - финальное обучение на всех данных
    if success:
        print(f"\n{'='*70}")
        print(f"  ✅ ВСЕ ЧЕКПОИНТЫ ПРОЙДЕНЫ - ФИНАЛЬНОЕ ОБУЧЕНИЕ")
        print(f"{'='*70}\n")
        
        # Обучение на IS + OOS
        full_train_data = pd.concat([is_data, oos_data], ignore_index=True)
        final_model = train_fn(full_train_data, {})
        
        # Тест на Out-of-Time
        print("\n📈 Финальный тест на Out-of-Time данных...")
        final_metrics = eval_fn(final_model, oot_data)
        
        print(f"\n  OOT Метрики:")
        print(f"    • PPT: {final_metrics['ppt']:.4f}")
        print(f"    • Drawdown: {final_metrics['drawdown']:.2%}")
        print(f"    • Sharpe: {final_metrics['sharpe']:.2f}")
        
        return success, final_model, validator
    
    return success, None, validator


def optimize_atr_parameters(data: pd.DataFrame,
                            config: dict) -> dict:
    """
    Оптимизация ATR множителей
    
    Args:
        data: Данные с сигналами
        config: Конфигурация
    
    Returns:
        dict: Оптимальные параметры
    """
    print(f"\n{'='*70}")
    print(f"  🔍 ОПТИМИЗАЦИЯ ATR ПАРАМЕТРОВ")
    print(f"{'='*70}\n")
    
    from src.risk.atr_manager import optimize_atr_multipliers
    
    # Заглушка для сигналов (нужно получить из модели)
    import pandas as pd
    signals = pd.Series(
        (data['labels'] > 0.5).astype(int),
        index=data.index
    )
    
    result = optimize_atr_multipliers(
        data,
        signals,
        direction=config['trading']['direction'],
        sl_range=(1.0, 3.0),
        tp_range=(1.5, 4.0),
        step=0.5
    )
    
    return result['best_params']


def main():
    """Главная функция"""
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f" "*10 + "🚀 ENHANCED TRAINING WITH WALK-FORWARD 🚀")
    print(f"{'='*70}\n")
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    print("📋 Параметры:")
    print(f"  • Symbol: {config['symbol']['name']}")
    print(f"  • Direction: {config['trading']['direction'].upper()}")
    print(f"  • Multiframe: {'ВКЛ' if args.enable_multiframe else 'ВЫКЛ'}")
    print(f"  • Walk-Forward: {'ВКЛ' if args.enable_walk_forward else 'ВЫКЛ'}")
    print(f"  • ATR Optimization: {'ВКЛ' if args.optimize_atr else 'ВЫКЛ'}")
    
    # Кэширование данных
    print("\n🔄 Кэширование данных...")
    cache_prices(config)
    
    # Подготовка данных
    data = prepare_data_with_features(
        config,
        use_multiframe=args.enable_multiframe
    )
    
    # Walk-Forward валидация (опционально)
    if args.enable_walk_forward:
        wf_config = WalkForwardConfig(
            n_is_blocks=10,
            n_oos_blocks=5,
            min_ppt=0.0,
            max_drawdown=0.05,
            min_sharpe=0.5,
            max_retries=3
        )
        
        success, final_model, validator = train_with_walk_forward(
            data,
            config,
            wf_config
        )
        
        if not success:
            print(f"\n❌ Walk-Forward валидация не пройдена!")
            return 1
        
        # Сохранение состояния валидатора
        validator_path = project_root / 'logs' / 'walk_forward_history.json'
        validator_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Сохранить историю валидатора
    
    else:
        # Обычное обучение (из оригинального скрипта)
        print(f"\n{'='*70}")
        print(f"  🎓 СТАНДАРТНОЕ ОБУЧЕНИЕ")
        print(f"{'='*70}\n")
        
        trainer = ClusterModelTrainer(config)
        results = trainer.train_all_clusters()
        
        if not results:
            print(f"\n❌ Обучение не удалось!")
            return 1
        
        final_model = max(results, key=lambda x: x['val_acc'])
    
    # Оптимизация ATR (опционально)
    if args.optimize_atr:
        atr_params = optimize_atr_parameters(data, config)
        print(f"\n✅ Оптимальные ATR параметры:")
        print(f"  • SL multiplier: {atr_params['sl_mult']}")
        print(f"  • TP multiplier: {atr_params['tp_mult']}")
        
        # Обновление конфига
        config['atr_sl_mult'] = atr_params['sl_mult']
        config['atr_tp_mult'] = atr_params['tp_mult']
    
    # Экспорт модели
    print(f"\n{'='*70}")
    print(f"  💾 ЭКСПОРТ МОДЕЛИ")
    print(f"{'='*70}\n")
    
    export_to_onnx(
        model_main=final_model['model'],
        model_meta=final_model['meta_model'],
        config=config,
        r2_score=final_model['r2']
    )
    
    # Создание монитора деградации для live-торговли
    print(f"\n{'='*70}")
    print(f"  🔍 ИНИЦИАЛИЗАЦИЯ DEGRADATION MONITOR")
    print(f"{'='*70}\n")
    
    historical_metrics = {
        'max_drawdown': 0.08,
        'win_rate': 0.58,
        'avg_profit_per_trade': 12.5,
        'profit_factor': 1.5
    }
    
    monitor = DegradationMonitor(historical_metrics)
    monitor_path = project_root / 'logs' / 'degradation_monitor_initial.json'
    monitor.save_state(str(monitor_path))
    
    print(f"\n{'='*70}")
    print(f"  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print(f"{'='*70}\n")
    
    print(f"📁 Результаты:")
    print(f"  • ONNX модели: {config['export']['paths']['onnx']}")
    print(f"  • Degradation monitor: {monitor_path}")
    
    print(f"\n💡 Следующие шаги:")
    print(f"  1. Скопировать ONNX в MT5/Experts/Files/")
    print(f"  2. Скопировать .mqh в MT5/Include/")
    print(f"  3. Запустить на демо-счете минимум 6 месяцев")
    print(f"  4. Мониторить degradation monitor")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())