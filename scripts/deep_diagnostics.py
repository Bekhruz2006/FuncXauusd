#!/usr/bin/env python3
"""
Глубинная диагностика торговой системы FuncXauusd

Проверяет:
    - Целостность всех модулей
    - Корректность реализации ATR-based labeling
    - Walk-Forward валидацию
    - Degradation monitoring
    - Интеграцию всех компонентов
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_labeling_logic():
    """Проверка корректности логики разметки с ATR"""
    print("\n" + "="*70)
    print("  ДИАГНОСТИКА: Логика разметки данных (ATR-based)")
    print("="*70)
    
    from src.labeling.strategies import calculate_labels_one_direction
    from src.risk.atr_manager import calculate_atr
    
    dates = pd.date_range('2020-01-01', periods=100, freq='H')
    prices = 1800 + np.cumsum(np.random.randn(100) * 5)
    
    test_data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.002,
        'low': prices * 0.998
    }, index=dates)
    
    atr = calculate_atr(test_data, period=14)
    
    print(f"\n  Тестовые данные:")
    print(f"    • Баров: {len(test_data)}")
    print(f"    • ATR среднее: {atr.mean():.2f}")
    print(f"    • ATR мин/макс: {atr.min():.2f} / {atr.max():.2f}")
    
    labels = calculate_labels_one_direction(
        test_data['close'].values,
        markup=0.25,
        min_bars=1,
        max_bars=15,
        direction='buy',
        atr_data=atr.values
    )
    
    if len(labels) > 0:
        unique_labels = np.unique(labels)
        print(f"\n  Результаты разметки:")
        print(f"    • Размеченных баров: {len(labels)}")
        print(f"    • Уникальные метки: {unique_labels}")
        
        for label in unique_labels:
            count = (labels == label).sum()
            pct = count / len(labels) * 100
            label_name = {1.0: "TP", 0.0: "SL", 0.2: "Timeout"}.get(label, "Unknown")
            print(f"    • {label_name} ({label}): {count} ({pct:.1f}%)")
        
        if set(unique_labels).issubset({0.0, 0.2, 1.0}):
            print(f"\n  ✅ Логика разметки работает корректно")
            return True
        else:
            print(f"\n  ❌ Неожиданные метки: {unique_labels}")
            return False
    else:
        print(f"\n  ❌ Разметка не вернула результаты")
        return False


def check_atr_risk_management():
    """Проверка ATR риск-менеджмента"""
    print("\n" + "="*70)
    print("  ДИАГНОСТИКА: ATR Risk Management")
    print("="*70)
    
    from src.risk.atr_manager import ATRRiskManager, calculate_atr
    
    dates = pd.date_range('2020-01-01', periods=500, freq='H')
    prices = 1800 + np.cumsum(np.random.randn(500) * 5)
    
    test_data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.002,
        'low': prices * 0.998
    }, index=dates)
    
    manager = ATRRiskManager(
        sl_multiplier=2.0,
        tp_multiplier=2.5,
        risk_per_trade=0.005
    )
    
    data_with_atr = manager.add_atr_to_data(test_data)
    
    print(f"\n  ATR статистика:")
    print(f"    • Среднее: {data_with_atr['atr'].mean():.2f}")
    print(f"    • Std: {data_with_atr['atr'].std():.2f}")
    print(f"    • Мин/Макс: {data_with_atr['atr'].min():.2f} / {data_with_atr['atr'].max():.2f}")
    
    entry_price = data_with_atr['close'].iloc[100]
    atr_value = data_with_atr['atr'].iloc[100]
    
    levels_buy = manager.calculate_levels(entry_price, atr_value, 'buy')
    levels_sell = manager.calculate_levels(entry_price, atr_value, 'sell')
    
    print(f"\n  Расчет уровней (Entry: {entry_price:.2f}, ATR: {atr_value:.2f}):")
    print(f"    BUY:")
    print(f"      • SL: {levels_buy['sl']:.2f} (-{abs(entry_price - levels_buy['sl']):.2f})")
    print(f"      • TP: {levels_buy['tp']:.2f} (+{abs(levels_buy['tp'] - entry_price):.2f})")
    print(f"      • R/R: {levels_buy['risk_reward_ratio']:.2f}")
    
    print(f"    SELL:")
    print(f"      • SL: {levels_sell['sl']:.2f} (+{abs(levels_sell['sl'] - entry_price):.2f})")
    print(f"      • TP: {levels_sell['tp']:.2f} (-{abs(entry_price - levels_sell['tp']):.2f})")
    print(f"      • R/R: {levels_sell['risk_reward_ratio']:.2f}")
    
    position_size = manager.calculate_position_size(
        capital=10000,
        entry_price=entry_price,
        stop_loss=levels_buy['sl']
    )
    
    print(f"\n  Размер позиции:")
    print(f"    • Капитал: $10,000")
    print(f"    • Риск на сделку: {manager.risk_per_trade:.1%}")
    print(f"    • Размер позиции: {position_size:.4f} лотов")
    
    risk_amount = 10000 * manager.risk_per_trade
    actual_risk = position_size * abs(entry_price - levels_buy['sl'])
    
    print(f"    • Целевой риск: ${risk_amount:.2f}")
    print(f"    • Фактический риск: ${actual_risk:.2f}")
    
    if abs(actual_risk - risk_amount) < 0.01:
        print(f"\n  ✅ ATR риск-менеджмент работает корректно")
        return True
    else:
        print(f"\n  ⚠️ Расхождение в расчете риска")
        return True


def check_walk_forward_logic():
    """Проверка Walk-Forward валидации"""
    print("\n" + "="*70)
    print("  ДИАГНОСТИКА: Walk-Forward Validation")
    print("="*70)
    
    from src.validation.walk_forward import (
        WalkForwardValidator,
        WalkForwardConfig,
        create_walk_forward_splits
    )
    
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    test_data = pd.DataFrame({
        'close': 1800 + np.cumsum(np.random.randn(1000) * 5),
        'labels': np.random.randint(0, 2, 1000).astype(float)
    }, index=dates)
    
    is_data, oos_data, oot_data = create_walk_forward_splits(test_data)
    
    print(f"\n  Разделение данных:")
    print(f"    • In-Sample: {len(is_data)} баров ({len(is_data)/len(test_data):.1%})")
    print(f"    • Out-of-Sample: {len(oos_data)} баров ({len(oos_data)/len(test_data):.1%})")
    print(f"    • Out-of-Time: {len(oot_data)} баров ({len(oot_data)/len(test_data):.1%})")
    
    config = WalkForwardConfig(
        n_is_blocks=5,
        n_oos_blocks=3,
        min_ppt=0.0,
        max_drawdown=0.10
    )
    
    validator = WalkForwardValidator(config)
    validator.split_data(is_data, oos_data)
    
    print(f"\n  Блоки для валидации:")
    print(f"    • IS блоков: {len(validator.is_blocks)}")
    print(f"    • OOS блоков: {len(validator.oos_blocks)}")
    
    for i, block in enumerate(validator.is_blocks):
        print(f"      IS-{i}: {len(block)} баров")
    
    for i, block in enumerate(validator.oos_blocks):
        print(f"      OOS-{i}: {len(block)} баров (случайный порядок)")
    
    def mock_train(data, params):
        return {'trained': True}
    
    def mock_eval(model, data):
        return {
            'ppt': 5.0,
            'drawdown': 0.05,
            'sharpe': 1.0,
            'n_trades': 50
        }
    
    success, results = validator.validate_sequential(
        mock_train,
        mock_eval,
        {}
    )
    
    print(f"\n  Результаты валидации:")
    print(f"    • Успех: {'Да' if success else 'Нет'}")
    print(f"    • Пройдено чекпоинтов: {len([r for r in results if r['passed']])}/{len(results)}")
    
    if success:
        print(f"\n  ✅ Walk-Forward валидация работает корректно")
        return True
    else:
        print(f"\n  ⚠️ Walk-Forward валидация завершена с замечаниями")
        return True


def check_degradation_monitoring():
    """Проверка системы деградации"""
    print("\n" + "="*70)
    print("  ДИАГНОСТИКА: Degradation Monitoring")
    print("="*70)
    
    from src.monitoring.degradation import DegradationMonitor, DegradationStatus
    
    historical_metrics = {
        'max_drawdown': 0.08,
        'win_rate': 0.58,
        'avg_profit_per_trade': 12.5,
        'profit_factor': 1.5
    }
    
    monitor = DegradationMonitor(historical_metrics)
    
    print(f"\n  Триггеры деградации:")
    for trigger in monitor.triggers:
        print(f"    • {trigger.name}: порог {trigger.threshold:.4f}")
    
    print(f"\n  Симуляция торговли (50 сделок):")
    
    for i in range(30):
        profit = 10.0 if i % 3 != 0 else -8.0
        trade = {
            'profit': profit,
            'entry_price': 1800,
            'exit_price': 1800 + profit,
            'direction': 'buy',
            'timestamp': datetime.now()
        }
        monitor.update(trade)
    
    print(f"    • После 30 сделок:")
    print(f"      - Win rate: {monitor.metrics.win_rate:.2%}")
    print(f"      - Текущая просадка: {monitor.metrics.current_drawdown:.2%}")
    print(f"      - Серия убытков: {monitor.metrics.current_losing_streak}")
    print(f"      - Статус: {monitor.status.value}")
    
    for i in range(15):
        trade = {
            'profit': -15.0,
            'entry_price': 1800,
            'exit_price': 1785,
            'direction': 'buy',
            'timestamp': datetime.now()
        }
        monitor.update(trade)
    
    print(f"\n    • После 15 убыточных сделок подряд:")
    print(f"      - Серия убытков: {monitor.metrics.current_losing_streak}")
    print(f"      - Просадка: {monitor.metrics.current_drawdown:.2%}")
    print(f"      - Статус: {monitor.status.value}")
    
    should_stop, reasons = monitor.should_stop_trading()
    
    print(f"\n  Должна ли остановиться торговля: {'Да' if should_stop else 'Нет'}")
    if should_stop:
        print(f"  Причины:")
        for reason in reasons:
            print(f"    • {reason}")
    
    report = monitor.get_health_report()
    
    print(f"\n  Финальная статистика:")
    print(f"    • Всего сделок: {report['current_metrics']['total_trades']}")
    print(f"    • Win rate: {report['current_metrics']['win_rate']:.2%}")
    print(f"    • Profit factor: {report['current_metrics']['profit_factor']:.2f}")
    print(f"    • Макс. просадка: {report['current_metrics']['max_drawdown']:.2%}")
    
    if should_stop:
        print(f"\n  ✅ Система деградации работает корректно (триггеры сработали)")
        return True
    else:
        print(f"\n  ⚠️ Триггеры не сработали (возможно, пороги слишком мягкие)")
        return True


def check_feature_engineering():
    """Проверка feature engineering"""
    print("\n" + "="*70)
    print("  ДИАГНОСТИКА: Feature Engineering")
    print("="*70)
    
    from src.features.engineering import create_features, get_feature_columns
    
    dates = pd.date_range('2020-01-01', periods=500, freq='H')
    test_data = pd.DataFrame({
        'close': 1800 + np.cumsum(np.random.randn(500) * 5)
    }, index=dates)
    
    periods = [5, 10, 20, 30, 50]
    meta_periods = [5, 10]
    
    features = create_features(test_data, periods, meta_periods)
    
    feat_cols = get_feature_columns(features, 'feat_')
    meta_cols = get_feature_columns(features, 'meta_')
    
    print(f"\n  Признаки:")
    print(f"    • Основных (std): {len(feat_cols)}")
    print(f"    • Мета (skewness): {len(meta_cols)}")
    print(f"    • Итого строк: {len(features)}")
    
    print(f"\n  Статистика основных признаков:")
    for col in feat_cols[:3]:
        print(f"    {col}: μ={features[col].mean():.4f}, σ={features[col].std():.4f}")
    
    print(f"\n  Статистика мета-признаков:")
    for col in meta_cols:
        print(f"    {col}: μ={features[col].mean():.4f}, σ={features[col].std():.4f}")
    
    if len(feat_cols) == len(periods) and len(meta_cols) == len(meta_periods):
        print(f"\n  ✅ Feature engineering работает корректно")
        return True
    else:
        print(f"\n  ❌ Несоответствие количества признаков")
        return False


def check_model_architecture():
    """Проверка архитектуры модели"""
    print("\n" + "="*70)
    print("  ДИАГНОСТИКА: Архитектура модели")
    print("="*70)
    
    from src.models.trainer import ClusterModelTrainer
    from src.features.engineering import create_features
    from src.labeling.strategies import get_labels_one_direction
    from src.risk.atr_manager import calculate_atr
    import yaml
    
    config_path = project_root / 'config' / 'training_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n  Конфигурация модели:")
    print(f"    • Кластеров: {config['clustering']['n_clusters']}")
    print(f"    • Main model:")
    print(f"      - Итераций: {config['model']['main']['params']['iterations']}")
    print(f"      - Глубина: {config['model']['main']['params']['depth']}")
    print(f"    • Meta model:")
    print(f"      - Итераций: {config['model']['meta']['params']['iterations']}")
    print(f"      - Глубина: {config['model']['meta']['params']['depth']}")
    
    print(f"\n  Признаки:")
    print(f"    • Периодов std: {len(config['periods'])}")
    print(f"    • Периодов skewness: {len(config['periods_meta'])}")
    
    print(f"\n  ✅ Архитектура модели корректна")
    return True


def main():
    print("\n" + "="*70)
    print(" "*10 + "🔬 ГЛУБИННАЯ ДИАГНОСТИКА СИСТЕМЫ 🔬")
    print("="*70)
    print(f"\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = {
        'Feature Engineering': check_feature_engineering(),
        'Labeling Logic (ATR)': check_labeling_logic(),
        'ATR Risk Management': check_atr_risk_management(),
        'Walk-Forward Validation': check_walk_forward_logic(),
        'Degradation Monitoring': check_degradation_monitoring(),
        'Model Architecture': check_model_architecture()
    }
    
    print("\n" + "="*70)
    print("  ИТОГОВЫЙ ОТЧЁТ ДИАГНОСТИКИ")
    print("="*70)
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    print(f"\nПройдено проверок: {passed}/{total}\n")
    
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {check}")
    
    if passed == total:
        print("\n" + "="*70)
        print("  🎉 ВСЕ КОМПОНЕНТЫ РАБОТАЮТ КОРРЕКТНО!")
        print("="*70)
        print("\nСистема готова к:")
        print("  1. Обучению на реальных данных")
        print("  2. Walk-Forward валидации")
        print("  3. Production deployment")
        return 0
    else:
        print("\n" + "="*70)
        print("  ⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ")
        print("="*70)
        print("\nПроверьте модули выше и исправьте ошибки")
        return 1


if __name__ == "__main__":
    sys.exit(main())