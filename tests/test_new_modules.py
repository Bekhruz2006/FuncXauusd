"""
Юнит-тесты для новых модулей

Тестирование:
    - Walk-Forward валидации
    - ATR Risk Manager
    - Degradation Monitor
    - Multiframe Features

Usage:
    pytest tests/test_new_modules.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    create_walk_forward_splits
)
from src.risk.atr_manager import (
    ATRRiskManager,
    calculate_atr,
    simulate_trade_with_atr
)
from src.monitoring.degradation import (
    DegradationMonitor,
    DegradationStatus
)


# ==================== FIXTURES ====================

@pytest.fixture
def sample_price_data():
    """Создание тестовых ценовых данных"""
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    
    # Генерация синтетических цен
    np.random.seed(42)
    base_price = 1800
    returns = np.random.normal(0, 5, 1000)
    prices = base_price + np.cumsum(returns)
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    return df


@pytest.fixture
def sample_labeled_data(sample_price_data):
    """Данные с метками"""
    data = sample_price_data.copy()
    data['labels'] = np.random.randint(0, 2, len(data)).astype(float)
    return data


@pytest.fixture
def atr_manager():
    """ATR Risk Manager"""
    return ATRRiskManager(
        sl_multiplier=2.0,
        tp_multiplier=2.5,
        risk_per_trade=0.005,
        max_bars_timeout=20
    )


@pytest.fixture
def historical_metrics():
    """Исторические метрики для монитора"""
    return {
        'max_drawdown': 0.08,
        'win_rate': 0.58,
        'avg_profit_per_trade': 12.5,
        'profit_factor': 1.5
    }


# ==================== ТЕСТЫ WALK-FORWARD ====================

class TestWalkForwardValidator:
    """Тесты Walk-Forward валидатора"""
    
    def test_config_creation(self):
        """Тест создания конфигурации"""
        config = WalkForwardConfig(
            n_is_blocks=10,
            n_oos_blocks=5,
            min_ppt=0.0,
            max_drawdown=0.05
        )
        
        assert config.n_is_blocks == 10
        assert config.n_oos_blocks == 5
        assert config.min_ppt == 0.0
        assert config.max_drawdown == 0.05
    
    def test_data_splits(self, sample_labeled_data):
        """Тест разделения данных"""
        is_data, oos_data, oot_data = create_walk_forward_splits(
            sample_labeled_data,
            is_ratio=0.6,
            oos_ratio=0.2
        )
        
        total = len(sample_labeled_data)
        
        assert len(is_data) == int(total * 0.6)
        assert len(oos_data) == int(total * 0.2)
        assert len(oot_data) > 0
        
        # Проверка временного порядка
        assert is_data.index[-1] <= oos_data.index[0]
        assert oos_data.index[-1] <= oot_data.index[0]
    
    def test_validator_initialization(self):
        """Тест инициализации валидатора"""
        config = WalkForwardConfig()
        validator = WalkForwardValidator(config)
        
        assert validator.config == config
        assert len(validator.is_blocks) == 0
        assert len(validator.oos_blocks) == 0
        assert len(validator.checkpoint_history) == 0
    
    def test_split_data(self, sample_labeled_data):
        """Тест разбиения на блоки"""
        config = WalkForwardConfig(n_is_blocks=5, n_oos_blocks=3)
        validator = WalkForwardValidator(config)
        
        is_data = sample_labeled_data.iloc[:600]
        oos_data = sample_labeled_data.iloc[600:800]
        
        validator.split_data(is_data, oos_data)
        
        assert len(validator.is_blocks) == 5
        assert len(validator.oos_blocks) == 3
        
        # Проверка размеров блоков
        for block in validator.is_blocks:
            assert len(block) > 0
        
        for block in validator.oos_blocks:
            assert len(block) > 0


# ==================== ТЕСТЫ ATR MANAGER ====================

class TestATRRiskManager:
    """Тесты ATR Risk Manager"""
    
    def test_calculate_atr(self, sample_price_data):
        """Тест расчета ATR"""
        atr = calculate_atr(sample_price_data, period=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_price_data)
        assert atr.notna().sum() > 0
        
        # ATR должен быть положительным
        assert (atr.dropna() >= 0).all()
    
    def test_calculate_levels_buy(self, atr_manager):
        """Тест расчета уровней для покупки"""
        levels = atr_manager.calculate_levels(
            entry_price=1800.0,
            atr_value=5.0,
            direction='buy'
        )
        
        assert 'sl' in levels
        assert 'tp' in levels
        assert 'risk_points' in levels
        
        # SL должен быть ниже entry
        assert levels['sl'] < 1800.0
        
        # TP должен быть выше entry
        assert levels['tp'] > 1800.0
        
        # Risk/Reward ratio
        assert levels['risk_reward_ratio'] > 1.0
    
    def test_calculate_levels_sell(self, atr_manager):
        """Тест расчета уровней для продажи"""
        levels = atr_manager.calculate_levels(
            entry_price=1800.0,
            atr_value=5.0,
            direction='sell'
        )
        
        # SL должен быть выше entry
        assert levels['sl'] > 1800.0
        
        # TP должен быть ниже entry
        assert levels['tp'] < 1800.0
    
    def test_calculate_position_size(self, atr_manager):
        """Тест расчета размера позиции"""
        size = atr_manager.calculate_position_size(
            capital=10000,
            entry_price=1800,
            stop_loss=1790,
            contract_size=1.0
        )
        
        assert size > 0
        
        # Риск должен быть 0.5% от капитала
        risk_amount = 10000 * 0.005
        expected_size = risk_amount / 10  # 10 пунктов до SL
        
        assert abs(size - expected_size) < 0.01
    
    def test_add_atr_to_data(self, atr_manager, sample_price_data):
        """Тест добавления ATR к данным"""
        result = atr_manager.add_atr_to_data(sample_price_data)
        
        assert 'atr' in result.columns
        assert len(result) == len(sample_price_data)
        assert result['atr'].notna().sum() > 0
    
    def test_validate_trade_conditions(self, atr_manager):
        """Тест валидации торговых условий"""
        # Нормальная волатильность
        valid, msg = atr_manager.validate_trade_conditions(
            atr_current=5.0,
            atr_avg=5.0
        )
        assert valid
        
        # Слишком низкая
        valid, msg = atr_manager.validate_trade_conditions(
            atr_current=1.0,
            atr_avg=5.0,
            min_atr_ratio=0.5
        )
        assert not valid
        assert "низкая" in msg.lower()
        
        # Слишком высокая
        valid, msg = atr_manager.validate_trade_conditions(
            atr_current=20.0,
            atr_avg=5.0,
            max_atr_ratio=3.0
        )
        assert not valid
        assert "высокая" in msg.lower()


# ==================== ТЕСТЫ DEGRADATION MONITOR ====================

class TestDegradationMonitor:
    """Тесты системы деградации"""
    
    def test_initialization(self, historical_metrics):
        """Тест инициализации монитора"""
        monitor = DegradationMonitor(historical_metrics)
        
        assert monitor.status == DegradationStatus.HEALTHY
        assert len(monitor.triggers) > 0
        assert monitor.metrics.total_trades == 0
    
    def test_update_winning_trade(self, historical_metrics):
        """Тест обновления после прибыльной сделки"""
        monitor = DegradationMonitor(historical_metrics)
        
        trade = {
            'profit': 10.0,
            'entry_price': 1800,
            'exit_price': 1810,
            'direction': 'buy',
            'timestamp': datetime.now()
        }
        
        status = monitor.update(trade)
        
        assert monitor.metrics.total_trades == 1
        assert monitor.metrics.winning_trades == 1
        assert monitor.metrics.current_winning_streak == 1
        assert monitor.metrics.current_losing_streak == 0
        assert monitor.metrics.total_profit == 10.0
    
    def test_update_losing_trade(self, historical_metrics):
        """Тест обновления после убыточной сделки"""
        monitor = DegradationMonitor(historical_metrics)
        
        trade = {
            'profit': -10.0,
            'entry_price': 1800,
            'exit_price': 1790,
            'direction': 'buy',
            'timestamp': datetime.now()
        }
        
        status = monitor.update(trade)
        
        assert monitor.metrics.total_trades == 1
        assert monitor.metrics.losing_trades == 1
        assert monitor.metrics.current_losing_streak == 1
        assert monitor.metrics.current_winning_streak == 0
        assert monitor.metrics.total_profit == -10.0
    
    def test_losing_streak_trigger(self, historical_metrics):
        """Тест триггера серии убытков"""
        monitor = DegradationMonitor(historical_metrics)
        
        # Симуляция 15 убыточных сделок подряд
        for i in range(15):
            trade = {
                'profit': -10.0,
                'entry_price': 1800,
                'exit_price': 1790,
                'direction': 'buy',
                'timestamp': datetime.now()
            }
            monitor.update(trade)
        
        # После 10 убытков должен сработать триггер
        should_stop, reasons = monitor.should_stop_trading()
        
        assert should_stop
        assert len(reasons) > 0
        assert monitor.status == DegradationStatus.STOPPED
    
    def test_drawdown_calculation(self, historical_metrics):
        """Тест расчета просадки"""
        monitor = DegradationMonitor(historical_metrics)
        
        # Прибыльные сделки
        for i in range(5):
            monitor.update({
                'profit': 10.0,
                'entry_price': 1800,
                'exit_price': 1810,
                'direction': 'buy',
                'timestamp': datetime.now()
            })
        
        peak_equity = monitor.metrics.peak_equity
        
        # Убыточные сделки
        for i in range(3):
            monitor.update({
                'profit': -20.0,
                'entry_price': 1800,
                'exit_price': 1780,
                'direction': 'buy',
                'timestamp': datetime.now()
            })
        
        # Просадка должна быть > 0
        assert monitor.metrics.current_drawdown > 0
        assert monitor.metrics.max_drawdown > 0
        
        # Peak не должен измениться
        assert monitor.metrics.peak_equity == peak_equity
    
    def test_get_health_report(self, historical_metrics):
        """Тест получения отчета"""
        monitor = DegradationMonitor(historical_metrics)
        
        # Несколько сделок
        for i in range(5):
            profit = 10.0 if i % 2 == 0 else -5.0
            monitor.update({
                'profit': profit,
                'entry_price': 1800,
                'exit_price': 1800 + profit,
                'direction': 'buy',
                'timestamp': datetime.now()
            })
        
        report = monitor.get_health_report()
        
        assert 'status' in report
        assert 'current_metrics' in report
        assert 'historical_comparison' in report
        assert 'triggers' in report
        
        assert report['current_metrics']['total_trades'] == 5
    
    def test_save_and_load_state(self, historical_metrics, tmp_path):
        """Тест сохранения и загрузки состояния"""
        monitor = DegradationMonitor(historical_metrics)
        
        # Добавляем сделки
        for i in range(3):
            monitor.update({
                'profit': 10.0,
                'entry_price': 1800,
                'exit_price': 1810,
                'direction': 'buy',
                'timestamp': datetime.now()
            })
        
        # Сохранение
        filepath = tmp_path / "monitor_state.json"
        monitor.save_state(str(filepath))
        
        assert filepath.exists()
        
        # Загрузка
        loaded = DegradationMonitor.load_state(str(filepath))
        
        assert loaded.metrics.total_trades == 3
        assert loaded.metrics.total_profit == 30.0


# ==================== ИНТЕГРАЦИОННЫЕ ТЕСТЫ ====================

class TestIntegration


# ==================== ЗАПУСК ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])