"""
Система мониторинга деградации модели

Реализация Этапа 6 из implementation_plan.md:
    - Отслеживание live-метрик в реальном времени
    - Автоматическая остановка при превышении порогов
    - Триггеры на серии убытков, просадку, падение прибыли
    - Требование полной перетренировки при деградации

Философия:
    Любая модель устаревает. Система деградации - это защита от "тихой смерти",
    когда убытки накапливаются медленно. Лучше остановиться и перетренироваться,
    чем продолжать торговать устаревшей моделью.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path


class DegradationStatus(Enum):
    """Статус системы деградации"""
    HEALTHY = "healthy"           # Все в порядке
    WARNING = "warning"           # Предупреждение
    CRITICAL = "critical"         # Критическое состояние
    STOPPED = "stopped"           # Торговля остановлена


@dataclass
class DegradationTrigger:
    """Триггер деградации"""
    name: str
    triggered: bool = False
    value: float = 0.0
    threshold: float = 0.0
    timestamp: Optional[datetime] = None
    message: str = ""


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    # Кумулятивные
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    
    # Серии
    current_losing_streak: int = 0
    max_losing_streak: int = 0
    current_winning_streak: int = 0
    
    # Просадка
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Прибыльность
    avg_profit_per_trade: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # История
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    
    # Временные метки
    last_trade_time: Optional[datetime] = None
    monitoring_start: Optional[datetime] = None


class DegradationMonitor:
    """
    Монитор деградации модели
    
    Отслеживает критические метрики и останавливает торговлю
    при обнаружении деградации модели.
    
    Триггеры остановки:
        1. Просадка > 120% от исторической
        2. Серия убытков >= 10 сделок
        3. Падение прибыли < 50% от ожидаемой
        4. Резкое падение Win Rate
        5. Негативный Profit Factor
    
    Attributes:
        config: Конфигурация триггеров
        metrics: Текущие метрики производительности
        triggers: Список триггеров деградации
        status: Текущий статус системы
    """
    
    def __init__(self,
                 historical_metrics: Dict,
                 config: Optional[Dict] = None):
        """
        Args:
            historical_metrics: Исторические метрики с OOT теста:
                {
                    'max_drawdown': 0.08,
                    'win_rate': 0.58,
                    'avg_profit_per_trade': 12.5,
                    'profit_factor': 1.5
                }
            config: Дополнительная конфигурация триггеров
        """
        self.historical_metrics = historical_metrics
        self.config = self._default_config()
        
        if config:
            self.config.update(config)
        
        self.metrics = PerformanceMetrics(
            monitoring_start=datetime.now()
        )
        
        self.triggers: List[DegradationTrigger] = []
        self.status = DegradationStatus.HEALTHY
        self._initialize_triggers()
    
    def _default_config(self) -> Dict:
        """Конфигурация по умолчанию"""
        return {
            # Триггер 1: Просадка
            'max_drawdown_multiplier': 1.2,  # 120% от исторической
            
            # Триггер 2: Серии убытков
            'max_losing_streak': 10,
            
            # Триггер 3: Падение прибыли
            'min_profit_ratio': 0.5,  # 50% от ожидаемой
            
            # Триггер 4: Win Rate
            'min_winrate_ratio': 0.7,  # 70% от исторического
            
            # Триггер 5: Profit Factor
            'min_profit_factor': 1.0,
            
            # Временные ограничения
            'min_trades_for_eval': 30,  # Минимум сделок для оценки
            
            # Логирование
            'log_path': './logs/degradation.log'
        }
    
    def _initialize_triggers(self) -> None:
        """Инициализация триггеров"""
        hist = self.historical_metrics
        cfg = self.config
        
        # Триггер 1: Просадка
        self.triggers.append(DegradationTrigger(
            name='max_drawdown',
            threshold=hist.get('max_drawdown', 0.1) * cfg['max_drawdown_multiplier'],
            message=f"Просадка превысила {cfg['max_drawdown_multiplier']:.0%} "
                   f"от исторической"
        ))
        
        # Триггер 2: Серия убытков
        self.triggers.append(DegradationTrigger(
            name='losing_streak',
            threshold=cfg['max_losing_streak'],
            message=f"Серия убытков достигла {cfg['max_losing_streak']} сделок"
        ))
        
        # Триггер 3: Падение прибыли
        expected_profit = hist.get('avg_profit_per_trade', 0)
        self.triggers.append(DegradationTrigger(
            name='profit_decline',
            threshold=expected_profit * cfg['min_profit_ratio'],
            message=f"Прибыль упала ниже {cfg['min_profit_ratio']:.0%} "
                   f"от ожидаемой"
        ))
        
        # Триггер 4: Win Rate
        expected_wr = hist.get('win_rate', 0.5)
        self.triggers.append(DegradationTrigger(
            name='winrate_decline',
            threshold=expected_wr * cfg['min_winrate_ratio'],
            message=f"Win Rate упал ниже {cfg['min_winrate_ratio']:.0%} "
                   f"от исторического"
        ))
        
        # Триггер 5: Profit Factor
        self.triggers.append(DegradationTrigger(
            name='negative_pf',
            threshold=cfg['min_profit_factor'],
            message=f"Profit Factor ниже {cfg['min_profit_factor']}"
        ))
    
    def update(self, trade_result: Dict) -> DegradationStatus:
        """
        Обновление метрик после сделки
        
        Args:
            trade_result: Результат сделки:
                {
                    'profit': float,
                    'entry_price': float,
                    'exit_price': float,
                    'direction': str,
                    'timestamp': datetime
                }
        
        Returns:
            DegradationStatus: Текущий статус системы
        
        Side Effects:
            - Обновляет self.metrics
            - Проверяет триггеры
            - Логирует события
        """
        profit = trade_result['profit']
        timestamp = trade_result.get('timestamp', datetime.now())
        
        # Обновление базовых метрик
        self.metrics.total_trades += 1
        self.metrics.total_profit += profit
        self.metrics.last_trade_time = timestamp
        
        if profit > 0:
            self.metrics.winning_trades += 1
            self.metrics.current_winning_streak += 1
            self.metrics.current_losing_streak = 0
        else:
            self.metrics.losing_trades += 1
            self.metrics.current_losing_streak += 1
            self.metrics.current_winning_streak = 0
            
            # Обновление максимальной серии убытков
            if self.metrics.current_losing_streak > self.metrics.max_losing_streak:
                self.metrics.max_losing_streak = self.metrics.current_losing_streak
        
        # Обновление equity
        self.metrics.current_equity += profit
        self.metrics.equity_curve.append(self.metrics.current_equity)
        
        # Обновление просадки
        if self.metrics.current_equity > self.metrics.peak_equity:
            self.metrics.peak_equity = self.metrics.current_equity
        
        self.metrics.current_drawdown = (
            (self.metrics.peak_equity - self.metrics.current_equity) /
            max(self.metrics.peak_equity, 1)
        )
        
        if self.metrics.current_drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = self.metrics.current_drawdown
        
        # Обновление агрегированных метрик
        self._update_aggregate_metrics()
        
        # История
        self.metrics.trade_history.append({
            **trade_result,
            'equity': self.metrics.current_equity,
            'drawdown': self.metrics.current_drawdown
        })
        
        # Проверка триггеров
        self._check_triggers()
        
        # Логирование
        self._log_update(trade_result)
        
        return self.status
    
    def _update_aggregate_metrics(self) -> None:
        """Обновление агрегированных метрик"""
        if self.metrics.total_trades == 0:
            return
        
        # Win Rate
        self.metrics.win_rate = (
            self.metrics.winning_trades / self.metrics.total_trades
        )
        
        # Средняя прибыль
        self.metrics.avg_profit_per_trade = (
            self.metrics.total_profit / self.metrics.total_trades
        )
        
        # Profit Factor
        winning_profit = sum(
            t['profit'] for t in self.metrics.trade_history
            if t['profit'] > 0
        ) if self.metrics.trade_history else 0
        
        losing_profit = abs(sum(
            t['profit'] for t in self.metrics.trade_history
            if t['profit'] < 0
        )) if self.metrics.trade_history else 1
        
        self.metrics.profit_factor = (
            winning_profit / losing_profit if losing_profit > 0 else 0
        )
    
    def _check_triggers(self) -> None:
        """Проверка всех триггеров"""
        # Проверяем только если достаточно сделок
        if self.metrics.total_trades < self.config['min_trades_for_eval']:
            return
        
        any_triggered = False
        
        for trigger in self.triggers:
            if trigger.name == 'max_drawdown':
                trigger.value = self.metrics.current_drawdown
                if self.metrics.current_drawdown > trigger.threshold:
                    trigger.triggered = True
                    trigger.timestamp = datetime.now()
                    any_triggered = True
            
            elif trigger.name == 'losing_streak':
                trigger.value = self.metrics.current_losing_streak
                if self.metrics.current_losing_streak >= trigger.threshold:
                    trigger.triggered = True
                    trigger.timestamp = datetime.now()
                    any_triggered = True
            
            elif trigger.name == 'profit_decline':
                trigger.value = self.metrics.avg_profit_per_trade
                if self.metrics.avg_profit_per_trade < trigger.threshold:
                    trigger.triggered = True
                    trigger.timestamp = datetime.now()
                    any_triggered = True
            
            elif trigger.name == 'winrate_decline':
                trigger.value = self.metrics.win_rate
                if self.metrics.win_rate < trigger.threshold:
                    trigger.triggered = True
                    trigger.timestamp = datetime.now()
                    any_triggered = True
            
            elif trigger.name == 'negative_pf':
                trigger.value = self.metrics.profit_factor
                if self.metrics.profit_factor < trigger.threshold:
                    trigger.triggered = True
                    trigger.timestamp = datetime.now()
                    any_triggered = True
        
        # Обновление статуса
        if any_triggered:
            # Если хотя бы один критический триггер - остановка
            critical_triggers = ['max_drawdown', 'losing_streak']
            if any(t.triggered and t.name in critical_triggers 
                   for t in self.triggers):
                self.status = DegradationStatus.STOPPED
            else:
                self.status = DegradationStatus.WARNING
        else:
            self.status = DegradationStatus.HEALTHY
    
    def should_stop_trading(self) -> Tuple[bool, List[str]]:
        """
        Проверка необходимости остановки торговли
        
        Returns:
            (should_stop, reasons): Нужно ли останавливаться и причины
        """
        if self.status == DegradationStatus.STOPPED:
            triggered = [t for t in self.triggers if t.triggered]
            reasons = [t.message for t in triggered]
            return True, reasons
        
        return False, []
    
    def get_health_report(self) -> Dict:
        """
        Детальный отчет о состоянии системы
        
        Returns:
            dict: Полный отчет с метриками и триггерами
        """
        return {
            'status': self.status.value,
            'monitoring_duration': (
                datetime.now() - self.metrics.monitoring_start
            ).total_seconds() / 3600 if self.metrics.monitoring_start else 0,
            
            'current_metrics': {
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'avg_profit': self.metrics.avg_profit_per_trade,
                'current_drawdown': self.metrics.current_drawdown,
                'max_drawdown': self.metrics.max_drawdown,
                'losing_streak': self.metrics.current_losing_streak,
                'profit_factor': self.metrics.profit_factor
            },
            
            'historical_comparison': {
                'drawdown_ratio': (
                    self.metrics.current_drawdown /
                    max(self.historical_metrics.get('max_drawdown', 0.1), 0.01)
                ),
                'winrate_ratio': (
                    self.metrics.win_rate /
                    max(self.historical_metrics.get('win_rate', 0.5), 0.01)
                ),
                'profit_ratio': (
                    self.metrics.avg_profit_per_trade /
                    max(self.historical_metrics.get('avg_profit_per_trade', 1), 0.01)
                )
            },
            
            'triggers': [
                {
                    'name': t.name,
                    'triggered': t.triggered,
                    'value': t.value,
                    'threshold': t.threshold,
                    'message': t.message
                }
                for t in self.triggers
            ]
        }
    
    def _log_update(self, trade_result: Dict) -> None:
        """Логирование обновления"""
        log_path = Path(self.config['log_path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'trade': trade_result,
            'status': self.status.value,
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'current_drawdown': self.metrics.current_drawdown,
                'losing_streak': self.metrics.current_losing_streak,
                'equity': self.metrics.current_equity
            }
        }
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def print_status(self) -> None:
        """Вывод текущего статуса в консоль"""
        report = self.get_health_report()
        
        # Иконки статуса
        status_icons = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '🔴',
            'stopped': '🛑'
        }
        
        print(f"\n{'='*70}")
        print(f"  {status_icons[report['status']]} DEGRADATION MONITOR STATUS")
        print(f"{'='*70}")
        
        print(f"\n📊 Текущие метрики:")
        cm = report['current_metrics']
        print(f"  • Сделок: {cm['total_trades']}")
        print(f"  • Win Rate: {cm['win_rate']:.2%}")
        print(f"  • Средняя прибыль: {cm['avg_profit']:.2f}")
        print(f"  • Просадка: {cm['current_drawdown']:.2%} "
              f"(макс: {cm['max_drawdown']:.2%})")
        print(f"  • Серия убытков: {cm['losing_streak']}")
        print(f"  • Profit Factor: {cm['profit_factor']:.2f}")
        
        print(f"\n📈 Сравнение с историей:")
        hc = report['historical_comparison']
        print(f"  • Просадка: {hc['drawdown_ratio']:.2f}x от исторической")
        print(f"  • Win Rate: {hc['winrate_ratio']:.2f}x от исторического")
        print(f"  • Прибыль: {hc['profit_ratio']:.2f}x от ожидаемой")
        
        print(f"\n🎯 Триггеры:")
        for t in report['triggers']:
            icon = '🔴' if t['triggered'] else '✅'
            print(f"  {icon} {t['name']}: {t['value']:.4f} "
                  f"(порог: {t['threshold']:.4f})")
        
        if self.status == DegradationStatus.STOPPED:
            print(f"\n{'='*70}")
            print(f"  🛑 ТОРГОВЛЯ ОСТАНОВЛЕНА!")
            print(f"  Требуется полная перетренировка модели.")
            print(f"{'='*70}")
        
        print()
    
    def save_state(self, filepath: str) -> None:
        """Сохранение состояния монитора"""
        state = {
            'config': self.config,
            'historical_metrics': self.historical_metrics,
            'status': self.status.value,
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'total_profit': self.metrics.total_profit,
                'current_losing_streak': self.metrics.current_losing_streak,
                'max_losing_streak': self.metrics.max_losing_streak,
                'peak_equity': self.metrics.peak_equity,
                'current_equity': self.metrics.current_equity,
                'current_drawdown': self.metrics.current_drawdown,
                'max_drawdown': self.metrics.max_drawdown,
                'equity_curve': self.metrics.equity_curve,
                'trade_history': self.metrics.trade_history
            },
            'triggers': [
                {
                    'name': t.name,
                    'triggered': t.triggered,
                    'value': t.value,
                    'threshold': t.threshold,
                    'message': t.message
                }
                for t in self.triggers
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✓ Состояние сохранено: {filepath}")
    
    @classmethod
    def load_state(cls, filepath: str) -> 'DegradationMonitor':
        """Загрузка состояния монитора"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        monitor = cls(state['historical_metrics'], state['config'])
        monitor.status = DegradationStatus(state['status'])
        
        # Восстановление метрик
        m = state['metrics']
        monitor.metrics.total_trades = m['total_trades']
        monitor.metrics.winning_trades = m['winning_trades']
        monitor.metrics.losing_trades = m['losing_trades']
        monitor.metrics.total_profit = m['total_profit']
        monitor.metrics.current_losing_streak = m['current_losing_streak']
        monitor.metrics.max_losing_streak = m['max_losing_streak']
        monitor.metrics.peak_equity = m['peak_equity']
        monitor.metrics.current_equity = m['current_equity']
        monitor.metrics.current_drawdown = m['current_drawdown']
        monitor.metrics.max_drawdown = m['max_drawdown']
        monitor.metrics.equity_curve = m['equity_curve']
        monitor.metrics.trade_history = m['trade_history']
        
        print(f"✓ Состояние загружено: {filepath}")
        
        return monitor


# === ИНТЕГРАЦИЯ С LIVE ТОРГОВЛЕЙ ===

class LiveTradingController:
    """
    Контроллер live-торговли с мониторингом деградации
    
    Обертка для торговой системы с автоматическим контролем
    """
    
    def __init__(self,
                 model,
                 historical_metrics: Dict,
                 config: Optional[Dict] = None):
        self.model = model
        self.monitor = DegradationMonitor(historical_metrics, config)
        self.is_active = True
    
    def execute_trade(self, trade_signal: Dict) -> Optional[Dict]:
        """
        Выполнение сделки с проверкой деградации
        
        Args:
            trade_signal: Сигнал на сделку
        
        Returns:
            dict: Результат сделки или None если остановлено
        """
        # Проверка статуса
        should_stop, reasons = self.monitor.should_stop_trading()
        
        if should_stop:
            print(f"\n🛑 ТОРГОВЛЯ ОСТАНОВЛЕНА!")
            print(f"Причины:")
            for reason in reasons:
                print(f"  • {reason}")
            
            self.is_active = False
            self.monitor.save_state('./logs/monitor_stopped.json')
            return None
        
        # Выполнение сделки (здесь должна быть реальная логика)
        # trade_result = self._execute_real_trade(trade_signal)
        trade_result = trade_signal  # Заглушка
        
        # Обновление монитора
        status = self.monitor.update(trade_result)
        
        if status == DegradationStatus.WARNING:
            print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Обнаружены признаки деградации")
            self.monitor.print_status()
        
        return trade_result
    
    def get_status(self) -> Dict:
        """Получение статуса системы"""
        return {
            'is_active': self.is_active,
            'health_report': self.monitor.get_health_report()
        }