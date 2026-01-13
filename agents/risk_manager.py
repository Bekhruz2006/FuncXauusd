"""
Модуль управления риском и позиционированием.
Эндогенное управление капиталом с адаптацией к волатильности и режимам.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from config.hyperparameters import HYPERPARAMS
from utils.logger import LOGGER


class RiskLevel(Enum):
    """Уровни риска"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class Position:
    """Представление открытой позиции"""
    position_id: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float  # в лотах
    stop_loss: float
    take_profits: List[float]
    partial_exit_fractions: List[float]
    direction: str  # "LONG" or "SHORT"
    
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    bars_held: int = 0
    tp_levels_hit: List[bool] = None
    remaining_size: float = 0.0
    
    def __post_init__(self):
        if self.tp_levels_hit is None:
            self.tp_levels_hit = [False] * len(self.take_profits)
        if self.remaining_size == 0.0:
            self.remaining_size = self.position_size
    
    def update_pnl(self, current_price: float) -> None:
        """Обновление нереализованной прибыли"""
        try:
            self.current_price = current_price
            
            if self.direction == "LONG":
                self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_size
            else:
                self.unrealized_pnl = (self.entry_price - current_price) * self.remaining_size
            
        except Exception as e:
            LOGGER.error(f"Ошибка обновления PnL: {e}")
    
    def to_dict(self) -> Dict:
        """Сериализация позиции"""
        return {
            'position_id': self.position_id,
            'entry_time': str(self.entry_time),
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profits': self.take_profits,
            'direction': self.direction,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'bars_held': self.bars_held,
            'remaining_size': self.remaining_size
        }


class RiskManager:
    """Менеджер риска с эндогенным управлением капиталом"""
    
    def __init__(self,
                 initial_capital: float,
                 max_drawdown: float = 0.15,
                 max_risk_per_trade: float = 0.025,
                 max_concurrent_positions: int = 3):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.max_risk_per_trade = max_risk_per_trade
        self.max_concurrent_positions = max_concurrent_positions
        
        self.equity_curve: List[float] = [initial_capital]
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        
        self.open_positions: List[Position] = []
        self.closed_positions: List[Dict] = []
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self.risk_adjustment_factor = 1.0
        
        LOGGER.info(f"Инициализация RiskManager: капитал={initial_capital}, max_dd={max_drawdown}")
    
    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss: float,
                                risk_percent: float,
                                atr: Optional[float] = None,
                                volatility_regime: str = "medium") -> float:
        """
        Расчет размера позиции на основе риска.
        
        Args:
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            risk_percent: Процент риска от капитала
            atr: Average True Range
            volatility_regime: Режим волатильности
        
        Returns:
            Размер позиции в лотах
        """
        try:
            # Валидация входных данных
            if entry_price <= 0 or stop_loss <= 0:
                LOGGER.error("Некорректные цены для расчета размера позиции")
                return 0.0
            
            if stop_loss >= entry_price:
                LOGGER.error("Stop loss должен быть ниже entry price для LONG")
                return 0.0
            
            # Корректировка риска по режиму волатильности
            volatility_multipliers = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.6
            }
            
            vol_multiplier = volatility_multipliers.get(volatility_regime, 1.0)
            adjusted_risk_percent = risk_percent * vol_multiplier * self.risk_adjustment_factor
            
            # Ограничение максимального риска
            adjusted_risk_percent = min(adjusted_risk_percent, self.max_risk_per_trade)
            
            # Расчет риска в долларах
            risk_amount = self.current_capital * adjusted_risk_percent
            
            # Расчет риска на 1 лот
            risk_per_lot = abs(entry_price - stop_loss)
            
            if risk_per_lot < 1e-6:
                LOGGER.error("Слишком узкий stop loss")
                return 0.0
            
            # Размер позиции
            position_size = risk_amount / risk_per_lot
            
            # Применение ограничений
            min_lot = HYPERPARAMS.risk.min_lot_size
            max_lot = HYPERPARAMS.risk.max_lot_size
            
            position_size = np.clip(position_size, min_lot, max_lot)
            
            # Округление до допустимого шага (0.01 лота)
            position_size = round(position_size / 0.01) * 0.01
            
            LOGGER.debug(f"Размер позиции: {position_size:.2f} лотов, риск: {adjusted_risk_percent*100:.2f}%")
            
            return position_size
            
        except Exception as e:
            LOGGER.error(f"Ошибка расчета размера позиции: {e}", exc_info=True)
            return HYPERPARAMS.risk.min_lot_size
    
    def can_open_position(self) -> Tuple[bool, str]:
        """
        Проверка возможности открытия новой позиции.
        
        Returns:
            (can_open, reason)
        """
        try:
            # Проверка максимального количества позиций
            if len(self.open_positions) >= self.max_concurrent_positions:
                return False, f"Достигнут лимит позиций: {self.max_concurrent_positions}"
            
            # Проверка текущей просадки
            if self.current_drawdown >= self.max_drawdown:
                return False, f"Превышена максимальная просадка: {self.current_drawdown*100:.2f}%"
            
            # Проверка достаточности капитала
            if self.current_capital < self.initial_capital * 0.5:
                return False, "Критически низкий капитал (< 50% от начального)"
            
            # Проверка risk adjustment factor
            if self.risk_adjustment_factor < 0.3:
                return False, "Риск-фактор слишком низкий (защита от серии убытков)"
            
            return True, "OK"
            
        except Exception as e:
            LOGGER.error(f"Ошибка проверки возможности входа: {e}")
            return False, f"Ошибка: {e}"
    
    def open_position(self,
                     position_id: str,
                     entry_time: pd.Timestamp,
                     entry_price: float,
                     position_size: float,
                     stop_loss: float,
                     take_profits: List[float],
                     partial_exit_fractions: List[float],
                     direction: str = "LONG") -> Optional[Position]:
        """Открытие новой позиции"""
        try:
            can_open, reason = self.can_open_position()
            
            if not can_open:
                LOGGER.warning(f"Невозможно открыть позицию: {reason}")
                return None
            
            position = Position(
                position_id=position_id,
                entry_time=entry_time,
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profits=take_profits,
                partial_exit_fractions=partial_exit_fractions,
                direction=direction
            )
            
            self.open_positions.append(position)
            self.total_trades += 1
            
            LOGGER.info(f"Открыта позиция {position_id}: {direction} {position_size} @ {entry_price}")
            
            return position
            
        except Exception as e:
            LOGGER.error(f"Ошибка открытия позиции: {e}", exc_info=True)
            return None
    
    def close_position(self,
                      position: Position,
                      exit_time: pd.Timestamp,
                      exit_price: float,
                      exit_reason: str,
                      partial: bool = False,
                      exit_fraction: float = 1.0) -> Dict:
        """Закрытие позиции (полное или частичное)"""
        try:
            if partial:
                closed_size = position.remaining_size * exit_fraction
                position.remaining_size -= closed_size
            else:
                closed_size = position.remaining_size
                position.remaining_size = 0.0
            
            # Расчет прибыли/убытка
            if position.direction == "LONG":
                pnl = (exit_price - position.entry_price) * closed_size
            else:
                pnl = (position.entry_price - exit_price) * closed_size
            
            # Учет комиссий и спреда
            spread_cost = HYPERPARAMS.risk.spread_points * closed_size * 0.1  # упрощенный расчет
            total_pnl = pnl - spread_cost
            
            # Обновление капитала
            self.current_capital += total_pnl
            self.equity_curve.append(self.current_capital)
            
            # Обновление пика и просадки
            if self.current_capital > self.peak_equity:
                self.peak_equity = self.current_capital
            
            self.current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
            
            # Статистика
            if total_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Адаптация риск-фактора
            self._adjust_risk_factor(total_pnl)
            
            # Запись закрытой позиции
            closed_trade = {
                'position_id': position.position_id,
                'entry_time': position.entry_time,
                'exit_time': exit_time,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'direction': position.direction,
                'size': closed_size,
                'pnl': total_pnl,
                'exit_reason': exit_reason,
                'bars_held': position.bars_held,
                'partial': partial
            }
            
            self.closed_positions.append(closed_trade)
            
            # Удаление полностью закрытой позиции
            if not partial or position.remaining_size < HYPERPARAMS.risk.min_lot_size:
                if position in self.open_positions:
                    self.open_positions.remove(position)
            
            LOGGER.info(f"Закрыта позиция {position.position_id}: PnL={total_pnl:.2f}, "
                       f"причина={exit_reason}, partial={partial}")
            
            return closed_trade
            
        except Exception as e:
            LOGGER.error(f"Ошибка закрытия позиции: {e}", exc_info=True)
            return {}
    
    def _adjust_risk_factor(self, pnl: float) -> None:
        """Адаптация risk adjustment factor на основе результатов"""
        try:
            # Увеличиваем после прибыльных сделок
            if pnl > 0:
                self.risk_adjustment_factor = min(self.risk_adjustment_factor * 1.05, 1.5)
            else:
                # Снижаем после убыточных
                self.risk_adjustment_factor = max(self.risk_adjustment_factor * 0.9, 0.3)
            
            # Дополнительное снижение при высокой просадке
            if self.current_drawdown > self.max_drawdown * 0.7:
                self.risk_adjustment_factor *= 0.8
            
        except Exception as e:
            LOGGER.error(f"Ошибка адаптации риск-фактора: {e}")
    
    def update_positions(self, current_time: pd.Timestamp, current_price: float) -> None:
        """Обновление всех открытых позиций"""
        try:
            for position in self.open_positions:
                position.update_pnl(current_price)
                position.bars_held += 1
            
        except Exception as e:
            LOGGER.error(f"Ошибка обновления позиций: {e}")
    
    def get_current_exposure(self) -> float:
        """Текущая экспозиция (сумма открытых позиций)"""
        try:
            total_exposure = sum(
                pos.remaining_size * pos.current_price
                for pos in self.open_positions
            )
            return total_exposure
            
        except Exception as e:
            LOGGER.error(f"Ошибка расчета экспозиции: {e}")
            return 0.0
    
    def get_total_unrealized_pnl(self) -> float:
        """Суммарная нереализованная прибыль/убыток"""
        try:
            return sum(pos.unrealized_pnl for pos in self.open_positions)
        except Exception:
            return 0.0
    
    def get_statistics(self) -> Dict:
        """Получение статистики риск-менеджера"""
        try:
            win_rate = self.winning_trades / (self.total_trades + 1e-8)
            
            realized_pnls = [trade['pnl'] for trade in self.closed_positions]
            
            if realized_pnls:
                avg_win = np.mean([pnl for pnl in realized_pnls if pnl > 0]) if any(pnl > 0 for pnl in realized_pnls) else 0
                avg_loss = np.mean([pnl for pnl in realized_pnls if pnl < 0]) if any(pnl < 0 for pnl in realized_pnls) else 0
                
                profit_factor = abs(sum([pnl for pnl in realized_pnls if pnl > 0]) / 
                                   (sum([pnl for pnl in realized_pnls if pnl < 0]) + 1e-8))
            else:
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            stats = {
                'current_capital': self.current_capital,
                'initial_capital': self.initial_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
                'peak_equity': self.peak_equity,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': max(self.current_drawdown, 
                                   max([0] + [(self.peak_equity - eq) / self.peak_equity 
                                             for eq in self.equity_curve])),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'open_positions': len(self.open_positions),
                'risk_adjustment_factor': self.risk_adjustment_factor,
                'total_unrealized_pnl': self.get_total_unrealized_pnl()
            }
            
            return stats
            
        except Exception as e:
            LOGGER.error(f"Ошибка получения статистики: {e}")
            return {}
    
    def reset(self) -> None:
        """Сброс состояния риск-менеджера"""
        try:
            self.current_capital = self.initial_capital
            self.equity_curve = [self.initial_capital]
            self.peak_equity = self.initial_capital
            self.current_drawdown = 0.0
            
            self.open_positions = []
            self.closed_positions = []
            
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            
            self.risk_adjustment_factor = 1.0
            
            LOGGER.debug("RiskManager сброшен")
            
        except Exception as e:
            LOGGER.error(f"Ошибка сброса RiskManager: {e}")


class PositionSizer:
    """Специализированный класс для различных методов sizing"""
    
    @staticmethod
    def fixed_fractional(capital: float,
                        risk_percent: float,
                        entry_price: float,
                        stop_loss: float) -> float:
        """Fixed Fractional position sizing"""
        try:
            risk_amount = capital * risk_percent
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit < 1e-6:
                return 0.0
            
            position_size = risk_amount / risk_per_unit
            return position_size
            
        except Exception as e:
            LOGGER.error(f"Ошибка fixed fractional sizing: {e}")
            return 0.0
    
    @staticmethod
    def kelly_criterion(win_rate: float,
                       avg_win: float,
                       avg_loss: float,
                       capital: float,
                       entry_price: float,
                       stop_loss: float,
                       fraction: float = 0.5) -> float:
        """
        Kelly Criterion position sizing.
        
        Args:
            win_rate: Процент выигрышных сделок
            avg_win: Средний выигрыш
            avg_loss: Средний убыток (положительное число)
            capital: Текущий капитал
            entry_price: Цена входа
            stop_loss: Стоп-лосс
            fraction: Доля Kelly (обычно 0.25-0.5 для безопасности)
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            # Kelly % = (W * R - L) / R
            # где W = win_rate, R = avg_win/avg_loss, L = 1 - W
            win_loss_ratio = avg_win / avg_loss
            kelly_percent = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Ограничиваем Kelly и применяем fractional
            kelly_percent = max(0, min(kelly_percent, 0.25))
            fractional_kelly = kelly_percent * fraction
            
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit < 1e-6:
                return 0.0
            
            position_size = (capital * fractional_kelly) / risk_per_unit
            
            return position_size
            
        except Exception as e:
            LOGGER.error(f"Ошибка Kelly Criterion sizing: {e}")
            return 0.0
    
    @staticmethod
    def volatility_adjusted(capital: float,
                           risk_percent: float,
                           entry_price: float,
                           stop_loss: float,
                           current_volatility: float,
                           baseline_volatility: float) -> float:
        """Position sizing с адаптацией к волатильности"""
        try:
            # Базовый размер
            base_size = PositionSizer.fixed_fractional(
                capital, risk_percent, entry_price, stop_loss
            )
            
            # Корректировка на волатильность
            volatility_ratio = baseline_volatility / (current_volatility + 1e-8)
            adjusted_size = base_size * volatility_ratio
            
            # Ограничение корректировки
            adjusted_size = np.clip(adjusted_size, base_size * 0.5, base_size * 1.5)
            
            return adjusted_size
            
        except Exception as e:
            LOGGER.error(f"Ошибка volatility adjusted sizing: {e}")
            return 0.0
    
    @staticmethod
    def optimal_f(trade_results: List[float],
                 capital: float,
                 num_periods: int = 100) -> float:
        """
        Optimal F method (Ralph Vince).
        
        Args:
            trade_results: История результатов сделок
            capital: Текущий капитал
            num_periods: Количество периодов для расчета
        """
        try:
            if not trade_results or len(trade_results) < 10:
                return 0.01  # Минимальная f
            
            # Используем только последние n сделок
            recent_results = trade_results[-num_periods:]
            
            # Largest loss
            largest_loss = abs(min(recent_results))
            
            if largest_loss < 1e-6:
                return 0.01
            
            # Поиск оптимальной f
            best_f = 0.01
            best_twi = -np.inf  # Terminal Wealth Index
            
            for f in np.arange(0.01, 0.5, 0.01):
                hprs = []  # Holding Period Returns
                
                for result in recent_results:
                    hpr = 1 + (f * result / largest_loss)
                    hprs.append(max(hpr, 0.01))  # Предотвращаем отрицательные
                
                # TWI = произведение всех HPR
                twi = np.prod(hprs)
                
                if twi > best_twi:
                    best_twi = twi
                    best_f = f
            
            # Консервативная корректировка (используем 50% optimal f)
            safe_f = best_f * 0.5
            
            return safe_f
            
        except Exception as e:
            LOGGER.error(f"Ошибка Optimal F sizing: {e}")
            return 0.01


class DrawdownProtector:
    """Защита от просадок с динамическим управлением риском"""
    
    def __init__(self,
                 max_drawdown: float = 0.15,
                 warning_threshold: float = 0.10):
        
        self.max_drawdown = max_drawdown
        self.warning_threshold = warning_threshold
        
        self.protection_active = False
        self.protection_triggered_at = None
    
    def check_drawdown(self, current_drawdown: float) -> Tuple[bool, str, float]:
        """
        Проверка уровня просадки.
        
        Returns:
            (should_reduce_risk, message, risk_multiplier)
        """
        try:
            if current_drawdown >= self.max_drawdown:
                self.protection_active = True
                return True, "КРИТИЧЕСКАЯ ПРОСАДКА - торговля заблокирована", 0.0
            
            elif current_drawdown >= self.warning_threshold:
                risk_multiplier = 1.0 - (current_drawdown / self.max_drawdown)
                return True, f"Предупреждение о просадке: {current_drawdown*100:.1f}%", risk_multiplier
            
            else:
                self.protection_active = False
                return False, "Просадка в норме", 1.0
                
        except Exception as e:
            LOGGER.error(f"Ошибка проверки просадки: {e}")
            return True, "Ошибка проверки", 0.5
    
    def get_allowed_risk(self,
                        base_risk: float,
                        current_drawdown: float) -> float:
        """Расчет допустимого риска с учетом просадки"""
        try:
            should_reduce, _, multiplier = self.check_drawdown(current_drawdown)
            
            if should_reduce:
                return base_risk * multiplier
            
            return base_risk
            
        except Exception as e:
            LOGGER.error(f"Ошибка расчета допустимого риска: {e}")
            return base_risk * 0.5


def calculate_optimal_stop_loss(entry_price: float,
                                atr: float,
                                atr_multiplier: float = 2.0,
                                min_risk_percent: float = 0.005,
                                max_risk_percent: float = 0.03) -> float:
    """
    Расчет оптимального стоп-лосса на основе ATR.
    
    Args:
        entry_price: Цена входа
        atr: Average True Range
        atr_multiplier: Множитель ATR
        min_risk_percent: Минимальный процент риска от цены
        max_risk_percent: Максимальный процент риска от цены
    
    Returns:
        Уровень стоп-лосса
    """
    try:
        # Базовый SL на основе ATR
        stop_distance = atr * atr_multiplier
        stop_loss = entry_price - stop_distance
        
        # Проверка минимального/максимального риска
        risk_percent = stop_distance / entry_price
        
        if risk_percent < min_risk_percent:
            stop_loss = entry_price * (1 - min_risk_percent)
        elif risk_percent > max_risk_percent:
            stop_loss = entry_price * (1 - max_risk_percent)
        
        return stop_loss
        
    except Exception as e:
        LOGGER.error(f"Ошибка расчета stop loss: {e}")
        return entry_price * 0.98  # Дефолтный SL 2%


def calculate_dynamic_take_profits(entry_price: float,
                                   atr: float,
                                   num_levels: int = 3,
                                   min_rr_ratio: float = 1.5,
                                   max_rr_ratio: float = 4.0) -> List[float]:
    """
    Расчет динамических уровней тейк-профита.
    
    Args:
        entry_price: Цена входа
        atr: Average True Range
        num_levels: Количество уровней TP
        min_rr_ratio: Минимальное соотношение риск/прибыль
        max_rr_ratio: Максимальное соотношение риск/прибыль
    
    Returns:
        Список уровней тейк-профита
    """
    try:
        take_profits = []
        
        # Равномерное распределение между min и max RR
        rr_ratios = np.linspace(min_rr_ratio, max_rr_ratio, num_levels)
        
        for rr in rr_ratios:
            tp_distance = atr * rr
            tp_level = entry_price + tp_distance
            take_profits.append(tp_level)
        
        return take_profits
        
    except Exception as e:
        LOGGER.error(f"Ошибка расчета take profits: {e}")
        return [entry_price * 1.02, entry_price * 1.04, entry_price * 1.06]