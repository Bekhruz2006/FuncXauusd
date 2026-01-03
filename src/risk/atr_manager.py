import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from numba import jit


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


class ATRRiskManager:
    def __init__(
        self,
        sl_multiplier: float = 2.0,
        tp_multiplier: float = 2.5,
        risk_per_trade: float = 0.005,
        atr_period: int = 14,
        max_bars_timeout: int = 20
    ):
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.risk_per_trade = risk_per_trade
        self.atr_period = atr_period
        self.max_bars_timeout = max_bars_timeout
        
    def calculate_levels(
        self,
        entry_price: float,
        atr_value: float,
        direction: str
    ) -> Dict:
        if direction == 'buy':
            sl = entry_price - (self.sl_multiplier * atr_value)
            tp = entry_price + (self.tp_multiplier * atr_value)
        else:
            sl = entry_price + (self.sl_multiplier * atr_value)
            tp = entry_price - (self.tp_multiplier * atr_value)
        
        risk_points = abs(entry_price - sl)
        reward_points = abs(tp - entry_price)
        
        return {
            'sl': sl,
            'tp': tp,
            'risk_points': risk_points,
            'reward_points': reward_points,
            'risk_reward_ratio': reward_points / risk_points if risk_points > 0 else 0
        }
    
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        contract_size: float = 1.0
    ) -> float:
        risk_amount = capital * self.risk_per_trade
        sl_distance = abs(entry_price - stop_loss)
        
        if sl_distance == 0:
            return 0.0
        
        position_size = risk_amount / (sl_distance * contract_size)
        
        return position_size
    
    def add_atr_to_data(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result['atr'] = calculate_atr(result, self.atr_period)
        return result
    
    def validate_trade_conditions(
        self,
        atr_current: float,
        atr_avg: float,
        min_atr_ratio: float = 0.5,
        max_atr_ratio: float = 2.0
    ) -> Tuple[bool, str]:
        ratio = atr_current / atr_avg if atr_avg > 0 else 0
        
        if ratio < min_atr_ratio:
            return False, f"Слишком низкая волатильность: {ratio:.2f}x"
        
        if ratio > max_atr_ratio:
            return False, f"Слишком высокая волатильность: {ratio:.2f}x"
        
        return True, "OK"


@jit(nopython=True)
def simulate_trade_with_atr(
    entry_price: float,
    sl: float,
    tp: float,
    future_prices: np.ndarray,
    max_bars: int
) -> Tuple[float, int, str]:
    for i in range(min(len(future_prices), max_bars)):
        price = future_prices[i]
        
        if (sl < entry_price and price <= sl) or (sl > entry_price and price >= sl):
            return price - entry_price if sl < entry_price else entry_price - price, i, 'sl'
        
        if (tp > entry_price and price >= tp) or (tp < entry_price and price <= tp):
            return tp - entry_price if tp > entry_price else entry_price - tp, i, 'tp'
    
    return future_prices[-1] - entry_price, max_bars, 'timeout'


def backtest_with_dynamic_atr(
    data: pd.DataFrame,
    signals: pd.Series,
    direction: str,
    manager: ATRRiskManager,
    initial_capital: float = 10000.0
) -> Dict:
    data_with_atr = manager.add_atr_to_data(data)
    
    trades = []
    capital = initial_capital
    
    for i in range(len(signals) - manager.max_bars_timeout):
        if signals.iloc[i] != 1:
            continue
        
        entry_price = data_with_atr['close'].iloc[i]
        atr_value = data_with_atr['atr'].iloc[i]
        
        if pd.isna(atr_value):
            continue
        
        levels = manager.calculate_levels(entry_price, atr_value, direction)
        
        future_prices = data_with_atr['close'].iloc[i+1:i+manager.max_bars_timeout+1].values
        
        profit, bars_held, exit_type = simulate_trade_with_atr(
            entry_price,
            levels['sl'],
            levels['tp'],
            future_prices,
            manager.max_bars_timeout
        )
        
        position_size = manager.calculate_position_size(
            capital,
            entry_price,
            levels['sl']
        )
        
        trade_profit = profit * position_size
        capital += trade_profit
        
        trades.append({
            'entry_price': entry_price,
            'exit_price': entry_price + profit,
            'profit': trade_profit,
            'bars_held': bars_held,
            'exit_type': exit_type,
            'capital': capital
        })
    
    if not trades:
        return {
            'total_trades': 0,
            'final_capital': initial_capital,
            'total_return': 0.0,
            'win_rate': 0.0
        }
    
    winning_trades = [t for t in trades if t['profit'] > 0]
    
    return {
        'total_trades': len(trades),
        'final_capital': capital,
        'total_return': (capital - initial_capital) / initial_capital,
        'win_rate': len(winning_trades) / len(trades),
        'trades': trades
    }


def optimize_atr_multipliers(
    data: pd.DataFrame,
    signals: pd.Series,
    direction: str,
    sl_range: Tuple[float, float] = (1.0, 3.0),
    tp_range: Tuple[float, float] = (1.5, 4.0),
    step: float = 0.5
) -> Dict:
    best_result = None
    best_params = None
    best_profit = -np.inf
    
    sl_values = np.arange(sl_range[0], sl_range[1] + step, step)
    tp_values = np.arange(tp_range[0], tp_range[1] + step, step)
    
    for sl_mult in sl_values:
        for tp_mult in tp_values:
            if tp_mult <= sl_mult:
                continue
            
            manager = ATRRiskManager(
                sl_multiplier=sl_mult,
                tp_multiplier=tp_mult
            )
            
            result = backtest_with_dynamic_atr(data, signals, direction, manager)
            
            if result['final_capital'] > best_profit:
                best_profit = result['final_capital']
                best_result = result
                best_params = {'sl_mult': sl_mult, 'tp_mult': tp_mult}
    
    return {
        'best_params': best_params,
        'best_result': best_result
    }