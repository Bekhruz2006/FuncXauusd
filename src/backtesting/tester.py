"""
Backtesting module
Fast testing of models on historical data with risk management
"""

import numpy as np
import pandas as pd
from numba import jit
from typing import Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt

from ..features.engineering import create_features


@jit(nopython=True)
def process_data_one_direction(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signals: np.ndarray,
    stop: float,
    take: float,
    direction: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process trading signals with stop-loss and take-profit
    Optimized with Numba JIT for performance
    
    Args:
        close: Close prices
        high: High prices
        low: Low prices
        signals: Trading signals (1 = trade, 0 = no trade)
        stop: Stop-loss in points
        take: Take-profit in points
        direction: 'buy' or 'sell'
        
    Returns:
        Tuple of (equity_curve, trades, trade_returns)
    """
    n = len(close)
    equity = np.ones(n)
    trades = np.zeros(n)
    returns = np.zeros(n)
    
    is_buy = direction == 'buy'
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    
    for i in range(n):
        # Check for signal to enter
        if not in_position and signals[i] == 1:
            in_position = True
            entry_price = close[i]
            entry_idx = i
            trades[i] = 1
            continue
        
        # Check exit conditions if in position
        if in_position:
            if is_buy:
                # Buy position
                sl_level = entry_price - stop
                tp_level = entry_price + take
                
                # Check stop-loss
                if low[i] <= sl_level:
                    pnl = -stop
                    returns[i] = pnl
                    equity[i] = equity[i-1] * (1 + pnl / entry_price)
                    in_position = False
                    continue
                
                # Check take-profit
                if high[i] >= tp_level:
                    pnl = take
                    returns[i] = pnl
                    equity[i] = equity[i-1] * (1 + pnl / entry_price)
                    in_position = False
                    continue
            else:
                # Sell position
                sl_level = entry_price + stop
                tp_level = entry_price - take
                
                # Check stop-loss
                if high[i] >= sl_level:
                    pnl = -stop
                    returns[i] = pnl
                    equity[i] = equity[i-1] * (1 + pnl / entry_price)
                    in_position = False
                    continue
                
                # Check take-profit
                if low[i] <= tp_level:
                    pnl = take
                    returns[i] = pnl
                    equity[i] = equity[i-1] * (1 + pnl / entry_price)
                    in_position = False
                    continue
            
            # Position still open
            equity[i] = equity[i-1]
    
    return equity, trades, returns


def tester_one_direction(
    dataset: pd.DataFrame,
    stop: float,
    take: float,
    forward: datetime,
    backward: datetime,
    markup: float,
    direction: str = 'buy',
    plt_show: bool = False
) -> float:
    """
    Test trading strategy on historical data
    
    Args:
        dataset: OHLCV DataFrame
        stop: Stop-loss in points
        take: Take-profit in points
        forward: Start date for testing
        backward: End date for testing
        markup: Markup threshold (not used in pure backtesting)
        direction: 'buy' or 'sell'
        plt_show: Show matplotlib plots
        
    Returns:
        R² score or performance metric
    """
    # Filter date range
    test_data = dataset[
        (dataset.index >= forward) & 
        (dataset.index <= backward)
    ]
    
    if len(test_data) < 100:
        print(f"⚠️ Insufficient test data: {len(test_data)} bars")
        return 0.0
    
    # For demonstration, generate random signals
    # In real usage, model predictions should be passed here
    signals = np.random.randint(0, 2, len(test_data))
    
    # Run backtest
    equity, trades, returns = process_data_one_direction(
        test_data['close'].values,
        test_data['high'].values,
        test_data['low'].values,
        signals,
        stop,
        take,
        direction
    )
    
    # Calculate metrics
    total_trades = np.sum(trades)
    winning_trades = np.sum(returns > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_return = (equity[-1] - 1) * 100
    max_equity = np.maximum.accumulate(equity)
    drawdown = (max_equity - equity) / max_equity
    max_drawdown = np.max(drawdown) * 100
    
    # R² as proxy for performance
    if total_trades > 0 and win_rate > 0.5:
        r2 = min(0.99, (win_rate - 0.5) * 2)  # Scale to [0, 1]
    else:
        r2 = 0.0
    
    if plt_show:
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, equity, label='Equity Curve', linewidth=2)
        plt.title(f'Backtest Results ({direction.upper()})')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 Backtest Results:")
        print(f"   Total Trades: {total_trades:.0f}")
        print(f"   Win Rate: {win_rate:.2%}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   R²: {r2:.4f}")
    
    return r2


def test_model_one_direction(
    dataset: pd.DataFrame,
    result: list,
    config: dict,
    plt: bool = False
) -> float:
    """
    Test trained model on out-of-time data
    
    Args:
        dataset: OHLCV DataFrame
        result: List containing [main_model, meta_model]
        config: Configuration dictionary
        plt: Show plots
        
    Returns:
        R² score
    """
    if len(result) < 2:
        raise ValueError("result must contain [main_model, meta_model]")
    
    model_main = result[0]
    model_meta = result[1]
    
    # Date ranges
    forward_date = pd.to_datetime(config['data']['forward'])
    full_forward_date = pd.to_datetime(config['data']['full_forward'])
    
    # Test on Out-of-Time data
    test_data = dataset[
        (dataset.index >= forward_date) &
        (dataset.index <= full_forward_date)
    ]
    
    if len(test_data) < 100:
        print(f"⚠️ Insufficient test data")
        return 0.0
    
    print(f"\n🧪 Testing on OOT data: {len(test_data):,} bars")
    print(f"   Period: {test_data.index[0].date()} → {test_data.index[-1].date()}")
    
    # Create features
    features_main, features_meta = create_features(
        test_data,
        config.get('periods', [5, 35, 65, 95, 125, 155, 185, 215, 245, 275]),
        config.get('periods_meta', [5]),
        verbose=False
    )
    
    # Align indices
    common_idx = features_main.index.intersection(features_meta.index)
    features_main = features_main.loc[common_idx]
    features_meta = features_meta.loc[common_idx]
    test_data = test_data.loc[common_idx]
    
    # Generate predictions
    predictions_main = model_main.predict(features_main)
    predictions_meta = model_meta.predict_proba(features_meta)[:, 1]
    
    # Combine predictions: main signal AND meta confidence
    threshold = 0.5
    combined_signals = (predictions_main == 1) & (predictions_meta > threshold)
    signals = combined_signals.astype(int)
    
    n_signals = np.sum(signals)
    print(f"   Generated signals: {n_signals}")
    
    if n_signals < 10:
        print(f"   ⚠️ Too few signals for reliable testing")
        return 0.0
    
    # Run backtest
    equity, trades, returns = process_data_one_direction(
        test_data['close'].values,
        test_data['high'].values,
        test_data['low'].values,
        signals,
        config['trading']['risk']['stop_loss'],
        config['trading']['risk']['take_profit'],
        config['trading']['direction']
    )
    
    # Calculate comprehensive metrics
    total_trades = np.sum(trades)
    winning_trades = np.sum(returns > 0)
    losing_trades = np.sum(returns < 0)
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    total_return = (equity[-1] - 1) * 100
    
    max_equity = np.maximum.accumulate(equity)
    drawdown = (max_equity - equity) / max_equity
    max_drawdown = np.max(drawdown) * 100
    
    # Sharpe ratio (simplified)
    if len(returns[returns != 0]) > 0:
        returns_pct = returns[returns != 0] / test_data['close'].values[returns != 0]
        sharpe = np.mean(returns_pct) / np.std(returns_pct) if np.std(returns_pct) > 0 else 0
    else:
        sharpe = 0
    
    # R² score (higher is better)
    r2 = min(0.99, max(0, (win_rate - 0.4) * 2))  # Scale win_rate to R²
    
    if plt:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        axes[0].plot(test_data.index, equity, label='Equity', linewidth=2, color='blue')
        axes[0].fill_between(test_data.index, 1, equity, alpha=0.3)
        axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title(f'Out-of-Time Backtest: {config["symbol"]["name"]} ({config["trading"]["direction"].upper()})', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Equity', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=11)
        
        # Drawdown
        axes[1].fill_between(test_data.index, 0, -drawdown * 100, 
                            color='red', alpha=0.5, label='Drawdown')
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        print(f"\n{'='*70}")
        print(f"  📊 OUT-OF-TIME BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"\n💰 Performance:")
        print(f"   Total Return:     {total_return:>8.2f}%")
        print(f"   Max Drawdown:     {max_drawdown:>8.2f}%")
        print(f"   Profit Factor:    {profit_factor:>8.2f}")
        print(f"   Sharpe Ratio:     {sharpe:>8.2f}")
        
        print(f"\n📈 Trading Statistics:")
        print(f"   Total Trades:     {total_trades:>8.0f}")
        print(f"   Winning Trades:   {winning_trades:>8.0f}")
        print(f"   Losing Trades:    {losing_trades:>8.0f}")
        print(f"   Win Rate:         {win_rate:>8.2%}")
        
        print(f"\n🎯 Quality Metrics:")
        print(f"   R² Score:         {r2:>8.4f}")
        print(f"   Signals/Day:      {n_signals/len(test_data)*24:>8.2f}")
        
        if total_return > 0 and max_drawdown < 15:
            print(f"\n✅ Model shows positive performance!")
        else:
            print(f"\n⚠️ Model needs improvement")
        
        print(f"\n{'='*70}\n")
    
    return r2


def calculate_walk_forward_metrics(
    dataset: pd.DataFrame,
    model,
    config: dict,
    n_folds: int = 5
) -> pd.DataFrame:
    """
    Perform walk-forward validation
    
    Args:
        dataset: Full dataset
        model: Trained model
        config: Configuration
        n_folds: Number of folds for walk-forward
        
    Returns:
        DataFrame with metrics for each fold
    """
    results = []
    
    data_length = len(dataset)
    fold_size = data_length // n_folds
    
    for i in range(n_folds):
        fold_start = i * fold_size
        fold_end = min((i + 1) * fold_size, data_length)
        
        fold_data = dataset.iloc[fold_start:fold_end]
        
        # Test on this fold
        # (Implementation would involve creating features and testing)
        # Placeholder for now
        fold_metrics = {
            'fold': i + 1,
            'start_date': fold_data.index[0],
            'end_date': fold_data.index[-1],
            'samples': len(fold_data)
        }
        
        results.append(fold_metrics)
    
    return pd.DataFrame(results)