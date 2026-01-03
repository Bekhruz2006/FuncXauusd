"""
Labeling strategies for supervised learning
Implements one-direction labeling with temporal penalty
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple


@njit
def calculate_labels_one_direction(
    close: np.ndarray,
    markup: float,
    direction: str,
    min_bars: int = 1,
    max_bars: int = 15
) -> np.ndarray:
    """
    Calculate labels for one-direction trading (buy or sell only)
    with temporal penalty - profit must be taken quickly
    
    Args:
        close: Close prices array
        markup: Threshold for signal (e.g., 0.25 for 25%)
        direction: 'buy' or 'sell'
        min_bars: Minimum bars to look forward
        max_bars: Maximum bars to look forward (timeout)
        
    Returns:
        Array of labels: 1 (signal), 0 (no signal), -1 (invalid/future)
        
    Logic:
        - For BUY: Look for price increase > markup within max_bars
        - For SELL: Look for price decrease > markup within max_bars
        - Apply temporal penalty: earlier profit = stronger signal
        - If profit not reached within max_bars, label = 0
    """
    n = len(close)
    labels = np.full(n, -1, dtype=np.int8)
    
    is_buy = direction.lower() == 'buy'
    
    # Can't label last max_bars (no future data)
    for i in range(n - max_bars):
        current_price = close[i]
        future_window = close[i + min_bars:i + max_bars + 1]
        
        if is_buy:
            # Buy: look for price going UP
            max_future = np.max(future_window)
            pct_change = (max_future - current_price) / current_price
            
            if pct_change >= markup:
                # Find how quickly profit was reached
                for j in range(len(future_window)):
                    if (future_window[j] - current_price) / current_price >= markup:
                        # Temporal bonus: earlier = better
                        time_factor = 1.0 - (j / max_bars) * 0.3  # 30% penalty for late
                        
                        if time_factor > 0.7:  # Strong signal
                            labels[i] = 1
                        break
            else:
                labels[i] = 0  # No profitable move
        else:
            # Sell: look for price going DOWN
            min_future = np.min(future_window)
            pct_change = (current_price - min_future) / current_price
            
            if pct_change >= markup:
                # Find how quickly profit was reached
                for j in range(len(future_window)):
                    if (current_price - future_window[j]) / current_price >= markup:
                        time_factor = 1.0 - (j / max_bars) * 0.3
                        
                        if time_factor > 0.7:
                            labels[i] = 1
                        break
            else:
                labels[i] = 0
    
    return labels


def get_labels_one_direction(
    data: pd.DataFrame,
    markup: float,
    direction: str = 'buy',
    min_bars: int = 1,
    max_bars: int = 15,
    verbose: bool = False
) -> pd.Series:
    """
    Wrapper for one-direction labeling with pandas integration
    
    Args:
        data: OHLCV DataFrame
        markup: Signal threshold (e.g., 0.25 for 25%)
        direction: 'buy' or 'sell'
        min_bars: Minimum bars to look forward
        max_bars: Maximum bars to look forward
        verbose: Print labeling statistics
        
    Returns:
        Series of labels aligned with data index
    """
    close = data['close'].values
    
    labels = calculate_labels_one_direction(
        close,
        markup,
        direction,
        min_bars,
        max_bars
    )
    
    labels_series = pd.Series(labels, index=data.index, name='label')
    
    # Filter out invalid labels (-1)
    valid_mask = labels_series != -1
    labels_series = labels_series[valid_mask]
    
    if verbose:
        print(f"\n📊 Labeling Statistics ({direction.upper()}):")
        print(f"   Markup threshold: {markup:.2%}")
        print(f"   Total samples: {len(labels_series):,}")
        print(f"   Signals (1): {(labels_series == 1).sum():,} ({(labels_series == 1).mean():.2%})")
        print(f"   No signals (0): {(labels_series == 0).sum():,} ({(labels_series == 0).mean():.2%})")
        
        balance = (labels_series == 1).mean()
        if balance < 0.1 or balance > 0.9:
            print(f"   ⚠️ Class imbalance detected! Consider adjusting markup.")
    
    return labels_series


def validate_class_balance(
    labels: pd.Series,
    min_positive_ratio: float = 0.1,
    max_positive_ratio: float = 0.9
) -> Tuple[bool, float]:
    """
    Validate that classes are reasonably balanced
    
    Args:
        labels: Series of binary labels
        min_positive_ratio: Minimum acceptable positive class ratio
        max_positive_ratio: Maximum acceptable positive class ratio
        
    Returns:
        Tuple of (is_valid, positive_ratio)
    """
    positive_ratio = (labels == 1).mean()
    is_valid = min_positive_ratio <= positive_ratio <= max_positive_ratio
    
    return is_valid, positive_ratio


def create_multiclass_labels(
    data: pd.DataFrame,
    markup_strong: float = 0.30,
    markup_weak: float = 0.15,
    direction: str = 'buy',
    max_bars: int = 15
) -> pd.Series:
    """
    Create multiclass labels: 0 (no signal), 1 (weak signal), 2 (strong signal)
    
    Args:
        data: OHLCV DataFrame
        markup_strong: Threshold for strong signal
        markup_weak: Threshold for weak signal
        direction: 'buy' or 'sell'
        max_bars: Look forward window
        
    Returns:
        Series of multiclass labels
    """
    close = data['close'].values
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    
    is_buy = direction.lower() == 'buy'
    
    for i in range(n - max_bars):
        current = close[i]
        future = close[i + 1:i + max_bars + 1]
        
        if is_buy:
            max_profit = (np.max(future) - current) / current
            
            if max_profit >= markup_strong:
                labels[i] = 2  # Strong signal
            elif max_profit >= markup_weak:
                labels[i] = 1  # Weak signal
        else:
            max_profit = (current - np.min(future)) / current
            
            if max_profit >= markup_strong:
                labels[i] = 2
            elif max_profit >= markup_weak:
                labels[i] = 1
    
    # Remove future data
    labels[-max_bars:] = -1
    
    labels_series = pd.Series(labels, index=data.index, name='label')
    labels_series = labels_series[labels_series != -1]
    
    return labels_series


@njit
def calculate_regression_target(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    direction: str,
    max_bars: int = 20
) -> np.ndarray:
    """
    Calculate continuous regression target with temporal penalty
    
    Target = (Max_Profit_Within_N_Bars) / ATR * time_factor
    
    Args:
        close: Close prices
        high: High prices
        low: Low prices
        atr: Average True Range
        direction: 'buy' or 'sell'
        max_bars: Maximum bars to look forward
        
    Returns:
        Array of regression targets
    """
    n = len(close)
    targets = np.full(n, np.nan)
    
    is_buy = direction.lower() == 'buy'
    
    for i in range(n - max_bars):
        current = close[i]
        current_atr = atr[i]
        
        if np.isnan(current_atr) or current_atr == 0:
            targets[i] = 0.0
            continue
        
        # Look at future bars
        future_highs = high[i + 1:i + max_bars + 1]
        future_lows = low[i + 1:i + max_bars + 1]
        
        if is_buy:
            max_profit = np.max(future_highs) - current
            
            # Find when profit was reached
            bars_to_profit = 0
            for j in range(len(future_highs)):
                if future_highs[j] >= (current + 1.5 * current_atr):
                    bars_to_profit = j + 1
                    break
            
            if bars_to_profit > 0:
                time_factor = 1.0 - (bars_to_profit / max_bars) * 0.5
                targets[i] = (max_profit / current_atr) * time_factor
            else:
                targets[i] = max_profit / current_atr * 0.3  # Penalty
        else:
            max_profit = current - np.min(future_lows)
            
            bars_to_profit = 0
            for j in range(len(future_lows)):
                if future_lows[j] <= (current - 1.5 * current_atr):
                    bars_to_profit = j + 1
                    break
            
            if bars_to_profit > 0:
                time_factor = 1.0 - (bars_to_profit / max_bars) * 0.5
                targets[i] = (max_profit / current_atr) * time_factor
            else:
                targets[i] = max_profit / current_atr * 0.3
    
    return targets


def get_regression_targets(
    data: pd.DataFrame,
    direction: str = 'buy',
    max_bars: int = 20,
    atr_period: int = 14
) -> pd.Series:
    """
    Create regression targets for continuous prediction
    
    Args:
        data: OHLCV DataFrame
        direction: 'buy' or 'sell'
        max_bars: Look forward window
        atr_period: Period for ATR calculation
        
    Returns:
        Series of regression targets
    """
    # Calculate ATR
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                             np.abs(low - np.roll(close, 1))))
    
    atr = np.convolve(tr, np.ones(atr_period) / atr_period, mode='same')
    
    # Calculate targets
    targets = calculate_regression_target(
        close, high, low, atr, direction, max_bars
    )
    
    targets_series = pd.Series(targets, index=data.index, name='target')
    targets_series = targets_series.dropna()
    
    return targets_series


def get_label_distribution(labels: pd.Series) -> dict:
    """
    Analyze label distribution
    
    Args:
        labels: Series of labels
        
    Returns:
        Dictionary with distribution statistics
    """
    dist = {
        'total': len(labels),
        'unique_values': labels.unique().tolist(),
        'value_counts': labels.value_counts().to_dict(),
        'proportions': (labels.value_counts() / len(labels)).to_dict()
    }
    
    return dist