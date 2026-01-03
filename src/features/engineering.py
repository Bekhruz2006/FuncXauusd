"""
Feature engineering module
Creates technical indicators and meta-features for ML models
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.signal import savgol_filter
from numba import njit


@njit
def _calculate_std_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast rolling standard deviation using Numba
    
    Args:
        prices: Price array
        period: Window size
        
    Returns:
        Array of standard deviations
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        result[i] = np.std(window)
    
    return result


@njit
def _calculate_skewness_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Fast rolling skewness using Numba
    
    Args:
        prices: Price array
        period: Window size
        
    Returns:
        Array of skewness values
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = prices[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if std == 0:
            result[i] = 0.0
        else:
            m3 = np.mean(((window - mean) / std) ** 3)
            result[i] = m3
    
    return result


def create_features(
    data: pd.DataFrame,
    periods: List[int],
    meta_periods: List[int],
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create main and meta features for ML model
    
    Main features: Standard deviation (std) for trading signals
    Meta features: Skewness for market regime clustering
    
    Args:
        data: OHLCV DataFrame
        periods: Periods for main features (std)
        meta_periods: Periods for meta features (skewness)
        verbose: Print progress
        
    Returns:
        Tuple of (features_main, features_meta) DataFrames
    """
    if verbose:
        print(f"🔧 Creating features...")
        print(f"   Main periods: {periods}")
        print(f"   Meta periods: {meta_periods}")
    
    close = data['close'].values
    n = len(close)
    
    # === MAIN FEATURES: Standard Deviation ===
    features_main = pd.DataFrame(index=data.index)
    
    for period in periods:
        col_name = f'std_{period}'
        features_main[col_name] = _calculate_std_numba(close, period)
    
    # === META FEATURES: Skewness ===
    features_meta = pd.DataFrame(index=data.index)
    
    for period in meta_periods:
        col_name = f'skew_{period}'
        features_meta[col_name] = _calculate_skewness_numba(close, period)
    
    # Remove NaN rows (from initial period)
    max_period = max(max(periods), max(meta_periods))
    features_main = features_main.iloc[max_period:]
    features_meta = features_meta.iloc[max_period:]
    
    if verbose:
        print(f"   ✅ Created {len(features_main.columns)} main features")
        print(f"   ✅ Created {len(features_meta.columns)} meta features")
        print(f"   Valid samples: {len(features_main):,}")
    
    return features_main, features_meta


def add_price_context(
    features: pd.DataFrame,
    data: pd.DataFrame,
    include_ohlc: bool = True
) -> pd.DataFrame:
    """
    Add raw price information to features
    
    Args:
        features: Existing features DataFrame
        data: OHLCV data
        include_ohlc: Include open, high, low, close
        
    Returns:
        Features with added price context
    """
    features = features.copy()
    
    # Align indices
    common_idx = features.index.intersection(data.index)
    features = features.loc[common_idx]
    data_aligned = data.loc[common_idx]
    
    if include_ohlc:
        features['open'] = data_aligned['open']
        features['high'] = data_aligned['high']
        features['low'] = data_aligned['low']
        features['close'] = data_aligned['close']
    
    features['volume'] = data_aligned['volume']
    
    return features


def smooth_features(
    features: pd.DataFrame,
    window_length: int = 5,
    polyorder: int = 2
) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter to smooth features
    
    Args:
        features: Features DataFrame
        window_length: Window length (must be odd)
        polyorder: Polynomial order
        
    Returns:
        Smoothed features DataFrame
    """
    if window_length % 2 == 0:
        window_length += 1
    
    smoothed = features.copy()
    
    for col in features.columns:
        if features[col].dtype in [np.float64, np.float32]:
            try:
                smoothed[col] = savgol_filter(
                    features[col].fillna(method='ffill'),
                    window_length,
                    polyorder,
                    mode='nearest'
                )
            except Exception:
                pass  # Keep original if smoothing fails
    
    return smoothed


def normalize_features(
    features: pd.DataFrame,
    method: str = 'standard'
) -> pd.DataFrame:
    """
    Normalize features
    
    Args:
        features: Features DataFrame
        method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        
    Returns:
        Normalized features
    """
    normalized = features.copy()
    
    for col in features.columns:
        values = features[col].values
        
        if method == 'standard':
            mean = np.nanmean(values)
            std = np.nanstd(values)
            if std > 0:
                normalized[col] = (values - mean) / std
        
        elif method == 'minmax':
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            if max_val > min_val:
                normalized[col] = (values - min_val) / (max_val - min_val)
    
    return normalized


def create_additional_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional technical indicators
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        DataFrame with indicators
    """
    indicators = pd.DataFrame(index=data.index)
    
    close = data['close']
    high = data['high']
    low = data['low']
    
    # ATR (Average True Range)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    indicators['atr_14'] = tr.rolling(14).mean()
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    indicators['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    indicators['sma_20'] = close.rolling(20).mean()
    indicators['sma_50'] = close.rolling(50).mean()
    indicators['ema_12'] = close.ewm(span=12).mean()
    indicators['ema_26'] = close.ewm(span=26).mean()
    
    # MACD
    indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
    indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    indicators['bb_upper'] = sma_20 + 2 * std_20
    indicators['bb_lower'] = sma_20 - 2 * std_20
    indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20
    
    return indicators


def validate_features(
    features: pd.DataFrame,
    verbose: bool = True
) -> dict:
    """
    Validate feature quality
    
    Args:
        features: Features DataFrame
        verbose: Print validation results
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'n_features': len(features.columns),
        'n_samples': len(features),
        'missing_values': features.isnull().sum().sum(),
        'infinite_values': np.isinf(features.values).sum(),
        'constant_features': (features.std() == 0).sum(),
        'feature_names': list(features.columns)
    }
    
    if verbose:
        print(f"\n🔍 Feature Validation:")
        print(f"   Features: {stats['n_features']}")
        print(f"   Samples: {stats['n_samples']:,}")
        print(f"   Missing: {stats['missing_values']}")
        print(f"   Infinite: {stats['infinite_values']}")
        print(f"   Constant: {stats['constant_features']}")
        
        if stats['missing_values'] > 0 or stats['infinite_values'] > 0:
            print(f"   ⚠️ Feature quality issues detected!")
    
    return stats


def get_feature_importance(
    model,
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained CatBoost model
        features: Features DataFrame
        
    Returns:
        DataFrame with feature importance scores
    """
    try:
        importance = model.get_feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': features.columns,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['cumulative'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()
        
        return importance_df
    
    except Exception as e:
        print(f"Warning: Could not extract feature importance: {e}")
        return pd.DataFrame()