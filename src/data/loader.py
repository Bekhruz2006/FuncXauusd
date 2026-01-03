"""
Data loading and caching module
Handles OHLCV data loading with intelligent caching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import pickle


# Global cache for performance
_PRICE_CACHE = None


def load_price_data(
    config: dict,
    use_cache: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and preprocess OHLCV price data
    
    Args:
        config: Configuration dictionary with data paths and parameters
        use_cache: Use cached data if available
        verbose: Print loading information
        
    Returns:
        DataFrame with OHLCV data, properly indexed and cleaned
        
    Raises:
        FileNotFoundError: If data file not found
        ValueError: If data is invalid or empty
    """
    global _PRICE_CACHE
    
    # Return cached data if available
    if use_cache and _PRICE_CACHE is not None:
        if verbose:
            print(f"✅ Using cached data ({len(_PRICE_CACHE)} bars)")
        return _PRICE_CACHE.copy()
    
    # Construct file path
    symbol = config['symbol']['name']
    data_path = Path(config['data']['paths']['raw']) / f"{symbol}.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please place {symbol}.csv in {config['data']['paths']['raw']}/"
        )
    
    if verbose:
        print(f"📂 Loading {data_path.name}...", end=" ")
    
    # Load CSV with semicolon separator
    try:
        df = pd.read_csv(
            data_path,
            sep=';',
            parse_dates=['Date'],
            index_col='Date'
        )
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")
    
    # Validate required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Convert to numeric, handle errors
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove invalid data
    initial_len = len(df)
    df = df.dropna()
    df = df[df['volume'] > 0]  # Remove zero volume bars
    df = df[df['high'] >= df['low']]  # Sanity check
    
    removed = initial_len - len(df)
    if removed > 0 and verbose:
        print(f"(removed {removed} invalid bars)")
    
    # Sort by date
    df = df.sort_index()
    
    # Apply date filters if specified
    if 'backward' in config['data']:
        start_date = pd.to_datetime(config['data']['backward'])
        df = df[df.index >= start_date]
    
    if 'full_forward' in config['data']:
        end_date = pd.to_datetime(config['data']['full_forward'])
        df = df[df.index <= end_date]
    
    if len(df) == 0:
        raise ValueError("No data remaining after filtering")
    
    if verbose:
        date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
        print(f"✅ Loaded {len(df):,} bars | {date_range}")
    
    # Cache for future use
    if use_cache:
        _PRICE_CACHE = df.copy()
    
    return df


def cache_prices(config: dict, verbose: bool = True) -> None:
    """
    Load and cache price data for fast repeated access
    
    Args:
        config: Configuration dictionary
        verbose: Print status messages
    """
    global _PRICE_CACHE
    
    if _PRICE_CACHE is not None:
        if verbose:
            print("✅ Data already cached")
        return
    
    _PRICE_CACHE = load_price_data(config, use_cache=False, verbose=verbose)


def get_cached_prices() -> Optional[pd.DataFrame]:
    """
    Get cached price data without reloading
    
    Returns:
        Cached DataFrame or None if not cached
    """
    if _PRICE_CACHE is None:
        return None
    return _PRICE_CACHE.copy()


def clear_cache() -> None:
    """Clear the price data cache"""
    global _PRICE_CACHE
    _PRICE_CACHE = None


def split_data(
    data: pd.DataFrame,
    forward_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into In-Sample and Out-of-Sample
    
    Args:
        data: Full dataset
        forward_date: Date to split on (YYYY-MM-DD)
        
    Returns:
        Tuple of (in_sample, out_of_sample) DataFrames
    """
    split_date = pd.to_datetime(forward_date)
    
    in_sample = data[data.index < split_date]
    out_of_sample = data[data.index >= split_date]
    
    return in_sample, out_of_sample


def validate_data_quality(data: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Validate data quality and return statistics
    
    Args:
        data: DataFrame to validate
        verbose: Print validation results
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_bars': len(data),
        'missing_values': data.isnull().sum().sum(),
        'zero_volume': (data['volume'] == 0).sum(),
        'invalid_ohlc': ((data['high'] < data['low']) | 
                        (data['high'] < data['open']) |
                        (data['high'] < data['close']) |
                        (data['low'] > data['open']) |
                        (data['low'] > data['close'])).sum(),
        'date_range': (data.index.min(), data.index.max()),
        'avg_volume': data['volume'].mean(),
        'price_range': (data['close'].min(), data['close'].max())
    }
    
    if verbose:
        print(f"\n📊 Data Quality Report:")
        print(f"  Total bars: {stats['total_bars']:,}")
        print(f"  Missing values: {stats['missing_values']}")
        print(f"  Zero volume bars: {stats['zero_volume']}")
        print(f"  Invalid OHLC: {stats['invalid_ohlc']}")
        print(f"  Date range: {stats['date_range'][0].date()} → {stats['date_range'][1].date()}")
        print(f"  Price range: ${stats['price_range'][0]:.2f} - ${stats['price_range'][1]:.2f}")
        
        if stats['missing_values'] > 0 or stats['invalid_ohlc'] > 0:
            print(f"  ⚠️ Data quality issues detected!")
    
    return stats


def save_processed_data(
    data: pd.DataFrame,
    config: dict,
    suffix: str = "processed"
) -> Path:
    """
    Save processed data to disk
    
    Args:
        data: DataFrame to save
        config: Configuration with output paths
        suffix: Filename suffix
        
    Returns:
        Path to saved file
    """
    output_dir = Path(config['data']['paths']['processed'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    symbol = config['symbol']['name']
    filename = f"{symbol}_{suffix}.parquet"
    output_path = output_dir / filename
    
    data.to_parquet(output_path, compression='snappy')
    
    return output_path


def load_processed_data(
    config: dict,
    suffix: str = "processed"
) -> Optional[pd.DataFrame]:
    """
    Load previously processed data
    
    Args:
        config: Configuration with paths
        suffix: Filename suffix
        
    Returns:
        DataFrame or None if not found
    """
    try:
        input_dir = Path(config['data']['paths']['processed'])
        symbol = config['symbol']['name']
        filename = f"{symbol}_{suffix}.parquet"
        input_path = input_dir / filename
        
        if not input_path.exists():
            return None
        
        return pd.read_parquet(input_path)
    except Exception:
        return None


# Convenient aliases
load_data = load_price_data
cache_data = cache_prices