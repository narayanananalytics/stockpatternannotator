"""
Utility functions for data loading and processing.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


def generate_sample_ohlc_data(
    n_periods: int = 100,
    start_date: str = '2024-01-01',
    timeframe: str = '1H',
    symbol: str = 'SAMPLE',
    volatility: float = 0.02,
    trend: float = 0.0001
) -> pd.DataFrame:
    """
    Generate sample OHLC data for testing.

    Args:
        n_periods: Number of periods to generate
        start_date: Start date for the data
        timeframe: Timeframe string (e.g., '1H', '1D')
        symbol: Symbol identifier
        volatility: Price volatility (standard deviation)
        trend: Price trend (mean change per period)

    Returns:
        DataFrame with OHLC data
    """
    # Create datetime index based on timeframe
    if timeframe == '1H':
        freq = 'H'
    elif timeframe == '1D':
        freq = 'D'
    elif timeframe == '4H':
        freq = '4H'
    elif timeframe == '15min':
        freq = '15min'
    else:
        freq = 'H'  # Default to hourly

    dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)

    # Generate price data with random walk
    np.random.seed(42)
    returns = np.random.normal(trend, volatility, n_periods)
    price = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from price
    data = []
    for i in range(n_periods):
        base_price = price[i]

        # Generate realistic OHLC
        daily_volatility = np.random.uniform(0.005, 0.02)
        open_price = base_price * (1 + np.random.normal(0, daily_volatility))
        close_price = base_price * (1 + np.random.normal(0, daily_volatility))

        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, daily_volatility / 2)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, daily_volatility / 2)))

        # Ensure high/low boundaries
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 10000)
        })

    df = pd.DataFrame(data, index=dates)
    return df


def load_ohlc_from_csv(
    filepath: str,
    datetime_col: str = 'timestamp',
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None
) -> pd.DataFrame:
    """
    Load OHLC data from CSV file.

    Args:
        filepath: Path to CSV file
        datetime_col: Name of datetime column
        symbol: Symbol identifier (optional)
        timeframe: Timeframe identifier (optional)

    Returns:
        DataFrame with OHLC data
    """
    df = pd.read_csv(filepath)

    # Convert datetime column to DatetimeIndex
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)

    # Standardize column names
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }

    df = df.rename(columns=column_mapping)

    return df


def calculate_pattern_statistics(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for detected patterns.

    Args:
        annotations: DataFrame with pattern annotations

    Returns:
        DataFrame with pattern statistics
    """
    if annotations.empty:
        return pd.DataFrame()

    stats = annotations.groupby('pattern_name').agg({
        'pattern_name': 'count',
        'confidence': ['mean', 'min', 'max']
    })

    stats.columns = ['count', 'avg_confidence', 'min_confidence', 'max_confidence']
    stats = stats.reset_index()

    return stats


def merge_annotations_with_ohlc(
    ohlc_data: pd.DataFrame,
    annotations: pd.DataFrame,
    how: str = 'left'
) -> pd.DataFrame:
    """
    Merge annotations back with OHLC data.

    Args:
        ohlc_data: Original OHLC DataFrame
        annotations: Annotations DataFrame
        how: Join type ('left', 'right', 'inner', 'outer')

    Returns:
        Merged DataFrame
    """
    # Create a copy to avoid modifying original
    result = ohlc_data.copy()

    # Add annotations based on timestamp
    if not annotations.empty:
        # Group annotations by timestamp
        annotation_groups = annotations.groupby('start_timestamp').apply(
            lambda x: x.to_dict('records')
        ).to_dict()

        result['patterns'] = result.index.map(
            lambda ts: annotation_groups.get(ts, [])
        )
    else:
        result['patterns'] = [[] for _ in range(len(result))]

    return result


def validate_ohlc_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate OHLC data format and quality.

    Args:
        data: OHLC DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_cols = ['open', 'high', 'low', 'close']

    # Check required columns
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        return False, f"Missing required columns: {missing}"

    # Check datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        return False, "Data must have a DatetimeIndex"

    # Check for NaN values
    if data[required_cols].isna().any().any():
        return False, "Data contains NaN values"

    # Check OHLC relationships
    invalid_high = (data['high'] < data[['open', 'close']].max(axis=1)).any()
    invalid_low = (data['low'] > data[['open', 'close']].min(axis=1)).any()

    if invalid_high or invalid_low:
        return False, "Invalid OHLC relationships (high/low constraints violated)"

    # Check for negative prices
    if (data[required_cols] < 0).any().any():
        return False, "Data contains negative prices"

    return True, "Data is valid"


def create_pattern_summary_report(annotations: pd.DataFrame) -> str:
    """
    Create a text summary report of detected patterns.

    Args:
        annotations: Annotations DataFrame

    Returns:
        Formatted text report
    """
    if annotations.empty:
        return "No patterns detected."

    report_lines = [
        "=" * 60,
        "PATTERN DETECTION SUMMARY",
        "=" * 60,
        ""
    ]

    # Overall statistics
    total_patterns = len(annotations)
    unique_patterns = annotations['pattern_name'].nunique()
    date_range = f"{annotations['start_timestamp'].min()} to {annotations['end_timestamp'].max()}"

    report_lines.extend([
        f"Total patterns detected: {total_patterns}",
        f"Unique pattern types: {unique_patterns}",
        f"Date range: {date_range}",
        ""
    ])

    # Pattern breakdown
    pattern_counts = annotations['pattern_name'].value_counts()

    report_lines.append("Pattern Breakdown:")
    report_lines.append("-" * 60)

    for pattern, count in pattern_counts.items():
        avg_conf = annotations[annotations['pattern_name'] == pattern]['confidence'].mean()
        report_lines.append(f"  {pattern:30s}: {count:4d} ({avg_conf:.2f} avg confidence)")

    report_lines.append("")

    # Symbol/Timeframe breakdown if available
    if 'symbol' in annotations.columns and annotations['symbol'].notna().any():
        report_lines.append("By Symbol:")
        report_lines.append("-" * 60)
        symbol_counts = annotations.groupby('symbol')['pattern_name'].count()
        for symbol, count in symbol_counts.items():
            report_lines.append(f"  {symbol:30s}: {count:4d} patterns")
        report_lines.append("")

    if 'timeframe' in annotations.columns and annotations['timeframe'].notna().any():
        report_lines.append("By Timeframe:")
        report_lines.append("-" * 60)
        tf_counts = annotations.groupby('timeframe')['pattern_name'].count()
        for tf, count in tf_counts.items():
            report_lines.append(f"  {tf:30s}: {count:4d} patterns")
        report_lines.append("")

    report_lines.append("=" * 60)

    return "\n".join(report_lines)
