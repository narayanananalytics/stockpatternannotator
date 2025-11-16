"""
Core pattern annotation functionality.
"""

from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

try:
    import vectorbtpro as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    import warnings
    warnings.warn(
        "vectorbtpro is not available. Pattern detection will use fallback methods. "
        "Install vectorbtpro for full functionality."
    )

from .patterns import PatternConfig, get_default_patterns
from .pivots import PivotDetector


class PatternAnnotator:
    """
    Main class for annotating OHLC data with candlestick patterns and support/resistance.
    """

    def __init__(
        self,
        pattern_config: Optional[PatternConfig] = None,
        pivot_detector: Optional[PivotDetector] = None
    ):
        """
        Initialize the pattern annotator.

        Args:
            pattern_config: Configuration for pattern detection
            pivot_detector: Pivot detector instance for support/resistance
        """
        self.pattern_config = pattern_config or get_default_patterns()
        self.pivot_detector = pivot_detector or PivotDetector()
        self.annotations = pd.DataFrame()

    def annotate(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        detect_patterns: bool = True,
        detect_pivots: bool = True
    ) -> pd.DataFrame:
        """
        Annotate OHLC data with patterns and pivots.

        Args:
            data: DataFrame with OHLC data (must have 'open', 'high', 'low', 'close' columns)
            symbol: Symbol identifier (optional)
            timeframe: Timeframe identifier (e.g., '1H', '1D')
            detect_patterns: Whether to detect candlestick patterns
            detect_pivots: Whether to detect pivot points

        Returns:
            DataFrame with annotations
        """
        annotations = []

        # Validate input data
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Ensure data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Detect candlestick patterns
        if detect_patterns:
            pattern_annotations = self._detect_patterns(data, symbol, timeframe)
            annotations.extend(pattern_annotations)

        # Detect pivot points
        if detect_pivots:
            pivot_annotations = self._detect_pivots(data, symbol, timeframe)
            annotations.extend(pivot_annotations)

        # Create annotations DataFrame
        if annotations:
            result = pd.DataFrame(annotations)
        else:
            result = pd.DataFrame(columns=[
                'symbol', 'timeframe', 'pattern_name',
                'start_timestamp', 'end_timestamp', 'confidence'
            ])

        self.annotations = result
        return result

    def annotate_multiple(
        self,
        data: pd.DataFrame,
        symbol_col: str = 'symbol',
        timeframe_col: str = 'timeframe',
        detect_patterns: bool = True,
        detect_pivots: bool = True
    ) -> pd.DataFrame:
        """
        Annotate OHLC data for multiple symbols and timeframes.

        Args:
            data: DataFrame with OHLC data and symbol/timeframe columns
            symbol_col: Name of symbol column
            timeframe_col: Name of timeframe column
            detect_patterns: Whether to detect candlestick patterns
            detect_pivots: Whether to detect pivot points

        Returns:
            DataFrame with all annotations
        """
        all_annotations = []

        # Get unique symbol-timeframe combinations
        if symbol_col in data.columns:
            symbols = data[symbol_col].unique()
        else:
            symbols = [None]

        if timeframe_col in data.columns:
            timeframes = data[timeframe_col].unique()
        else:
            timeframes = [None]

        # Process each combination
        for symbol in symbols:
            for timeframe in timeframes:
                # Filter data
                mask = pd.Series([True] * len(data), index=data.index)
                if symbol is not None:
                    mask &= data[symbol_col] == symbol
                if timeframe is not None:
                    mask &= data[timeframe_col] == timeframe

                subset = data[mask].copy()

                if len(subset) == 0:
                    continue

                # Remove grouping columns for annotation
                subset = subset.drop(columns=[col for col in [symbol_col, timeframe_col]
                                              if col in subset.columns])

                # Annotate this subset
                annotations = self.annotate(
                    subset,
                    symbol=symbol,
                    timeframe=timeframe,
                    detect_patterns=detect_patterns,
                    detect_pivots=detect_pivots
                )

                all_annotations.append(annotations)

        # Combine all annotations
        if all_annotations:
            result = pd.concat(all_annotations, ignore_index=True)
        else:
            result = pd.DataFrame(columns=[
                'symbol', 'timeframe', 'pattern_name',
                'start_timestamp', 'end_timestamp', 'confidence'
            ])

        self.annotations = result
        return result

    def _detect_patterns(
        self,
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Dict]:
        """Detect candlestick patterns in the data."""
        annotations = []

        if VBT_AVAILABLE:
            # Use vectorbtpro for pattern detection
            annotations = self._detect_patterns_vbt(data, symbol, timeframe)
        else:
            # Use fallback pattern detection
            annotations = self._detect_patterns_fallback(data, symbol, timeframe)

        return annotations

    def _detect_patterns_vbt(
        self,
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Dict]:
        """Detect patterns using vectorbtpro."""
        annotations = []

        # Note: This is a conceptual implementation
        # vectorbtpro's actual API may differ - adjust as needed
        try:
            for pattern_name in self.pattern_config.get_all_patterns():
                window_size = self.pattern_config.get_pattern_window_size(pattern_name)

                # Use vectorbtpro's pattern recognition
                # The actual implementation depends on vectorbtpro's API
                # This is a placeholder that shows the intended structure
                pattern_annotations = self._search_pattern_vbt(
                    data, pattern_name, window_size, symbol, timeframe
                )
                annotations.extend(pattern_annotations)

        except Exception as e:
            import warnings
            warnings.warn(f"Error in vectorbtpro pattern detection: {e}. Using fallback.")
            return self._detect_patterns_fallback(data, symbol, timeframe)

        return annotations

    def _search_pattern_vbt(
        self,
        data: pd.DataFrame,
        pattern_name: str,
        window_size: int,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Dict]:
        """Search for a specific pattern using vectorbtpro."""
        # Placeholder for vectorbtpro pattern search
        # Actual implementation depends on vectorbtpro's pattern recognition API
        return []

    def _detect_patterns_fallback(
        self,
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Dict]:
        """Fallback pattern detection using basic logic."""
        annotations = []

        # Detect basic patterns using simple heuristics
        for pattern_name in self.pattern_config.single_candle_patterns:
            if pattern_name == "DOJI":
                matches = self._detect_doji(data)
            elif pattern_name == "HAMMER":
                matches = self._detect_hammer(data)
            elif pattern_name == "SHOOTING_STAR":
                matches = self._detect_shooting_star(data)
            else:
                continue

            for idx in matches:
                annotations.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'pattern_name': pattern_name,
                    'start_timestamp': data.index[idx],
                    'end_timestamp': data.index[idx],
                    'confidence': 0.8  # Default confidence
                })

        return annotations

    def _detect_doji(self, data: pd.DataFrame) -> List[int]:
        """Detect Doji patterns."""
        body = abs(data['close'] - data['open'])
        range_hl = data['high'] - data['low']

        # Doji: body is very small compared to range
        doji_mask = (body / range_hl < 0.1) & (range_hl > 0)
        return doji_mask[doji_mask].index.tolist()

    def _detect_hammer(self, data: pd.DataFrame) -> List[int]:
        """Detect Hammer patterns."""
        body = abs(data['close'] - data['open'])
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        # Hammer: long lower shadow, small upper shadow, small body at top
        hammer_mask = (
            (lower_shadow > 2 * body) &
            (upper_shadow < 0.3 * body) &
            (body > 0)
        )
        return hammer_mask[hammer_mask].index.tolist()

    def _detect_shooting_star(self, data: pd.DataFrame) -> List[int]:
        """Detect Shooting Star patterns."""
        body = abs(data['close'] - data['open'])
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)

        # Shooting Star: long upper shadow, small lower shadow, small body at bottom
        shooting_star_mask = (
            (upper_shadow > 2 * body) &
            (lower_shadow < 0.3 * body) &
            (body > 0)
        )
        return shooting_star_mask[shooting_star_mask].index.tolist()

    def _detect_pivots(
        self,
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> List[Dict]:
        """Detect pivot points for support/resistance."""
        annotations = []

        # Detect pivots using the pivot detector
        pivots_df = self.pivot_detector.detect_pivots(
            data['high'],
            data['low'],
            data['close']
        )

        # Convert pivots to annotations
        for _, pivot in pivots_df.iterrows():
            annotations.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'pattern_name': pivot['type'],
                'start_timestamp': pivot['timestamp'],
                'end_timestamp': pivot['timestamp'],
                'confidence': pivot['strength']
            })

        return annotations

    def get_annotations(self) -> pd.DataFrame:
        """Get the current annotations."""
        return self.annotations

    def filter_annotations(
        self,
        pattern_names: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Filter annotations based on criteria.

        Args:
            pattern_names: List of pattern names to include
            min_confidence: Minimum confidence threshold
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Filtered DataFrame
        """
        result = self.annotations.copy()

        if pattern_names is not None:
            result = result[result['pattern_name'].isin(pattern_names)]

        if min_confidence is not None:
            result = result[result['confidence'] >= min_confidence]

        if symbol is not None:
            result = result[result['symbol'] == symbol]

        if timeframe is not None:
            result = result[result['timeframe'] == timeframe]

        if start_date is not None:
            result = result[result['start_timestamp'] >= start_date]

        if end_date is not None:
            result = result[result['end_timestamp'] <= end_date]

        return result

    def export_annotations(self, filepath: str, format: str = 'csv') -> None:
        """
        Export annotations to file.

        Args:
            filepath: Output file path
            format: Export format ('csv', 'parquet', 'json')
        """
        if format == 'csv':
            self.annotations.to_csv(filepath, index=False)
        elif format == 'parquet':
            self.annotations.to_parquet(filepath, index=False)
        elif format == 'json':
            self.annotations.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
