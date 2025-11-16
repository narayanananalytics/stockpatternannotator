"""
Pattern validation and probability analysis module.

This module provides backtesting functionality to validate pattern predictions
and calculate the probability of bullish/bearish outcomes over different time horizons.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class PatternValidator:
    """Validate patterns by analyzing actual price movements after pattern detection."""

    def __init__(
        self,
        forecast_horizons: List[int] = [1, 3, 5, 10],
        price_change_threshold: float = 0.0,
        require_minimum_samples: int = 5
    ):
        """
        Initialize pattern validator.

        Args:
            forecast_horizons: List of candle counts to look forward (e.g., [1, 3, 5, 10])
            price_change_threshold: Minimum % change to consider directional (default 0.0)
            require_minimum_samples: Minimum samples required for probability calculation
        """
        self.forecast_horizons = sorted(forecast_horizons)
        self.price_change_threshold = price_change_threshold
        self.require_minimum_samples = require_minimum_samples
        self.validation_results = None

    def validate_patterns(
        self,
        ohlc_data: pd.DataFrame,
        annotations: pd.DataFrame,
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Validate patterns by checking actual price movements after detection.

        Args:
            ohlc_data: OHLC data with DatetimeIndex
            annotations: Pattern annotations DataFrame
            price_column: Column to use for price analysis ('close', 'high', 'low', etc.)

        Returns:
            DataFrame with validation results for each annotation
        """
        if annotations.empty:
            return pd.DataFrame()

        validation_results = []

        # Group by symbol and timeframe for efficiency
        if 'symbol' in annotations.columns and 'timeframe' in annotations.columns:
            groups = annotations.groupby(['symbol', 'timeframe'])
        else:
            # Single group if no symbol/timeframe
            groups = [(('ALL', 'ALL'), annotations)]

        for (symbol, timeframe), group_annotations in groups:
            # Get corresponding OHLC data
            if 'symbol' in ohlc_data.columns:
                ohlc_subset = ohlc_data[
                    (ohlc_data['symbol'] == symbol) &
                    (ohlc_data['timeframe'] == timeframe)
                ].sort_index()
            else:
                ohlc_subset = ohlc_data.sort_index()

            if ohlc_subset.empty:
                continue

            # Validate each annotation
            for _, annotation in group_annotations.iterrows():
                result = self._validate_single_pattern(
                    annotation,
                    ohlc_subset,
                    price_column
                )
                if result:
                    validation_results.append(result)

        if validation_results:
            self.validation_results = pd.DataFrame(validation_results)
        else:
            self.validation_results = pd.DataFrame()

        return self.validation_results

    def _validate_single_pattern(
        self,
        annotation: pd.Series,
        ohlc_data: pd.DataFrame,
        price_column: str
    ) -> Optional[Dict]:
        """
        Validate a single pattern annotation.

        Args:
            annotation: Single annotation row
            ohlc_data: OHLC data for this symbol/timeframe
            price_column: Price column to analyze

        Returns:
            Dictionary with validation results or None if cannot validate
        """
        pattern_timestamp = annotation['start_timestamp']

        # Find the pattern timestamp in the OHLC data
        if pattern_timestamp not in ohlc_data.index:
            return None

        pattern_idx = ohlc_data.index.get_loc(pattern_timestamp)
        pattern_price = ohlc_data.iloc[pattern_idx][price_column]

        result = {
            'symbol': annotation.get('symbol', 'N/A'),
            'timeframe': annotation.get('timeframe', 'N/A'),
            'pattern_name': annotation['pattern_name'],
            'pattern_timestamp': pattern_timestamp,
            'pattern_price': pattern_price,
            'confidence': annotation.get('confidence', None)
        }

        # Calculate outcomes for each forecast horizon
        for horizon in self.forecast_horizons:
            future_idx = pattern_idx + horizon

            if future_idx < len(ohlc_data):
                future_price = ohlc_data.iloc[future_idx][price_column]
                future_timestamp = ohlc_data.index[future_idx]

                # Calculate price change
                price_change = future_price - pattern_price
                price_change_pct = (price_change / pattern_price) * 100

                # Determine direction
                if price_change_pct > self.price_change_threshold:
                    direction = 'bullish'
                elif price_change_pct < -self.price_change_threshold:
                    direction = 'bearish'
                else:
                    direction = 'neutral'

                result[f'h{horizon}_price'] = future_price
                result[f'h{horizon}_change'] = price_change
                result[f'h{horizon}_change_pct'] = price_change_pct
                result[f'h{horizon}_direction'] = direction
                result[f'h{horizon}_timestamp'] = future_timestamp
            else:
                # Not enough future data
                result[f'h{horizon}_price'] = None
                result[f'h{horizon}_change'] = None
                result[f'h{horizon}_change_pct'] = None
                result[f'h{horizon}_direction'] = None
                result[f'h{horizon}_timestamp'] = None

        return result

    def calculate_probabilities(
        self,
        validation_results: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate probability of bullish/bearish outcomes for each pattern type.

        Args:
            validation_results: Validation results DataFrame (uses self.validation_results if None)

        Returns:
            DataFrame with probabilities for each pattern and horizon
        """
        if validation_results is None:
            validation_results = self.validation_results

        if validation_results is None or validation_results.empty:
            return pd.DataFrame()

        probabilities = []

        # Group by pattern name
        for pattern_name in validation_results['pattern_name'].unique():
            pattern_data = validation_results[
                validation_results['pattern_name'] == pattern_name
            ]

            prob_row = {
                'pattern_name': pattern_name,
                'total_samples': len(pattern_data)
            }

            # Calculate probabilities for each horizon
            for horizon in self.forecast_horizons:
                direction_col = f'h{horizon}_direction'

                if direction_col in pattern_data.columns:
                    # Get valid samples (non-null)
                    valid_data = pattern_data[pattern_data[direction_col].notna()]
                    n_valid = len(valid_data)

                    if n_valid >= self.require_minimum_samples:
                        # Count outcomes
                        bullish_count = (valid_data[direction_col] == 'bullish').sum()
                        bearish_count = (valid_data[direction_col] == 'bearish').sum()
                        neutral_count = (valid_data[direction_col] == 'neutral').sum()

                        # Calculate probabilities
                        prob_row[f'h{horizon}_samples'] = n_valid
                        prob_row[f'h{horizon}_bullish_prob'] = (bullish_count / n_valid) * 100
                        prob_row[f'h{horizon}_bearish_prob'] = (bearish_count / n_valid) * 100
                        prob_row[f'h{horizon}_neutral_prob'] = (neutral_count / n_valid) * 100

                        # Average price change
                        avg_change = valid_data[f'h{horizon}_change_pct'].mean()
                        prob_row[f'h{horizon}_avg_change_pct'] = avg_change

                        # Win rate (bullish if avg change is positive, bearish if negative)
                        if avg_change > 0:
                            prob_row[f'h{horizon}_win_rate'] = prob_row[f'h{horizon}_bullish_prob']
                        else:
                            prob_row[f'h{horizon}_win_rate'] = prob_row[f'h{horizon}_bearish_prob']
                    else:
                        # Insufficient samples
                        prob_row[f'h{horizon}_samples'] = n_valid
                        prob_row[f'h{horizon}_bullish_prob'] = None
                        prob_row[f'h{horizon}_bearish_prob'] = None
                        prob_row[f'h{horizon}_neutral_prob'] = None
                        prob_row[f'h{horizon}_avg_change_pct'] = None
                        prob_row[f'h{horizon}_win_rate'] = None

            probabilities.append(prob_row)

        return pd.DataFrame(probabilities)

    def calculate_probabilities_by_symbol(
        self,
        validation_results: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate probabilities grouped by symbol and pattern.

        Args:
            validation_results: Validation results DataFrame

        Returns:
            DataFrame with probabilities for each symbol-pattern combination
        """
        if validation_results is None:
            validation_results = self.validation_results

        if validation_results is None or validation_results.empty:
            return pd.DataFrame()

        probabilities = []

        # Group by symbol and pattern
        for (symbol, pattern_name) in validation_results.groupby(['symbol', 'pattern_name']).groups.keys():
            pattern_data = validation_results[
                (validation_results['symbol'] == symbol) &
                (validation_results['pattern_name'] == pattern_name)
            ]

            prob_row = {
                'symbol': symbol,
                'pattern_name': pattern_name,
                'total_samples': len(pattern_data)
            }

            # Calculate probabilities for each horizon
            for horizon in self.forecast_horizons:
                direction_col = f'h{horizon}_direction'

                if direction_col in pattern_data.columns:
                    valid_data = pattern_data[pattern_data[direction_col].notna()]
                    n_valid = len(valid_data)

                    if n_valid >= self.require_minimum_samples:
                        bullish_count = (valid_data[direction_col] == 'bullish').sum()
                        bearish_count = (valid_data[direction_col] == 'bearish').sum()

                        prob_row[f'h{horizon}_samples'] = n_valid
                        prob_row[f'h{horizon}_bullish_prob'] = (bullish_count / n_valid) * 100
                        prob_row[f'h{horizon}_bearish_prob'] = (bearish_count / n_valid) * 100
                        prob_row[f'h{horizon}_avg_change_pct'] = valid_data[f'h{horizon}_change_pct'].mean()
                    else:
                        prob_row[f'h{horizon}_samples'] = n_valid
                        prob_row[f'h{horizon}_bullish_prob'] = None
                        prob_row[f'h{horizon}_bearish_prob'] = None
                        prob_row[f'h{horizon}_avg_change_pct'] = None

            probabilities.append(prob_row)

        return pd.DataFrame(probabilities)

    def generate_probability_report(
        self,
        probabilities: Optional[pd.DataFrame] = None,
        min_win_rate: float = 50.0
    ) -> str:
        """
        Generate a formatted text report of probabilities.

        Args:
            probabilities: Probabilities DataFrame (calculates if None)
            min_win_rate: Minimum win rate to highlight (default 50%)

        Returns:
            Formatted report string
        """
        if probabilities is None:
            probabilities = self.calculate_probabilities()

        if probabilities.empty:
            return "No probability data available."

        lines = []
        lines.append("=" * 80)
        lines.append("PATTERN PROBABILITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        for _, row in probabilities.iterrows():
            pattern_name = row['pattern_name']
            total_samples = row['total_samples']

            lines.append(f"Pattern: {pattern_name}")
            lines.append(f"Total Samples: {total_samples}")
            lines.append("-" * 80)

            # Check if we have valid data for any horizon
            has_data = False
            for horizon in self.forecast_horizons:
                samples_col = f'h{horizon}_samples'
                if samples_col in row and row[samples_col] >= self.require_minimum_samples:
                    has_data = True
                    break

            if not has_data:
                lines.append("  Insufficient data for probability analysis")
                lines.append("")
                continue

            # Show results for each horizon
            lines.append(f"  {'Horizon':<10} {'Samples':<10} {'Bullish %':<12} {'Bearish %':<12} {'Avg Change %':<15} {'Win Rate %':<12}")
            lines.append(f"  {'-'*9} {'-'*9} {'-'*11} {'-'*11} {'-'*14} {'-'*11}")

            for horizon in self.forecast_horizons:
                samples = row.get(f'h{horizon}_samples', 0)

                if samples >= self.require_minimum_samples:
                    bullish_prob = row.get(f'h{horizon}_bullish_prob', 0)
                    bearish_prob = row.get(f'h{horizon}_bearish_prob', 0)
                    avg_change = row.get(f'h{horizon}_avg_change_pct', 0)
                    win_rate = row.get(f'h{horizon}_win_rate', 0)

                    # Highlight good win rates
                    marker = "★" if win_rate >= min_win_rate else " "

                    lines.append(
                        f"{marker} {horizon:<9} {samples:<10} "
                        f"{bullish_prob:>10.1f}% {bearish_prob:>10.1f}% "
                        f"{avg_change:>13.2f}% {win_rate:>10.1f}%"
                    )
                else:
                    lines.append(f"  {horizon:<9} {samples:<10} {'Insufficient data'}")

            lines.append("")

        lines.append("=" * 80)
        lines.append("Legend:")
        lines.append("  ★ = Win rate >= 50%")
        lines.append(f"  Minimum samples required: {self.require_minimum_samples}")
        lines.append(f"  Price change threshold: {self.price_change_threshold}%")
        lines.append("=" * 80)

        return "\n".join(lines)

    def get_best_patterns(
        self,
        probabilities: Optional[pd.DataFrame] = None,
        horizon: int = 5,
        min_win_rate: float = 60.0,
        min_samples: int = 10
    ) -> pd.DataFrame:
        """
        Get patterns with the best predictive performance.

        Args:
            probabilities: Probabilities DataFrame
            horizon: Forecast horizon to analyze
            min_win_rate: Minimum win rate threshold
            min_samples: Minimum samples required

        Returns:
            DataFrame with best performing patterns
        """
        if probabilities is None:
            probabilities = self.calculate_probabilities()

        if probabilities.empty:
            return pd.DataFrame()

        # Filter by horizon
        samples_col = f'h{horizon}_samples'
        win_rate_col = f'h{horizon}_win_rate'

        if samples_col not in probabilities.columns or win_rate_col not in probabilities.columns:
            return pd.DataFrame()

        # Filter
        filtered = probabilities[
            (probabilities[samples_col] >= min_samples) &
            (probabilities[win_rate_col] >= min_win_rate)
        ].copy()

        # Sort by win rate
        filtered = filtered.sort_values(win_rate_col, ascending=False)

        return filtered

    def get_validation_summary(self) -> Dict:
        """
        Get summary statistics of validation results.

        Returns:
            Dictionary with summary statistics
        """
        if self.validation_results is None or self.validation_results.empty:
            return {
                'total_validations': 0,
                'patterns_analyzed': 0,
                'horizons_analyzed': len(self.forecast_horizons)
            }

        summary = {
            'total_validations': len(self.validation_results),
            'patterns_analyzed': self.validation_results['pattern_name'].nunique(),
            'horizons_analyzed': len(self.forecast_horizons),
            'symbols_analyzed': self.validation_results['symbol'].nunique() if 'symbol' in self.validation_results.columns else 1,
            'date_range': (
                self.validation_results['pattern_timestamp'].min(),
                self.validation_results['pattern_timestamp'].max()
            )
        }

        # Count valid samples per horizon
        for horizon in self.forecast_horizons:
            direction_col = f'h{horizon}_direction'
            if direction_col in self.validation_results.columns:
                valid_count = self.validation_results[direction_col].notna().sum()
                summary[f'h{horizon}_valid_samples'] = valid_count

        return summary
