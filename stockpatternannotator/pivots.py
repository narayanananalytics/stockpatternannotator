"""
Pivot detection for support and resistance levels.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


class PivotDetector:
    """Detect pivot points for support and resistance levels."""

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
        min_strength: float = 0.0
    ):
        """
        Initialize pivot detector.

        Args:
            left_bars: Number of bars to the left for pivot detection
            right_bars: Number of bars to the right for pivot detection
            min_strength: Minimum strength threshold for pivot (0.0 to 1.0)
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.min_strength = min_strength

    def detect_pivots(
        self,
        high: pd.Series,
        low: pd.Series,
        close: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Detect pivot highs and lows.

        Args:
            high: High prices
            low: Low prices
            close: Close prices (optional, for strength calculation)

        Returns:
            DataFrame with pivot information
        """
        pivots = []

        # Detect pivot highs
        pivot_highs = self._detect_pivot_highs(high)
        for idx, price in pivot_highs:
            strength = self._calculate_strength(high, idx, is_high=True)
            if strength >= self.min_strength:
                pivots.append({
                    'timestamp': high.index[idx],
                    'type': 'PIVOT_HIGH',
                    'price': price,
                    'strength': strength,
                    'level': 'resistance'
                })

        # Detect pivot lows
        pivot_lows = self._detect_pivot_lows(low)
        for idx, price in pivot_lows:
            strength = self._calculate_strength(low, idx, is_high=False)
            if strength >= self.min_strength:
                pivots.append({
                    'timestamp': low.index[idx],
                    'type': 'PIVOT_LOW',
                    'price': price,
                    'strength': strength,
                    'level': 'support'
                })

        return pd.DataFrame(pivots)

    def _detect_pivot_highs(self, high: pd.Series) -> List[Tuple[int, float]]:
        """Detect pivot high points."""
        pivots = []
        n = len(high)

        for i in range(self.left_bars, n - self.right_bars):
            is_pivot = True

            # Check if this is higher than left bars
            for j in range(i - self.left_bars, i):
                if high.iloc[i] <= high.iloc[j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            # Check if this is higher than right bars
            for j in range(i + 1, i + self.right_bars + 1):
                if high.iloc[i] <= high.iloc[j]:
                    is_pivot = False
                    break

            if is_pivot:
                pivots.append((i, high.iloc[i]))

        return pivots

    def _detect_pivot_lows(self, low: pd.Series) -> List[Tuple[int, float]]:
        """Detect pivot low points."""
        pivots = []
        n = len(low)

        for i in range(self.left_bars, n - self.right_bars):
            is_pivot = True

            # Check if this is lower than left bars
            for j in range(i - self.left_bars, i):
                if low.iloc[i] >= low.iloc[j]:
                    is_pivot = False
                    break

            if not is_pivot:
                continue

            # Check if this is lower than right bars
            for j in range(i + 1, i + self.right_bars + 1):
                if low.iloc[i] >= low.iloc[j]:
                    is_pivot = False
                    break

            if is_pivot:
                pivots.append((i, low.iloc[i]))

        return pivots

    def _calculate_strength(
        self,
        prices: pd.Series,
        idx: int,
        is_high: bool
    ) -> float:
        """
        Calculate pivot strength based on price deviation.

        Args:
            prices: Price series
            idx: Pivot index
            is_high: Whether this is a pivot high

        Returns:
            Strength value between 0 and 1
        """
        window_start = max(0, idx - self.left_bars)
        window_end = min(len(prices), idx + self.right_bars + 1)
        window = prices.iloc[window_start:window_end]

        if len(window) < 2:
            return 0.0

        pivot_price = prices.iloc[idx]
        mean_price = window.mean()
        std_price = window.std()

        if std_price == 0:
            return 0.0

        # Normalize strength (z-score approach)
        deviation = abs(pivot_price - mean_price) / std_price
        strength = min(1.0, deviation / 2.0)  # Cap at 1.0

        return strength

    def identify_support_resistance_zones(
        self,
        pivots_df: pd.DataFrame,
        price_tolerance: float = 0.02
    ) -> pd.DataFrame:
        """
        Group pivots into support and resistance zones.

        Args:
            pivots_df: DataFrame with pivot information
            price_tolerance: Price tolerance for grouping (as percentage)

        Returns:
            DataFrame with support/resistance zones
        """
        if pivots_df.empty:
            return pd.DataFrame(columns=['level_type', 'price', 'count', 'avg_strength', 'touches'])

        zones = []

        # Group by level type (support/resistance)
        for level_type in ['support', 'resistance']:
            level_pivots = pivots_df[pivots_df['level'] == level_type].copy()

            if level_pivots.empty:
                continue

            # Sort by price
            level_pivots = level_pivots.sort_values('price')

            # Group nearby pivots into zones
            current_zone = []
            for _, pivot in level_pivots.iterrows():
                if not current_zone:
                    current_zone.append(pivot)
                else:
                    zone_price = np.mean([p['price'] for p in current_zone])
                    if abs(pivot['price'] - zone_price) / zone_price <= price_tolerance:
                        current_zone.append(pivot)
                    else:
                        # Save current zone and start new one
                        zones.append(self._create_zone_summary(current_zone, level_type))
                        current_zone = [pivot]

            # Don't forget the last zone
            if current_zone:
                zones.append(self._create_zone_summary(current_zone, level_type))

        return pd.DataFrame(zones)

    def _create_zone_summary(self, pivots: List, level_type: str) -> Dict:
        """Create summary for a support/resistance zone."""
        prices = [p['price'] for p in pivots]
        strengths = [p['strength'] for p in pivots]

        return {
            'level_type': level_type,
            'price': np.mean(prices),
            'price_min': np.min(prices),
            'price_max': np.max(prices),
            'count': len(pivots),
            'avg_strength': np.mean(strengths),
            'touches': len(pivots)
        }
