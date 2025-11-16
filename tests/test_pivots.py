"""
Tests for the PivotDetector class.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import PivotDetector
from stockpatternannotator.utils import generate_sample_ohlc_data


class TestPivotDetector(unittest.TestCase):
    """Test cases for PivotDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = PivotDetector(left_bars=5, right_bars=5)
        self.sample_data = generate_sample_ohlc_data(n_periods=100, timeframe='1D')

    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.left_bars, 5)
        self.assertEqual(self.detector.right_bars, 5)
        self.assertEqual(self.detector.min_strength, 0.0)

    def test_detect_pivots(self):
        """Test pivot detection."""
        pivots = self.detector.detect_pivots(
            self.sample_data['high'],
            self.sample_data['low']
        )

        self.assertIsInstance(pivots, pd.DataFrame)

        if len(pivots) > 0:
            # Check required columns
            self.assertIn('timestamp', pivots.columns)
            self.assertIn('type', pivots.columns)
            self.assertIn('price', pivots.columns)
            self.assertIn('strength', pivots.columns)
            self.assertIn('level', pivots.columns)

            # Check pivot types
            for pivot_type in pivots['type'].unique():
                self.assertIn(pivot_type, ['PIVOT_HIGH', 'PIVOT_LOW'])

            # Check level types
            for level in pivots['level'].unique():
                self.assertIn(level, ['support', 'resistance'])

    def test_pivot_highs_detection(self):
        """Test detection of pivot highs."""
        # Create simple data with clear high
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        high_prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101,
                      100, 99, 98, 97, 96, 95, 96, 97, 98, 99]

        high_series = pd.Series(high_prices, index=dates)
        low_series = pd.Series([p - 2 for p in high_prices], index=dates)

        pivots = self.detector.detect_pivots(high_series, low_series)

        # Should detect the high at index 5
        pivot_highs = pivots[pivots['type'] == 'PIVOT_HIGH']
        self.assertTrue(len(pivot_highs) > 0)

    def test_pivot_lows_detection(self):
        """Test detection of pivot lows."""
        # Create simple data with clear low
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        low_prices = [100, 99, 98, 97, 96, 95, 96, 97, 98, 99,
                     100, 101, 102, 103, 104, 105, 104, 103, 102, 101]

        low_series = pd.Series(low_prices, index=dates)
        high_series = pd.Series([p + 2 for p in low_prices], index=dates)

        pivots = self.detector.detect_pivots(high_series, low_series)

        # Should detect the low at index 5
        pivot_lows = pivots[pivots['type'] == 'PIVOT_LOW']
        self.assertTrue(len(pivot_lows) > 0)

    def test_min_strength_filter(self):
        """Test minimum strength filtering."""
        # Create detector with high strength threshold
        strong_detector = PivotDetector(left_bars=5, right_bars=5, min_strength=0.8)

        pivots = strong_detector.detect_pivots(
            self.sample_data['high'],
            self.sample_data['low']
        )

        # All detected pivots should have strength >= 0.8
        if len(pivots) > 0:
            self.assertTrue(all(pivots['strength'] >= 0.8))

    def test_support_resistance_zones(self):
        """Test support/resistance zone identification."""
        pivots = self.detector.detect_pivots(
            self.sample_data['high'],
            self.sample_data['low']
        )

        if len(pivots) > 0:
            zones = self.detector.identify_support_resistance_zones(
                pivots,
                price_tolerance=0.02
            )

            self.assertIsInstance(zones, pd.DataFrame)

            if len(zones) > 0:
                # Check required columns
                self.assertIn('level_type', zones.columns)
                self.assertIn('price', zones.columns)
                self.assertIn('count', zones.columns)
                self.assertIn('avg_strength', zones.columns)

    def test_empty_data(self):
        """Test with insufficient data."""
        # Create very short series
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        short_high = pd.Series([100, 101, 102, 101, 100], index=dates)
        short_low = pd.Series([98, 99, 100, 99, 98], index=dates)

        pivots = self.detector.detect_pivots(short_high, short_low)

        # Should return empty or minimal pivots (not enough bars on both sides)
        self.assertIsInstance(pivots, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
