"""
Tests for the PatternAnnotator class.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import PatternAnnotator, PatternConfig, PivotDetector
from stockpatternannotator.utils import generate_sample_ohlc_data, validate_ohlc_data


class TestPatternAnnotator(unittest.TestCase):
    """Test cases for PatternAnnotator."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = generate_sample_ohlc_data(n_periods=100, timeframe='1H')
        self.annotator = PatternAnnotator()

    def test_annotator_initialization(self):
        """Test annotator initialization."""
        self.assertIsNotNone(self.annotator)
        self.assertIsInstance(self.annotator.pattern_config, PatternConfig)
        self.assertIsInstance(self.annotator.pivot_detector, PivotDetector)

    def test_annotate_basic(self):
        """Test basic annotation."""
        annotations = self.annotator.annotate(
            data=self.sample_data,
            symbol='TEST',
            timeframe='1H'
        )

        self.assertIsInstance(annotations, pd.DataFrame)
        self.assertTrue(len(annotations) >= 0)

        # Check column names
        expected_cols = ['symbol', 'timeframe', 'pattern_name',
                        'start_timestamp', 'end_timestamp', 'confidence']
        for col in expected_cols:
            self.assertIn(col, annotations.columns)

    def test_annotate_with_invalid_data(self):
        """Test annotation with invalid data."""
        # Missing required columns
        invalid_data = pd.DataFrame({'price': [1, 2, 3]})

        with self.assertRaises(ValueError):
            self.annotator.annotate(invalid_data)

    def test_annotate_patterns_only(self):
        """Test annotation with patterns only."""
        annotations = self.annotator.annotate(
            data=self.sample_data,
            symbol='TEST',
            timeframe='1H',
            detect_patterns=True,
            detect_pivots=False
        )

        # Should have only pattern annotations, no pivots
        pivot_patterns = ['PIVOT_HIGH', 'PIVOT_LOW']
        pivot_count = annotations[annotations['pattern_name'].isin(pivot_patterns)].shape[0]
        self.assertEqual(pivot_count, 0)

    def test_annotate_pivots_only(self):
        """Test annotation with pivots only."""
        annotations = self.annotator.annotate(
            data=self.sample_data,
            symbol='TEST',
            timeframe='1H',
            detect_patterns=False,
            detect_pivots=True
        )

        # Should have only pivot annotations
        if len(annotations) > 0:
            pivot_patterns = ['PIVOT_HIGH', 'PIVOT_LOW']
            for pattern in annotations['pattern_name'].unique():
                self.assertIn(pattern, pivot_patterns)

    def test_filter_annotations(self):
        """Test annotation filtering."""
        # First annotate
        self.annotator.annotate(
            data=self.sample_data,
            symbol='TEST',
            timeframe='1H'
        )

        # Filter by pattern name
        doji_annotations = self.annotator.filter_annotations(
            pattern_names=['DOJI']
        )

        if len(doji_annotations) > 0:
            self.assertTrue(all(doji_annotations['pattern_name'] == 'DOJI'))

        # Filter by symbol
        symbol_annotations = self.annotator.filter_annotations(symbol='TEST')
        if len(symbol_annotations) > 0:
            self.assertTrue(all(symbol_annotations['symbol'] == 'TEST'))

    def test_export_annotations(self):
        """Test exporting annotations."""
        import tempfile

        # Annotate
        self.annotator.annotate(
            data=self.sample_data,
            symbol='TEST',
            timeframe='1H'
        )

        # Export to CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            csv_path = f.name

        try:
            self.annotator.export_annotations(csv_path, format='csv')
            self.assertTrue(os.path.exists(csv_path))

            # Verify exported data
            exported = pd.read_csv(csv_path)
            self.assertEqual(len(exported), len(self.annotator.annotations))
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_annotate_multiple(self):
        """Test multiple symbol/timeframe annotation."""
        # Create multi-symbol data
        data1 = generate_sample_ohlc_data(n_periods=50, timeframe='1H')
        data1['symbol'] = 'AAPL'
        data1['timeframe'] = '1H'

        data2 = generate_sample_ohlc_data(n_periods=50, timeframe='1D')
        data2['symbol'] = 'GOOGL'
        data2['timeframe'] = '1D'

        combined = pd.concat([data1, data2])

        # Annotate
        annotations = self.annotator.annotate_multiple(
            data=combined,
            symbol_col='symbol',
            timeframe_col='timeframe'
        )

        self.assertIsInstance(annotations, pd.DataFrame)

        # Check both symbols present if annotations exist
        if len(annotations) > 0:
            symbols = annotations['symbol'].unique()
            self.assertTrue(len(symbols) > 0)


class TestPatternConfig(unittest.TestCase):
    """Test cases for PatternConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PatternConfig()

        self.assertIsInstance(config.single_candle_patterns, list)
        self.assertIsInstance(config.multi_candle_patterns, list)
        self.assertEqual(config.min_similarity, 0.85)
        self.assertTrue(len(config.get_all_patterns()) > 0)

    def test_custom_config(self):
        """Test custom configuration."""
        config = PatternConfig(
            single_candle_patterns=['DOJI'],
            multi_candle_patterns=['ENGULFING'],
            min_similarity=0.90
        )

        self.assertEqual(config.single_candle_patterns, ['DOJI'])
        self.assertEqual(config.multi_candle_patterns, ['ENGULFING'])
        self.assertEqual(config.min_similarity, 0.90)

    def test_pattern_window_size(self):
        """Test pattern window size calculation."""
        config = PatternConfig()

        # Single candle should have window 1
        self.assertEqual(config.get_pattern_window_size('DOJI'), 1)

        # Multi-candle should have window > 1
        self.assertEqual(config.get_pattern_window_size('ENGULFING'), 2)
        self.assertEqual(config.get_pattern_window_size('MORNING_STAR'), 3)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_generate_sample_data(self):
        """Test sample data generation."""
        data = generate_sample_ohlc_data(n_periods=100, timeframe='1H')

        self.assertEqual(len(data), 100)
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        self.assertIn('open', data.columns)
        self.assertIn('high', data.columns)
        self.assertIn('low', data.columns)
        self.assertIn('close', data.columns)

    def test_validate_ohlc_data(self):
        """Test OHLC data validation."""
        # Valid data
        valid_data = generate_sample_ohlc_data(n_periods=50)
        is_valid, message = validate_ohlc_data(valid_data)
        self.assertTrue(is_valid)

        # Invalid data - missing columns
        invalid_data = pd.DataFrame({'price': [1, 2, 3]})
        is_valid, message = validate_ohlc_data(invalid_data)
        self.assertFalse(is_valid)

        # Invalid data - wrong index type
        invalid_data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [2, 3, 4],
            'low': [0.5, 1.5, 2.5],
            'close': [1.5, 2.5, 3.5]
        })
        is_valid, message = validate_ohlc_data(invalid_data)
        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()
