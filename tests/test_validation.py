"""
Tests for the PatternValidator class.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import PatternValidator
from stockpatternannotator.utils import generate_sample_ohlc_data


class TestPatternValidator(unittest.TestCase):
    """Test cases for PatternValidator."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate sample OHLC data
        self.ohlc_data = generate_sample_ohlc_data(
            n_periods=100,
            timeframe='1D',
            volatility=0.02
        )

        # Create sample annotations
        # Add some at various points in the data
        annotations_data = []
        for i in [10, 20, 30, 40, 50]:
            if i < len(self.ohlc_data):
                annotations_data.append({
                    'symbol': 'TEST',
                    'timeframe': '1D',
                    'pattern_name': 'DOJI' if i % 2 == 0 else 'HAMMER',
                    'start_timestamp': self.ohlc_data.index[i],
                    'end_timestamp': self.ohlc_data.index[i],
                    'confidence': 0.85
                })

        self.annotations = pd.DataFrame(annotations_data)

        # Create validator
        self.validator = PatternValidator(
            forecast_horizons=[1, 3, 5],
            price_change_threshold=0.0,
            require_minimum_samples=2
        )

    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.forecast_horizons, [1, 3, 5])
        self.assertEqual(self.validator.price_change_threshold, 0.0)
        self.assertEqual(self.validator.require_minimum_samples, 2)

    def test_validate_patterns(self):
        """Test pattern validation."""
        results = self.validator.validate_patterns(
            ohlc_data=self.ohlc_data,
            annotations=self.annotations
        )

        # Should have results for each annotation
        self.assertEqual(len(results), len(self.annotations))

        # Check required columns
        required_cols = [
            'symbol', 'timeframe', 'pattern_name',
            'pattern_timestamp', 'pattern_price'
        ]
        for col in required_cols:
            self.assertIn(col, results.columns)

        # Check horizon columns
        for horizon in [1, 3, 5]:
            self.assertIn(f'h{horizon}_price', results.columns)
            self.assertIn(f'h{horizon}_change', results.columns)
            self.assertIn(f'h{horizon}_change_pct', results.columns)
            self.assertIn(f'h{horizon}_direction', results.columns)

    def test_validate_single_pattern(self):
        """Test validation of a single pattern."""
        annotation = self.annotations.iloc[0]

        result = self.validator._validate_single_pattern(
            annotation,
            self.ohlc_data,
            'close'
        )

        self.assertIsNotNone(result)
        self.assertEqual(result['pattern_name'], annotation['pattern_name'])
        self.assertIn('h1_direction', result)

    def test_calculate_probabilities(self):
        """Test probability calculation."""
        # First validate patterns
        validation_results = self.validator.validate_patterns(
            ohlc_data=self.ohlc_data,
            annotations=self.annotations
        )

        # Calculate probabilities
        probabilities = self.validator.calculate_probabilities(validation_results)

        self.assertFalse(probabilities.empty)
        self.assertIn('pattern_name', probabilities.columns)
        self.assertIn('total_samples', probabilities.columns)

        # Check horizon probability columns
        for horizon in [1, 3, 5]:
            self.assertIn(f'h{horizon}_bullish_prob', probabilities.columns)
            self.assertIn(f'h{horizon}_bearish_prob', probabilities.columns)

    def test_direction_classification(self):
        """Test that direction is classified correctly."""
        # Create controlled data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        ohlc = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100, 102, 104, 98, 96, 101, 103, 99, 105, 107]  # Varied closes
        }, index=dates)

        annotations = pd.DataFrame([{
            'symbol': 'TEST',
            'timeframe': '1D',
            'pattern_name': 'TEST_PATTERN',
            'start_timestamp': dates[0],
            'end_timestamp': dates[0],
            'confidence': 0.85
        }])

        validator = PatternValidator(
            forecast_horizons=[1, 2],
            price_change_threshold=0.0
        )

        results = validator.validate_patterns(ohlc, annotations)

        # Check direction after 1 day (100 -> 102 = bullish)
        self.assertEqual(results.iloc[0]['h1_direction'], 'bullish')

        # Check direction after 2 days (100 -> 104 = bullish)
        self.assertEqual(results.iloc[0]['h2_direction'], 'bullish')

    def test_insufficient_future_data(self):
        """Test handling when not enough future data available."""
        # Create annotation near the end of data
        last_idx = len(self.ohlc_data) - 2
        annotation = pd.DataFrame([{
            'symbol': 'TEST',
            'timeframe': '1D',
            'pattern_name': 'TEST',
            'start_timestamp': self.ohlc_data.index[last_idx],
            'end_timestamp': self.ohlc_data.index[last_idx],
            'confidence': 0.85
        }])

        validator = PatternValidator(forecast_horizons=[1, 5, 10])
        results = validator.validate_patterns(self.ohlc_data, annotation)

        # h1 should have data, h5 and h10 should be null
        self.assertIsNotNone(results.iloc[0]['h1_direction'])
        self.assertIsNone(results.iloc[0]['h5_direction'])
        self.assertIsNone(results.iloc[0]['h10_direction'])

    def test_generate_probability_report(self):
        """Test probability report generation."""
        validation_results = self.validator.validate_patterns(
            self.ohlc_data,
            self.annotations
        )

        probabilities = self.validator.calculate_probabilities(validation_results)
        report = self.validator.generate_probability_report(probabilities)

        self.assertIsInstance(report, str)
        self.assertIn('PATTERN PROBABILITY ANALYSIS REPORT', report)
        self.assertIn('DOJI', report)

    def test_get_best_patterns(self):
        """Test getting best patterns."""
        validation_results = self.validator.validate_patterns(
            self.ohlc_data,
            self.annotations
        )

        probabilities = self.validator.calculate_probabilities(validation_results)

        best = self.validator.get_best_patterns(
            probabilities,
            horizon=5,
            min_win_rate=0.0,  # Set to 0 to get all patterns
            min_samples=1
        )

        # Should return DataFrame
        self.assertIsInstance(best, pd.DataFrame)

    def test_get_validation_summary(self):
        """Test validation summary."""
        validation_results = self.validator.validate_patterns(
            self.ohlc_data,
            self.annotations
        )

        summary = self.validator.get_validation_summary()

        self.assertIn('total_validations', summary)
        self.assertIn('patterns_analyzed', summary)
        self.assertIn('horizons_analyzed', summary)
        self.assertEqual(summary['total_validations'], len(validation_results))

    def test_empty_annotations(self):
        """Test with empty annotations."""
        empty_annotations = pd.DataFrame()

        results = self.validator.validate_patterns(
            self.ohlc_data,
            empty_annotations
        )

        self.assertTrue(results.empty)

        probabilities = self.validator.calculate_probabilities(results)
        self.assertTrue(probabilities.empty)

    def test_price_change_threshold(self):
        """Test price change threshold for direction classification."""
        # Create data with small price changes
        dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        ohlc = pd.DataFrame({
            'open': [100] * 5,
            'high': [100.5] * 5,
            'low': [99.5] * 5,
            'close': [100, 100.1, 100.2, 99.9, 99.8]  # Small changes
        }, index=dates)

        annotations = pd.DataFrame([{
            'symbol': 'TEST',
            'timeframe': '1D',
            'pattern_name': 'TEST',
            'start_timestamp': dates[0],
            'end_timestamp': dates[0],
            'confidence': 0.85
        }])

        # With 0% threshold, should classify as bullish/bearish
        validator_zero = PatternValidator(
            forecast_horizons=[1],
            price_change_threshold=0.0
        )
        results_zero = validator_zero.validate_patterns(ohlc, annotations)
        self.assertEqual(results_zero.iloc[0]['h1_direction'], 'bullish')

        # With 1% threshold, should classify as neutral
        validator_high = PatternValidator(
            forecast_horizons=[1],
            price_change_threshold=1.0
        )
        results_high = validator_high.validate_patterns(ohlc, annotations)
        self.assertEqual(results_high.iloc[0]['h1_direction'], 'neutral')


if __name__ == '__main__':
    unittest.main()
