"""
Tests for the DatabaseManager class.
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from stockpatternannotator.database import DatabaseManager
    from stockpatternannotator.utils import generate_sample_ohlc_data
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


@unittest.skipUnless(DB_AVAILABLE, "sqlalchemy not available")
class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_url = f'sqlite:///{self.temp_db.name}'
        self.db = DatabaseManager(database_url=self.db_url)

        # Generate sample data
        self.sample_data = generate_sample_ohlc_data(n_periods=50, timeframe='1H')
        self.sample_data['symbol'] = 'TEST'
        self.sample_data['timeframe'] = '1H'

    def tearDown(self):
        """Clean up after tests."""
        self.db.close()
        os.unlink(self.temp_db.name)

    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db.engine)
        self.assertIsNotNone(self.db.metadata)
        self.assertIsNotNone(self.db.ohlc_table)
        self.assertIsNotNone(self.db.annotations_table)

    def test_save_ohlc_data(self):
        """Test saving OHLC data."""
        rows_saved = self.db.save_ohlc_data(self.sample_data)
        self.assertEqual(rows_saved, len(self.sample_data))

        # Verify data was saved
        loaded_data = self.db.load_ohlc_data(symbol='TEST', timeframe='1H')
        self.assertEqual(len(loaded_data), len(self.sample_data))

    def test_save_duplicate_ohlc_data(self):
        """Test that duplicate OHLC data is not saved twice."""
        # Save data first time
        rows_saved_1 = self.db.save_ohlc_data(self.sample_data)
        self.assertEqual(rows_saved_1, len(self.sample_data))

        # Try to save same data again
        rows_saved_2 = self.db.save_ohlc_data(self.sample_data)
        self.assertEqual(rows_saved_2, 0)  # No new rows should be saved

        # Verify total count
        loaded_data = self.db.load_ohlc_data(symbol='TEST', timeframe='1H')
        self.assertEqual(len(loaded_data), len(self.sample_data))

    def test_load_ohlc_data_with_filters(self):
        """Test loading OHLC data with filters."""
        self.db.save_ohlc_data(self.sample_data)

        # Test symbol filter
        data = self.db.load_ohlc_data(symbol='TEST')
        self.assertEqual(len(data), len(self.sample_data))

        # Test non-existent symbol
        data = self.db.load_ohlc_data(symbol='NONEXISTENT')
        self.assertTrue(data.empty)

        # Test date range filter
        midpoint = len(self.sample_data) // 2
        start_date = self.sample_data.index[midpoint]
        data = self.db.load_ohlc_data(symbol='TEST', start_date=start_date)
        self.assertLess(len(data), len(self.sample_data))

    def test_save_annotations(self):
        """Test saving annotations."""
        annotations = pd.DataFrame({
            'symbol': ['TEST', 'TEST'],
            'timeframe': ['1H', '1H'],
            'pattern_name': ['DOJI', 'HAMMER'],
            'start_timestamp': [datetime.now(), datetime.now()],
            'end_timestamp': [datetime.now(), datetime.now()],
            'confidence': [0.85, 0.90]
        })

        rows_saved = self.db.save_annotations(annotations)
        self.assertEqual(rows_saved, len(annotations))

        # Verify
        loaded = self.db.load_annotations(symbol='TEST')
        self.assertEqual(len(loaded), len(annotations))

    def test_load_annotations_with_filters(self):
        """Test loading annotations with filters."""
        annotations = pd.DataFrame({
            'symbol': ['TEST1', 'TEST2', 'TEST1'],
            'timeframe': ['1H', '1H', '1H'],
            'pattern_name': ['DOJI', 'HAMMER', 'DOJI'],
            'start_timestamp': [datetime.now(), datetime.now(), datetime.now()],
            'end_timestamp': [datetime.now(), datetime.now(), datetime.now()],
            'confidence': [0.85, 0.90, 0.88]
        })

        self.db.save_annotations(annotations)

        # Filter by symbol
        test1_annotations = self.db.load_annotations(symbol='TEST1')
        self.assertEqual(len(test1_annotations), 2)

        # Filter by pattern
        doji_annotations = self.db.load_annotations(pattern_name='DOJI')
        self.assertEqual(len(doji_annotations), 2)

        # Combine filters
        test1_doji = self.db.load_annotations(symbol='TEST1', pattern_name='DOJI')
        self.assertEqual(len(test1_doji), 2)

    def test_get_available_symbols(self):
        """Test getting available symbols."""
        # Initially empty
        symbols = self.db.get_available_symbols()
        self.assertEqual(len(symbols), 0)

        # Add data
        self.db.save_ohlc_data(self.sample_data)

        # Check again
        symbols = self.db.get_available_symbols()
        self.assertEqual(len(symbols), 1)
        self.assertIn('TEST', symbols)

    def test_get_data_summary(self):
        """Test getting data summary."""
        self.db.save_ohlc_data(self.sample_data)

        summary = self.db.get_data_summary()
        self.assertFalse(summary.empty)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]['symbol'], 'TEST')
        self.assertEqual(summary.iloc[0]['record_count'], len(self.sample_data))

    def test_get_annotation_summary(self):
        """Test getting annotation summary."""
        annotations = pd.DataFrame({
            'symbol': ['TEST', 'TEST', 'TEST'],
            'timeframe': ['1H', '1H', '1H'],
            'pattern_name': ['DOJI', 'DOJI', 'HAMMER'],
            'start_timestamp': [datetime.now()] * 3,
            'end_timestamp': [datetime.now()] * 3,
            'confidence': [0.85, 0.90, 0.88]
        })

        self.db.save_annotations(annotations)

        summary = self.db.get_annotation_summary()
        self.assertFalse(summary.empty)
        # Should have 2 rows: one for DOJI, one for HAMMER
        self.assertEqual(len(summary), 2)

    def test_delete_ohlc_data(self):
        """Test deleting OHLC data."""
        self.db.save_ohlc_data(self.sample_data)

        # Delete by symbol
        deleted = self.db.delete_ohlc_data(symbol='TEST')
        self.assertEqual(deleted, len(self.sample_data))

        # Verify deletion
        data = self.db.load_ohlc_data(symbol='TEST')
        self.assertTrue(data.empty)

    def test_delete_annotations(self):
        """Test deleting annotations."""
        annotations = pd.DataFrame({
            'symbol': ['TEST'] * 3,
            'timeframe': ['1H'] * 3,
            'pattern_name': ['DOJI', 'DOJI', 'HAMMER'],
            'start_timestamp': [datetime.now()] * 3,
            'end_timestamp': [datetime.now()] * 3,
            'confidence': [0.85, 0.90, 0.88]
        })

        self.db.save_annotations(annotations)

        # Delete by pattern
        deleted = self.db.delete_annotations(pattern_name='DOJI')
        self.assertEqual(deleted, 2)

        # Verify
        remaining = self.db.load_annotations()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining.iloc[0]['pattern_name'], 'HAMMER')


if __name__ == '__main__':
    unittest.main()
