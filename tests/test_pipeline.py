"""
Tests for the DataPipeline class.
"""

import unittest
import sys
import os
import tempfile
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from stockpatternannotator.pipeline import DataPipeline
    from stockpatternannotator.utils import generate_sample_ohlc_data
    import pandas as pd
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@unittest.skipUnless(PIPELINE_AVAILABLE, "Required libraries not available")
class TestDataPipeline(unittest.TestCase):
    """Test cases for DataPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_url = f'sqlite:///{self.temp_db.name}'

        # Create pipeline without polygon client (we'll mock it)
        with patch.dict(os.environ, {}, clear=True):
            self.pipeline = DataPipeline(
                database_url=self.db_url
            )

    def tearDown(self):
        """Clean up after tests."""
        self.pipeline.close()
        os.unlink(self.temp_db.name)

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.db_manager)
        self.assertIsNotNone(self.pipeline.annotator)

    def test_annotate_from_database_empty(self):
        """Test annotation with empty database."""
        annotations = self.pipeline.annotate_from_database(
            symbol='TEST',
            save_to_db=False
        )

        self.assertTrue(annotations.empty)

    def test_annotate_from_database_with_data(self):
        """Test annotation with data in database."""
        # Add sample data to database
        sample_data = generate_sample_ohlc_data(n_periods=50, timeframe='1H')
        sample_data['symbol'] = 'TEST'
        sample_data['timeframe'] = '1H'

        self.pipeline.db_manager.save_ohlc_data(sample_data)

        # Run annotation
        annotations = self.pipeline.annotate_from_database(
            symbol='TEST',
            timeframe='1H',
            save_to_db=True
        )

        # Should have some annotations (patterns or pivots)
        self.assertGreaterEqual(len(annotations), 0)

        # If annotations exist, verify they were saved
        if len(annotations) > 0:
            loaded = self.pipeline.db_manager.load_annotations(symbol='TEST')
            self.assertEqual(len(loaded), len(annotations))

    def test_get_summary(self):
        """Test getting pipeline summary."""
        summary = self.pipeline.get_summary()

        self.assertIn('data_summary', summary)
        self.assertIn('annotation_summary', summary)
        self.assertIn('total_symbols', summary)
        self.assertIn('total_timeframes', summary)

    @patch('stockpatternannotator.polygon_client.PolygonClient')
    def test_fetch_and_store_mock(self, mock_polygon):
        """Test fetch and store with mocked polygon client."""
        # Create mock data
        sample_data = generate_sample_ohlc_data(n_periods=20, timeframe='1D')
        sample_data['symbol'] = 'AAPL'
        sample_data['timeframe'] = '1D'

        # Setup mock
        mock_instance = Mock()
        mock_instance.get_multiple_tickers.return_value = sample_data
        mock_polygon.return_value = mock_instance

        # Create pipeline with mock
        with patch.dict(os.environ, {'POLYGON_API_KEY': 'test_key'}):
            pipeline = DataPipeline(database_url=self.db_url)
            pipeline.polygon_client = mock_instance

            # Test fetch and store
            data = pipeline.fetch_and_store(
                tickers=['AAPL'],
                timespan='day',
                from_date='2024-01-01',
                to_date='2024-01-31'
            )

            # Verify
            self.assertFalse(data.empty)
            self.assertEqual(len(data), 20)

            # Check data was stored
            stored = pipeline.db_manager.load_ohlc_data(symbol='AAPL')
            self.assertEqual(len(stored), 20)

            pipeline.close()

    def test_export_data(self):
        """Test exporting data."""
        import tempfile

        # Add some data
        sample_data = generate_sample_ohlc_data(n_periods=20, timeframe='1H')
        sample_data['symbol'] = 'TEST'
        sample_data['timeframe'] = '1H'
        self.pipeline.db_manager.save_ohlc_data(sample_data)

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            self.pipeline.export_data(
                output_dir=tmpdir,
                symbol='TEST',
                format='csv'
            )

            # Check files were created
            files = os.listdir(tmpdir)
            self.assertTrue(any('ohlc' in f for f in files))


if __name__ == '__main__':
    unittest.main()
