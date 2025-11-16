"""
Tests for the PolygonClient class.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from stockpatternannotator.polygon_client import PolygonClient
    POLYGON_CLIENT_AVAILABLE = True
except ImportError:
    POLYGON_CLIENT_AVAILABLE = False


@unittest.skipUnless(POLYGON_CLIENT_AVAILABLE, "requests library not available")
class TestPolygonClient(unittest.TestCase):
    """Test cases for PolygonClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        self.client = PolygonClient(api_key=self.api_key)

    def test_client_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertEqual(self.client.rate_limit_delay, 0.1)

    def test_timeframe_string_conversion(self):
        """Test timeframe string conversion."""
        self.assertEqual(self.client._get_timeframe_string('minute', 1), '1m')
        self.assertEqual(self.client._get_timeframe_string('hour', 1), '1H')
        self.assertEqual(self.client._get_timeframe_string('day', 1), '1D')
        self.assertEqual(self.client._get_timeframe_string('week', 1), '1W')
        self.assertEqual(self.client._get_timeframe_string('minute', 5), '5m')
        self.assertEqual(self.client._get_timeframe_string('hour', 4), '4H')

    @patch('stockpatternannotator.polygon_client.requests.Session.get')
    def test_get_aggregates_success(self, mock_get):
        """Test successful aggregate data fetch."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [
                {
                    'o': 100.0,
                    'h': 105.0,
                    'l': 99.0,
                    'c': 103.0,
                    'v': 1000000,
                    't': 1640995200000,  # 2022-01-01 00:00:00 UTC
                    'n': 100
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test
        df = self.client.get_aggregates(
            ticker='AAPL',
            timespan='day',
            from_date='2022-01-01',
            to_date='2022-01-02',
            multiplier=1
        )

        # Verify
        self.assertFalse(df.empty)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('symbol', df.columns)
        self.assertEqual(df['symbol'].iloc[0], 'AAPL')

    @patch('stockpatternannotator.polygon_client.requests.Session.get')
    def test_get_aggregates_no_data(self, mock_get):
        """Test aggregate fetch with no data."""
        # Mock response with no results
        mock_response = Mock()
        mock_response.json.return_value = {'results': []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test
        df = self.client.get_aggregates(
            ticker='INVALID',
            timespan='day',
            from_date='2022-01-01',
            to_date='2022-01-02'
        )

        # Verify
        self.assertTrue(df.empty)

    def test_date_conversion(self):
        """Test date conversion in get_aggregates."""
        # This is tested implicitly in the success test, but we can add specific checks
        with patch('stockpatternannotator.polygon_client.requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {'results': []}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # Test with datetime objects
            from_date = datetime(2022, 1, 1)
            to_date = datetime(2022, 1, 31)

            self.client.get_aggregates(
                ticker='AAPL',
                timespan='day',
                from_date=from_date,
                to_date=to_date
            )

            # Verify the call was made (dates should be converted to strings)
            self.assertTrue(mock_get.called)


class TestPolygonClientCreation(unittest.TestCase):
    """Test client creation functions."""

    @patch.dict(os.environ, {'POLYGON_API_KEY': 'env_api_key'})
    def test_create_client_from_env(self):
        """Test creating client from environment variable."""
        if not POLYGON_CLIENT_AVAILABLE:
            self.skipTest("requests library not available")

        from stockpatternannotator.polygon_client import create_polygon_client

        client = create_polygon_client()
        self.assertEqual(client.api_key, 'env_api_key')

    def test_create_client_no_key(self):
        """Test creating client without API key raises error."""
        if not POLYGON_CLIENT_AVAILABLE:
            self.skipTest("requests library not available")

        from stockpatternannotator.polygon_client import create_polygon_client

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                create_polygon_client()


if __name__ == '__main__':
    unittest.main()
