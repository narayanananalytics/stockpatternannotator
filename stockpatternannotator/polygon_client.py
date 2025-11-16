"""
Polygon.io REST API client for fetching OHLC data.
"""

from typing import List, Optional, Dict, Union
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("requests library not available. Install it to use polygon.io integration.")


class PolygonClient:
    """Client for interacting with Polygon.io REST API."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str, rate_limit_delay: float = 0.1):
        """
        Initialize Polygon.io client.

        Args:
            api_key: Polygon.io API key
            rate_limit_delay: Delay between requests in seconds (to respect rate limits)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for polygon.io integration. Install with: pip install requests")

        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def get_aggregates(
        self,
        ticker: str,
        timespan: str,
        from_date: Union[str, datetime],
        to_date: Union[str, datetime],
        multiplier: int = 1,
        limit: int = 50000,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Get aggregate bars (OHLC data) for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            timespan: Size of time window ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            multiplier: Size of the timespan multiplier (e.g., 1 for 1 day, 5 for 5 minutes)
            limit: Maximum number of results
            adjusted: Whether to adjust for splits

        Returns:
            DataFrame with OHLC data
        """
        # Convert dates to string format
        if isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')

        # Build URL
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

        # Parameters
        params = {
            'adjusted': 'true' if adjusted else 'false',
            'sort': 'asc',
            'limit': limit,
            'apiKey': self.api_key
        }

        # Make request with retry logic
        data = self._make_request(url, params)

        # Parse results
        if 'results' not in data or not data['results']:
            warnings.warn(f"No data returned for {ticker} from {from_date} to {to_date}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data['results'])

        # Rename columns to match OHLC convention
        column_mapping = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            't': 'timestamp',
            'n': 'transactions'
        }
        df = df.rename(columns=column_mapping)

        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # Sort by timestamp
        df = df.sort_index()

        # Add metadata
        df['symbol'] = ticker
        df['timeframe'] = self._get_timeframe_string(timespan, multiplier)

        return df

    def get_multiple_tickers(
        self,
        tickers: List[str],
        timespan: str,
        from_date: Union[str, datetime],
        to_date: Union[str, datetime],
        multiplier: int = 1,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLC data for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            timespan: Size of time window
            from_date: Start date
            to_date: End date
            multiplier: Size of the timespan multiplier
            adjusted: Whether to adjust for splits

        Returns:
            Combined DataFrame with all tickers
        """
        all_data = []

        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            try:
                df = self.get_aggregates(
                    ticker=ticker,
                    timespan=timespan,
                    from_date=from_date,
                    to_date=to_date,
                    multiplier=multiplier,
                    adjusted=adjusted
                )

                if not df.empty:
                    all_data.append(df)

                # Respect rate limits
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                warnings.warn(f"Error fetching data for {ticker}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined = pd.concat(all_data, ignore_index=False)
        return combined

    def get_ticker_details(self, ticker: str) -> Dict:
        """
        Get details about a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker details
        """
        url = f"{self.BASE_URL}/v3/reference/tickers/{ticker}"
        params = {'apiKey': self.api_key}

        data = self._make_request(url, params)
        return data.get('results', {})

    def get_market_status(self) -> Dict:
        """
        Get current market status.

        Returns:
            Dictionary with market status information
        """
        url = f"{self.BASE_URL}/v1/marketstatus/now"
        params = {'apiKey': self.api_key}

        data = self._make_request(url, params)
        return data

    def _make_request(
        self,
        url: str,
        params: Dict,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict:
        """
        Make HTTP request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Returns:
            Response data as dictionary

        Raises:
            Exception: If request fails after all retries
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    warnings.warn(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise Exception(f"Request failed after {max_retries} attempts: {e}")

    def _get_timeframe_string(self, timespan: str, multiplier: int) -> str:
        """
        Convert polygon timespan to standard timeframe string.

        Args:
            timespan: Polygon timespan (minute, hour, day, etc.)
            multiplier: Timespan multiplier

        Returns:
            Timeframe string (e.g., '1H', '5m', '1D')
        """
        mapping = {
            'minute': 'm',
            'hour': 'H',
            'day': 'D',
            'week': 'W',
            'month': 'M',
            'quarter': 'Q',
            'year': 'Y'
        }

        suffix = mapping.get(timespan, timespan)
        return f"{multiplier}{suffix}"

    def get_previous_close(self, ticker: str) -> Dict:
        """
        Get previous day's close for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with previous close data
        """
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        params = {'apiKey': self.api_key, 'adjusted': 'true'}

        data = self._make_request(url, params)
        return data.get('results', [{}])[0] if data.get('results') else {}

    def search_tickers(
        self,
        query: str,
        market: str = 'stocks',
        active: bool = True,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search for tickers.

        Args:
            query: Search query
            market: Market type ('stocks', 'crypto', 'fx', 'otc')
            active: Only return active tickers
            limit: Maximum number of results

        Returns:
            List of matching tickers
        """
        url = f"{self.BASE_URL}/v3/reference/tickers"
        params = {
            'search': query,
            'market': market,
            'active': active,
            'limit': limit,
            'apiKey': self.api_key
        }

        data = self._make_request(url, params)
        return data.get('results', [])

    def validate_api_key(self) -> bool:
        """
        Validate the API key by making a simple request.

        Returns:
            True if API key is valid, False otherwise
        """
        try:
            self.get_market_status()
            return True
        except Exception:
            return False


def create_polygon_client(api_key: Optional[str] = None) -> PolygonClient:
    """
    Create a Polygon.io client, with API key from environment if not provided.

    Args:
        api_key: Polygon.io API key (if None, will try to get from environment)

    Returns:
        PolygonClient instance

    Raises:
        ValueError: If no API key is provided or found
    """
    if api_key is None:
        import os
        api_key = os.getenv('POLYGON_API_KEY')

    if not api_key:
        raise ValueError(
            "No Polygon.io API key provided. Either pass api_key parameter or set "
            "POLYGON_API_KEY environment variable."
        )

    return PolygonClient(api_key)
