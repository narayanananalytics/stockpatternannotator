"""
Data pipeline for fetching, storing, and annotating stock data.
"""

from typing import List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import warnings

from .polygon_client import PolygonClient, create_polygon_client
from .database import DatabaseManager, create_database_manager
from .annotator import PatternAnnotator
from .patterns import PatternConfig
from .pivots import PivotDetector
from .validation import PatternValidator


class DataPipeline:
    """
    Complete pipeline for fetching data from Polygon.io, storing in database, and running annotations.
    """

    def __init__(
        self,
        polygon_api_key: Optional[str] = None,
        database_url: Optional[str] = None,
        pattern_config: Optional[PatternConfig] = None,
        pivot_detector: Optional[PivotDetector] = None
    ):
        """
        Initialize data pipeline.

        Args:
            polygon_api_key: Polygon.io API key (or set POLYGON_API_KEY env var)
            database_url: Database connection string (or set DATABASE_URL env var)
            pattern_config: Pattern detection configuration
            pivot_detector: Pivot detection configuration
        """
        self.polygon_client = create_polygon_client(polygon_api_key) if polygon_api_key or self._has_polygon_key() else None
        self.db_manager = create_database_manager(database_url)
        self.annotator = PatternAnnotator(
            pattern_config=pattern_config,
            pivot_detector=pivot_detector
        )

    def _has_polygon_key(self) -> bool:
        """Check if polygon API key is available in environment."""
        import os
        return bool(os.getenv('POLYGON_API_KEY'))

    def fetch_and_store(
        self,
        tickers: Union[str, List[str]],
        timespan: str,
        from_date: Union[str, datetime],
        to_date: Union[str, datetime],
        multiplier: int = 1,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data from Polygon.io and store in database.

        Args:
            tickers: Single ticker or list of tickers
            timespan: Time window size ('minute', 'hour', 'day', etc.)
            from_date: Start date
            to_date: End date
            multiplier: Timespan multiplier
            adjusted: Whether to adjust for splits

        Returns:
            DataFrame with fetched data
        """
        if not self.polygon_client:
            raise ValueError("Polygon.io client not initialized. Provide API key or set POLYGON_API_KEY environment variable.")

        # Convert single ticker to list
        if isinstance(tickers, str):
            tickers = [tickers]

        # Fetch data
        print(f"Fetching data for {len(tickers)} ticker(s) from Polygon.io...")
        data = self.polygon_client.get_multiple_tickers(
            tickers=tickers,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date,
            multiplier=multiplier,
            adjusted=adjusted
        )

        if data.empty:
            warnings.warn("No data fetched from Polygon.io")
            return data

        print(f"Fetched {len(data)} records")

        # Store in database
        print("Storing data in database...")
        rows_saved = self.db_manager.save_ohlc_data(data, if_exists='append')
        print(f"Saved {rows_saved} new records to database")

        return data

    def annotate_from_database(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        detect_patterns: bool = True,
        detect_pivots: bool = True,
        save_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Load data from database, run pattern detection, and optionally save annotations.

        Args:
            symbol: Filter by symbol (if None, process all symbols)
            timeframe: Filter by timeframe (if None, process all timeframes)
            start_date: Start date filter
            end_date: End date filter
            detect_patterns: Whether to detect patterns
            detect_pivots: Whether to detect pivots
            save_to_db: Whether to save annotations to database

        Returns:
            DataFrame with annotations
        """
        # Load data from database
        print("Loading data from database...")
        data = self.db_manager.load_ohlc_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if data.empty:
            warnings.warn("No data found in database matching criteria")
            return pd.DataFrame()

        print(f"Loaded {len(data)} records")

        # Determine if we need to process multiple symbols/timeframes
        has_multiple = 'symbol' in data.columns or 'timeframe' in data.columns

        # Run annotations
        print("Running pattern detection...")
        if has_multiple:
            annotations = self.annotator.annotate_multiple(
                data=data,
                symbol_col='symbol',
                timeframe_col='timeframe',
                detect_patterns=detect_patterns,
                detect_pivots=detect_pivots
            )
        else:
            annotations = self.annotator.annotate(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                detect_patterns=detect_patterns,
                detect_pivots=detect_pivots
            )

        print(f"Detected {len(annotations)} patterns/pivots")

        # Save to database if requested
        if save_to_db and not annotations.empty:
            print("Saving annotations to database...")
            rows_saved = self.db_manager.save_annotations(annotations, if_exists='append')
            print(f"Saved {rows_saved} annotations to database")

        return annotations

    def run_full_pipeline(
        self,
        tickers: Union[str, List[str]],
        timespan: str,
        from_date: Union[str, datetime],
        to_date: Union[str, datetime],
        multiplier: int = 1,
        detect_patterns: bool = True,
        detect_pivots: bool = True
    ) -> pd.DataFrame:
        """
        Run complete pipeline: fetch -> store -> annotate.

        Args:
            tickers: Single ticker or list of tickers
            timespan: Time window size
            from_date: Start date
            to_date: End date
            multiplier: Timespan multiplier
            detect_patterns: Whether to detect patterns
            detect_pivots: Whether to detect pivots

        Returns:
            DataFrame with annotations
        """
        print("=" * 70)
        print("RUNNING FULL DATA PIPELINE")
        print("=" * 70)
        print()

        # Step 1: Fetch and store
        print("STEP 1: Fetch data from Polygon.io")
        print("-" * 70)
        data = self.fetch_and_store(
            tickers=tickers,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date,
            multiplier=multiplier
        )
        print()

        if data.empty:
            print("No data fetched. Pipeline stopped.")
            return pd.DataFrame()

        # Step 2: Annotate
        print("STEP 2: Run pattern detection")
        print("-" * 70)

        # Get unique symbols and timeframes from fetched data
        symbols = data['symbol'].unique() if 'symbol' in data.columns else [None]
        timeframes = data['timeframe'].unique() if 'timeframe' in data.columns else [None]

        all_annotations = []

        for symbol in symbols:
            for timeframe in timeframes:
                annotations = self.annotate_from_database(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=from_date if isinstance(from_date, datetime) else pd.to_datetime(from_date),
                    end_date=to_date if isinstance(to_date, datetime) else pd.to_datetime(to_date),
                    detect_patterns=detect_patterns,
                    detect_pivots=detect_pivots,
                    save_to_db=True
                )

                if not annotations.empty:
                    all_annotations.append(annotations)

        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        if all_annotations:
            return pd.concat(all_annotations, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_summary(self) -> dict:
        """
        Get summary of data and annotations in database.

        Returns:
            Dictionary with summary information
        """
        data_summary = self.db_manager.get_data_summary()
        annotation_summary = self.db_manager.get_annotation_summary()

        return {
            'data_summary': data_summary,
            'annotation_summary': annotation_summary,
            'total_symbols': len(self.db_manager.get_available_symbols()),
            'total_timeframes': len(self.db_manager.get_available_timeframes()),
        }

    def print_summary(self):
        """Print a formatted summary of database contents."""
        summary = self.get_summary()

        print("=" * 70)
        print("DATABASE SUMMARY")
        print("=" * 70)
        print()

        print(f"Total Symbols: {summary['total_symbols']}")
        print(f"Total Timeframes: {summary['total_timeframes']}")
        print()

        if not summary['data_summary'].empty:
            print("OHLC Data:")
            print("-" * 70)
            print(summary['data_summary'].to_string(index=False))
            print()

        if not summary['annotation_summary'].empty:
            print("Annotations:")
            print("-" * 70)
            print(summary['annotation_summary'].to_string(index=False))
            print()

        print("=" * 70)

    def update_existing_data(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 7,
        timespan: str = 'day',
        multiplier: int = 1
    ):
        """
        Update existing data in database with recent data.

        Args:
            symbols: List of symbols to update (if None, update all existing symbols)
            days_back: Number of days to fetch
            timespan: Timespan
            multiplier: Multiplier
        """
        if symbols is None:
            symbols = self.db_manager.get_available_symbols()

        if not symbols:
            print("No symbols found in database to update")
            return

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)

        print(f"Updating {len(symbols)} symbols with data from last {days_back} days...")

        self.fetch_and_store(
            tickers=symbols,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date,
            multiplier=multiplier
        )

    def backfill_annotations(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ):
        """
        Regenerate annotations for existing data.

        Args:
            symbol: Filter by symbol (if None, process all)
            timeframe: Filter by timeframe (if None, process all)
        """
        print("Backfilling annotations for existing data...")

        # Delete existing annotations for the specified criteria
        deleted = self.db_manager.delete_annotations(
            symbol=symbol,
            timeframe=timeframe
        )
        print(f"Deleted {deleted} existing annotations")

        # Regenerate annotations
        annotations = self.annotate_from_database(
            symbol=symbol,
            timeframe=timeframe,
            save_to_db=True
        )

        print(f"Generated {len(annotations)} new annotations")

    def export_data(
        self,
        output_dir: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        format: str = 'csv'
    ):
        """
        Export data and annotations to files.

        Args:
            output_dir: Output directory
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            format: Export format ('csv', 'parquet', 'json')
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export OHLC data
        ohlc_data = self.db_manager.load_ohlc_data(symbol=symbol, timeframe=timeframe)
        if not ohlc_data.empty:
            suffix = f"_{symbol}" if symbol else ""
            suffix += f"_{timeframe}" if timeframe else ""

            ohlc_path = os.path.join(output_dir, f'ohlc_data{suffix}.{format}')

            if format == 'csv':
                ohlc_data.to_csv(ohlc_path)
            elif format == 'parquet':
                ohlc_data.to_parquet(ohlc_path)
            elif format == 'json':
                ohlc_data.to_json(ohlc_path, orient='records', date_format='iso')

            print(f"Exported OHLC data to: {ohlc_path}")

        # Export annotations
        annotations = self.db_manager.load_annotations(symbol=symbol, timeframe=timeframe)
        if not annotations.empty:
            suffix = f"_{symbol}" if symbol else ""
            suffix += f"_{timeframe}" if timeframe else ""

            ann_path = os.path.join(output_dir, f'annotations{suffix}.{format}')

            if format == 'csv':
                annotations.to_csv(ann_path, index=False)
            elif format == 'parquet':
                annotations.to_parquet(ann_path, index=False)
            elif format == 'json':
                annotations.to_json(ann_path, orient='records', date_format='iso')

            print(f"Exported annotations to: {ann_path}")

    def validate_patterns(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        forecast_horizons: List[int] = [1, 3, 5, 10],
        price_column: str = 'close',
        calculate_probabilities: bool = True,
        save_to_db: bool = False
    ) -> dict:
        """
        Validate patterns and calculate probability of bullish/bearish outcomes.

        Args:
            symbol: Filter by symbol (None for all)
            timeframe: Filter by timeframe (None for all)
            forecast_horizons: List of candle counts to look forward
            price_column: Price column to analyze
            calculate_probabilities: Whether to calculate probabilities
            save_to_db: Whether to save validation results to database (future feature)

        Returns:
            Dictionary with validation results and probabilities
        """
        print("=" * 70)
        print("PATTERN VALIDATION AND PROBABILITY ANALYSIS")
        print("=" * 70)
        print()

        # Load OHLC data
        print("Loading OHLC data...")
        ohlc_data = self.db_manager.load_ohlc_data(
            symbol=symbol,
            timeframe=timeframe
        )

        if ohlc_data.empty:
            print("No OHLC data found")
            return {
                'validation_results': pd.DataFrame(),
                'probabilities': pd.DataFrame(),
                'summary': {}
            }

        print(f"Loaded {len(ohlc_data)} OHLC records")

        # Load annotations
        print("Loading annotations...")
        annotations = self.db_manager.load_annotations(
            symbol=symbol,
            timeframe=timeframe
        )

        if annotations.empty:
            print("No annotations found")
            return {
                'validation_results': pd.DataFrame(),
                'probabilities': pd.DataFrame(),
                'summary': {}
            }

        print(f"Loaded {len(annotations)} annotations")
        print()

        # Create validator
        print(f"Validating patterns with forecast horizons: {forecast_horizons}")
        validator = PatternValidator(
            forecast_horizons=forecast_horizons,
            price_change_threshold=0.0,
            require_minimum_samples=5
        )

        # Run validation
        validation_results = validator.validate_patterns(
            ohlc_data=ohlc_data,
            annotations=annotations,
            price_column=price_column
        )

        print(f"Validated {len(validation_results)} pattern instances")
        print()

        # Calculate probabilities
        probabilities = pd.DataFrame()
        if calculate_probabilities and not validation_results.empty:
            print("Calculating probabilities...")
            probabilities = validator.calculate_probabilities()
            print(f"Calculated probabilities for {len(probabilities)} pattern types")
            print()

            # Print report
            report = validator.generate_probability_report(probabilities)
            print(report)

        # Get summary
        summary = validator.get_validation_summary()

        return {
            'validation_results': validation_results,
            'probabilities': probabilities,
            'probabilities_by_symbol': validator.calculate_probabilities_by_symbol() if not validation_results.empty else pd.DataFrame(),
            'summary': summary,
            'validator': validator
        }

    def get_pattern_probabilities(
        self,
        pattern_name: Optional[str] = None,
        symbol: Optional[str] = None,
        forecast_horizons: List[int] = [1, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Quick method to get probabilities for specific patterns.

        Args:
            pattern_name: Filter by pattern name (None for all)
            symbol: Filter by symbol (None for all)
            forecast_horizons: Forecast horizons to analyze

        Returns:
            DataFrame with probabilities
        """
        # Load data
        ohlc_data = self.db_manager.load_ohlc_data(symbol=symbol)
        annotations = self.db_manager.load_annotations(
            symbol=symbol,
            pattern_name=pattern_name
        )

        if ohlc_data.empty or annotations.empty:
            return pd.DataFrame()

        # Validate and calculate probabilities
        validator = PatternValidator(forecast_horizons=forecast_horizons)
        validation_results = validator.validate_patterns(ohlc_data, annotations)

        if validation_results.empty:
            return pd.DataFrame()

        return validator.calculate_probabilities()

    def close(self):
        """Close all connections."""
        self.db_manager.close()
