"""
Complete pipeline example using Polygon.io integration.

This script demonstrates:
- Fetching data from Polygon.io
- Storing data in database
- Running pattern detection
- Querying and analyzing results
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import DataPipeline, PatternConfig, PivotDetector


def main():
    print("=" * 70)
    print("Stock Pattern Annotator - Polygon.io Pipeline Example")
    print("=" * 70)
    print()

    # IMPORTANT: Set your Polygon.io API key
    # Option 1: Set environment variable POLYGON_API_KEY
    # Option 2: Pass it directly (shown below)

    # For this example, we'll check for environment variable
    api_key = os.getenv('POLYGON_API_KEY')

    if not api_key:
        print("ERROR: Polygon.io API key not found!")
        print()
        print("Please set your API key using one of these methods:")
        print("  1. Environment variable: export POLYGON_API_KEY='your_key_here'")
        print("  2. Create .env file with: POLYGON_API_KEY=your_key_here")
        print()
        print("Get your free API key at: https://polygon.io/")
        return

    # Configure pattern detection
    pattern_config = PatternConfig(
        single_candle_patterns=['DOJI', 'HAMMER', 'SHOOTING_STAR'],
        multi_candle_patterns=[],
        min_similarity=0.80,
        use_parallel=False
    )

    # Configure pivot detection
    pivot_detector = PivotDetector(
        left_bars=5,
        right_bars=5,
        min_strength=0.3
    )

    # Create pipeline
    # Database will be created automatically as SQLite file
    print("Initializing pipeline...")
    pipeline = DataPipeline(
        polygon_api_key=api_key,
        database_url='sqlite:///stockpatterns_example.db',
        pattern_config=pattern_config,
        pivot_detector=pivot_detector
    )
    print("Pipeline initialized")
    print()

    # Define parameters
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    to_date = datetime.now()
    from_date = to_date - timedelta(days=30)  # Last 30 days

    print(f"Fetching data for: {', '.join(tickers)}")
    print(f"Date range: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
    print()

    # Run the complete pipeline
    try:
        annotations = pipeline.run_full_pipeline(
            tickers=tickers,
            timespan='day',  # Daily data
            from_date=from_date,
            to_date=to_date,
            multiplier=1,
            detect_patterns=True,
            detect_pivots=True
        )

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()

        if not annotations.empty:
            print(f"Total patterns detected: {len(annotations)}")
            print()

            # Show pattern breakdown
            pattern_counts = annotations['pattern_name'].value_counts()
            print("Pattern breakdown:")
            for pattern, count in pattern_counts.items():
                print(f"  {pattern:30s}: {count:4d}")
            print()

            # Show by symbol
            print("By symbol:")
            for symbol in tickers:
                symbol_annotations = annotations[annotations['symbol'] == symbol]
                print(f"  {symbol:10s}: {len(symbol_annotations):4d} patterns")
            print()

            # Show recent patterns
            print("Recent patterns (last 10):")
            print("-" * 70)
            recent = annotations.sort_values('start_timestamp', ascending=False).head(10)
            print(recent[['symbol', 'pattern_name', 'start_timestamp', 'confidence']].to_string(index=False))
            print()

        else:
            print("No patterns detected")
            print()

        # Show database summary
        print()
        pipeline.print_summary()

        # Export data
        print()
        print("Exporting data...")
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        pipeline.export_data(output_dir, format='csv')
        print()

        print("=" * 70)
        print("EXAMPLE COMPLETE")
        print("=" * 70)
        print()
        print(f"Database saved to: stockpatterns_example.db")
        print(f"Exported files saved to: {output_dir}")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        pipeline.close()


if __name__ == '__main__':
    main()
