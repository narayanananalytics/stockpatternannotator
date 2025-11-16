"""
Database query and analysis example.

This script demonstrates:
- Loading data from database
- Querying annotations
- Analyzing patterns over time
- Working with stored data
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import DatabaseManager
import pandas as pd


def main():
    print("=" * 70)
    print("Stock Pattern Annotator - Database Query Example")
    print("=" * 70)
    print()

    # Connect to database (make sure it exists from previous example)
    db_path = 'sqlite:///stockpatterns_example.db'

    if not os.path.exists('stockpatterns_example.db'):
        print("ERROR: Database not found!")
        print("Please run polygon_pipeline_example.py first to create the database.")
        return

    print("Connecting to database...")
    db = DatabaseManager(database_url=db_path)
    print("Connected")
    print()

    # Get overview
    print("DATABASE OVERVIEW")
    print("-" * 70)
    summary = db.get_data_summary()
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No data found in database")
    print()

    # Get available symbols
    symbols = db.get_available_symbols()
    print(f"Available symbols: {', '.join(symbols)}")
    print()

    if not symbols:
        print("No data in database. Run polygon_pipeline_example.py first.")
        db.close()
        return

    # Query 1: Get all data for a specific symbol
    print("=" * 70)
    print("QUERY 1: OHLC data for AAPL")
    print("-" * 70)
    aapl_data = db.load_ohlc_data(symbol='AAPL', timeframe='1D')
    if not aapl_data.empty:
        print(f"Found {len(aapl_data)} records")
        print()
        print("First 5 records:")
        print(aapl_data.head().to_string())
        print()
        print("Last 5 records:")
        print(aapl_data.tail().to_string())
    else:
        print("No data found for AAPL")
    print()

    # Query 2: Get annotations for a symbol
    print("=" * 70)
    print("QUERY 2: Annotations for AAPL")
    print("-" * 70)
    aapl_annotations = db.load_annotations(symbol='AAPL')
    if not aapl_annotations.empty:
        print(f"Found {len(aapl_annotations)} annotations")
        print()

        # Pattern distribution
        pattern_dist = aapl_annotations['pattern_name'].value_counts()
        print("Pattern distribution:")
        for pattern, count in pattern_dist.items():
            print(f"  {pattern:30s}: {count:4d}")
        print()

        # Recent annotations
        print("Recent annotations:")
        recent = aapl_annotations.sort_values('start_timestamp', ascending=False).head(10)
        print(recent[['pattern_name', 'start_timestamp', 'confidence']].to_string(index=False))
    else:
        print("No annotations found for AAPL")
    print()

    # Query 3: Filter by pattern type
    print("=" * 70)
    print("QUERY 3: All DOJI patterns")
    print("-" * 70)
    doji_patterns = db.load_annotations(pattern_name='DOJI')
    if not doji_patterns.empty:
        print(f"Found {len(doji_patterns)} DOJI patterns")
        print()
        print("By symbol:")
        for symbol in doji_patterns['symbol'].unique():
            count = len(doji_patterns[doji_patterns['symbol'] == symbol])
            avg_conf = doji_patterns[doji_patterns['symbol'] == symbol]['confidence'].mean()
            print(f"  {symbol:10s}: {count:4d} (avg confidence: {avg_conf:.2f})")
    else:
        print("No DOJI patterns found")
    print()

    # Query 4: Date range query
    print("=" * 70)
    print("QUERY 4: Recent data (last 7 days)")
    print("-" * 70)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    recent_annotations = db.load_annotations(
        start_date=start_date,
        end_date=end_date
    )

    if not recent_annotations.empty:
        print(f"Found {len(recent_annotations)} annotations in last 7 days")
        print()
        print("Daily pattern count:")
        recent_annotations['date'] = pd.to_datetime(recent_annotations['start_timestamp']).dt.date
        daily_counts = recent_annotations.groupby('date').size()
        for date, count in daily_counts.items():
            print(f"  {date}: {count}")
    else:
        print("No recent annotations found")
    print()

    # Query 5: Pivot analysis
    print("=" * 70)
    print("QUERY 5: Support and Resistance levels")
    print("-" * 70)
    pivots = db.load_annotations(pattern_name='PIVOT_HIGH')
    pivot_lows = db.load_annotations(pattern_name='PIVOT_LOW')

    print(f"Pivot Highs (Resistance): {len(pivots)}")
    print(f"Pivot Lows (Support): {len(pivot_lows)}")
    print()

    if not pivots.empty:
        print("Recent resistance levels:")
        recent_pivots = pivots.sort_values('start_timestamp', ascending=False).head(5)
        print(recent_pivots[['symbol', 'start_timestamp', 'confidence']].to_string(index=False))
    print()

    # Get annotation summary
    print("=" * 70)
    print("ANNOTATION SUMMARY")
    print("-" * 70)
    ann_summary = db.get_annotation_summary()
    if not ann_summary.empty:
        print(ann_summary.to_string(index=False))
    print()

    # Clean up
    print("=" * 70)
    db.close()
    print("Example complete!")


if __name__ == '__main__':
    main()
