"""
Example for updating existing data and backfilling annotations.

This script demonstrates:
- Updating database with recent data
- Backfilling annotations for existing data
- Incremental data updates
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import DataPipeline, PatternConfig, PivotDetector


def main():
    print("=" * 70)
    print("Stock Pattern Annotator - Data Update Example")
    print("=" * 70)
    print()

    # Check for API key
    api_key = os.getenv('POLYGON_API_KEY')

    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable not set")
        return

    # Check for database
    if not os.path.exists('stockpatterns_example.db'):
        print("ERROR: Database not found!")
        print("Please run polygon_pipeline_example.py first.")
        return

    # Create pipeline
    print("Initializing pipeline...")
    pipeline = DataPipeline(
        polygon_api_key=api_key,
        database_url='sqlite:///stockpatterns_example.db',
        pattern_config=PatternConfig(
            single_candle_patterns=['DOJI', 'HAMMER', 'SHOOTING_STAR'],
            min_similarity=0.80
        ),
        pivot_detector=PivotDetector()
    )
    print()

    # Show current database status
    print("CURRENT DATABASE STATUS")
    pipeline.print_summary()
    print()

    # Update existing symbols with recent data
    print("=" * 70)
    print("UPDATING EXISTING SYMBOLS")
    print("-" * 70)
    print("Fetching last 7 days of data for existing symbols...")

    try:
        pipeline.update_existing_data(
            days_back=7,
            timespan='day',
            multiplier=1
        )
        print("Update complete")
        print()

        # Re-run annotations on new data
        print("Running annotations on updated data...")
        annotations = pipeline.annotate_from_database(
            save_to_db=True
        )
        print(f"Detected {len(annotations)} new patterns")
        print()

    except Exception as e:
        print(f"Error during update: {e}")

    # Backfill annotations (regenerate all annotations)
    print("=" * 70)
    print("BACKFILLING ANNOTATIONS")
    print("-" * 70)
    print("This will delete and regenerate all annotations...")

    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        try:
            pipeline.backfill_annotations()
            print("Backfill complete")
        except Exception as e:
            print(f"Error during backfill: {e}")
    else:
        print("Skipped backfill")
    print()

    # Show updated database status
    print("=" * 70)
    print("UPDATED DATABASE STATUS")
    pipeline.print_summary()

    # Clean up
    pipeline.close()
    print()
    print("Example complete!")


if __name__ == '__main__':
    main()
