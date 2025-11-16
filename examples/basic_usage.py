"""
Basic usage example for Stock Pattern Annotator.

This script demonstrates:
- Generating sample OHLC data
- Detecting candlestick patterns
- Detecting support/resistance levels
- Exporting annotations
"""

import sys
import os

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import PatternAnnotator, PatternConfig, PivotDetector
from stockpatternannotator.utils import (
    generate_sample_ohlc_data,
    calculate_pattern_statistics,
    create_pattern_summary_report
)


def main():
    print("=" * 70)
    print("Stock Pattern Annotator - Basic Usage Example")
    print("=" * 70)
    print()

    # Step 1: Generate sample OHLC data
    print("Step 1: Generating sample OHLC data...")
    ohlc_data = generate_sample_ohlc_data(
        n_periods=200,
        start_date='2024-01-01',
        timeframe='1H',
        symbol='SAMPLE',
        volatility=0.02,
        trend=0.0005
    )
    print(f"  Generated {len(ohlc_data)} periods of data")
    print(f"  Date range: {ohlc_data.index[0]} to {ohlc_data.index[-1]}")
    print()

    # Step 2: Configure pattern detection
    print("Step 2: Configuring pattern detection...")
    pattern_config = PatternConfig(
        single_candle_patterns=['DOJI', 'HAMMER', 'SHOOTING_STAR'],
        multi_candle_patterns=[],  # Not using multi-candle in this example
        min_similarity=0.80,
        use_parallel=False  # Set to True for large datasets
    )
    print(f"  Enabled patterns: {pattern_config.get_all_patterns()}")
    print()

    # Step 3: Configure pivot detection
    print("Step 3: Configuring pivot detection...")
    pivot_detector = PivotDetector(
        left_bars=5,
        right_bars=5,
        min_strength=0.3
    )
    print(f"  Pivot detection: left_bars=5, right_bars=5, min_strength=0.3")
    print()

    # Step 4: Create annotator and run detection
    print("Step 4: Running pattern and pivot detection...")
    annotator = PatternAnnotator(
        pattern_config=pattern_config,
        pivot_detector=pivot_detector
    )

    annotations = annotator.annotate(
        data=ohlc_data,
        symbol='SAMPLE',
        timeframe='1H',
        detect_patterns=True,
        detect_pivots=True
    )
    print(f"  Detected {len(annotations)} total annotations")
    print()

    # Step 5: Display results
    print("Step 5: Displaying results...")
    print()

    # Show pattern statistics
    stats = calculate_pattern_statistics(annotations)
    if not stats.empty:
        print("Pattern Statistics:")
        print("-" * 70)
        print(stats.to_string(index=False))
        print()

    # Show summary report
    print(create_pattern_summary_report(annotations))
    print()

    # Show sample annotations
    print("Sample Annotations (first 10):")
    print("-" * 70)
    if not annotations.empty:
        print(annotations.head(10).to_string(index=False))
    else:
        print("No annotations found")
    print()

    # Step 6: Filter annotations
    print("Step 6: Filtering annotations...")
    pattern_only = annotator.filter_annotations(
        pattern_names=['DOJI', 'HAMMER', 'SHOOTING_STAR'],
        min_confidence=0.5
    )
    print(f"  Candlestick patterns only: {len(pattern_only)} annotations")

    pivot_only = annotator.filter_annotations(
        pattern_names=['PIVOT_HIGH', 'PIVOT_LOW']
    )
    print(f"  Pivot points only: {len(pivot_only)} annotations")
    print()

    # Step 7: Export annotations
    print("Step 7: Exporting annotations...")
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'annotations.csv')
    annotator.export_annotations(csv_path, format='csv')
    print(f"  Exported to: {csv_path}")
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
