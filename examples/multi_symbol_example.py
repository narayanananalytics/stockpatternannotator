"""
Multi-symbol and multi-timeframe example for Stock Pattern Annotator.

This script demonstrates:
- Processing multiple symbols
- Processing multiple timeframes
- Comparing patterns across symbols/timeframes
"""

import sys
import os
import pandas as pd

# Add parent directory to path to import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import PatternAnnotator, PatternConfig, PivotDetector
from stockpatternannotator.utils import (
    generate_sample_ohlc_data,
    create_pattern_summary_report
)


def main():
    print("=" * 70)
    print("Stock Pattern Annotator - Multi-Symbol/Timeframe Example")
    print("=" * 70)
    print()

    # Step 1: Generate data for multiple symbols and timeframes
    print("Step 1: Generating data for multiple symbols and timeframes...")

    symbols = ['AAPL', 'GOOGL', 'MSFT']
    timeframes = ['1H', '4H', '1D']

    all_data = []

    for symbol in symbols:
        for timeframe in timeframes:
            # Generate sample data
            data = generate_sample_ohlc_data(
                n_periods=150,
                start_date='2024-01-01',
                timeframe=timeframe,
                symbol=symbol,
                volatility=0.015 if timeframe == '1H' else 0.02,
                trend=0.0003 if symbol == 'AAPL' else 0.0002
            )

            # Add symbol and timeframe columns
            data['symbol'] = symbol
            data['timeframe'] = timeframe

            all_data.append(data)

    # Combine all data
    combined_data = pd.concat(all_data)
    print(f"  Generated data for {len(symbols)} symbols x {len(timeframes)} timeframes")
    print(f"  Total records: {len(combined_data)}")
    print()

    # Step 2: Configure annotator
    print("Step 2: Configuring annotator...")
    pattern_config = PatternConfig(
        single_candle_patterns=['DOJI', 'HAMMER', 'SHOOTING_STAR'],
        multi_candle_patterns=[],
        min_similarity=0.75
    )

    pivot_detector = PivotDetector(
        left_bars=5,
        right_bars=5,
        min_strength=0.2
    )

    annotator = PatternAnnotator(
        pattern_config=pattern_config,
        pivot_detector=pivot_detector
    )
    print("  Configuration complete")
    print()

    # Step 3: Annotate all data
    print("Step 3: Annotating all symbol-timeframe combinations...")
    annotations = annotator.annotate_multiple(
        data=combined_data,
        symbol_col='symbol',
        timeframe_col='timeframe',
        detect_patterns=True,
        detect_pivots=True
    )
    print(f"  Total annotations: {len(annotations)}")
    print()

    # Step 4: Analyze results by symbol
    print("Step 4: Analyzing results by symbol...")
    print("-" * 70)

    for symbol in symbols:
        symbol_annotations = annotator.filter_annotations(symbol=symbol)
        print(f"\n{symbol}:")
        print(f"  Total patterns: {len(symbol_annotations)}")

        # Count by pattern type
        pattern_counts = symbol_annotations['pattern_name'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"    {pattern}: {count}")

    print()

    # Step 5: Analyze results by timeframe
    print("Step 5: Analyzing results by timeframe...")
    print("-" * 70)

    for timeframe in timeframes:
        tf_annotations = annotator.filter_annotations(timeframe=timeframe)
        print(f"\n{timeframe}:")
        print(f"  Total patterns: {len(tf_annotations)}")

        # Count by pattern type
        pattern_counts = tf_annotations['pattern_name'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"    {pattern}: {count}")

    print()

    # Step 6: Find symbols with most patterns
    print("Step 6: Pattern density analysis...")
    print("-" * 70)

    pattern_density = annotations.groupby(['symbol', 'timeframe']).size().reset_index(name='pattern_count')
    pattern_density = pattern_density.sort_values('pattern_count', ascending=False)

    print("\nTop 5 symbol-timeframe combinations by pattern count:")
    print(pattern_density.head(5).to_string(index=False))
    print()

    # Step 7: Export results
    print("Step 7: Exporting results...")
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Export all annotations
    csv_path = os.path.join(output_dir, 'multi_symbol_annotations.csv')
    annotator.export_annotations(csv_path, format='csv')
    print(f"  Exported all annotations to: {csv_path}")

    # Export by symbol
    for symbol in symbols:
        symbol_annotations = annotator.filter_annotations(symbol=symbol)
        symbol_path = os.path.join(output_dir, f'annotations_{symbol}.csv')
        symbol_annotations.to_csv(symbol_path, index=False)
        print(f"  Exported {symbol} annotations to: {symbol_path}")

    print()

    # Step 8: Summary report
    print("Step 8: Overall Summary")
    print(create_pattern_summary_report(annotations))

    print("=" * 70)
    print("Multi-symbol example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
