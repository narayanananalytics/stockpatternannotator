"""
Standalone pattern validation example.

This example shows how to use PatternValidator directly
without the full pipeline, useful for custom workflows.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import PatternAnnotator, PatternValidator
from stockpatternannotator.utils import generate_sample_ohlc_data
import pandas as pd


def main():
    print("=" * 70)
    print("Standalone Pattern Validation Example")
    print("=" * 70)
    print()

    # Step 1: Generate sample data
    print("Step 1: Generating sample OHLC data...")
    data = generate_sample_ohlc_data(
        n_periods=200,
        timeframe='1D',
        volatility=0.02,
        trend=0.001
    )
    print(f"Generated {len(data)} periods of data")
    print()

    # Step 2: Detect patterns
    print("Step 2: Detecting patterns...")
    annotator = PatternAnnotator()
    annotations = annotator.annotate(
        data=data,
        symbol='SAMPLE',
        timeframe='1D',
        detect_patterns=True,
        detect_pivots=True
    )
    print(f"Detected {len(annotations)} patterns")
    print()

    # Show pattern distribution
    if not annotations.empty:
        print("Pattern distribution:")
        pattern_counts = annotations['pattern_name'].value_counts()
        for pattern, count in pattern_counts.items():
            print(f"  {pattern:30s}: {count:4d}")
        print()

    # Step 3: Validate patterns
    print("Step 3: Validating patterns...")
    print("-" * 70)

    # Create validator with custom horizons
    forecast_horizons = [1, 2, 3, 5, 10, 15, 20]
    validator = PatternValidator(
        forecast_horizons=forecast_horizons,
        price_change_threshold=0.5,  # 0.5% threshold for directional movement
        require_minimum_samples=3     # Need at least 3 samples
    )

    # Run validation
    validation_results = validator.validate_patterns(
        ohlc_data=data,
        annotations=annotations,
        price_column='close'
    )

    print(f"Validated {len(validation_results)} pattern instances")
    print()

    # Step 4: Calculate probabilities
    print("Step 4: Calculating probabilities...")
    probabilities = validator.calculate_probabilities()

    if not probabilities.empty:
        print(f"Calculated probabilities for {len(probabilities)} pattern types")
        print()

        # Show detailed results
        print("=" * 70)
        print("PROBABILITY RESULTS")
        print("=" * 70)
        print()

        for _, row in probabilities.iterrows():
            pattern_name = row['pattern_name']
            total_samples = row['total_samples']

            print(f"Pattern: {pattern_name} (Total samples: {total_samples})")
            print("-" * 70)

            # Show results for each horizon
            has_data = False
            for horizon in forecast_horizons:
                samples = row.get(f'h{horizon}_samples', 0)
                if samples >= validator.require_minimum_samples:
                    has_data = True
                    bullish_prob = row.get(f'h{horizon}_bullish_prob', 0)
                    bearish_prob = row.get(f'h{horizon}_bearish_prob', 0)
                    avg_change = row.get(f'h{horizon}_avg_change_pct', 0)

                    print(f"  {horizon:2d} candles ahead: "
                          f"Bullish={bullish_prob:5.1f}% | "
                          f"Bearish={bearish_prob:5.1f}% | "
                          f"Avg Change={avg_change:+6.2f}% | "
                          f"Samples={samples}")

            if not has_data:
                print("  Insufficient data for analysis")

            print()

    # Step 5: Generate report
    print("=" * 70)
    print("DETAILED PROBABILITY REPORT")
    report = validator.generate_probability_report(probabilities, min_win_rate=50.0)
    print(report)
    print()

    # Step 6: Analyze best patterns
    print("=" * 70)
    print("BEST PERFORMING PATTERNS")
    print("=" * 70)
    print()

    for horizon in [5, 10]:
        best_patterns = validator.get_best_patterns(
            probabilities=probabilities,
            horizon=horizon,
            min_win_rate=50.0,
            min_samples=3
        )

        if not best_patterns.empty:
            print(f"\nBest patterns for {horizon}-candle forecast:")
            print("-" * 70)

            for _, pattern in best_patterns.iterrows():
                win_rate = pattern[f'h{horizon}_win_rate']
                avg_change = pattern[f'h{horizon}_avg_change_pct']
                samples = pattern[f'h{horizon}_samples']

                print(f"  {pattern['pattern_name']:30s}: "
                      f"Win Rate={win_rate:5.1f}% | "
                      f"Avg Change={avg_change:+6.2f}% | "
                      f"Samples={int(samples)}")
        else:
            print(f"\nNo patterns meet criteria for {horizon}-candle forecast")

    print()

    # Step 7: Show individual validation results (sample)
    print("=" * 70)
    print("SAMPLE VALIDATION RESULTS (First 10)")
    print("=" * 70)
    print()

    if not validation_results.empty:
        # Select columns to display
        display_cols = [
            'pattern_name',
            'pattern_timestamp',
            'pattern_price',
            'h5_price',
            'h5_change_pct',
            'h5_direction'
        ]
        available_cols = [col for col in display_cols if col in validation_results.columns]

        print(validation_results[available_cols].head(10).to_string(index=False))
        print()

    # Step 8: Summary statistics
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    summary = validator.get_validation_summary()

    print(f"Total validations: {summary['total_validations']}")
    print(f"Patterns analyzed: {summary['patterns_analyzed']}")
    print(f"Horizons analyzed: {summary['horizons_analyzed']}")

    print()
    print("Valid samples per horizon:")
    for horizon in forecast_horizons:
        key = f'h{horizon}_valid_samples'
        if key in summary:
            print(f"  {horizon:2d} candles: {summary[key]:4d} samples")

    print()
    print("=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
