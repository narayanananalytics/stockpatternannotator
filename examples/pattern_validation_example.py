"""
Pattern validation and probability analysis example.

This script demonstrates:
- Validating historical pattern predictions
- Calculating probability of bullish/bearish outcomes
- Analyzing pattern effectiveness across different time horizons
- Identifying the most predictive patterns
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockpatternannotator import DataPipeline, PatternValidator


def main():
    print("=" * 70)
    print("Stock Pattern Annotator - Pattern Validation Example")
    print("=" * 70)
    print()

    # Check for API key
    api_key = os.getenv('POLYGON_API_KEY')

    if not api_key:
        print("WARNING: POLYGON_API_KEY not set.")
        print("This example will use existing database data.")
        print("To fetch new data, set POLYGON_API_KEY environment variable.")
        print()

    # Check for database
    if not os.path.exists('stockpatterns_example.db'):
        print("ERROR: Database not found!")
        print("Please run polygon_pipeline_example.py first to create sample data.")
        return

    # Create pipeline
    print("Initializing pipeline...")
    pipeline = DataPipeline(
        polygon_api_key=api_key,
        database_url='sqlite:///stockpatterns_example.db'
    )
    print()

    # If we have API key, fetch some recent data
    if api_key:
        print("Fetching recent data to ensure we have enough for validation...")
        try:
            pipeline.fetch_and_store(
                tickers=['AAPL', 'GOOGL', 'MSFT'],
                timespan='day',
                from_date=datetime.now() - timedelta(days=90),
                to_date=datetime.now(),
                multiplier=1
            )
            print()

            # Run pattern detection
            print("Running pattern detection...")
            pipeline.annotate_from_database(save_to_db=True)
            print()
        except Exception as e:
            print(f"Note: Could not fetch new data: {e}")
            print("Proceeding with existing data...")
            print()

    # Validate patterns with multiple forecast horizons
    print("=" * 70)
    print("STEP 1: Validate patterns across multiple time horizons")
    print("=" * 70)
    print()

    forecast_horizons = [1, 3, 5, 10, 20]  # Look forward 1, 3, 5, 10, 20 candles

    results = pipeline.validate_patterns(
        forecast_horizons=forecast_horizons,
        calculate_probabilities=True
    )

    print()

    # Analyze results
    if results['probabilities'].empty:
        print("No probability data available. Need more historical data.")
        pipeline.close()
        return

    # Show best patterns for different horizons
    print("=" * 70)
    print("STEP 2: Identify best performing patterns")
    print("=" * 70)
    print()

    validator = results['validator']

    for horizon in [5, 10]:
        print(f"\nBest patterns for {horizon}-candle forecast:")
        print("-" * 70)

        best_patterns = validator.get_best_patterns(
            probabilities=results['probabilities'],
            horizon=horizon,
            min_win_rate=55.0,  # At least 55% win rate
            min_samples=5
        )

        if not best_patterns.empty:
            # Show relevant columns
            cols_to_show = [
                'pattern_name',
                f'h{horizon}_samples',
                f'h{horizon}_bullish_prob',
                f'h{horizon}_bearish_prob',
                f'h{horizon}_avg_change_pct',
                f'h{horizon}_win_rate'
            ]

            print(best_patterns[cols_to_show].to_string(index=False))
            print()
        else:
            print("No patterns meet the criteria (55% win rate, 5+ samples)")
            print()

    # Analyze by symbol
    print("=" * 70)
    print("STEP 3: Probability analysis by symbol")
    print("=" * 70)
    print()

    prob_by_symbol = results['probabilities_by_symbol']

    if not prob_by_symbol.empty:
        # Show for a specific pattern if available
        available_patterns = prob_by_symbol['pattern_name'].unique()
        if 'DOJI' in available_patterns:
            print("DOJI pattern performance by symbol (5-candle forecast):")
            print("-" * 70)

            doji_data = prob_by_symbol[prob_by_symbol['pattern_name'] == 'DOJI']
            cols = ['symbol', 'h5_samples', 'h5_bullish_prob', 'h5_bearish_prob', 'h5_avg_change_pct']
            available_cols = [c for c in cols if c in doji_data.columns]

            if available_cols:
                print(doji_data[available_cols].to_string(index=False))
            print()

    # Show validation summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    summary = results['summary']
    print(f"Total validations: {summary['total_validations']}")
    print(f"Patterns analyzed: {summary['patterns_analyzed']}")
    print(f"Symbols analyzed: {summary['symbols_analyzed']}")
    print(f"Horizons analyzed: {summary['horizons_analyzed']}")

    if 'date_range' in summary:
        start, end = summary['date_range']
        print(f"Date range: {start} to {end}")

    print()
    print("Valid samples per horizon:")
    for horizon in forecast_horizons:
        key = f'h{horizon}_valid_samples'
        if key in summary:
            print(f"  {horizon} candles: {summary[key]} samples")

    print()

    # Export validation results
    print("=" * 70)
    print("EXPORTING VALIDATION RESULTS")
    print("=" * 70)
    print()

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Export validation results
    validation_path = os.path.join(output_dir, 'pattern_validation_results.csv')
    results['validation_results'].to_csv(validation_path, index=False)
    print(f"Validation results saved to: {validation_path}")

    # Export probabilities
    prob_path = os.path.join(output_dir, 'pattern_probabilities.csv')
    results['probabilities'].to_csv(prob_path, index=False)
    print(f"Probabilities saved to: {prob_path}")

    # Export probabilities by symbol
    if not prob_by_symbol.empty:
        prob_by_symbol_path = os.path.join(output_dir, 'pattern_probabilities_by_symbol.csv')
        prob_by_symbol.to_csv(prob_by_symbol_path, index=False)
        print(f"Probabilities by symbol saved to: {prob_by_symbol_path}")

    print()

    # Show actionable insights
    print("=" * 70)
    print("ACTIONABLE INSIGHTS")
    print("=" * 70)
    print()

    # Find best overall pattern
    best_overall = results['probabilities'].copy()
    if 'h5_win_rate' in best_overall.columns:
        best_overall = best_overall.sort_values('h5_win_rate', ascending=False)

        if len(best_overall) > 0 and best_overall.iloc[0]['h5_win_rate'] > 50:
            best = best_overall.iloc[0]
            print(f"🎯 Best performing pattern (5-candle forecast):")
            print(f"   Pattern: {best['pattern_name']}")
            print(f"   Win rate: {best['h5_win_rate']:.1f}%")
            print(f"   Average change: {best['h5_avg_change_pct']:.2f}%")
            print(f"   Samples: {int(best['h5_samples'])}")
            print()

    # Show patterns to avoid
    if 'h5_win_rate' in best_overall.columns:
        worst = best_overall[best_overall['h5_win_rate'].notna()].tail(3)

        if len(worst) > 0:
            print(f"⚠️  Patterns with lowest win rates (5-candle forecast):")
            for _, pattern in worst.iterrows():
                if pattern['h5_samples'] >= 5:
                    print(f"   {pattern['pattern_name']}: {pattern['h5_win_rate']:.1f}% "
                          f"(avg change: {pattern['h5_avg_change_pct']:.2f}%)")
            print()

    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()
    print("Use this information to:")
    print("  1. Focus on patterns with high win rates")
    print("  2. Avoid patterns with poor predictive power")
    print("  3. Choose appropriate forecast horizons for your trading strategy")
    print("  4. Understand pattern performance varies by symbol")

    # Clean up
    pipeline.close()


if __name__ == '__main__':
    main()
