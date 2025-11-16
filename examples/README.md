# Examples

This directory contains example scripts demonstrating how to use the Stock Pattern Annotator.

## Available Examples

### basic_usage.py
Demonstrates the fundamental features:
- Generating sample OHLC data
- Configuring pattern detection
- Detecting candlestick patterns and pivots
- Filtering and exporting annotations

Run with:
```bash
python basic_usage.py
```

### multi_symbol_example.py
Shows advanced usage with multiple symbols and timeframes:
- Processing multiple symbols (AAPL, GOOGL, MSFT)
- Processing multiple timeframes (1H, 4H, 1D)
- Analyzing patterns across different instruments
- Comparing pattern density

Run with:
```bash
python multi_symbol_example.py
```

## Output

Examples create an `output/` directory with exported annotations in CSV format.

## Customization

Feel free to modify these examples to suit your needs:
- Change the pattern selection
- Adjust similarity thresholds
- Use your own OHLC data
- Experiment with different pivot detection parameters
