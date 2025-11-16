# Stock Pattern Annotator

A flexible and efficient tool for annotating OHLC (Open, High, Low, Close) candlestick data with common chart patterns and support/resistance levels using vectorbtpro.

## Features

- **Candlestick Pattern Detection**: Detect both single-candle and multi-candle patterns
  - Single-candle: Doji, Hammer, Shooting Star, Inverted Hammer, Spinning Top, etc.
  - Multi-candle: Engulfing, Harami, Morning Star, Evening Star, Three White Soldiers, etc.

- **Support/Resistance Detection**: Identify pivot points representing potential support and resistance levels

- **Flexible Configuration**: Customize which patterns to detect, similarity thresholds, and window sizes

- **Multi-Symbol/Timeframe Support**: Process multiple symbols and timeframes in a single run

- **Parallel Processing**: Utilize vectorbtpro's multithreading capabilities for fast processing

- **Separate Annotation Storage**: Keep pattern annotations separate from OHLC data with full metadata linkage

- **Multiple Export Formats**: Export annotations to CSV, Parquet, or JSON

## Installation

### Prerequisites

- Python 3.8 or higher
- vectorbtpro (requires license for full functionality)
- pandas
- numpy

### Install from source

```bash
git clone https://github.com/narayanananalytics/stockpatternannotator.git
cd stockpatternannotator
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: vectorbtpro requires a commercial license. The tool includes fallback pattern detection methods that work without vectorbtpro, but with reduced functionality.

## Quick Start

### Basic Usage

```python
from stockpatternannotator import PatternAnnotator, PatternConfig, PivotDetector
from stockpatternannotator.utils import generate_sample_ohlc_data

# Generate sample data
ohlc_data = generate_sample_ohlc_data(n_periods=200, timeframe='1H')

# Configure pattern detection
pattern_config = PatternConfig(
    single_candle_patterns=['DOJI', 'HAMMER', 'SHOOTING_STAR'],
    min_similarity=0.85
)

# Configure pivot detection
pivot_detector = PivotDetector(left_bars=5, right_bars=5)

# Create annotator
annotator = PatternAnnotator(
    pattern_config=pattern_config,
    pivot_detector=pivot_detector
)

# Detect patterns and pivots
annotations = annotator.annotate(
    data=ohlc_data,
    symbol='SAMPLE',
    timeframe='1H',
    detect_patterns=True,
    detect_pivots=True
)

# View results
print(annotations)
```

### Multi-Symbol/Timeframe Processing

```python
import pandas as pd
from stockpatternannotator import PatternAnnotator

# Load data with symbol and timeframe columns
# Assuming data has columns: timestamp, open, high, low, close, symbol, timeframe
data = pd.read_csv('your_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp')

# Create annotator with default configuration
annotator = PatternAnnotator()

# Process all symbols and timeframes
annotations = annotator.annotate_multiple(
    data=data,
    symbol_col='symbol',
    timeframe_col='timeframe'
)

# Filter results
aapl_patterns = annotator.filter_annotations(symbol='AAPL')
pivot_points = annotator.filter_annotations(pattern_names=['PIVOT_HIGH', 'PIVOT_LOW'])

# Export results
annotator.export_annotations('output/annotations.csv', format='csv')
```

## Configuration

### Pattern Configuration

```python
from stockpatternannotator import PatternConfig

config = PatternConfig(
    # Select specific patterns
    single_candle_patterns=['DOJI', 'HAMMER', 'SHOOTING_STAR'],
    multi_candle_patterns=['ENGULFING', 'MORNING_STAR'],

    # Set similarity threshold (0.0 to 1.0)
    min_similarity=0.85,

    # Enable parallel processing
    use_parallel=True,

    # Show progress bar
    show_progress=True
)
```

### Pivot Detection Configuration

```python
from stockpatternannotator import PivotDetector

pivot_detector = PivotDetector(
    left_bars=5,      # Number of bars to the left
    right_bars=5,     # Number of bars to the right
    min_strength=0.3  # Minimum pivot strength (0.0 to 1.0)
)
```

## Annotation Format

Annotations are stored in a DataFrame with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | string | Symbol identifier |
| `timeframe` | string | Timeframe (e.g., '1H', '1D') |
| `pattern_name` | string | Pattern name (e.g., 'HAMMER', 'PIVOT_HIGH') |
| `start_timestamp` | datetime | Pattern start time |
| `end_timestamp` | datetime | Pattern end time (same as start for single-candle) |
| `confidence` | float | Confidence score (0.0 to 1.0) |

## Examples

Complete examples are provided in the `examples/` directory:

- **basic_usage.py**: Simple pattern detection with sample data
- **multi_symbol_example.py**: Processing multiple symbols and timeframes

Run examples:

```bash
python examples/basic_usage.py
python examples/multi_symbol_example.py
```

## Supported Patterns

### Single-Candle Patterns
- DOJI
- HAMMER
- INVERTED_HAMMER
- SHOOTING_STAR
- HANGING_MAN
- SPINNING_TOP
- MARUBOZU
- DRAGONFLY_DOJI
- GRAVESTONE_DOJI

### Multi-Candle Patterns
- ENGULFING (Bullish/Bearish)
- HARAMI (Bullish/Bearish)
- PIERCING_LINE
- DARK_CLOUD_COVER
- MORNING_STAR
- EVENING_STAR
- THREE_WHITE_SOLDIERS
- THREE_BLACK_CROWS
- RISING_THREE_METHODS
- FALLING_THREE_METHODS

### Pivot Patterns
- PIVOT_HIGH (Potential resistance)
- PIVOT_LOW (Potential support)

## API Reference

### PatternAnnotator

Main class for pattern annotation.

**Methods:**
- `annotate(data, symbol, timeframe, detect_patterns, detect_pivots)`: Annotate single dataset
- `annotate_multiple(data, symbol_col, timeframe_col, ...)`: Annotate multiple symbols/timeframes
- `filter_annotations(pattern_names, min_confidence, ...)`: Filter annotations
- `export_annotations(filepath, format)`: Export to file

### PatternConfig

Configuration for pattern detection.

**Parameters:**
- `single_candle_patterns`: List of single-candle pattern names
- `multi_candle_patterns`: List of multi-candle pattern names
- `min_similarity`: Minimum similarity threshold (0.0-1.0)
- `use_parallel`: Enable parallel processing
- `show_progress`: Show progress bar

### PivotDetector

Pivot point detection for support/resistance.

**Parameters:**
- `left_bars`: Number of bars to the left for pivot detection
- `right_bars`: Number of bars to the right for pivot detection
- `min_strength`: Minimum pivot strength threshold (0.0-1.0)

**Methods:**
- `detect_pivots(high, low, close)`: Detect pivot points
- `identify_support_resistance_zones(pivots_df, price_tolerance)`: Group pivots into zones

## Utilities

The `utils` module provides helper functions:

- `generate_sample_ohlc_data()`: Generate sample OHLC data for testing
- `load_ohlc_from_csv()`: Load OHLC data from CSV files
- `calculate_pattern_statistics()`: Calculate pattern statistics
- `create_pattern_summary_report()`: Generate text summary report
- `validate_ohlc_data()`: Validate OHLC data format and quality
- `merge_annotations_with_ohlc()`: Merge annotations back with OHLC data

## Data Requirements

Input OHLC data must:
- Have a `DatetimeIndex`
- Contain columns: `open`, `high`, `low`, `close`
- Follow OHLC constraints: `high >= max(open, close)` and `low <= min(open, close)`
- Not contain NaN or negative values

Use `validate_ohlc_data()` to check your data before processing.

## Performance Considerations

For large datasets:
- Enable parallel processing: `pattern_config.use_parallel = True`
- Process data in chunks by symbol/timeframe
- Use appropriate window sizes for pattern detection
- Consider using Parquet format for faster I/O

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [vectorbtpro](https://vectorbt.pro/) for pattern recognition
- Uses [pandas](https://pandas.pydata.org/) for data manipulation

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

## Roadmap

- [ ] Add more pattern types
- [ ] Implement pattern strength scoring
- [ ] Add visualization utilities
- [ ] Support for custom pattern definitions
- [ ] Integration with popular data providers
- [ ] Performance optimizations for very large datasets
- [ ] Real-time pattern detection mode

## Citation

If you use this tool in your research, please cite:

```
Stock Pattern Annotator (2024)
GitHub: https://github.com/narayanananalytics/stockpatternannotator
```
