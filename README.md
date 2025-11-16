# Stock Pattern Annotator

A flexible and efficient tool for annotating OHLC (Open, High, Low, Close) candlestick data with common chart patterns and support/resistance levels using vectorbtpro, with integrated Polygon.io data fetching and database storage.

## Features

### Core Pattern Detection
- **Candlestick Pattern Detection**: Detect both single-candle and multi-candle patterns
  - Single-candle: Doji, Hammer, Shooting Star, Inverted Hammer, Spinning Top, etc.
  - Multi-candle: Engulfing, Harami, Morning Star, Evening Star, Three White Soldiers, etc.

- **Support/Resistance Detection**: Identify pivot points representing potential support and resistance levels

- **Flexible Configuration**: Customize which patterns to detect, similarity thresholds, and window sizes

### Data Integration (v0.2.0)
- **Polygon.io Integration**: Fetch real-time and historical market data directly from Polygon.io REST API
- **Database Storage**: Store OHLC data and annotations in SQLite, PostgreSQL, or MySQL databases
- **Complete Data Pipeline**: Automated workflow from data fetching → storage → pattern detection
- **Incremental Updates**: Update existing data with recent market information
- **Query Interface**: Flexible querying of stored data and annotations

### Pattern Validation & Probability Analysis (NEW in v0.3.0)
- **Backtesting**: Validate historical pattern predictions against actual price movements
- **Probability Calculation**: Calculate bullish/bearish probabilities for each pattern type
- **Multi-Horizon Analysis**: Analyze pattern effectiveness across different time horizons (1, 3, 5, 10+ candles)
- **Win Rate Tracking**: Identify which patterns actually predict price movements
- **Statistical Validation**: Require minimum sample sizes for statistical significance
- **Performance Reports**: Generate detailed reports showing pattern effectiveness

### Performance & Scalability
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

### Polygon.io Integration (Complete Pipeline)

```python
from stockpatternannotator import DataPipeline
from datetime import datetime, timedelta
import os

# Set your Polygon.io API key
os.environ['POLYGON_API_KEY'] = 'your_api_key_here'

# Create pipeline (uses SQLite by default)
pipeline = DataPipeline()

# Run complete pipeline: fetch → store → annotate
tickers = ['AAPL', 'GOOGL', 'MSFT']
to_date = datetime.now()
from_date = to_date - timedelta(days=30)

annotations = pipeline.run_full_pipeline(
    tickers=tickers,
    timespan='day',
    from_date=from_date,
    to_date=to_date,
    detect_patterns=True,
    detect_pivots=True
)

# View summary
pipeline.print_summary()

# Export results
pipeline.export_data('output', format='csv')
```

### Pattern Validation & Probability Analysis (NEW)

```python
from stockpatternannotator import DataPipeline

# Create pipeline with existing database
pipeline = DataPipeline(database_url='sqlite:///stockpatterns.db')

# Validate patterns and calculate probabilities
results = pipeline.validate_patterns(
    forecast_horizons=[1, 3, 5, 10, 20],  # Look forward X candles
    calculate_probabilities=True
)

# View probabilities
probabilities = results['probabilities']
print(probabilities)

# Get best performing patterns
validator = results['validator']
best_patterns = validator.get_best_patterns(
    horizon=5,          # 5-candle forecast
    min_win_rate=60.0,  # At least 60% win rate
    min_samples=10      # At least 10 samples
)

print(best_patterns)

# Example output:
# Pattern: DOJI
#   Horizon  Samples  Bullish %  Bearish %  Avg Change %  Win Rate %
#   1        45       55.6%      44.4%      +0.23%        55.6%
#   3        45       48.9%      51.1%      -0.15%        51.1%
#   5        45       62.2%      37.8%      +0.87%        62.2%  ★
#   10       42       57.1%      42.9%      +1.24%        57.1%  ★
```

### Database Operations

```python
from stockpatternannotator import DatabaseManager
from datetime import datetime, timedelta

# Connect to database
db = DatabaseManager(database_url='sqlite:///stockpatterns.db')

# Query OHLC data
data = db.load_ohlc_data(
    symbol='AAPL',
    timeframe='1D',
    start_date=datetime(2024, 1, 1),
    end_date=datetime.now()
)

# Query annotations
annotations = db.load_annotations(
    symbol='AAPL',
    pattern_name='DOJI'
)

# Get summary
summary = db.get_data_summary()
print(summary)

# Close connection
db.close()
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

### Database Configuration

```python
from stockpatternannotator import DatabaseManager

# SQLite (default, file-based)
db = DatabaseManager(database_url='sqlite:///stockpatterns.db')

# PostgreSQL
db = DatabaseManager(database_url='postgresql://user:password@localhost:5432/stockpatterns')

# MySQL
db = DatabaseManager(database_url='mysql+pymysql://user:password@localhost:3306/stockpatterns')
```

### Polygon.io Configuration

```python
from stockpatternannotator import PolygonClient
import os

# Option 1: Environment variable
os.environ['POLYGON_API_KEY'] = 'your_api_key_here'

# Option 2: Direct initialization
client = PolygonClient(api_key='your_api_key_here', rate_limit_delay=0.1)

# Fetch data
data = client.get_aggregates(
    ticker='AAPL',
    timespan='day',
    from_date='2024-01-01',
    to_date='2024-12-31',
    multiplier=1
)
```

### Environment Variables

Create a `.env` file in your project root:

```bash
# Polygon.io API Key
POLYGON_API_KEY=your_polygon_api_key_here

# Database URL (optional, defaults to SQLite)
DATABASE_URL=sqlite:///stockpatterns.db

# Rate limiting for Polygon.io (optional)
POLYGON_RATE_LIMIT=0.1
```

Then load it in your code:

```python
from dotenv import load_dotenv
load_dotenv()

# Now environment variables are available
from stockpatternannotator import DataPipeline
pipeline = DataPipeline()  # Automatically uses env vars
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

### Core Pattern Detection Examples
- **basic_usage.py**: Simple pattern detection with sample data
- **multi_symbol_example.py**: Processing multiple symbols and timeframes

### Polygon.io Integration Examples
- **polygon_pipeline_example.py**: Complete pipeline with Polygon.io data fetching
- **database_query_example.py**: Querying and analyzing stored data
- **update_data_example.py**: Updating existing data and backfilling annotations

### Pattern Validation Examples (NEW in v0.3.0)
- **pattern_validation_example.py**: Full validation workflow with probability analysis
- **standalone_validation_example.py**: Using PatternValidator independently

Run examples:

```bash
# Basic pattern detection
python examples/basic_usage.py
python examples/multi_symbol_example.py

# Polygon.io integration (requires API key)
export POLYGON_API_KEY='your_key_here'
python examples/polygon_pipeline_example.py
python examples/database_query_example.py

# Pattern validation (requires existing data in database)
python examples/pattern_validation_example.py
python examples/standalone_validation_example.py
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

### DataPipeline

Complete data pipeline for fetching, storing, and annotating.

**Methods:**
- `fetch_and_store(tickers, timespan, from_date, to_date)`: Fetch from Polygon.io and store
- `annotate_from_database(symbol, timeframe, ...)`: Load from DB and annotate
- `run_full_pipeline(tickers, timespan, from_date, to_date)`: Complete fetch→store→annotate workflow
- `validate_patterns(symbol, timeframe, forecast_horizons, ...)`: Validate patterns and calculate probabilities (NEW in v0.3.0)
- `get_pattern_probabilities(pattern_name, symbol, forecast_horizons)`: Quick probability lookup (NEW in v0.3.0)
- `get_summary()`: Get database statistics
- `update_existing_data(symbols, days_back)`: Update with recent data
- `backfill_annotations(symbol, timeframe)`: Regenerate annotations
- `export_data(output_dir, format)`: Export data and annotations

### PatternValidator (NEW in v0.3.0)

Validate patterns by analyzing actual outcomes.

**Parameters:**
- `forecast_horizons`: List of candle counts to look forward (e.g., [1, 3, 5, 10])
- `price_change_threshold`: Minimum % change to consider directional (default 0.0)
- `require_minimum_samples`: Minimum samples for probability calculation (default 5)

**Methods:**
- `validate_patterns(ohlc_data, annotations, price_column)`: Validate all patterns
- `calculate_probabilities(validation_results)`: Calculate probability statistics
- `calculate_probabilities_by_symbol(validation_results)`: Probabilities grouped by symbol
- `generate_probability_report(probabilities, min_win_rate)`: Generate formatted report
- `get_best_patterns(probabilities, horizon, min_win_rate, min_samples)`: Find best performing patterns
- `get_validation_summary()`: Get validation statistics

### PolygonClient (NEW)

Client for Polygon.io REST API.

**Methods:**
- `get_aggregates(ticker, timespan, from_date, to_date)`: Get OHLC data for ticker
- `get_multiple_tickers(tickers, timespan, from_date, to_date)`: Get data for multiple tickers
- `get_ticker_details(ticker)`: Get ticker information
- `get_previous_close(ticker)`: Get previous day's close
- `search_tickers(query)`: Search for tickers

### DatabaseManager (NEW)

Database operations for OHLC data and annotations.

**Methods:**
- `save_ohlc_data(data, if_exists)`: Save OHLC data
- `save_annotations(annotations, if_exists)`: Save annotations
- `load_ohlc_data(symbol, timeframe, start_date, end_date)`: Load OHLC data
- `load_annotations(symbol, timeframe, pattern_name)`: Load annotations
- `get_available_symbols()`: Get list of symbols
- `get_data_summary()`: Get data summary statistics
- `get_annotation_summary()`: Get annotation statistics
- `delete_ohlc_data(symbol, timeframe)`: Delete OHLC data
- `delete_annotations(symbol, timeframe)`: Delete annotations

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

## Use Cases

### Trading Strategy Development
1. **Backtest pattern effectiveness** - Use validation to identify which patterns actually predict price movements
2. **Optimize entry/exit timing** - Analyze different forecast horizons to find optimal holding periods
3. **Filter weak signals** - Eliminate patterns with low win rates or statistical significance
4. **Symbol-specific analysis** - Identify patterns that work better for specific stocks

### Risk Management
- **Set confidence thresholds** - Only trade patterns with proven win rates above your threshold
- **Position sizing** - Scale position size based on pattern probability and historical performance
- **Stop-loss optimization** - Use average price changes to set appropriate stop-loss levels

### Research & Analysis
- **Pattern comparison** - Compare effectiveness of different technical patterns
- **Market regime analysis** - Understand how pattern performance changes over time
- **Statistical validation** - Ensure patterns have statistical significance before trading

## Performance Considerations

For large datasets:
- Enable parallel processing: `pattern_config.use_parallel = True`
- Process data in chunks by symbol/timeframe
- Use appropriate window sizes for pattern detection
- Consider using Parquet format for faster I/O
- Use database indexes for efficient querying

## Roadmap

- [x] Core pattern detection
- [x] Polygon.io integration
- [x] Database storage
- [x] Pattern validation and probability analysis
- [ ] Add more pattern types
- [ ] Pattern strength scoring enhancements
- [ ] Visualization utilities (charts with pattern overlays)
- [ ] Support for custom pattern definitions
- [ ] Real-time pattern detection mode
- [ ] Machine learning pattern classification
- [ ] Integration with additional data providers

## Citation

If you use this tool in your research, please cite:

```
Stock Pattern Annotator (2024)
GitHub: https://github.com/narayanananalytics/stockpatternannotator
```
