# Data Loaders

This folder defines how raw market data enter the system.

## Files

### `base_loader.py`

Defines the abstract loader contract:

- input: `ticker`, `start`, `end`
- output: `pandas.DataFrame`
- required columns: `open`, `high`, `low`, `close`, `volume`

The abstract interface exists so that changing data vendors does not change the rest of the codebase.

### `price_loader.py`

Implements the loader using `yfinance`. The loader cleans the column names, ensures the required schema exists, sorts the index, and converts values to numeric form.

### `registry.py`

Maps asset classes such as `equity`, `crypto`, `forex`, and `commodity` to loader instances. The current registry uses the same Yahoo-based implementation for all supported asset types, which is practical for a baseline system but intentionally leaves room for specialized loaders later.

## Why use an abstract base loader

Financial research systems typically evolve from one data source to many. An abstract loader prevents the rest of the pipeline from depending on vendor-specific quirks such as column names, timestamp conventions, or symbol syntax.

## Why `yfinance` is acceptable here

Yahoo Finance is not an institutional market data feed, but it is useful for research prototyping because:

- it covers multiple asset classes
- it is widely accessible
- it keeps the baseline easy to run

The main cost is that data quality and field semantics should always be validated before drawing strong conclusions from a backtest.

## References

- yfinance project: https://github.com/ranaroussi/yfinance
- pandas data structures and indexing: https://pandas.pydata.org/docs/
