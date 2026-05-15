# Preprocessing

This folder converts raw asset histories into aligned, normalized, fixed-length windows suitable for sequence modeling.

## Files

### `normalizer.py`

Implements per-asset rolling z-score normalization with train-only fitting. The object stores historical training context and applies that context when transforming later data, so validation and test windows are normalized consistently without refitting.

### `sequencer.py`

Builds sliding windows with shape `(N, T, F)`, where:

- `N` is the number of windows
- `T` is the lookback length
- `F` is the feature dimension

This is the natural representation expected by the LSTM autoencoder.

### `alignment.py`

Aligns multiple asset data frames on a common date index, forward-fills gaps up to a configurable limit, and drops dates where any asset would require more than the allowed fill budget.

## Theory behind each choice

### Rolling z-score normalization

The idea is to compare the current value of a feature to its recent history rather than to the entire sample. This is especially useful for:

- volatility-sensitive price features
- volume features with regime shifts
- multi-asset universes with very different scales

The implementation uses historical rolling statistics shifted by one step so the normalization at time `t` depends only on information available before `t`.

### Sliding windows

Sequence models need contiguous local histories. A fixed window length is a practical bias: it assumes that the most useful temporal information lies within a bounded lookback horizon. That is rarely perfectly true in finance, but it simplifies training and controls memory usage.

### Limited forward fill

Forward-fill is often necessary because real market panels are sparse or asynchronous. The limit is crucial because unlimited filling silently turns stale prices into fake observations. In portfolio problems, that can understate realized risk and distort correlation structure.

## References

- pandas rolling window operations: https://pandas.pydata.org/docs/reference/window.html
- Lopez de Prado (2018), data leakage and validation discipline: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
