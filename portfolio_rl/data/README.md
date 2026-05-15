# Data Layer

This folder contains the market data access layer and the preprocessing steps that convert raw OHLCV histories into model-safe tensors.

## Why data handling is isolated

In a portfolio system, data errors propagate everywhere. If missing dates are handled poorly, the encoder learns distorted patterns. If normalization leaks future information, the reported RL performance is overstated. Keeping loaders and preprocessing isolated makes those assumptions inspectable.

## Components

### `loaders/`

Responsible for fetching raw market histories under a shared interface.

### `preprocessing/`

Responsible for transforming raw histories into aligned and normalized sequences without contaminating evaluation splits.

## Key design decisions

### Shared OHLCV schema

Every loader returns `[open, high, low, close, volume]`. A shared schema matters because the downstream encoder expects the same feature order across assets and experiments.

### Train-only normalization

Normalization is fit on training history and reused out of sample. This is one of the most important anti-leakage decisions in the package. Even a simple mean or standard deviation computed on the full sample can inject future volatility information into the training process.

### Alignment before modeling

Assets trade on different calendars and may have gaps. The alignment stage constructs a common timestamp index, forward-fills only up to a limited horizon, and drops dates that would otherwise require excessive imputation. This is a compromise between preserving sample size and preserving data integrity.

## Theory background

Financial time series are heteroskedastic and non-stationary. That means the scale of returns, volume, and intraday ranges changes over time. Using static transformations is often weaker than local normalization because the model sees values under inconsistent regimes. Rolling statistics do not solve non-stationarity, but they reduce its most obvious scale effects.

Another key issue is asynchronous observations. Multi-asset portfolio decisions are cross-sectional by nature, so the policy state must represent all assets on a common time axis. Alignment is therefore not a convenience step but part of the modeling definition.

## References

- Lopez de Prado (2018), leakage-aware validation in financial ML: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
- pandas documentation for time-series alignment operations: https://pandas.pydata.org/docs/
