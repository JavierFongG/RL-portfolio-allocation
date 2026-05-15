"""Yahoo Finance-backed loader that returns normalized OHLCV price data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from portfolio_rl.data.loaders.base_loader import BaseLoader

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - import guard
    yf = None
    _YFINANCE_IMPORT_ERROR = exc
else:
    _YFINANCE_IMPORT_ERROR = None


@dataclass
class PriceLoader(BaseLoader):
    """Loads price data from Yahoo Finance and standardizes its schema."""

    auto_adjust: bool = False
    interval: str = "1d"

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return a cleaned OHLCV frame with lowercase columns."""
        if yf is None:  # pragma: no cover - runtime dependency guard
            raise ImportError("yfinance is required to use PriceLoader") from _YFINANCE_IMPORT_ERROR

        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=self.interval,
            auto_adjust=self.auto_adjust,
            progress=False,
            actions=False,
        )
        if raw.empty:
            raise ValueError(f"No data returned for ticker={ticker!r} between {start} and {end}")

        frame = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame = frame.loc[:, [col for col in self.REQUIRED_COLUMNS if col in frame.columns]].copy()
        missing = [col for col in self.REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"Ticker {ticker!r} missing required columns: {missing}")

        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame.sort_index()
        frame = frame.astype(float)
        return frame

