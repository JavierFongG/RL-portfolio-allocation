"""Abstract interface for fetching OHLCV market data frames."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseLoader(ABC):
    """Defines the loader contract used by the rest of the package."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch a date-indexed OHLCV frame for the requested ticker."""

