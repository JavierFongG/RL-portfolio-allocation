"""Registry helpers that map asset classes to concrete price loaders."""

from __future__ import annotations

from typing import Dict

from portfolio_rl.data.loaders.base_loader import BaseLoader
from portfolio_rl.data.loaders.price_loader import PriceLoader


_LOADERS: Dict[str, BaseLoader] = {
    "equity": PriceLoader(),
    "crypto": PriceLoader(),
    "forex": PriceLoader(),
    "commodity": PriceLoader(),
}


def get_loader(asset_type: str) -> BaseLoader:
    """Return the loader configured for the requested asset type."""
    try:
        return _LOADERS[asset_type]
    except KeyError as exc:
        valid = ", ".join(sorted(_LOADERS))
        raise KeyError(f"Unknown asset_type={asset_type!r}. Expected one of: {valid}") from exc

