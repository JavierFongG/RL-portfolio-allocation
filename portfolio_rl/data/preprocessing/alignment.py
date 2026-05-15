"""Date alignment helpers for combining multiple asset histories safely."""

from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def align_frames(frames: Iterable[pd.DataFrame], max_forward_fill: int = 1) -> List[pd.DataFrame]:
    """Align all frames on a shared index while respecting a forward-fill limit."""
    aligned_input = [frame.sort_index().copy() for frame in frames]
    if not aligned_input:
        raise ValueError("At least one DataFrame is required for alignment")

    common_index = aligned_input[0].index
    for frame in aligned_input[1:]:
        common_index = common_index.union(frame.index)
    common_index = common_index.sort_values()

    aligned = []
    valid_mask = pd.Series(True, index=common_index)
    for frame in aligned_input:
        reindexed = frame.reindex(common_index).ffill(limit=max_forward_fill)
        valid_mask &= ~reindexed.isnull().any(axis=1)
        aligned.append(reindexed)

    filtered_index = common_index[valid_mask]
    if filtered_index.empty:
        raise ValueError("No common aligned dates remain after applying the forward-fill limit")

    return [frame.loc[filtered_index].copy() for frame in aligned]
