"""Processor for irregular time series that preserves timestamps and masks.

Unlike :class:`TimeseriesProcessor` which resamples to a uniform grid,
this processor keeps the raw irregular observations along with:
- An observation mask (1 = observed, 0 = missing/padding)
- Normalized timestamps in [0, 1]

This is essential for models like FuseMoE that use temporal attention
(mTAND) to interpolate irregular observations onto a learned grid.

Output schema: ``("value", "mask", "time")``
"""

from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("irregular_timeseries")
class IrregularTimeseriesProcessor(FeatureProcessor):
    """Processor for irregular time series with timestamps and masks.

    Input:
        A tuple of ``(timestamps, values)`` where:
        - ``timestamps``: either ``List[datetime]`` (legacy) or a numeric
          sequence the caller has ALREADY anchored and normalised
        - ``values``: ``np.ndarray`` of shape ``(T, F)`` with NaN for missing

    Output:
        A tuple of three tensors:
        - ``value``: ``(max_len, F)`` — observation values (NaN → 0)
        - ``mask``: ``(max_len, F)`` — 1 where observed, 0 otherwise
        - ``time``: ``(max_len,)`` — timestamps normalized to [0, 1]

    Two timestamp conventions are supported:

    **Numeric (preferred).** The caller passes floats and owns the whole time
    convention — origin and scale. This is the only way to guarantee that two
    fields of the same sample (e.g. vitals and clinical notes) end up on ONE
    clock: the processor sees each field independently and cannot align them.
    Pass ``time_window_hours=1.0`` so this path is a pure pass-through.

    **Datetime (legacy).** Timestamps are anchored at ``timestamps[0]`` — the
    first observation *of that field* — and divided by ``time_window_hours``.
    Because the anchor is per-field, two modalities of the same sample get
    different origins. Kept only for callers that predate the numeric path;
    do not use it for multimodal tasks.

    Args:
        max_len: Maximum number of observations to keep. Longer sequences are
            truncated from the FRONT (the most recent ``max_len`` observations
            are kept); shorter ones are zero-padded.
        time_window_hours: Normalisation denominator for the datetime path,
            and a plain divisor for the numeric path (use 1.0 there).
    """

    def __init__(self, max_len: int = 512, time_window_hours: float = 48.0):
        self.max_len = max_len
        self.time_window_hours = time_window_hours
        self.n_features = None

    def fit(self, samples: Any, field: str) -> None:
        for sample in samples:
            if field in sample and sample[field] is not None:
                _, values = sample[field]
                values = np.asarray(values, dtype=np.float32)
                if values.ndim == 2:
                    self.n_features = values.shape[1]
                    break
                elif values.ndim == 1:
                    self.n_features = 1
                    break

    def process(
        self, value: Tuple[List[datetime], np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return zero tensors for missing modalities (e.g. vitals for non-ICU patients)
        if value is None:
            F = self.n_features or 1
            return (
                torch.zeros(self.max_len, F, dtype=torch.float32),
                torch.zeros(self.max_len, F, dtype=torch.float32),
                torch.zeros(self.max_len, dtype=torch.float32),
            )

        timestamps, values = value
        values = np.asarray(values, dtype=np.float32)

        if values.ndim == 1:
            values = values[:, None]

        T, F = values.shape

        # Build observation mask (1 = observed, 0 = NaN/missing)
        mask = (~np.isnan(values)).astype(np.float32)

        # Replace NaN with 0 in values
        values = np.nan_to_num(values, nan=0.0)

        # Normalized timestamps. Numeric input is already anchored by the
        # caller (one clock for every field of the sample); datetime input
        # falls back to the legacy per-field anchor.
        if len(timestamps) > 0:
            if isinstance(timestamps[0], (datetime, np.datetime64)):
                start_time = timestamps[0]
                time_hours = np.array(
                    [(t - start_time).total_seconds() / 3600.0 for t in timestamps],
                    dtype=np.float32,
                )
            else:
                time_hours = np.asarray(timestamps, dtype=np.float32)
            time_norm = time_hours / self.time_window_hours
        else:
            time_norm = np.zeros(T, dtype=np.float32)

        # Truncate if too long — keep the MOST RECENT observations.
        # Keeping the first max_len would drop the end of the window, which for
        # a prediction task is precisely the informative part: the grid slot the
        # classifier reads is the last one. Timestamps are sorted ascending, so
        # a tail slice is the recent window.
        if T > self.max_len:
            values = values[-self.max_len :]
            mask = mask[-self.max_len :]
            time_norm = time_norm[-self.max_len :]
            T = self.max_len

        # Pad if too short
        if T < self.max_len:
            pad_len = self.max_len - T
            values = np.pad(values, ((0, pad_len), (0, 0)), constant_values=0.0)
            mask = np.pad(mask, ((0, pad_len), (0, 0)), constant_values=0.0)
            time_norm = np.pad(time_norm, (0, pad_len), constant_values=0.0)

        return (
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(time_norm, dtype=torch.float32),
        )

    def size(self) -> int:
        return self.n_features

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value", "mask", "time")

    def dim(self) -> tuple[int, ...]:
        return (2, 2, 1)

    def spatial(self) -> tuple[bool, ...]:
        return (True, False)

    def __repr__(self):
        return (
            f"IrregularTimeseriesProcessor(max_len={self.max_len}, "
            f"time_window_hours={self.time_window_hours})"
        )
