"""Vectorized route geometry for the first 2D AMR simulation milestone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RectangularLoop:
    """Clockwise-free rectangular centerline parameterized by arc length.

    Robots move counter-clockwise from the lower-left corner: east, north,
    west, then south. Dynamics and collision checks remain in route coordinate
    ``s``; this class provides the 2D embedding used for plots and later sensor
    geometry.
    """

    width: float
    height: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("width must be finite and strictly positive")
        if not np.isfinite(self.height) or self.height <= 0.0:
            raise ValueError("height must be finite and strictly positive")

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.width + self.height)

    def to_xy_heading(
        self, s: ArrayLike
    ) -> tuple[float | FloatArray, float | FloatArray, float | FloatArray]:
        """Map route coordinate(s) to ``x``, ``y``, and heading radians."""

        raw = np.asarray(s, dtype=float)
        if not np.all(np.isfinite(raw)):
            raise ValueError("s must contain only finite values")
        arc = np.mod(raw, self.perimeter)
        x = np.empty_like(arc)
        y = np.empty_like(arc)
        heading = np.empty_like(arc)

        first_end = self.width
        second_end = self.width + self.height
        third_end = 2.0 * self.width + self.height

        first = arc < first_end
        second = (arc >= first_end) & (arc < second_end)
        third = (arc >= second_end) & (arc < third_end)
        fourth = arc >= third_end

        x[first] = arc[first]
        y[first] = 0.0
        heading[first] = 0.0

        x[second] = self.width
        y[second] = arc[second] - self.width
        heading[second] = np.pi / 2.0

        x[third] = self.width - (arc[third] - second_end)
        y[third] = self.height
        heading[third] = np.pi

        x[fourth] = 0.0
        y[fourth] = self.height - (arc[fourth] - third_end)
        heading[fourth] = -np.pi / 2.0

        if raw.ndim == 0:
            return float(x), float(y), float(heading)
        return x, y, heading

