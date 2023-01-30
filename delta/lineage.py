"""
Cell- and lineage-related objects and functions.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

Pole = npt.NDArray[np.int16]


@dataclass
class CellFeatures:
    new_pole: Pole
    """Location of "Young" cell pole created after division of the septum."""
    old_pole: Pole
    """Location of "Old" cell pole that is maintained at division"""
    length: float
    """Cell length, in pixels. Long axis of a rotated bounding box."""
    width: float
    """Cell width, pixels. Short axis of a rotated bounding box."""
    area: float
    """Cell area, in square pixels. Sum of pixels inside the contour."""
    perimeter: float
    """Cell perimeter length, in pixels. Sum of pixels along the contour as 
    computed by opencv's ``arcLength()`` function"""
    fluo: List[float]
    """Average (mean) fluorescence level(s) within cell contour."""
    edges: str
    """String describing image edges touched by the cell. Can be a combination 
    of the following: '-x', '+x', '-y', '+y'. Empty otherwise."""
    growthrate_length: float = np.nan
    """Growth rate of the cell length, see documentation."""
    growthrate_area: float = np.nan
    """Growth rate of the cell area, see documentation."""

    def swap_poles(self) -> None:
        self.new_pole, self.old_pole = self.old_pole, self.new_pole
