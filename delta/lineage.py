"""
Module for cell and lineage-related objects and functions.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt

Pole = npt.NDArray[np.int16]


@dataclass
class CellFeatures:
    new_pole: Pole
    old_pole: Pole
    length: float
    width: float
    area: float
    perimeter: float
    fluo: List[float]
    edges: str
    growthrate_length: float = np.nan
    growthrate_area: float = np.nan

    def swap_poles(self) -> None:
        self.new_pole, self.old_pole = self.old_pole, self.new_pole
