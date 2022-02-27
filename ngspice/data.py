"""Data types."""

from functools import cached_property
from enum import Enum
from dataclasses import dataclass
import numpy as np


class SimulationType(Enum):
    AC = "AC Analysis"
    DC = "DC transfer characteristic"
    DISTO = "DISTORTION - 2nd harmonic"
    NOISE = "Noise Spectral Density Curves"
    OP = "Operating Point"
    PZ = "Pole-Zero Analysis"
    SENS = "Sensitivity Analysis"
    TF = "Transfer Function"
    TRAN = "Transient Analysis"


@dataclass
class Vector:
    """Vector information."""

    name: str
    data: np.ndarray


@dataclass
class Solution:
    """Solution information."""

    name: str
    type: SimulationType
    vectors: dict[str, Vector]

    def __post_init__(self):
        self.type = SimulationType(self.type)

    @property
    def row_names(self):
        return list(self.vectors)

    @cached_property
    def xaxis(self):
        for vector in self.vectors.values():
            if self._is_xaxis(vector):
                return vector.data

        # No axis found (e.g. it's an .op).
        return None

    @cached_property
    def ydata(self):
        return {
            name: vector.data
            for name, vector in self.vectors.items() if not self._is_xaxis(vector)
        }

    def _is_xaxis(self, vector):
        return vector.name.endswith("-sweep")
