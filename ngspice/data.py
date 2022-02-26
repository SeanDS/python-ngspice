"""Data types."""

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
    vectors: list[Vector]

    def __post_init__(self):
        self.type = SimulationType(self.type)

    @property
    def row_names(self):
        return [vector.name for vector in self.vectors]
