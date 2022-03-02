"""Data types."""

from functools import cached_property
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np


class SimulationType(Enum):
    AC = auto()
    DC = auto()
    DISTORTION = auto()
    NOISE = auto()
    OPERATING_POINT = auto()
    POLE_ZERO = auto()
    SENSITIVITY = auto()
    TRANSFER_FUNCTION = auto()
    TRANSIENT = auto()

    @classmethod
    def from_description(cls, description):
        descriptions = {
            "AC Analysis": cls.AC,
            "DC transfer characteristic": cls.DC,
            "DISTORTION - 2nd harmonic": cls.DISTORTION,
            "DISTORTION - 3rd harmonic": cls.DISTORTION,
            "Noise Spectral Density Curves": cls.NOISE,
            "Integrated Noise": cls.NOISE,
            "Operating Point": cls.OPERATING_POINT,
            "Pole-Zero Analysis": cls.POLE_ZERO,
            "Sensitivity Analysis": cls.SENSITIVITY,
            "Transfer Function": cls.TRANSFER_FUNCTION,
            "Transient Analysis": cls.TRANSIENT
        }

        try:
            simtype = descriptions[description]
        except KeyError:
            raise ValueError(f"Unrecognized simulation type description {repr(description)}")

        return simtype


@dataclass(frozen=True)
class Vector:
    """Vector information."""

    name: str
    data: np.ndarray


@dataclass(frozen=True)
class Solution:
    """Solution information."""

    name: str
    type: SimulationType
    vectors: dict

    @property
    def row_names(self):
        return list(self.vectors)

    @cached_property
    def xvector(self):
        for vector in self.vectors.values():
            if self._is_xaxis(vector):
                return vector

        # No axis found (e.g. it's an .op).
        return None

    @cached_property
    def xdata(self):
        try:
            return self.xvector.data
        except AttributeError:
            # No axis found (e.g. it's an .op).
            return None

    @cached_property
    def yvectors(self):
        return {
            name: vector
            for name, vector in self.vectors.items() if not self._is_xaxis(vector)
        }

    @cached_property
    def ydata(self):
        return {name: vector.data for name, vector in self.yvectors.items()}

    def _is_xaxis(self, vector):
        name = vector.name

        if self.type in (
            SimulationType.AC,
            SimulationType.NOISE,
            SimulationType.SENSITIVITY,
            SimulationType.DISTORTION
        ):
            return name == "frequency"
        elif self.type == SimulationType.DC:
            return name.endswith("-sweep")
        elif self.type == SimulationType.TRANSIENT:
            return name == "time"

        return False
