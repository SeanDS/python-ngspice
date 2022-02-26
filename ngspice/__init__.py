"""Ngspice for Python."""

# Get package version.
try:
    from ._version import version as __version__
except ImportError:
    raise FileNotFoundError("Could not find _version.py. Ensure you have run setup.")

from .ngspice import run, run_file, Session, SimulationError
from .data import Vector, Solution
