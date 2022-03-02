"""Solution data tests."""

import numpy as np
from ngspice import run
from ..util import dedent_multiline


def test_data_type():
    """Check axis data are numpy arrays."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 0 1k
            V1 n1 0 DC 1
            .ac dec 10 100 1000
            .end
            """
        )
    )
    ac1 = solutions["ac1"]
    assert isinstance(ac1.xdata, np.ndarray)
    assert all(isinstance(vec.data, np.ndarray) for vec in ac1.yvectors.values())
