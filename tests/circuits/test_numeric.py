"""Circuit tests with numerical solutions to compare to."""

import numpy as np
import pytest
from ngspice import run
from ..util import dedent_multiline


def test_voltage_divider():
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 n2 1k
            R2 n2 0 2k
            V1 n1 0 DC 1
            .op
            .end
            """
        )
    )

    assert solutions["op1"].yvectors["n1"].data == pytest.approx(np.array([1]))
    assert solutions["op1"].yvectors["n2"].data == pytest.approx(np.array([2/3]))


def test_johnson_noise():
    solutions = run(
        dedent_multiline(
            """
            Noise simulation
            R1 n1 n2 1k
            R2 n2 0 2k
            V1 n1 0 DC 1 AC 1
            .noise v(n2) V1 dec 3 100 1000
            .end
            """
        )
    )

    def vjohnson(r):
        k = 1.380649e-23
        T = 300.15  # == 27°C; the ngspice default.
        return np.sqrt(4 * k * T * r)

    # Noise in the middle of the voltage divider is equivalent to the noise of the equivalent
    # parallel resistance.
    r_eff = 1 / (1 / 1e3 + 1 / 2e3)
    n_eff = vjohnson(r_eff)

    assert np.allclose(
        solutions["noise1"].yvectors["onoise_spectrum"].data, n_eff, rtol=1e-15, atol=1e-15
    )
