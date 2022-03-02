"""Run function tests.

Note: tests of the behaviour in running simulations (e.g. error handling) should instead go in
`test_session.py`.
"""

from io import StringIO
from itertools import zip_longest
import pytest
from ngspice import run, run_file, SimulationError
from ngspice.testing import assert_solutions_equal
from .util import dedent_multiline


scripts = pytest.mark.parametrize(
    "script",
    (
        dedent_multiline(
            """
            Test script
            R1 n1 n2 1k
            R2 n2 0 10k
            V1 n1 0 DC 1
            .op
            .end
            """
        ),
    )
)


@scripts
def test_run(script):
    """Run netlist from string."""
    assert run(script)


@scripts
def test_run_file__str(script_path, script):
    """Run netlist from path string."""
    with script_path.open("w") as fobj:
        fobj.write(script)
    assert run_file(str(script_path))


@scripts
def test_run_file__path(script_path, script):
    """Run netlist from :class:`pathlib.Path`."""
    with script_path.open("w") as fobj:
        fobj.write(script)
    assert run_file(script_path)


@scripts
def test_run_file__stringio(script):
    """Run netlist from :class:`io.StringIO`."""
    assert run_file(StringIO(script))


@scripts
def test_run_results_same_as_run_file_results(script_path, script):
    """Check runs via string and file give same results when the script is the same."""
    with script_path.open("w") as fobj:
        fobj.write(script)

    sols1 = run(script)
    sols2 = run_file(str(script_path))

    for (name1, sol1), (name2, sol2) in zip_longest(sols1.items(), sols2.items()):
        assert name1 == name2
        assert_solutions_equal(sol1, sol2)
