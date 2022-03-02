"""Session tests."""

import time
import pytest
from ngspice import Session, SimulationError
from ngspice.testing import dedent_multiline


@pytest.fixture
def session():
    return Session()


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


invalid_scripts = pytest.mark.parametrize(
    "script,exception_type",
    (
        pytest.param(
            dedent_multiline(
                """
                Test script
                R1 n10 n20 1k
                R2 n20 0 10k
                Q1 n10 0 DC 1
                """
            ),
            ValueError,
            id="no-end-control-statement"
        ),
        pytest.param(
            dedent_multiline(
                """
                Test script
                R1 n10 n20 1k
                A1
                .op
                .end
                """
            ),
            SimulationError,
            id="invalid-element"
        ),
    )
)


@scripts
def test_read(session, script):
    session.read(script)


@scripts
def test_read_file(session, script_path, script):
    with script_path.open("w+") as fobj:
        fobj.write(script)
        fobj.seek(0)
        session.read_file(fobj)


@scripts
def test_session_run(session, script):
    session.read(script)
    session.run()

    assert session.solutions()


@invalid_scripts
def test_invalid_script_read(session, script, exception_type):
    with pytest.raises(exception_type):
        session.read(script)


@invalid_scripts
def test_cannot_run_after_invalid_script_read_error(session, script, exception_type):
    """Test running after an invalid script read results in an error."""
    with pytest.raises(exception_type):
        session.read(script)

    with pytest.raises(SimulationError):
        session.run()


def test_can_run_new_script_after_invalid_script_read_error(session):
    """Check that the session can be reused after an ngspice failure.

    This tests the C++ wrapper's DLL reloading capability.
    """
    with pytest.raises(SimulationError):
        # Script with invalid element, triggering an error.
        session.read(
            dedent_multiline(
                """
                Test script
                R1 n10 n20 1k
                A1
                .op
                .end
                """
            )
        )

    # Now run something that should work.
    session.read(
        dedent_multiline(
            """
            Test script
            R1 n1 n2 1k
            R2 n2 0 10k
            V1 n1 0 DC 1
            .op
            .end
            """
        )
    )
    session.run()
    assert session.solutions()
