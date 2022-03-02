"""Solution axes tests."""

from ngspice import run
from ..util import dedent_multiline


def test_ac():
    """AC analysis axes."""
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
    assert list(solutions.keys()) == ["ac1"]
    ac1 = solutions["ac1"]
    assert ac1.xvector.name == "frequency"
    assert list(ac1.yvectors.keys()) == ["n1", "v1#branch"]


def test_dc():
    """DC analysis axes."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 0 1k
            V1 n1 0 DC 1
            .dc V1 0 3 1
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["dc1"]
    dc1 = solutions["dc1"]
    assert dc1.xvector.name == "v-sweep"
    assert list(dc1.yvectors.keys()) == ["n1", "v1#branch"]


def test_disto():
    """Distortion analysis axes."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 0 1k
            V1 n1 0 DC 1
            .disto dec 10 100 1000
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["disto1", "disto2"]
    # The first solution is apparently "AC values of all node voltages and branch currents at the
    # harmonic frequencies 2F_1" (manual 15.3.3).
    disto1 = solutions["disto1"]
    assert disto1.xvector.name == "frequency"
    assert list(disto1.yvectors.keys()) == ["n1", "v1#branch"]
    # The second solution is apparently "AC values of all node voltages and branch currents vs. the
    # input frequency F_1" (manual 15.3.3).
    disto2 = solutions["disto2"]
    assert disto2.xvector.name == "frequency"
    assert list(disto2.yvectors.keys()) == ["n1", "v1#branch"]


def test_noise():
    """Noise analysis axes."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 n2 1k
            R2 n2 0 2k
            V1 n1 0 DC 1 AC 1
            .noise v(n1) V1 dec 10 100 1000
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["noise1", "noise2"]
    # The first solution is the spectrum.
    noise1 = solutions["noise1"]
    assert noise1.xvector.name == "frequency"
    assert list(noise1.yvectors.keys()) == ["inoise_spectrum", "onoise_spectrum"]
    # Second result is the integrated noise. There's no axis.
    noise2 = solutions["noise2"]
    assert noise2.xvector == None
    assert list(noise2.yvectors.keys()) == ["inoise_total", "onoise_total"]


def test_op():
    """Operating point analysis axes."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 0 1k
            V1 n1 0 DC 1
            .op
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["op1"]
    op1 = solutions["op1"]
    assert op1.xvector is None
    assert list(op1.yvectors.keys()) == ["n1", "v1#branch"]


def test_pz():
    """Pole-zero analysis axes."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 n2 1k
            C1 n2 0 1u
            R2 n2 n3 1k
            C2 n3 0 1u
            R3 n3 n4 1k
            C3 n4 0 1u
            V1 n1 0 DC 1
            .pz n1 n2 n3 n4 vol pz
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["pz1"]
    pz1 = solutions["pz1"]
    assert pz1.xvector is None
    assert list(pz1.yvectors.keys()) == ["pole(1)", "pole(2)", "zero(1)"]


def test_sens_dc():
    """Sensitivity analysis axes (DC)."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 0 1k
            V1 n1 0 DC 1
            .sens V(n1, n2)
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["sens1"]
    sens1 = solutions["sens1"]
    assert sens1.xvector is None
    assert list(sens1.yvectors.keys()) == [
        "r1",
        "r1:bv_max",
        "r1:ef",
        "r1:lf",
        "r1:wf",
        "r1_bv_max",
        "r1_l",
        "r1_m",
        "r1_scale",
        "r1_w",
        "v1"
    ]


def test_sens_ac():
    """Sensitivity analysis axes (AC)."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 0 1k
            V1 n1 0 DC 1
            .sens V(n1) AC dec 10 100 1000
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["sens1"]
    sens1 = solutions["sens1"]
    assert sens1.xvector.name == "frequency"
    assert list(sens1.yvectors.keys()) == [
        "r1",
        "r1:bv_max",
        "r1:ef",
        "r1:lf",
        "r1:wf",
        "r1_ac",
        "r1_bv_max",
        "r1_l",
        "r1_m",
        "r1_scale",
        "r1_w",
        "v1",
        "v1_acmag"
    ]


def test_tf():
    """Transfer function analysis axes."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 n2 1k
            R2 n2 0 1k
            V1 n1 0 DC 1
            .tf v(n2) V1
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["tf1"]
    tf1 = solutions["tf1"]
    # Transfer function analysis is done at DC.
    assert tf1.xvector is None
    assert list(tf1.yvectors.keys()) == [
        "Transfer_function", "output_impedance_at_V(n2)", "v1#Input_impedance"
    ]


def test_tran():
    """Transient analysis axes."""
    solutions = run(
        dedent_multiline(
            """
            Test simulation
            R1 n1 0 1k
            V1 n1 0 DC 3.5
            .tran 1 10
            .end
            """
        )
    )
    assert list(solutions.keys()) == ["tran1"]
    tran1 = solutions["tran1"]
    assert tran1.xvector.name == "time"
    assert list(tran1.yvectors.keys()) == ["n1", "v1#branch"]
