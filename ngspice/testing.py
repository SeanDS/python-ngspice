"""Testing tools."""

from itertools import zip_longest
import numpy as np
from .data import Solution, Vector


def assert_solutions_equal(solution1, solution2):
    if not isinstance(solution1, Solution) or not isinstance(solution2, Solution):
        raise NotImplementedError(
            f"Can only compare solution objects (got {type(solution1)} and {type(solution2)})"
        )

    assert solution1.name == solution2.name
    assert solution1.type == solution2.type

    veciter = zip_longest(solution1.vectors.items(), solution2.vectors.items())
    for (name1, vector1), (name2, vector2) in veciter:
        assert name1 == name2
        assert_vectors_equal(vector1, vector2)


def assert_vectors_equal(vector1, vector2):
    if not isinstance(vector1, Vector) or not isinstance(vector2, Vector):
        raise NotImplementedError(
            f"Can only compare vector objects (got {type(vector1)} and {type(vector2)})"
        )

    assert vector1.name == vector2.name
    np.testing.assert_almost_equal(vector1.data, vector2.data)
