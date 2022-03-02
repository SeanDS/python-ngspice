"""Shared test fixtures."""

import pytest


@pytest.fixture
def script_path(tmp_path):
    return tmp_path / "script.net"
