"""Test utilities."""

from textwrap import dedent


def dedent_multiline(text):
    """Dedent multiline text, stripping preceding and succeeding newlines.

    This is useful for specifying multiline strings in tests without having to compensate for the
    indentation.
    """
    return dedent(text).strip()
