"""Utilities."""

from os import fspath, PathLike
from pathlib import Path
from contextlib import closing, nullcontext


def file_lines(filename):
    """Get file lines regardless of whether a string or a file object is provided.

    Parameters
    ----------
    filename : file, :class:`str`, or :class:`pathlib.Path`
        File or filename to read.

    Returns
    -------
    :class:`list`
        The lines in the file.

    Raises
    ------
    TypeError
        If `filename` is not a type that can be read.
    """
    if isinstance(filename, PathLike):
        filename = fspath(filename)

    if isinstance(filename, str):
        fid = open(filename, "r")
        fid_context = closing(fid)
    else:
        fid = filename
        fid_context = nullcontext(fid)

    with fid_context:
        try:
            return fid.readlines()
        except AttributeError as e:
            raise TypeError(
                f"filename must be a string or a file handle (got {type(filename)})."
            ) from e
