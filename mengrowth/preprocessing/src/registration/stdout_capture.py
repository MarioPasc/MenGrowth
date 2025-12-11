"""Utilities for capturing stdout during registration operations."""

import sys
import io
from contextlib import contextmanager
from typing import Generator


@contextmanager
def capture_stdout() -> Generator[io.StringIO, None, None]:
    """Context manager to capture stdout.

    This is useful for capturing diagnostic output from ANTs registration
    operations that write directly to stdout.

    Yields:
        StringIO object containing captured stdout

    Example:
        with capture_stdout() as captured:
            ants.registration(...)
        output = captured.getvalue()
    """
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        yield captured
    finally:
        sys.stdout = old_stdout
