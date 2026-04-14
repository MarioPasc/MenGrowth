"""Custom exceptions for the SynthSeg pipeline."""

from __future__ import annotations


class SynthSegError(Exception):
    """Base class for errors raised by the SynthSeg pipeline."""


class SynthSegConfigError(SynthSegError):
    """Raised when SynthSeg configuration is invalid."""


class SynthSegInputError(SynthSegError):
    """Raised when an expected input file is missing or malformed."""


class SynthSegRuntimeError(SynthSegError):
    """Raised when the SynthSeg subprocess fails in an unrecoverable way."""


class SynthSegOutputError(SynthSegError):
    """Raised when SynthSeg output does not satisfy post-run validation."""
