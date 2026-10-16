"""Helpers shared by the NAPS command-line entry points (argparse actions and log verbosity)."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: Names of the loggers whose level the ``--verbose`` / ``--quiet`` flags control.
PACKAGE_LOGGERS = ("NAPS Logger", "naps")


class ConfirmFile(argparse.Action):
    """Store the value of an option after confirming that it names an existing file.

    Raises:
        FileNotFoundError: If the file does not exist (an :class:`OSError`, as before).
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        """Validate ``values`` and store it on ``namespace``."""
        value = str(values)
        if not Path(value).is_file():
            raise FileNotFoundError(f"{value} not found")
        setattr(namespace, self.dest, value)


def add_verbosity_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive ``--verbose`` / ``--quiet`` flags (default: INFO, as before)."""
    group = parser.add_argument_group("logging")
    exclusive = group.add_mutually_exclusive_group()
    exclusive.add_argument(
        "-v", "--verbose", action="store_true", help="Also log debugging messages."
    )
    exclusive.add_argument(
        "-q", "--quiet", action="store_true", help="Only log warnings and errors."
    )


@contextmanager
def log_verbosity(verbose: bool = False, quiet: bool = False) -> Generator[None]:
    """Temporarily set the NAPS log level for ``--verbose`` / ``--quiet`` (no flag: unchanged).

    The previous levels are restored on exit, so repeated in-process calls (tests, notebooks) do not
    leak the setting.
    """
    loggers = [logging.getLogger(name) for name in PACKAGE_LOGGERS]
    previous = [lg.level for lg in loggers]
    level = logging.DEBUG if verbose else logging.WARNING if quiet else None
    if level is not None:
        for lg in loggers:
            lg.setLevel(level)
    try:
        yield
    finally:
        for lg, old in zip(loggers, previous, strict=True):
            lg.setLevel(old)
