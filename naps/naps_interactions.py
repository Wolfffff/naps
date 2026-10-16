#!/usr/bin/env python
"""``naps-interactions``: assign interactions between tracked animals from an analysis h5."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from naps.utils.cli import ConfirmFile, add_verbosity_arguments, log_verbosity
from naps.utils.tracking import TrackingArray

logger = logging.getLogger("NAPS Logger")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``naps-interactions`` argument parser."""
    interaction_parser = argparse.ArgumentParser(
        prog="naps-interactions",
        description="NAPS -- assign interactions between tracked animals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = interaction_parser.add_argument_group("input files")
    inputs.add_argument(
        "--h5-file",
        help="Defines the input h5 file",
        type=str,
        action=ConfirmFile,
        required=True,
    )
    inputs.add_argument(
        "--yaml-model",
        help="Defines the YAML model file",
        type=str,
        action=ConfirmFile,
        required=True,
    )

    general = interaction_parser.add_argument_group("general options")
    general.add_argument(
        "--frame-start",
        help="Defines the frame to start interaction assignment",
        type=int,
        default=0,
    )
    general.add_argument(
        "--frame-end",
        help="Defines the frame to end interaction assignment (exclusive; -1 means all frames)",
        type=int,
        default=-1,
    )
    general.add_argument("--threads", help="Number of threads to use", type=int, default=1)

    outputs = interaction_parser.add_argument_group("output options")
    outputs.add_argument(
        "--out-file",
        help="Defines the output tsv file. Defaults to: out.tsv",
        type=str,
        default="out.tsv",
    )

    add_verbosity_arguments(interaction_parser)
    return interaction_parser


def interactionParser(  # noqa: N802 (public camelCase API)
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the ``naps-interactions`` arguments.

    Args:
        argv: Arguments to parse (default: ``sys.argv[1:]``).

    Raises:
        FileNotFoundError: If ``--h5-file`` or ``--yaml-model`` does not exist.
    """
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``naps-interactions``.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        The process exit status (0).
    """
    interaction_args = interactionParser(argv)

    with log_verbosity(interaction_args.verbose, interaction_args.quiet):
        logger.info("Loading the tracking array...")
        t0 = time.perf_counter()

        # Create the tracking array from the analysis h5 and the yaml model
        interactions = TrackingArray.fromH5(
            interaction_args.h5_file,
            model_yaml=interaction_args.yaml_model,
            threads=interaction_args.threads,
        )
        logger.info("Done loading in %.2f seconds.", time.perf_counter() - t0)

        logger.info("Starting interaction assignment...")
        t0 = time.perf_counter()

        interactions.assign(
            interaction_args.out_file,
            frame_start=interaction_args.frame_start,
            frame_end=interaction_args.frame_end,
        )
        logger.info("Done with assignment in %.2f seconds.", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
