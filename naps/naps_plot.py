#!/usr/bin/env python
"""``naps-plot``: draw interaction models onto the video of a SLEAP file."""

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
    """Build the ``naps-plot`` argument parser."""
    plot_parser = argparse.ArgumentParser(
        prog="naps-plot",
        description="NAPS -- plot interaction models onto a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = plot_parser.add_argument_group("input files")
    inputs.add_argument(
        "--sleap-file",
        help="Defines the input SLEAP file",
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
    inputs.add_argument(
        "--interactions-file",
        help="Defines the interactions file",
        type=str,
        required=True,
    )
    inputs.add_argument(
        "--limit-interactions",
        help="Limits the interactions to plot",
        type=str,
        nargs="+",
    )
    inputs.add_argument(
        "--video-file", help="Defines the input video file", type=str, required=True
    )
    inputs.add_argument(
        "--models-to-plot",
        help="Defines the models to plot",
        type=str,
        nargs="+",
        required=True,
    )

    general = plot_parser.add_argument_group("general options")
    general.add_argument(
        "--frame-start",
        help="Defines the first frame to draw",
        type=int,
        default=0,
    )
    general.add_argument(
        "--frame-end",
        help="Defines the last frame to draw (inclusive; -1 means the last frame)",
        type=int,
        default=-1,
    )
    general.add_argument(
        "--plot-color",
        help="Defines the color of the models (currently not forwarded to the plot)",
        type=str,
        default="red",
    )
    general.add_argument(
        "--line-width",
        help="Defines the line width of the models (currently not forwarded to the plot)",
        type=int,
        default=2,
    )

    outputs = plot_parser.add_argument_group("output options")
    outputs.add_argument(
        "--out-file", help="Defines the output video file", type=str, required=True
    )

    add_verbosity_arguments(plot_parser)
    return plot_parser


def plotParser(  # noqa: N802 (public camelCase API)
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the ``naps-plot`` arguments.

    Args:
        argv: Arguments to parse (default: ``sys.argv[1:]``).

    Raises:
        FileNotFoundError: If ``--sleap-file`` or ``--yaml-model`` does not exist.
    """
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``naps-plot``.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        The process exit status (0).
    """
    plot_args = plotParser(argv)

    with log_verbosity(plot_args.verbose, plot_args.quiet):
        logger.info("Loading the tracking array...")
        t0 = time.perf_counter()

        # Create the tracking array object using a SLEAP file and the yaml model
        tracking_array = TrackingArray.fromSLEAP(
            plot_args.sleap_file, model_yaml=plot_args.yaml_model
        )
        logger.info("Done loading in %.2f seconds.", time.perf_counter() - t0)

        logger.info("Starting plotting...")
        logger.info("Plotting frames %s to %s", plot_args.frame_start, plot_args.frame_end)
        t0 = time.perf_counter()

        tracking_array.plot(
            plot_args.interactions_file,
            plot_args.models_to_plot,
            plot_args.video_file,
            plot_args.out_file,
            frame_start=plot_args.frame_start,
            frame_end=plot_args.frame_end,
            limit_interactions=plot_args.limit_interactions,
        )
        logger.info("Done plotting in %.2f seconds.", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
