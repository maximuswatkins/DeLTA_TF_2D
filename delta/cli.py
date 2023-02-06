"""Command-line interface of DeLTA."""

import argparse
from typing import List, Optional


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse a list of arguments, typically given on the commandline.

    Arguments
    ---------
    args : Optional[List[str]]
        List of command-line arguments.

    Returns
    -------
    namespace : argparse.Namespace
        Parsed and typed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="delta",
        description="Deep Learning for Time-Lapse Analysis",
    )
    parser.add_argument(
        "presets",
        choices=["2D", "mothermachine"],
        help="Whether to run in 2D or mothermachine mode",
    )
    parser.add_argument(
        "--bioformats",
        action="store_true",
        help="Use bioformats to read the input file (for example for nd2 files)",
    )
    parser.add_argument("-i", "--input", help="Input file or directory", required=True)
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory (by default `delta_results` inside the input directory)",
    )
    parser.add_argument(
        "--prototype",
        help=(
            "Format of the image filenames, "
            'for example "Position%%02d_Channel%%02d_Frame%%06d.tif"'
        ),
    )
    parser.add_argument(
        "--order",
        help=(
            "Combination of some or all of the letters 'p', 'c' and 't', "
            "respectively for Position, Channel and Timepoint, "
            'in the order that they appear in the prototype (default "pct")'
        ),
        default="pct",
    )
    parser.add_argument(
        "--indexing",
        help="Whether position, channel and timepoint start with 0 or 1 (default: 1)",
        default=1,
    )

    return parser.parse_args(args)
