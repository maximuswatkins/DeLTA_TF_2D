"""Command-line interface of DeLTA."""

import argparse
from typing import List, Optional


def to_int_list(list_spec: str) -> List[int]:
    """
    Convert a list string specification ("0-2,4,7-10") to a list of integers.

    Parameters
    ----------
    list_spec : str
        List string specification.

    Returns
    -------
    int_list : List[int]
        List of integers.
    """
    int_list = []
    for group in list_spec.split(","):
        if "-" not in group:
            int_list.append(int(group))
        else:
            start, end = (int(x) for x in group.split("-"))
            for x in range(start, end + 1):
                int_list.append(x)
    return int_list


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse a list of arguments, typically given on the commandline.

    Parameters
    ----------
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
    parser.add_argument("-i", "--input", help="Input file or directory", required=True)
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory (by default `delta_results` inside the input directory)",
    )
    parser.add_argument(
        "--positions", help="Positions to process, ex.: 0-2,4,7-10 (default: all)"
    )
    parser.add_argument(
        "--nbframes", type=int, help="Number of frames to process (default: all)"
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

    parsed_args = parser.parse_args(args)
    if parsed_args.positions is not None:
        parsed_args.positions = to_int_list(parsed_args.positions)

    return parsed_args
