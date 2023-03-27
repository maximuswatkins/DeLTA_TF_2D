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


def to_interval(interval_spec: str) -> slice:
    """
    Convert an interval string specification ("35-150") to a slice.

    Parameters
    ----------
    interval_spec: str
        Interval string specification.

    Returns
    -------
    interval : slice
        Interval.
    """
    if "-" not in interval_spec:
        only_value = int(interval_spec)
        return slice(only_value, only_value + 1)
    start, end = interval_spec.split("-")
    if not start:
        return slice(int(end) + 1)
    if not end:
        return slice(int(start), None)
    return slice(int(start), int(end) + 1)


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
        "-c",
        "--config",
        help="Configuration file.  Can be either `2D`, `mothermachine`, "
        "or a path to a previously saved custom config file.",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input file or directory. Can include `{p}`, `{c}` and `{t}` as "
        "placeholders for the position, channel, and frame number.  "
        "Example for micromanager: "
        "`/path/to/folder/Pos{p}/img_channel{c}_position{p}_time{t}_z000.tif`",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory (by default `delta_results` inside the input directory)",
    )
    parser.add_argument(
        "--positions", help="Positions to process, ex.: 0-2,4,7-10 (default: all)"
    )
    parser.add_argument(
        "--frames",
        type=str,
        help="Range of frames to process, ex.: -150 (up to frame 150), "
        "15- (from frame 15), 15-30 (frames 15 to 30), 40 (just frame 40) (default: all)",
    )

    parsed_args = parser.parse_args(args)
    if parsed_args.positions is not None:
        parsed_args.positions = to_int_list(parsed_args.positions)
    if parsed_args.frames is not None:
        parsed_args.frames = to_interval(parsed_args.frames)

    return parsed_args
