#!/usr/bin/env python3
"""Convert DeLTA standalone segmentation/tracking outputs to CTC result format.

Expected sequence layout:
    SEQUENCE/
        t000.tif
        t001.tif
        ...
        segmentation/
            t000.tif          # binary DeLTA masks
            t001.tif
            ...
        tracking/
            t000Cell000001.tif  # predicted correspondence into t001
            t000Cell000002.tif
            ...

Output:
    OUTPUT_RES/
        mask000.tif
        mask001.tif
        ...
        res_track.txt

The converter labels binary segmentations into frame instances, then uses each
DeLTA per-cell prediction mask to associate a source instance in frame t with
one or two instances in frame t+1. Two accepted children are treated as a cell
division and receive new track IDs whose parent is the source track.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile


FRAME_RE = re.compile(r"^t(?P<frame>\d+)$")
TRACK_RE = re.compile(r"^t(?P<frame>\d+)Cell(?P<cell>\d+)$")


@dataclass
class TrackRecord:
    begin: int
    end: int
    parent: int = 0


def read_2d(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image at {path}, got {arr.shape}")
    return arr


def label_binary_mask(path: Path) -> np.ndarray:
    arr = read_2d(path)
    binary = (arr > 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    return labels.astype(np.int32)


def threshold_prediction(path: Path, threshold: float) -> np.ndarray:
    arr = read_2d(path)

    if np.issubdtype(arr.dtype, np.floating):
        return arr > threshold

    # Saved DeLTA masks are commonly 0/255. Treat threshold <= 1 as a
    # probability-style threshold and scale it for integer images.
    if threshold <= 1.0 and arr.max(initial=0) > 1:
        return arr > (threshold * float(np.iinfo(arr.dtype).max))

    return arr > threshold


def frame_files(seg_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(seg_dir.glob("t*.tif*")):
        match = FRAME_RE.match(path.stem)
        if match:
            result[int(match.group("frame"))] = path
    return result


def prediction_files(tracking_dir: Path) -> dict[tuple[int, int], Path]:
    result: dict[tuple[int, int], Path] = {}
    for path in sorted(tracking_dir.glob("t*Cell*.tif*")):
        match = TRACK_RE.match(path.stem)
        if match:
            key = (int(match.group("frame")), int(match.group("cell")))
            result[key] = path
    return result


def overlap_scores(prediction: np.ndarray, target_instances: np.ndarray) -> list[tuple[int, float, int]]:
    """Return (target_label, fraction_of_target_covered, overlap_pixels)."""
    if prediction.shape != target_instances.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target mask "
            f"shape {target_instances.shape}"
        )

    target_ids = np.unique(target_instances[prediction])
    target_ids = target_ids[target_ids > 0]
    scores: list[tuple[int, float, int]] = []

    for target_id in target_ids:
        target = target_instances == target_id
        overlap = int(np.count_nonzero(prediction & target))
        target_area = int(np.count_nonzero(target))
        fraction = overlap / target_area if target_area else 0.0
        scores.append((int(target_id), fraction, overlap))

    scores.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return scores


def convert(
    sequence_dir: Path,
    output_dir: Path,
    *,
    segmentation_dir: Path | None,
    tracking_dir: Path | None,
    prediction_threshold: float,
    min_target_coverage: float,
    division_coverage: float,
    min_overlap_pixels: int,
) -> None:
    segmentation_dir = segmentation_dir or sequence_dir / "segmentation"
    tracking_dir = tracking_dir or sequence_dir / "tracking"

    if not segmentation_dir.is_dir():
        raise FileNotFoundError(f"Segmentation directory not found: {segmentation_dir}")
    if not tracking_dir.is_dir():
        raise FileNotFoundError(f"Tracking directory not found: {tracking_dir}")

    seg_files = frame_files(segmentation_dir)
    pred_files = prediction_files(tracking_dir)
    if not seg_files:
        raise RuntimeError(f"No tXXX.tif segmentation masks found in {segmentation_dir}")

    frames = sorted(seg_files)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_instances = {frame: label_binary_mask(seg_files[frame]) for frame in frames}
    frame_track_ids: dict[int, np.ndarray] = {}
    tracks: dict[int, TrackRecord] = {}
    next_track_id = 1

    # Initialise every instance in the first frame as a new track.
    first = frames[0]
    first_instances = frame_instances[first]
    first_track_mask = np.zeros_like(first_instances, dtype=np.uint16)
    source_label_to_track: dict[int, int] = {}

    for label_id in np.unique(first_instances):
        if label_id == 0:
            continue
        track_id = next_track_id
        next_track_id += 1
        source_label_to_track[int(label_id)] = track_id
        tracks[track_id] = TrackRecord(begin=first, end=first, parent=0)
        first_track_mask[first_instances == label_id] = track_id

    frame_track_ids[first] = first_track_mask

    for prev_frame, frame in zip(frames[:-1], frames[1:]):
        prev_instances = frame_instances[prev_frame]
        target_instances = frame_instances[frame]
        target_track_mask = np.zeros_like(target_instances, dtype=np.uint16)

        proposals: dict[int, list[tuple[float, int, int]]] = {}
        # target_label -> [(coverage, overlap_pixels, source_label), ...]

        for source_label in np.unique(prev_instances):
            if source_label == 0:
                continue

            prediction_path = pred_files.get((prev_frame, int(source_label)))
            if prediction_path is None:
                continue

            prediction = threshold_prediction(prediction_path, prediction_threshold)
            for target_label, coverage, overlap in overlap_scores(prediction, target_instances):
                if coverage >= min_target_coverage and overlap >= min_overlap_pixels:
                    proposals.setdefault(target_label, []).append(
                        (coverage, overlap, int(source_label))
                    )

        # Resolve each target to its strongest source.
        source_to_targets: dict[int, list[tuple[int, float, int]]] = {}
        for target_label, candidates in proposals.items():
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            coverage, overlap, source_label = candidates[0]
            source_to_targets.setdefault(source_label, []).append(
                (target_label, coverage, overlap)
            )

        assigned_targets: set[int] = set()
        next_source_label_to_track: dict[int, int] = {}

        for source_label, matches in source_to_targets.items():
            source_track = source_label_to_track.get(source_label)
            if source_track is None:
                continue

            matches.sort(key=lambda item: (item[1], item[2]), reverse=True)
            strong = [m for m in matches if m[1] >= division_coverage]

            if len(strong) >= 2:
                # Division: parent ends in prev_frame; two strongest daughters
                # start in the current frame with the parent recorded.
                for target_label, _, _ in strong[:2]:
                    daughter_track = next_track_id
                    next_track_id += 1
                    tracks[daughter_track] = TrackRecord(
                        begin=frame, end=frame, parent=source_track
                    )
                    target_track_mask[target_instances == target_label] = daughter_track
                    next_source_label_to_track[target_label] = daughter_track
                    assigned_targets.add(target_label)
            else:
                # Normal continuation: strongest target keeps the same track ID.
                target_label, _, _ = matches[0]
                tracks[source_track].end = frame
                target_track_mask[target_instances == target_label] = source_track
                next_source_label_to_track[target_label] = source_track
                assigned_targets.add(target_label)

        # Any unmatched target is a new independent track.
        for target_label in np.unique(target_instances):
            target_label = int(target_label)
            if target_label == 0 or target_label in assigned_targets:
                continue

            track_id = next_track_id
            next_track_id += 1
            tracks[track_id] = TrackRecord(begin=frame, end=frame, parent=0)
            target_track_mask[target_instances == target_label] = track_id
            next_source_label_to_track[target_label] = track_id

        frame_track_ids[frame] = target_track_mask
        source_label_to_track = next_source_label_to_track

    digits = max(3, max(len(str(frame)) for frame in frames))
    for frame in frames:
        tifffile.imwrite(
            output_dir / f"mask{frame:0{digits}d}.tif",
            frame_track_ids[frame].astype(np.uint16),
            photometric="minisblack",
        )

    with open(output_dir / "res_track.txt", "w", encoding="utf-8") as handle:
        for track_id in sorted(tracks):
            record = tracks[track_id]
            handle.write(
                f"{track_id} {record.begin} {record.end} {record.parent}\n"
            )

    print(f"Wrote {len(frames)} mask files to {output_dir}")
    print(f"Wrote {len(tracks)} tracks to {output_dir / 'res_track.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeLTA per-cell tracking outputs into CTC mask*.tif and res_track.txt"
    )
    parser.add_argument("sequence_dir", type=Path, help="CTC sequence directory, e.g. test/05")
    parser.add_argument("output_dir", type=Path, help="CTC result directory, e.g. test/05_RES")
    parser.add_argument("--segmentation-dir", type=Path, default=None)
    parser.add_argument("--tracking-dir", type=Path, default=None)
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.5,
        help="Threshold for DeLTA per-cell correspondence masks (default: 0.5)",
    )
    parser.add_argument(
        "--min-target-coverage",
        type=float,
        default=0.15,
        help="Minimum fraction of a target instance covered to accept a link (default: 0.15)",
    )
    parser.add_argument(
        "--division-coverage",
        type=float,
        default=0.35,
        help="Coverage required for each daughter in a division (default: 0.35)",
    )
    parser.add_argument(
        "--min-overlap-pixels",
        type=int,
        default=3,
        help="Minimum overlapping pixels required for a link (default: 3)",
    )
    args = parser.parse_args()

    convert(
        args.sequence_dir,
        args.output_dir,
        segmentation_dir=args.segmentation_dir,
        tracking_dir=args.tracking_dir,
        prediction_threshold=args.prediction_threshold,
        min_target_coverage=args.min_target_coverage,
        division_coverage=args.division_coverage,
        min_overlap_pixels=args.min_overlap_pixels,
    )


if __name__ == "__main__":
    main()