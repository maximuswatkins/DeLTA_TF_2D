"""
Run the tracking U-Net on data out of the segmentation U-Net.

Please make sure you have run the segmentation script first.

Images are processed in batches of 512, although the number of actual samples
run through the tracking U-Net will depend on the number of cells in each
image.

Format of images needs to be Position-Number-Chamber-Number-Frame-Number.fext
or Position-Number-Frame-Number.fext

If there are chambers, then it is assumed to be run in the mothermachine.
If there are no chambers, then images will be cropped as done for DeLTA 2D
@author: jblugagne
"""

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

import delta
from delta.data import predictCompilefromseg_track, saveResult_track
from delta.model import unet_track
from delta.utilities import listfiles, read_image
from frame_timer import (
    FrameMetricsRecorder,
    count_instance_cells,
    count_binary_cells,
    find_gt_mask_for_frame,
    frame_name_from_tracking_stem,
)

import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import tensorflow as tf
tf.config.optimizer.set_jit(False)
tf.config.run_functions_eagerly(True)
### Parameters:

# Default config ("2D" or "MOTHERMACHINE")
config = delta.config.DEFAULT_CONFIG_2D
# Uncomment if you already have a config file
# config = delta.config.Config.read("/path/to/your/config.yml")

# Image sequence to segment: (replace with your own)
#inputs_folder = delta.config._DELTA_DIR.parent / "tests/data/movie_2D_tif"
#segmentation_folder = inputs_folder / "segmentation"

parser = argparse.ArgumentParser()
default_model_path = "/home/max/MaxWork/DeLTA_TF_New/models/My2D/track/my2d_track_model.keras"
parser.add_argument(
    "--inputs_folder",
    nargs="?",
    default=str(delta.config._DELTA_DIR.parent / "tests/data/movie_2D_tif"),
    help="Folder containing one image sequence.",
)
parser.add_argument(
    "--model-path",
    type=str,
    default=default_model_path,
    help="Path to the tracking model (.keras).",
)
parser.add_argument(
    "--model-tag",
    type=str,
    default=None,
    help="Optional model identifier used in metrics output path. Defaults to model filename stem.",
)
parser.add_argument(
    "--frame-metrics-csv",
    type=str,
    default=None,
    help="Path to write per-frame tracking metrics CSV (overrides model-specific default path).",
)
parser.add_argument(
    "--disable-frame-metrics",
    action="store_true",
    help="Disable per-frame metrics recording.",
)
parser.add_argument(
    "--gt-seg-dir",
    type=str,
    default=None,
    help="Optional CTC GT SEG directory override. If omitted, uses <seq>_GT/SEG when available.",
)
args = parser.parse_args()

inputs_folder = Path(args.inputs_folder)
segmentation_folder = inputs_folder / "segmentation"


### /Parameters

# Output folder:
outputs_folder = inputs_folder / "tracking"
outputs_folder.mkdir(exist_ok=True)

# List files to read
unprocessed = listfiles(inputs_folder, {".png", ".tif"})

# Get original image size:
imsize = read_image(unprocessed[0]).shape

# Load up model:
#model = config.model("track")
import tensorflow as tf

model_path = Path(args.model_path)
model = tf.keras.models.load_model(str(model_path), compile=False)


def infer_gt_seg_dir(sequence_dir: Path) -> Path | None:
    gt_seg_dir = sequence_dir.parent / f"{sequence_dir.name}_GT" / "SEG"
    if gt_seg_dir.is_dir():
        return gt_seg_dir
    return None


gt_seg_dir = Path(args.gt_seg_dir) if args.gt_seg_dir is not None else infer_gt_seg_dir(inputs_folder)
model_tag = args.model_tag if args.model_tag else model_path.stem
metrics_csv = (
    Path(args.frame_metrics_csv)
    if args.frame_metrics_csv is not None
    else outputs_folder / "metrics" / model_tag / "frame_metrics.csv"
)
write_metrics = not args.disable_frame_metrics
all_detected_counts = Counter()
all_elapsed_by_frame = Counter()

if write_metrics:
    print(f"Writing frame metrics to: {metrics_csv}")

# Process
while unprocessed:
    batch_start = time.perf_counter()
    # Pop out filenames
    ps = min(5, len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]

    print("\n###### Now processing: ")
    for file in to_process:
        print(file)

    # Get data:
    inputs, seg_filenames, boxes = predictCompilefromseg_track(
        inputs_folder,
        segmentation_folder,
        files_list=to_process,
        target_size=config.target_size_track,
        crop_windows=config.crop_windows,
    )

    print("Cells to track: %d" % len(inputs))

    # Predict (have to do it in batches otherwise run into memory issues):
    results = np.empty_like(inputs, shape=inputs.shape[:3] + (1,))
    for i in range(0, len(inputs), 32):
        j = min((len(inputs), i + 32))
        #results[i:j] = model.predict(inputs[i:j], verbose=1, batch_size=1)
        batch = tf.convert_to_tensor(inputs[i:j], dtype=tf.float32)
        results[i:j] = model(batch, training=False).numpy()
    # Paste results into masks of original imsize:
    if config.crop_windows:
        _results = np.zeros_like(results, shape=(len(results),) + imsize + (1,), dtype=np.float32,)
        for r, mask in enumerate(results):
            cb, fb = boxes[r]
            cb.patch(_results[r, :, :, 0], fb.crop(results[r, :, :, 0]))
        results = _results

    # Save (use the filenames list from the data compiler)
    saveResult_track(outputs_folder, results, files_list=seg_filenames)

    if write_metrics:
        elapsed = time.perf_counter() - batch_start

        # Record runtime for every raw frame in this batch.
        elapsed_per_input_frame = elapsed / len(to_process)
        for frame_path in to_process:
            all_elapsed_by_frame[frame_path.stem] += elapsed_per_input_frame

        # Count detected cells from DeLTA's per-cell output filenames.
        frame_cell_counts = Counter(
            frame_name_from_tracking_stem(seg_filename.stem)
            for seg_filename in seg_filenames
        )

        for frame_name, cell_count in frame_cell_counts.items():
            all_detected_counts[frame_name] += cell_count

if write_metrics:
    raw_frames = sorted(listfiles(inputs_folder, {".png", ".tif"}))

    with FrameMetricsRecorder(metrics_csv) as metrics_recorder:
        for frame_path in raw_frames:
            frame_name = frame_path.stem

            gt_mask_path = find_gt_mask_for_frame(gt_seg_dir, frame_name)
            gt_cells = -1 if gt_mask_path is None else count_instance_cells(gt_mask_path)

            seg_mask_path = segmentation_folder / frame_path.name
            detected_cells = count_binary_cells(seg_mask_path) if seg_mask_path.exists() else 0

            elapsed_seconds = all_elapsed_by_frame.get(frame_name, 0.0)

            metrics_recorder.record_frame(
                frame_name=frame_name,
                elapsed_seconds=elapsed_seconds,
                gt_cells=gt_cells,
                detected_cells=detected_cells,
            )
