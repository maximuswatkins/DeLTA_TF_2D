"""
Run the segmentation U-Net.

For mother machine data, it runs on images of cropped out and resized single
chambers as fed to it in Pipeline processing.

The images are processed by batches of 4096 to prevent memory issues.

@author: jblugagne
"""
from pathlib import Path
import argparse
import time  # add this

import numpy as np

import delta
import delta.utilities as utils
from delta.data import postprocess, predictGenerator_seg, readreshape, saveResult_seg
from delta.model import unet_seg
from delta.utilities import listfiles

from frame_timer import (
    FrameMetricsRecorder,
    count_binary_cells,
    find_gt_mask_for_frame,
)

### Parameters:

# Default config ("2D" or "MOTHERMACHINE")
config = delta.config.DEFAULT_CONFIG_2D
config.crop_windows = True
config.target_size_seg = (256, 256)
# Uncomment if you already have a config file
# config = delta.config.Config.read("/path/to/your/config.yml")

# Image sequence to segment: (replace with your own)
# inputs_folder = delta.config._DELTA_DIR.parent / "tests/data/movie_2D_tif/"

# Image sequence to segment (argparse version):
parser = argparse.ArgumentParser()
parser.add_argument(
    "inputs_folder",
    nargs="?",
    default=str(delta.config._DELTA_DIR.parent / "tests/data/movie_2D_tif"),
    help="Folder containing one image sequence.",
)
parser.add_argument(
    "--frame-metrics-csv",
    type=str,
    default=None,
    help="Path to write per-frame segmentation metrics CSV.",
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

def infer_gt_seg_dir(sequence_dir: Path) -> Path | None:
    gt_seg_dir = sequence_dir.parent / f"{sequence_dir.name}_GT" / "SEG"
    return gt_seg_dir if gt_seg_dir.is_dir() else None

outputs_folder = inputs_folder / "segmentation"
outputs_folder.mkdir(exist_ok=True)

gt_seg_dir = Path(args.gt_seg_dir) if args.gt_seg_dir is not None else infer_gt_seg_dir(inputs_folder)

metrics_csv = (
    Path(args.frame_metrics_csv)
    if args.frame_metrics_csv is not None
    else outputs_folder / "metrics" / "frame_metrics.csv"
)

write_metrics = not args.disable_frame_metrics
if write_metrics:
    print(f"Writing frame metrics to: {metrics_csv}")
    metrics_recorder = FrameMetricsRecorder(metrics_csv)
else:
    metrics_recorder = None

### /Parameters

# List files in inputs folder:
unprocessed = listfiles(inputs_folder, {".tif", ".png"})

# Load up model:
# model = config.model("seg")
import tensorflow as tf
model = tf.keras.models.load_model(
    "/home/max/MaxWork/DeLTA_TF_New/models/My2D/seg/my2d_seg_model.keras",
    compile=False,
)

# Process
while unprocessed:
    batch_start = time.perf_counter()

    ps = min(4096, len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]

    predGene = predictGenerator_seg(
        inputs_folder,
        files_list=to_process,
        target_size=config.target_size_seg,
        crop_windows=config.crop_windows,
    )

    if not config.crop_windows:
        predGene_tf = ((x,) for x in predGene)
        results = model.predict(predGene_tf, verbose=1)[:, :, :, 0]
    else:
        img = readreshape(
            inputs_folder / to_process[0],
            target_size=config.target_size_seg,
            method="pad",
        )
        results = np.zeros((len(to_process), img.shape[0], img.shape[1], 1))

        for i in range(len(to_process)):
            windows, loc_y, loc_x = utils.create_windows(
                next(predGene)[0, :, :], target_size=config.target_size_seg
            )
            windows = windows[:, :, :, np.newaxis]
            pred = model.predict(windows, verbose=1, steps=windows.shape[0])
            pred = utils.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
            pred = pred[np.newaxis, :, :, np.newaxis]
            results[i] = pred

    results = postprocess(results, crop=config.crop_windows)

    if write_metrics:
        elapsed = time.perf_counter() - batch_start
        elapsed_per_frame = elapsed / len(to_process)

        for i, frame_path in enumerate(to_process):
            frame_name = frame_path.stem

            gt_mask_path = find_gt_mask_for_frame(gt_seg_dir, frame_name)
            gt_cells = -1 if gt_mask_path is None else count_binary_cells(gt_mask_path)

            detected_cells = count_binary_cells(results[i])

            metrics_recorder.record_frame(
                frame_name=frame_name,
                elapsed_seconds=elapsed_per_frame,
                gt_cells=gt_cells,
                detected_cells=detected_cells,
            )

    saveResult_seg(outputs_folder, results, files_list=to_process)

if metrics_recorder is not None:
    metrics_recorder.close()
