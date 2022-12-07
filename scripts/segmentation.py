"""
This script runs the segmentation U-Net
For mother machine data, it runs on images of cropped out and resized single
chambers as fed to it in Pipeline processing.

The images are processed by batches of 4096 to prevent memory issues.

@author: jblugagne
"""
import numpy as np

import delta.utilities as utils
from delta.data import postprocess, predictGenerator_seg, readreshape, saveResult_seg
from delta.model import unet_seg
from delta.utilities import cfg, listfiles

### Parameters:

# Configuration presets: ("2D" or "mothermachine")
cfg.load_config(presets="2D")

# Image sequence to segment: (replace with your own)
inputs_folder = cfg.eval_movie  # or cfg.eval_movie + "/cropped_rois" for mother machine

### /Parameters


# Outputs folder:
outputs_folder = inputs_folder / "segmentation"
outputs_folder.mkdir(exist_ok=True)

# List files in inputs folder:
unprocessed = listfiles(inputs_folder, {".tif", ".png"})

# Load up model:
model = unet_seg(input_size=cfg.target_size_seg + (1,))
model.load_weights(cfg.model_file_seg)

# Process
while unprocessed:
    # Pop out filenames
    ps = min(4096, len(unprocessed))  # 4096 at a time
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]

    # Input data generator:
    predGene = predictGenerator_seg(
        inputs_folder,
        files_list=to_process,
        target_size=cfg.target_size_seg,
        crop_windows=cfg.crop_windows,
    )

    # mother machine: Don't crop images into windows
    if not cfg.crop_windows:
        # Predictions:
        results = model.predict(predGene, verbose=1)[:, :, :, 0]

    # 2D: Cut into overlapping windows
    else:
        img = readreshape(
            inputs_folder / to_process[0],
            target_size=cfg.target_size_seg,
            method="pad",
        )
        # Create array to store predictions
        results = np.zeros((len(to_process), img.shape[0], img.shape[1], 1))
        # Crop, segment, stitch and store predictions in results
        for i in range(len(to_process)):
            # Crop each frame into overlapping windows:
            windows, loc_y, loc_x = utils.create_windows(
                next(predGene)[0, :, :], target_size=cfg.target_size_seg
            )
            # We have to play around with tensor dimensions to conform to
            # tensorflow's functions:
            windows = windows[:, :, :, np.newaxis]
            # Predictions:
            pred = model.predict(windows, verbose=1, steps=windows.shape[0])
            # Stich prediction frames back together:
            pred = utils.stitch_pic(pred[:, :, :, 0], loc_y, loc_x)
            pred = pred[np.newaxis, :, :, np.newaxis]  # Mess around with dims

            results[i] = pred

    # Post process results (binarize + light morphology-based cleaning):
    results = postprocess(results, crop=cfg.crop_windows)

    # Save to disk:
    saveResult_seg(outputs_folder, results, files_list=to_process)
