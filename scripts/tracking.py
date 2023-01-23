"""
This script runs the tracking U-Net on data out of the segmentation U-Net.
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

import numpy as np

import delta
from delta.data import predictCompilefromseg_track, saveResult_track
from delta.model import unet_track
from delta.utilities import listfiles, read_image

### Parameters:

# Default config ("2D" or "MOTHERMACHINE")
config = delta.config.DEFAULT_CONFIG_2D
# Uncomment if you already have a config file
# config = delta.config.Config.read("/path/to/your/config.yml")

# Image sequence to segment: (replace with your own)
inputs_folder = delta.config._DELTA_DIR.parent / "tests/data/movie_2D_tif"
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
model = config.model("track")

# Process
while unprocessed:
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
        results[i:j] = model.predict(inputs[i:j], verbose=1, batch_size=1)

    # Paste results into masks of original imsize:
    if config.crop_windows:
        _results = np.zeros_like(results, shape=(len(results),) + imsize + (1,))
        for r, mask in enumerate(results):
            cb, fb = boxes[r]
            cb.patch(_results[r, :, :, 0], fb.crop(results[r, :, :, 0]))
        results = _results

    # Save (use the filenames list from the data compiler)
    saveResult_track(outputs_folder, results, files_list=seg_filenames)
